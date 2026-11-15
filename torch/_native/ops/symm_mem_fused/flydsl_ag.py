# mypy: allow-untyped-defs
"""FlyDSL fused all-gather + matmul (symm_mem::fused_all_gather_matmul) on gfx950.

One state per (K, N, TP, B layout, rows per rank, group, device), running one kernel (flydsl_kernels.agmm_gfx950):
blocks [0, KC) pull the peers' shards into the gathered A chunk by chunk, then every block computes its tile, peer
tiles waiting on their chunk flag. One call, all on the current stream: bump the device epoch, stage the shard into
half (epoch & 1) of the symmetric buffer and into this rank's rows of the gathered A, flush (one L2 write-back per
XCD, which the peers' reads of the staged half need), the symmetric-memory barrier (every rank's staging is in, and
every peer is done reading this half from two calls ago), then the fused kernel.

A state is built at its first call: a collective symmetric-memory rendezvous and a FlyDSL compile (which also runs the
kernel once, with no copy blocks and the flags far past any target), so every rank must make its calls in the same
order, and the first call of a state must not be under CUDA graph capture (later calls may be: all state lives on the
device). The gathered A is written on every call, return_A or not. A chunk flag still missing after
flydsl_common.wait_ms() fails the launch with a [FATAL] line instead of hanging; the barrier times out the same way
from a state's second call on.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import triton
import triton.language as tl

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch._native.instrumentation import instrument_triton_kernel
from torch.distributed.distributed_c10d import _resolve_process_group

from . import flydsl_kernels as K
from .flydsl_common import (
    arg,
    b_transposed,
    bump,
    CHUNK_ROWS,
    control_block,
    gemm_param,
    order_streams,
    SLOT,
    STAGE_BLOCK,
    wait_ms,
)


FLUSH_PROGRAMS = 64  # workgroups go round-robin over the 8 XCDs: 8 per XCD
# (K, N / TP, TP) -> copy blocks, the best measured at M 65536 (Llama 3.1 70B); else KC_DEFAULT.
KC = {
    (8192, 7168, 8): 28,
    (8192, 1280, 8): 28,
    (8192, 3584, 8): 28,
    (8192, 1024, 8): 28,
    (8192, 14336, 4): 12,
    (8192, 2560, 4): 15,
    (8192, 7168, 4): 12,
    (8192, 2048, 4): 15,
    (8192, 28672, 2): 6,
    (8192, 5120, 2): 8,
    (8192, 14336, 2): 6,
    (8192, 4096, 2): 8,
}
KC_DEFAULT = {8: 28, 4: 12, 2: 8}
_states: dict = {}


@instrument_triton_kernel("symm_mem::fused_all_gather_matmul")
@triton.jit
def _stage3(src, mine, own, ep, n, BUFS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask)
    # BUFS is a power of two; & keeps the half in range once the int32 epoch wraps (% would go negative), and the
    # kernel's copy blocks pick their source half the same way.
    tl.store(mine + (tl.load(ep) & (BUFS - 1)) * n + offs, v, mask=mask)
    tl.store(own + offs, v, mask=mask)


@instrument_triton_kernel("symm_mem::fused_all_gather_matmul")
@triton.jit
def _flush(sink):
    # Launched after _stage3: a system-scope release writes back its XCD's whole L2, which the peers' xGMI reads of
    # the staged half need (dirty lines are not written back at the kernel boundary).
    tl.atomic_add(sink, 0, sem="release", scope="sys")


class _State:
    def __init__(self, m, k, n, b_t, kc, rank, world, group_name, a0, b):
        self.m, self.k, self.n, self.kc, self.rank, self.world = (
            m,
            k,
            n,
            kc,
            rank,
            world,
        )
        lm = m // world
        self.shard, self.nch = lm * k // 2, lm // CHUNK_ROWS
        self.shard_b, self.ch_b = lm * k * 2, CHUNK_ROWS * k * 2
        if lm % CHUNK_ROWS or kc % (world - 1) or self.ch_b % (512 * 16 * 8):
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): {lm} rows per rank, K {k} and "
                f"{kc} copy blocks don't fit the chunking"
            )
        if (m // 256) % 32 or (m // 256) * (n // 256) % 8:
            raise RuntimeError(
                f"fused_all_gather_matmul (FlyDSL): M {m}, N {n} don't fit the "
                "XCD-interleaved tile order"
            )
        self.buf = symm_mem.empty(
            2 * self.shard, dtype=torch.int32, device=torch.device("cuda")
        )
        self.hdl = symm_mem.rendezvous(self.buf, group_name)
        self.flags = torch.zeros(
            world * self.nch * SLOT, dtype=torch.int32, device="cuda"
        )
        self.ep = control_block("cuda")
        self.bar_ms = 0
        self.sink = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.mult = kc // (world - 1)
        self.prm = gemm_param(m, n, k, b_t)
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once: with kc 0 (no copy blocks) on flags far past any target.
        never = torch.full_like(self.flags, 1 << 30)
        c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.agmm_gfx950,
            arg(c),
            arg(a0),
            arg(b),
            never.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            rank,
            world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            a0.data_ptr(),
            0,
            self.shard_b,
            self.ch_b,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a_shard, b):
        gat = torch.empty(self.m, self.k, dtype=torch.bfloat16, device="cuda")
        c = torch.empty(self.m, self.n, dtype=torch.bfloat16, device="cuda")
        g32 = gat.view(torch.int32).view(-1)
        bump[(1,)](self.ep, num_warps=1)
        _stage3[(triton.cdiv(self.shard, STAGE_BLOCK),)](
            a_shard.view(torch.int32),
            self.buf,
            g32[self.rank * self.shard :],
            self.ep,
            self.shard,
            BUFS=2,
            BLOCK=STAGE_BLOCK,
        )
        _flush[(FLUSH_PROGRAMS,)](self.sink, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = wait_ms()
        self.gemm(
            c,
            gat,
            b,
            self.flags.data_ptr(),
            self.ep.data_ptr(),
            self.mult,
            SLOT * 4,
            self.rank,
            self.world,
            self.nch,
            self.hdl.buffer_ptrs_dev,
            gat.data_ptr(),
            self.kc,
            self.shard_b,
            self.ch_b,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return gat, c


def supported(A_shard, Bs, gather_dim, group_name):
    """(K, N, TP, B transposed) if the kernel takes this call, else None."""
    if gather_dim != 0 or len(Bs) != 1:
        return None
    B = Bs[0]
    if (
        A_shard.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or not A_shard.is_cuda
    ):
        return None
    if (
        A_shard.dim() != 2
        or B.dim() != 2
        or not A_shard.is_contiguous()
        or B.shape[0] != A_shard.shape[1]
    ):
        return None
    b_t = b_transposed(B)
    world = _resolve_process_group(group_name).size()
    (lm, k), n = A_shard.shape, B.shape[1]
    if b_t is None or world not in KC_DEFAULT:
        return None
    tm = (
        lm * world // 256
    )  # tile rows: the XCD-interleaved order needs whole groups of 32 and tiles in 8s
    if lm % 256 or n % 256 or k % 128 or tm % 32 or tm * (n // 256) % 8:
        return None
    # The gathered A and the output are addressed per tile; B's buffer offsets (one descriptor over K x N) and
    # _stage3's int32 offsets into the shard are still 32-bit.
    if k * n * 2 >= 2**32 or lm * k * 2 >= 2**32:
        return None
    return k, n, world, b_t


def run(A_shard, Bs, gather_dim, group_name, return_A=True):
    k, n, world, b_t = supported(A_shard, Bs, gather_dim, group_name)
    lm = A_shard.shape[0]
    skey = (k, n, world, b_t, lm, group_name, A_shard.device.index)
    with torch.cuda.device(A_shard.device):
        st = _states.get(skey)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_all_gather_matmul (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows {lm}) is "
                    "under CUDA graph capture; it compiles and rendezvouses, so make one eager call first"
                )
            a0 = torch.empty(lm * world, k, dtype=torch.bfloat16, device=A_shard.device)
            st = _State(
                lm * world,
                k,
                n,
                b_t,
                KC.get((k, n, world), KC_DEFAULT[world]),
                _resolve_process_group(group_name).rank(),
                world,
                group_name,
                a0,
                Bs[0],
            )
            _states[skey] = st
        order_streams(st)
        gat, c = st(A_shard, Bs[0])
    st.calls += 1
    return (gat if return_A else None), [c]
