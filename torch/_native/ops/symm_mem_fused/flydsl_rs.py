# mypy: allow-untyped-defs
"""FlyDSL fused matmul + reduce-scatter (symm_mem::fused_matmul_reduce_scatter) on gfx950.

One state per (K, N, TP, B layout, rows, group, device), running one kernel (flydsl_kernels.rsmm_gfx950): every
block computes one tile of this rank's partial into its symmetric staging buffer in reduce-scatter order and signals
the chunk's owner; blocks [0, R) then sum this rank's chunk of every source's staging into the output. One call, all on
the current stream: bump the device epoch, the symmetric-memory barrier (every peer is done reading this rank's
staging from the previous call), the kernel. The staging buffer is M x N bf16 per state.

A state is built at its first call: a collective symmetric-memory rendezvous and a FlyDSL compile (which also runs the
kernel once, with no reducers and every counter add into local scratch), so every rank must make its calls in the
same order, and the first call of a state must not be under CUDA graph capture (later calls may be). A band counter
still short after flydsl_common.wait_ms() fails the launch with a [FATAL] line instead of hanging; the barrier times out
the same way from a state's second call on.
"""

from __future__ import annotations

import flydsl.compiler as flyc

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group

from . import flydsl_kernels as K
from .flydsl_common import (
    arg,
    b_transposed,
    bump,
    control_block,
    gemm_param,
    order_streams,
    wait_ms,
)


# Rows per chunk band: the reducers sum a band once all its tiles from every source are in.
BAND_ROWS = {8: 256, 4: 512, 2: 1024}
# (K, N, TP) -> reducer blocks, the best measured at M 65536 (Llama 3.1 70B); else R_DEFAULT.
R = {
    (1024, 8192, 8): 28,
    (3584, 8192, 8): 16,
    (1280, 8192, 8): 28,
    (7168, 8192, 8): 16,
    (2048, 8192, 4): 16,
    (7168, 8192, 4): 8,
    (2560, 8192, 4): 16,
    (14336, 8192, 4): 16,
    (4096, 8192, 2): 16,
    (14336, 8192, 2): 8,
    (5120, 8192, 2): 16,
    (28672, 8192, 2): 8,
}
R_DEFAULT = {8: 28, 4: 12, 2: 8}
_states: dict = {}


class _State:
    def __init__(self, m, k, n, b_t, r, br, rank, world, group_name, a0, b):
        self.m, self.k, self.n, self.r, self.rank, self.world = m, k, n, r, rank, world
        self.lm, self.br = m // world, br
        self.nb, self.rpb = self.lm // br, br // 256
        tiles, g = (m // 256) * (n // 256), self.rpb * (n // 256)
        if self.lm % br or br % 256 or n % 256 or tiles % (8 * g):
            raise RuntimeError(
                f"fused_matmul_reduce_scatter (FlyDSL): M {m}, N {n} don't fit "
                f"{br}-row bands"
            )
        if br * n % (512 * 8 * 2):
            raise RuntimeError(
                f"fused_matmul_reduce_scatter (FlyDSL): N {n} doesn't give whole "
                f"reducer items per {br}-row band"
            )
        self.buf = symm_mem.empty(
            m * n + self.nb * 32 * 2,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        self.hdl = symm_mem.rendezvous(self.buf, group_name)
        self.stage = self.buf[: m * n].view(m, n)
        self.cnt = self.buf[m * n :].view(torch.int32)
        self.cnt.zero_()  # peers add only after this call's barrier, which this rank reaches after the zeroing
        self.cnt_off = m * n * 2
        self.ep = control_block("cuda")
        self.bar_ms = 0
        self.prm = gemm_param(m, n, k, b_t)
        self.calls, self.stream = 0, None
        # Compiling also runs the kernel once with no reducers and every peer entry on this rank's own staging, counter
        # offset 0: counter adds land on staging words every call rewrites.
        table = torch.full(
            (world,), self.stage.data_ptr(), dtype=torch.int64, device="cuda"
        )
        s = torch.cuda.current_stream().cuda_stream
        self.gemm = flyc.compile(
            K.rsmm_gfx950,
            arg(self.stage),
            arg(a0),
            arg(b),
            table.data_ptr(),
            0,
            self.ep.data_ptr(),
            self.stage.data_ptr(),
            rank,
            world,
            0,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            s,
        )
        torch.cuda.synchronize()

    def __call__(self, a, b):
        out = torch.empty(self.lm, self.n, dtype=torch.bfloat16, device="cuda")
        bump[(1,)](self.ep, num_warps=1)
        self.hdl.barrier(channel=0, timeout_ms=self.bar_ms)
        self.bar_ms = wait_ms()
        self.gemm(
            self.stage,
            a,
            b,
            self.hdl.buffer_ptrs_dev,
            self.cnt_off,
            self.ep.data_ptr(),
            out.data_ptr(),
            self.rank,
            self.world,
            self.r,
            self.nb,
            self.rpb,
            self.lm,
            self.prm,
            torch.cuda.current_stream().cuda_stream,
        )
        return out


def supported(A, B, reduce_op, scatter_dim, group_name):
    """(K, N, TP, B transposed) if the kernel takes this call, else None."""
    if reduce_op != "sum" or scatter_dim != 0:
        return None
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16 or not A.is_cuda:
        return None
    if (
        A.dim() != 2
        or B.dim() != 2
        or not A.is_contiguous()
        or B.shape[0] != A.shape[1]
    ):
        return None
    b_t = b_transposed(B)
    world = _resolve_process_group(group_name).size()
    (m, k), n = A.shape, B.shape[1]
    if b_t is None or world not in R_DEFAULT:
        return None
    br = BAND_ROWS[world]
    g = (
        br // 256 * (n // 256)
    )  # tiles per chunk band: the XCD-interleaved order takes them in groups of 8 g
    if (
        m % world
        or (m // world) % br
        or n % 256
        or k % 128
        or (m // 256) * (n // 256) % (8 * g)
    ):
        return None
    if br * n % (512 * 8 * 2):  # whole reducer items per band
        return None
    # A and the staging are addressed per tile; B's buffer offsets (one descriptor over K x N) and the TP 8 reducers'
    # (one descriptor per chunk, M / TP x N) are still 32-bit.
    if k * n * 2 >= 2**32 or (m // world) * n * 2 >= 2**32:
        return None
    if world == 8:
        # The TP 8 kernel's XCD order: XCD x takes 4 tile rows x (world * g / 32) tile columns of each band step, which
        # covers every tile only if these divide (N % 1024); otherwise tiles repeat, some chunks never get their counter
        # adds, and the owners' reducers spin forever.
        rows = world * (br // 256)
        if rows % 4 or 8 % (rows // 4) or world * g % 32:
            return None
    return k, n, world, b_t


def run(A, B, reduce_op, scatter_dim, group_name):
    k, n, world, b_t = supported(A, B, reduce_op, scatter_dim, group_name)
    m = A.shape[0]
    skey = (k, n, world, b_t, m // world, group_name, A.device.index)
    with torch.cuda.device(A.device):
        st = _states.get(skey)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused_matmul_reduce_scatter (FlyDSL): the first call at (K {k}, N {n}, TP {world}, rows "
                    f"{m // world}) is under CUDA graph capture; it compiles and rendezvouses, so make one eager "
                    "call first"
                )
            st = _State(
                m,
                k,
                n,
                b_t,
                R.get((k, n, world), R_DEFAULT[world]),
                BAND_ROWS[world],
                _resolve_process_group(group_name).rank(),
                world,
                group_name,
                A,
                B,
            )
            _states[skey] = st
        order_streams(st)
        out = st(A, B)
    st.calls += 1
    return out
