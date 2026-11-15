# Owner(s): ["oncall: distributed"]
"""FlyDSL overrides of symm_mem::fused_all_gather_matmul / fused_matmul_reduce_scatter (torch/_native/ops/symm_mem_fused).

TestRegistration needs no GPU. FlyAsyncTPTest follows AsyncTPTest (test_symmetric_memory.py) at shapes the predicates
admit, checking each call against PyTorch's reference (_fused_*_fallback), plus calls the predicates reject and the
registry's off switch. It needs FlyDSL and gfx950; world size = visible GPUs (2, 4 or 8).

  python test/distributed/test_symmetric_memory_flydsl.py
"""

import os


os.environ.setdefault("TORCH_SYMM_MEM_FLYDSL", "1")

import subprocess
import sys
import unittest

import torch
import torch.distributed as dist
from torch._native import flydsl_utils as fu
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    isRocmArchAnyOf,
    MI350_ARCH,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)


if torch.version.hip is None or not PLATFORM_SUPPORTS_SYMM_MEM:
    print("ROCm symmetric memory not available, skipping tests")
    sys.exit(0)


requires_flydsl = skip_but_pass_in_sandcastle_if(
    not (fu.runtime_available() and fu._version_is_ok()),
    "FlyDSL is not installed",
)

BF = torch.bfloat16
# (M, K, N, B layout), admitted at TP 2 / 4 / 8
AG_SHAPES = {
    "k384_n768_col": (8192, 384, 768, "col"),
    "k1152_n1280_row": (16384, 1152, 1280, "row"),
}
RS_SHAPES = {
    "row": {
        8: (2048, 128, 1024, "row"),
        4: (4096, 128, 256, "row"),
        2: (8192, 128, 256, "row"),
    },
    "col": {
        8: (4096, 640, 3072, "col"),
        4: (8192, 640, 768, "col"),
        2: (8192, 640, 768, "col"),
    },
}


def _tol(ref):
    # The op's own path multiplies per chunk, the reference in one mm: fp32 results differ in accumulation order.
    return {"rtol": 1e-5, "atol": 1e-3} if ref.dtype == torch.float32 else {}


_PROBE = (
    "import torch; from torch._native import registry; "
    "print(sorted(k[1] for k in registry._graphs if k[0] == 'symm_mem'), torch.cuda.is_initialized())"
)


def _probe(env_value):
    env = {k: v for k, v in os.environ.items() if k != "TORCH_SYMM_MEM_FLYDSL"}
    if env_value is not None:
        env["TORCH_SYMM_MEM_FLYDSL"] = env_value
    return (
        subprocess.check_output([sys.executable, "-c", _PROBE], env=env, text=True)
        .strip()
        .splitlines()[-1]
    )


class TestRegistration(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_off_by_default(self):
        self.assertEqual(_probe(None), "[] False")
        self.assertEqual(_probe("0"), "[] False")

    @requires_flydsl
    def test_opt_in_registers_without_initializing_cuda(self):
        self.assertEqual(
            _probe("1"),
            "['fused_all_gather_matmul', 'fused_matmul_reduce_scatter'] False",
        )


def weight(k, n, lay):
    """B [K, N]: W.t() (strides (1, K)) or row-major, as TorchTitan passes it."""
    return (
        torch.rand(n, k, dtype=BF, device="cuda").t()
        if lay == "col"
        else torch.rand(k, n, dtype=BF, device="cuda")
    )


@requires_flydsl
@instantiate_parametrized_tests
class FlyAsyncTPTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        if not isRocmArchAnyOf(MI350_ARCH):
            raise unittest.SkipTest(f"the FlyDSL kernels run on {MI350_ARCH} only")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42 + self.rank)
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("warn")
        torch.utils.deterministic.fill_uninitialized_memory = True

    @staticmethod
    def _calls(op):
        from torch._native.ops.symm_mem_fused import flydsl_impl

        return flydsl_impl._calls[op, "flydsl"]

    def _routed(self, op, before, flydsl):
        msg = (
            "the call did not go to FlyDSL"
            if flydsl
            else "FlyDSL took a call its predicate should reject"
        )
        self.assertEqual(self._calls(op), before + int(flydsl), msg)

    def _ag(self, A_shard, B, return_A, flydsl=True, gather_dim=0):
        before = self._calls("ag")
        out = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard,
            [B],
            gather_dim=gather_dim,
            group_name=dist.group.WORLD.group_name,
            return_A=return_A,
        )
        self._routed("ag", before, flydsl)
        return out

    def _rs(self, A, B, flydsl=True):
        before = self._calls("rs")
        out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "sum", 0, dist.group.WORLD.group_name
        )
        self._routed("rs", before, flydsl)
        return out

    def _check_ag(self, A_shard, B, ag, mm, return_A):
        from torch.distributed._symmetric_memory import (
            _fused_all_gather_matmul_fallback,
        )

        ag0, mm0 = _fused_all_gather_matmul_fallback(
            A_shard,
            [B],
            gather_dim=0,
            group_name=dist.group.WORLD.group_name,
            return_A=True,
        )
        if return_A:
            torch.testing.assert_close(ag, ag0, rtol=0, atol=0)
            self.assertEqual(ag.stride(), ag0.stride())
        else:
            self.assertIsNone(ag)
        torch.testing.assert_close(mm[0], mm0[0], **_tol(mm0[0]))
        self.assertEqual(mm[0].stride(), mm0[0].stride())

    def _check_rs(self, A, B, out):
        from torch.distributed._symmetric_memory import (
            _fused_matmul_reduce_scatter_fallback,
        )

        out0 = _fused_matmul_reduce_scatter_fallback(
            A, B, "sum", scatter_dim=0, group_name=dist.group.WORLD.group_name
        )
        torch.testing.assert_close(out, out0, **_tol(out0))
        self.assertEqual(out.stride(), out0.stride())

    @skip_if_lt_x_gpu(2)
    @parametrize("shape", list(AG_SHAPES))
    @parametrize("return_A", [True, False])
    def test_fused_all_gather_matmul(self, shape: str, return_A: bool) -> None:
        self._init_process()
        m, k, n, lay = AG_SHAPES[shape]
        A_shard = torch.rand(m // self.world_size, k, dtype=BF, device="cuda")
        B = weight(k, n, lay)
        ag, mm = self._ag(A_shard, B, return_A)
        self._check_ag(A_shard, B, ag, mm, return_A)

    @skip_if_lt_x_gpu(2)
    @parametrize("layout", list(RS_SHAPES))
    def test_fused_matmul_reduce_scatter(self, layout: str) -> None:
        self._init_process()
        m, k, n, lay = RS_SHAPES[layout][self.world_size]
        A = torch.rand(m, k, dtype=BF, device="cuda")
        B = weight(k, n, lay)
        self._check_rs(A, B, self._rs(A, B))

    @skip_if_lt_x_gpu(2)
    def test_rejected_calls_fall_through(self) -> None:
        """fp32, a K the tiles don't divide, and gather / scatter dim 1 go to the op's own implementation."""
        self._init_process()
        m, k, n, lay = AG_SHAPES["k384_n768_col"]
        for A_shard, B in (
            (
                torch.rand(m // self.world_size, k, device="cuda"),
                torch.rand(k, n, device="cuda"),
            ),
            (
                torch.rand(m // self.world_size, k + 64, dtype=BF, device="cuda"),
                weight(k + 64, n, lay),
            ),
        ):
            ag, mm = self._ag(A_shard, B, True, flydsl=False)
            self._check_ag(A_shard, B, ag, mm, True)
        self._ag(
            torch.rand(4, 512, k, dtype=BF, device="cuda"),
            weight(k, n, lay),
            True,
            flydsl=False,
            gather_dim=1,
        )
        m, k, n, lay = RS_SHAPES["col"][self.world_size]
        A, B = torch.rand(m, k, device="cuda"), torch.rand(k, n, device="cuda")
        self._check_rs(A, B, self._rs(A, B, flydsl=False))

    @skip_if_lt_x_gpu(2)
    def test_32bit_bounds(self) -> None:
        """A gathered A or output of 4 GiB is admitted (both are addressed per tile). Still 32-bit, so rejected at
        4 GiB: B (both ops), the AG shard, and the RS output chunk (M / TP x N)."""
        self._init_process()
        from torch._native.ops.symm_mem_fused import flydsl_impl

        g, w = dist.group.WORLD.group_name, self.world_size
        k, n = 128, 65536
        B = weight(k, n, "col")
        rows = 2**32 // (2 * n)  # an M x N bf16 output of exactly 4 GiB
        A_shard = torch.empty(rows // w, k, dtype=BF, device="cuda")
        self.assertTrue(flydsl_impl._ag_cond(A_shard, [B], 0, g), "AG, 4 GiB output")
        self.assertTrue(
            flydsl_impl._rs_cond(
                torch.empty(rows, k, dtype=BF, device="cuda"), B, "sum", 0, g
            )
        )
        A = torch.empty(rows * w, k, dtype=BF, device="cuda")
        self.assertFalse(
            flydsl_impl._rs_cond(A, B, "sum", 0, g), "RS, 4 GiB output chunk"
        )
        del A_shard, A, B
        # B one tile-aligned step below 4 GiB, then exactly 4 GiB.
        k = 32768
        W = torch.empty(n, k, dtype=BF, device="cuda")
        A_shard = torch.empty(8192 // w, k, dtype=BF, device="cuda")
        A = torch.empty(8192, k, dtype=BF, device="cuda")
        for cols, admitted in ((n - 1024, True), (n, False)):
            B = W[:cols].t()
            self.assertEqual(
                flydsl_impl._ag_cond(A_shard, [B], 0, g),
                admitted,
                f"AG, B {k} x {cols}",
            )
            self.assertEqual(
                flydsl_impl._rs_cond(A, B, "sum", 0, g), admitted, f"RS, B {k} x {cols}"
            )

    @skip_if_lt_x_gpu(2)
    def test_deregistered(self) -> None:
        """With the FlyDSL overrides deregistered an admitted call takes the op's own path; re-enabling restores them."""
        self._init_process()
        m, k, n, lay = AG_SHAPES["k384_n768_col"]
        A_shard, B = (
            torch.rand(m // self.world_size, k, dtype=BF, device="cuda"),
            weight(k, n, lay),
        )
        fu.deregister_op_overrides()
        try:
            ag, mm = self._ag(A_shard, B, True, flydsl=False)
            self._check_ag(A_shard, B, ag, mm, True)
        finally:
            torch._native.registry.reenable_op_overrides(enable_dsl_names="flydsl")
        ag, mm = self._ag(A_shard, B, True)
        self._check_ag(A_shard, B, ag, mm, True)

    @skip_if_lt_x_gpu(2)
    def test_graph_capture(self) -> None:
        """One eager call per op builds the states; then both ops in one CUDA graph, replayed with new inputs."""
        self._init_process()
        m, k, n, lay = AG_SHAPES["k384_n768_col"]
        A_shard, Bg = (
            torch.rand(m // self.world_size, k, dtype=BF, device="cuda"),
            weight(k, n, lay),
        )
        m, k, n, lay = RS_SHAPES["col"][self.world_size]
        A, Br = torch.rand(m, k, dtype=BF, device="cuda"), weight(k, n, lay)
        self._ag(A_shard, Bg, True)
        self._rs(A, Br)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ag, mm = self._ag(A_shard, Bg, True)
            out = self._rs(A, Br)
        for _ in range(2):
            A_shard.copy_(torch.rand_like(A_shard))
            A.copy_(torch.rand_like(A))
            Bg.copy_(torch.rand_like(Bg))
            Br.copy_(torch.rand_like(Br))
            graph.replay()
            torch.cuda.synchronize()
            self._check_ag(A_shard, Bg, ag, mm, True)
            self._check_rs(A, Br, out)


if __name__ == "__main__":
    run_tests()
