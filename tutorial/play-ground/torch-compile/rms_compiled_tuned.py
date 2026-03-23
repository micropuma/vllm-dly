# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_0 = async_compile.cpp_pybinding(['const double*', 'double*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const double* in_ptr0,
                       double* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                out_ptr0[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /tmp/torchinductor_douliyang/w5/cw5dtgokpne53lpqajdsand7zxyfle7q7vuogrcdbdh55sz6nb2z.py
# Topologically Sorted Source Nodes: [x, pow_1, var, rsqrt, x_1, x_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   var => mean
#   x => convert_element_type
#   x_1 => mul_7
#   x_2 => convert_element_type_1
# Graph fragment:
#   %arg2_1 : Tensor "f16[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %buf2 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=buf2]
#   %device_put_default : Tensor "f64[1][1]cuda:0" = PlaceHolder[target=device_put_default]
#   %unbind_int : [num_users=1] = call_function[target=torch.ops.aten.unbind.int](args = (%device_put_default,), kwargs = {})
#   %convert_element_type : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %convert_element_type_default_1), kwargs = {})
#   %rsqrt : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_7 : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.float16), kwargs = {})
#   return %buf2,%convert_element_type_1
triton_red_fused__to_copy_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_red_fused__to_copy_mean_mul_pow_rsqrt_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp64', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '0F8E6B2A3476BD3493EAF879E6446E2A685A9E4A09C943EF2B03B0DF5D73507F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [1, 1])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = ks0
        tmp9 = tmp8.to(tl.float32)
        tmp10 = (tmp4 / tmp9)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp10 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp7 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp17, r0_mask & xmask)
''', device_str='cuda')

def partition_0(args):
    arg2_1, buf1, s27, s77 = args
    args.clear()
    s27 = s27
    s77 = s77
    assert_size_stride(arg2_1, (s77, s27), (s27, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((s77, s27), (s27, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x, pow_1, var, rsqrt, x_1, x_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mean_mul_pow_rsqrt_1.run(arg2_1, buf1, buf3, s27, s77, s27, stream=stream0)
        del arg2_1
        del buf1
    return (buf3, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s77 = arg0_1
        s27 = arg1_1
        buf0 = empty_strided_cpu_pinned((1, ), (1, ), torch.float64)
        cpp_fused_0(arg3_1, buf0)
        del arg3_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, ), (1, ), torch.float64)
            buf1.copy_(buf0, True)
            del buf0
        partition0_args = [arg2_1, buf1, s27, s77]
        del arg2_1, buf1
        (buf3,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf3, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2048
    arg1_1 = 4096
    arg2_1 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
