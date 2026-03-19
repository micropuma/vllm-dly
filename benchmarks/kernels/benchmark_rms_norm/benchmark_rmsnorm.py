# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from torch import nn

from vllm import _custom_ops as vllm_ops
from vllm.triton_utils import triton
import triton.language as tl

"""
This benchmark evaluates the performance of RMSNorm implementations across different providers:
- HuggingFace: A naive PyTorch implementation of RMSNorm.
- FlashInfer: The implementation of RMSNorm in the FlashInfer library.
- vLLM: The implementation of RMSNorm in the vLLM library.
- Torch Compile: A PyTorch implementation of RMSNorm that is compiled with torch.compile.
- Triton: A Triton implementation of RMSNorm.

usage example:
- python benchmarks/kernels/benchmark_rmsnorm.py
- add export TRITON_PRINT_AUTOTUNING=1 to print the triton autotuning results
"""

def _rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_dtype = x.dtype
    orig_shape = x.shape

    x = x.to(torch.float32)
    x.view(-1, x.shape[-1])

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight

    return x.view(orig_shape)

    
def _fused_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    orig_x_shape = x.shape
    orig_res_shape = residual.shape

    x = x.to(torch.float32)
    x.view(-1, x.shape[-1])
    residual = residual.to(torch.float32)

    x += residual.view(-1, residual.shape[-1])
    residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight

    return x.view(orig_x_shape), residual.view(orig_res_shape)

torch._dynamo.config.recompile_limit = 8888

# TODO(leon): benchmark torch.compile with hard-coded shape versus dynamic shape  
# currently have to set dyanmic to be True to avoid recompilation for different shapes, 
# but this may lead to suboptimal performance. 

# TORCH_LOGS=recompiles for debugging torch.compile behavior
_RMSNorm_Torch_Compile = torch.compile(
    _rmsnorm_naive,
    fullgraph=True,
    dynamic=False,
)
_FusedAddRMSNorm_Torch_Compile = torch.compile(
    _fused_add_rmsnorm,
    fullgraph=True,
    dynamic=False,
)

def rmsnorm_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if residual is None:  
        return _RMSNorm_Torch_Compile(x, weight, eps)
    else:
        return _FusedAddRMSNorm_Torch_Compile(x, weight, residual, eps)

class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output

@triton.jit
def _rms_norm_kernel(
    X_ptr,
    Output_ptr,
    Weight_ptr,
    Residual_ptr,
    stride_x,
    stride_out,
    stride_res,
    hidden_size,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # HAS_RESIDUAL is tl.constexpr — resolved at compile time, zero warp divergence.
    # Triton compiles two separate kernel binaries (one per constexpr value).
    #
    # No autotune: the fused path modifies buffers in-place, which would corrupt
    # autotune trials. Config is selected manually in the Python wrapper.
    #
    # HAS_RESIDUAL=False:  Output = rms_norm(X) * W
    # HAS_RESIDUAL=True:   Residual += X;  Output = rms_norm(Residual) * W
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Out_row = Output_ptr + row * stride_out
    if HAS_RESIDUAL:
        Res_row = Residual_ptr + row * stride_res

    # Phase 1: sum of squares (+ fused residual add when HAS_RESIDUAL)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            res = tl.load(Res_row + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + res
            tl.store(Res_row + cols, x, mask=mask)
        _var += x * x

    var = tl.sum(_var, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)

    # Phase 2: normalize × weight
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        if HAS_RESIDUAL:
            x = tl.load(Res_row + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Out_row + cols, y, mask=mask)


def _select_block_config(
    hidden_size: int,
    num_tokens: int,
    element_size: int,
) -> tuple[int, int]:
    """Mirror the vLLM CUDA kernel's two-step config strategy.

    Step 1 — decode vs prefill threshold (matches CUDA's max_block_size):
        num_tokens < 256  →  max_threads = 1024  (decode: few rows, big block)
        num_tokens >= 256 →  max_threads = 256   (prefill: many rows, small block
                                                   for higher SM occupancy)

    Step 2 — vectorization width (matches CUDA's calculated_vec_size):
        Target 128-bit (16-byte) loads — LDG.128.
        vec_size = gcd(16 // element_size, hidden_size)
          fp16/bf16: 16/2=8  → up to 8 elements per transaction
          fp32:      16/4=4  → up to 4 elements per transaction

    Step 3 — threads (matches CUDA's block_size):
        threads = min(hidden_size // vec_size, max_threads)
        Round up to next power of 2 for Triton alignment requirement.

    Triton mapping:
        BLOCK_SIZE = vec_size × threads   (total elements per loop iteration,
                                           equivalent to CUDA block_size × vec_size)
        num_warps  = threads // 32
    """
    import math

    # Step 1: decode vs prefill
    max_threads = 1024 if num_tokens < 256 else 256

    # Step 2: 128-bit vectorization width
    vec_size = math.gcd(16 // element_size, hidden_size)

    # Step 3: actual threads needed to cover one row in a single pass
    threads = min(hidden_size // vec_size, max_threads)
    # Triton requires BLOCK_SIZE to be a power of 2; round threads up accordingly
    threads = triton.next_power_of_2(threads)
    threads = min(threads, max_threads)   # re-clamp after rounding up

    num_warps = max(1, threads // 32)

    # BLOCK_SIZE = elements per loop iteration = vec_size × threads
    BLOCK_SIZE = triton.next_power_of_2(vec_size * threads)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)    # cap against extreme register pressure

    return BLOCK_SIZE, num_warps


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert weight.is_contiguous(), "weight must be contiguous"
    assert x.stride(-1) == 1, "innermost dim of input x must be contiguous"

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    num_tokens, hidden_size = x.shape
    BLOCK_SIZE, num_warps = _select_block_config(hidden_size, num_tokens,
                                                  x.element_size())
    grid = (num_tokens,)

    if residual is not None:
        # Fused path: residual += x, then x = norm(residual) * w (both in-place)
        _rms_norm_kernel[grid](
            x, x, weight, residual,
            x.stride(0), x.stride(0), residual.stride(0),
            hidden_size, eps,
            HAS_RESIDUAL=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return x.view(orig_shape), residual.view(orig_shape)
    else:
        output = torch.empty_like(x)
        _rms_norm_kernel[grid](
            x, output, weight, x,
            x.stride(0), output.stride(0), x.stride(0),
            hidden_size, eps,
            HAS_RESIDUAL=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return output.view(orig_shape)


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_flashinfer = rmsnorm_flashinfer(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_torch_compile = rmsnorm_torch_compile(
        x.clone(), weight, residual.clone() if residual is not None else None   
    )
    output_triton = rmsnorm_triton(
        x.clone(), weight, residual.clone() if residual is not None else None
    )

    if use_residual:
        output_naive = output_naive[0]
        output_flashinfer = output_flashinfer[0]
        output_vllm = output_vllm[0]
        output_torch_compile = output_torch_compile[0]
        output_triton = output_triton[0]

    print(f"Naive output={output_naive}")
    print(f"FlashInfer output={output_flashinfer}")
    print(f"vLLM output={output_vllm}")
    print(f"Torch Compile output={output_torch_compile}")
    print(f"Triton output={output_triton}")

    all_match = (
        torch.allclose(output_naive, output_flashinfer, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_torch_compile, output_vllm, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_triton, output_vllm, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_torch_compile, output_triton, atol=1e-2, rtol=1e-2)
    )
    if all_match:
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [2**i for i in range(0, 7, 2)]
seq_length_range = [2**i for i in range(6, 11, 1)]
head_num_range = [32, 48]
configs = list(itertools.product(head_num_range, batch_size_range, seq_length_range))


def get_benchmark(use_residual):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "flashinfer", "vllm", "torch_compile", "triton"],
            line_names=["HuggingFace", "FlashInfer", "vLLM", "Torch Compile", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-"), ("purple", "-")],
            ylabel="us",
            plot_name=f"rmsnorm-perf-{'with' if use_residual else 'without'}-residual",
            args={},
        )
    )
    def benchmark(head_num, batch_size, seq_len, provider):
        dtype = torch.bfloat16
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "flashinfer":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_flashinfer(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "torch_compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_torch_compile(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_triton(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size (2nd dimension) of the sequence",
    )
    parser.add_argument(
        "--use-residual", action="store_true", help="Whether to use residual connection"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/rmsnorm/",
        help="Path to save rmsnorm benchmark results",
    )

    args = parser.parse_args()

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        use_residual=args.use_residual,
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual)
    # Run performance benchmark
    import os
    os.makedirs(args.save_path, exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=args.save_path)