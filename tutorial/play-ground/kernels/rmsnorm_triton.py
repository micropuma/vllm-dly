"""
Triton RMSNorm kernels that mirror the CUDA implementation in csrc/layernorm_kernels.cu.

Two kernels:
  1. rms_norm_kernel        — pure RMSNorm:  y = x / rms(x) * w
  2. fused_add_rms_norm_kernel — residual += x; y = residual / rms(residual) * w

Optimization strategies carried over from the CUDA version:
  - Adaptive num_warps (decode vs. prefill) via @triton.autotune
  - FP32 intermediate accumulation for numerical stability
  - Single-pass when BLOCK_SIZE >= hidden_size; looped otherwise
  - Triton compiler auto-vectorizes tl.load/tl.store to wide transactions (LDG.128)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------
# Analogous to the CUDA kernel's adaptive block size:
#   - Large BLOCK_SIZE + many warps  → fewer loop iters, good for small num_tokens (decode)
#   - Smaller BLOCK_SIZE + fewer warps → higher occupancy, good for large num_tokens (prefill)

def _rms_norm_configs():
    configs = []
    for BLOCK_SIZE in [1024, 2048, 4096, 8192]:
        for num_warps in [4, 8, 16]:
            configs.append(
                triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_warps=num_warps)
            )
    return configs


# ---------------------------------------------------------------------------
# Kernel 1: rms_norm  (read-only input → safe for autotune)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_rms_norm_configs(), key=["hidden_size"])
@triton.jit
def _rms_norm_kernel(
    X_ptr,        # [num_tokens, hidden_size]
    Y_ptr,        # [num_tokens, hidden_size]
    W_ptr,        # [hidden_size]
    stride_x,     # X.stride(0)
    stride_y,     # Y.stride(0)
    hidden_size,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    # Phase 1: sum of squares in FP32
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)

    # Phase 2: normalize × weight
    if IS_FP16:
        OUT_DTYPE: tl.constexpr = tl.float16
    elif IS_BF16:
        OUT_DTYPE: tl.constexpr = tl.bfloat16
    else:
        OUT_DTYPE: tl.constexpr = tl.float32

    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_row + cols, y.to(OUT_DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: fused_add_rms_norm  (in-place → NO autotune, manual config)
# ---------------------------------------------------------------------------
# Autotune corrupts in-place buffers between trials, so we select config
# manually — the same strategy the CUDA kernel uses.

@triton.jit
def _fused_add_rms_norm_kernel(
    Input_ptr,      # [num_tokens, hidden_size] — receives normalized output
    Residual_ptr,   # [num_tokens, hidden_size] — updated in-place: residual += input
    W_ptr,          # [hidden_size]
    stride_input,   # Input.stride(0)
    stride_res,     # Residual.stride(0)
    hidden_size,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Inp_row = Input_ptr + row * stride_input
    Res_row = Residual_ptr + row * stride_res

    if IS_FP16:
        OUT_DTYPE: tl.constexpr = tl.float16
    elif IS_BF16:
        OUT_DTYPE: tl.constexpr = tl.bfloat16
    else:
        OUT_DTYPE: tl.constexpr = tl.float32

    # Phase 1: fused residual add + sum-of-squares
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        inp = tl.load(Inp_row + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Res_row + cols, mask=mask, other=0.0).to(tl.float32)
        z = inp + res
        tl.store(Res_row + cols, z.to(OUT_DTYPE), mask=mask)
        _var += z * z

    var = tl.sum(_var, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)

    # Phase 2: normalize × weight → write to input
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        res = tl.load(Res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = res * rstd * w
        tl.store(Inp_row + cols, y.to(OUT_DTYPE), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def rms_norm(
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):
    assert out.is_contiguous()
    assert weight.is_contiguous()
    assert x.stride(-1) == 1, "innermost dim must be contiguous"
    assert x.dim() == 2, "Triton version only supports 2D input [num_tokens, hidden_size]"

    num_tokens, hidden_size = x.shape
    grid = (num_tokens,)

    _rms_norm_kernel[grid](
        x, out, weight,
        x.stride(0), out.stride(0),
        hidden_size, epsilon,
        IS_FP16=(x.dtype == torch.float16),
        IS_BF16=(x.dtype == torch.bfloat16),
    )
    return out


def _select_block_config(hidden_size, num_tokens):
    """Mirror the CUDA kernel's adaptive block/warp selection."""
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # More warps for decode (small grid), fewer for prefill (large grid, more occupancy)
    if num_tokens < 256:
        num_warps = 16
    else:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def fused_add_rms_norm(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):
    assert residual.is_contiguous()
    assert weight.is_contiguous()
    assert inp.stride(-1) == 1, "innermost dim must be contiguous"
    assert inp.dim() == 2, "Triton version only supports 2D input"

    num_tokens, hidden_size = inp.shape
    grid = (num_tokens,)
    BLOCK_SIZE, num_warps = _select_block_config(hidden_size, num_tokens)

    _fused_add_rms_norm_kernel[grid](
        inp, residual, weight,
        inp.stride(0), residual.stride(0),
        hidden_size, epsilon,
        IS_FP16=(inp.dtype == torch.float16),
        IS_BF16=(inp.dtype == torch.bfloat16),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------
def _pytorch_rms_norm(x, weight, eps):
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(var + eps)).to(x.dtype) * weight


def _pytorch_fused_add_rms_norm(inp, residual, weight, eps):
    residual_out = residual + inp
    x_fp32 = residual_out.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = (x_fp32 * torch.rsqrt(var + eps)).to(inp.dtype) * weight
    return normalized, residual_out


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
def test_correctness():
    torch.manual_seed(42)
    device = "cuda"

    tol = {torch.float16: 1e-2, torch.bfloat16: 0.1, torch.float32: 1e-4}
    all_pass = True

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for num_tokens, hidden_size in [(1, 4096), (32, 4096), (2048, 4096), (128, 8192)]:
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            w = torch.randn(hidden_size, dtype=dtype, device=device)
            eps = 1e-6

            # --- Test rms_norm ---
            out_triton = torch.empty_like(x)
            rms_norm(out_triton, x, w, eps)
            out_ref = _pytorch_rms_norm(x, w, eps)
            max_diff = (out_triton - out_ref).abs().max().item()
            ok = max_diff < tol[dtype]
            all_pass &= ok
            status = "PASS" if ok else "FAIL"
            print(f"[rms_norm] {status}  dtype={dtype}, shape=({num_tokens}, {hidden_size}), max_diff={max_diff:.6f}")

            # --- Test fused_add_rms_norm ---
            inp_triton = x.clone()
            res_triton = torch.randn_like(x)
            res_copy = res_triton.clone()
            fused_add_rms_norm(inp_triton, res_triton, w, eps)
            out_ref2, res_ref = _pytorch_fused_add_rms_norm(x, res_copy, w, eps)

            max_diff_out = (inp_triton - out_ref2).abs().max().item()
            max_diff_res = (res_triton - res_ref).abs().max().item()
            ok = max_diff_out < tol[dtype] and max_diff_res < tol[dtype]
            all_pass &= ok
            status = "PASS" if ok else "FAIL"
            print(f"[fused]    {status}  dtype={dtype}, shape=({num_tokens}, {hidden_size}), "
                  f"max_diff_out={max_diff_out:.6f}, max_diff_res={max_diff_res:.6f}")

    print(f"\nAll correctness tests {'PASSED' if all_pass else 'FAILED'}.")
    return all_pass


# ---------------------------------------------------------------------------
# Benchmark: Triton vs vLLM CUDA vs PyTorch
# ---------------------------------------------------------------------------
def benchmark():
    try:
        from vllm._custom_ops import rms_norm as vllm_rms_norm
        from vllm._custom_ops import fused_add_rms_norm as vllm_fused_add_rms_norm
        has_vllm = True
    except ImportError:
        has_vllm = False
        print("vLLM not importable, skipping vLLM CUDA comparison.\n")

    configs = [
        (1,    4096),    # decode: single token
        (32,   4096),    # decode: small batch
        (256,  4096),    # prefill boundary
        (2048, 4096),    # prefill: medium
        (8192, 4096),    # prefill: large
        (2048, 8192),    # large hidden
    ]
    dtype = torch.float16
    eps = 1e-6
    device = "cuda"

    # ---- rms_norm ----
    print(f"{'='*90}")
    print(f"  [rms_norm]")
    print(f"{'Config':>20s} | {'Triton (us)':>12s} | {'vLLM CUDA (us)':>14s} | {'PyTorch (us)':>12s} | {'Tri/CUDA':>8s}")
    print(f"{'-'*90}")

    for num_tokens, hidden_size in configs:
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)
        out = torch.empty_like(x)

        triton_ms = triton.testing.do_bench(lambda: rms_norm(out, x, w, eps), warmup=100, rep=500)

        if has_vllm:
            cuda_ms = triton.testing.do_bench(lambda: vllm_rms_norm(out, x, w, eps), warmup=100, rep=500)
        else:
            cuda_ms = float("nan")

        pytorch_ms = triton.testing.do_bench(lambda: _pytorch_rms_norm(x, w, eps), warmup=100, rep=500)

        ratio = f"{triton_ms / cuda_ms:.2f}x" if has_vllm else "N/A"
        label = f"({num_tokens}, {hidden_size})"
        print(f"{label:>20s} | {triton_ms*1000:>12.1f} | {cuda_ms*1000:>14.1f} | {pytorch_ms*1000:>12.1f} | {ratio:>8s}")

    # ---- fused_add_rms_norm ----
    print(f"\n{'='*90}")
    print(f"  [fused_add_rms_norm]")
    print(f"{'Config':>20s} | {'Triton (us)':>12s} | {'vLLM CUDA (us)':>14s} | {'PyTorch (us)':>12s} | {'Tri/CUDA':>8s}")
    print(f"{'-'*90}")

    for num_tokens, hidden_size in configs:
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)
        res = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        def triton_fn():
            inp_t = x.clone()
            res_t = res.clone()
            fused_add_rms_norm(inp_t, res_t, w, eps)

        triton_ms = triton.testing.do_bench(triton_fn, warmup=100, rep=500)

        if has_vllm:
            def cuda_fn():
                inp_c = x.clone()
                res_c = res.clone()
                vllm_fused_add_rms_norm(inp_c, res_c, w, eps)
            cuda_ms = triton.testing.do_bench(cuda_fn, warmup=100, rep=500)
        else:
            cuda_ms = float("nan")

        def pytorch_fn():
            _pytorch_fused_add_rms_norm(x, res.clone(), w, eps)

        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        ratio = f"{triton_ms / cuda_ms:.2f}x" if has_vllm else "N/A"
        label = f"({num_tokens}, {hidden_size})"
        print(f"{label:>20s} | {triton_ms*1000:>12.1f} | {cuda_ms*1000:>14.1f} | {pytorch_ms*1000:>12.1f} | {ratio:>8s}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    args = parser.parse_args()

    if not args.test and not args.bench:
        args.test = True
        args.bench = True

    if args.test:
        test_correctness()
        print()
    if args.bench:
        benchmark()
