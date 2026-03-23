"""
Minimal RMSNorm Triton kernel with exhaustive triton.autotune.

No fused-residual path — pure RMSNorm only:  y = x / rms(x) * w

The original benchmark_rmsnorm.py uses heuristic config selection
(_select_block_config). This file replaces that with triton.autotune
so Triton's autotuner benchmarks every config and picks the fastest.

Usage:
    # Run correctness test + autotune benchmark for all shapes
    python benchmark_rmsnorm_autotune.py

    # Print TRITON_PRINT_AUTOTUNING=1 to see Triton's internal autotune log
    TRITON_PRINT_AUTOTUNING=1 python benchmark_rmsnorm_autotune.py

    # Only run correctness tests
    python benchmark_rmsnorm_autotune.py --test-only

    # Only run benchmark (skip correctness)
    python benchmark_rmsnorm_autotune.py --bench-only
"""

import itertools
import time

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exhaustive autotune configurations
# ---------------------------------------------------------------------------
# We sweep BLOCK_SIZE × num_warps × num_stages to let Triton find the best
# combination for each (num_tokens, hidden_size) pair.

def _exhaustive_configs():
    configs = []
    for BLOCK_SIZE in [256, 512, 1024, 2048, 4096, 8192]:
        for num_warps in [1, 2, 4, 8, 16, 32]:
            for num_stages in [1, 2, 3, 4]:
                if num_warps * 32 > BLOCK_SIZE:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE": BLOCK_SIZE},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


ALL_CONFIGS = _exhaustive_configs()
print(f"[autotune] Total config candidates: {len(ALL_CONFIGS)}")


# ---------------------------------------------------------------------------
# Minimal autotuned RMSNorm kernel
# ---------------------------------------------------------------------------
# key=["hidden_size"] means Triton re-tunes whenever hidden_size changes.
# Adding "num_tokens" would make it also tune per token count, but that
# generates too many cache entries; hidden_size is the dominant factor.

@triton.autotune(configs=ALL_CONFIGS, key=["hidden_size"])
@triton.jit
def _rmsnorm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x,
    stride_y,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)

    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_row + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Also autotune on both (num_tokens, hidden_size) for fine-grained analysis
# ---------------------------------------------------------------------------

@triton.autotune(configs=ALL_CONFIGS, key=["num_tokens", "hidden_size"])
@triton.jit
def _rmsnorm_kernel_2d_key(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x,
    stride_y,
    num_tokens,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X_ptr + row * stride_x
    Y_row = Y_ptr + row * stride_y

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)

    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_row + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def rmsnorm_autotune(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """RMSNorm with autotune keyed on hidden_size only."""
    assert weight.is_contiguous()
    assert x.stride(-1) == 1
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    num_tokens, hidden_size = x.shape
    output = torch.empty_like(x)
    grid = (num_tokens,)
    _rmsnorm_kernel[grid](
        x, output, weight,
        x.stride(0), output.stride(0),
        hidden_size, eps,
    )
    return output.view(orig_shape)


def rmsnorm_autotune_2d(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """RMSNorm with autotune keyed on (num_tokens, hidden_size)."""
    assert weight.is_contiguous()
    assert x.stride(-1) == 1
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    num_tokens, hidden_size = x.shape
    output = torch.empty_like(x)
    grid = (num_tokens,)
    _rmsnorm_kernel_2d_key[grid](
        x, output, weight,
        x.stride(0), output.stride(0),
        num_tokens, hidden_size, eps,
    )
    return output.view(orig_shape)


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(var + eps)).to(x.dtype) * weight


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness():
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6
    tol = {torch.float16: 1e-2, torch.bfloat16: 0.1, torch.float32: 1e-4}
    all_pass = True

    shapes = [
        (1, 896),
        (1, 2048),
        (1, 4096),
        (32, 4096),
        (128, 4096),
        (512, 4096),
        (2048, 4096),
        (1, 8192),
        (128, 8192),
    ]

    for dtype in [torch.float16, torch.bfloat16]:
        for num_tokens, hidden_size in shapes:
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            w = torch.randn(hidden_size, dtype=dtype, device=device)

            ref = rmsnorm_pytorch(x, w, eps)
            out1 = rmsnorm_autotune(x, w, eps)
            out2 = rmsnorm_autotune_2d(x, w, eps)

            diff1 = (out1 - ref).abs().max().item()
            diff2 = (out2 - ref).abs().max().item()
            ok1 = diff1 < tol[dtype]
            ok2 = diff2 < tol[dtype]
            all_pass &= ok1 & ok2
            tag1 = "PASS" if ok1 else "FAIL"
            tag2 = "PASS" if ok2 else "FAIL"
            print(f"  [{tag1}] 1d-key  dtype={dtype}, shape=({num_tokens:>5}, {hidden_size:>5}), max_diff={diff1:.6f}")
            print(f"  [{tag2}] 2d-key  dtype={dtype}, shape=({num_tokens:>5}, {hidden_size:>5}), max_diff={diff2:.6f}")

    print(f"\nAll correctness tests {'PASSED' if all_pass else 'FAILED'}.")
    return all_pass


# ---------------------------------------------------------------------------
# Extract the winning config from triton's autotune cache
# ---------------------------------------------------------------------------

def _get_best_config(kernel):
    """Return dict of { key_tuple: triton.Config } from the autotuner cache."""
    if not hasattr(kernel, "cache"):
        return {}
    results = {}
    for key, config in kernel.cache.items():
        results[key] = config
    return results


# ---------------------------------------------------------------------------
# Comprehensive shape benchmark + config report
# ---------------------------------------------------------------------------

# Shapes that cover real model hidden sizes (Qwen3-0.6B=896, LLaMA-7B=4096,
# LLaMA-13B=5120, LLaMA-70B=8192, Qwen-72B=8192, Mixtral=4096, etc.)
# crossed with token counts spanning decode → large prefill.

HIDDEN_SIZES = [896, 1536, 2048, 3072, 4096, 5120, 6144, 8192]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

ALL_SHAPES = list(itertools.product(TOKEN_COUNTS, HIDDEN_SIZES))


def run_benchmark():
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    warmup = 200
    rep = 500

    # --- Phase 1: trigger autotune for every shape (2d-key kernel) ---
    print(f"\n{'='*100}")
    print(f"  Phase 1: Autotuning {len(ALL_SHAPES)} shapes ({len(ALL_CONFIGS)} configs each)")
    print(f"{'='*100}")
    t0 = time.time()

    for i, (num_tokens, hidden_size) in enumerate(ALL_SHAPES):
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)
        _ = rmsnorm_autotune_2d(x, w, eps)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  ... autotuned {i+1}/{len(ALL_SHAPES)} shapes")

    autotune_time = time.time() - t0
    print(f"  Autotune complete in {autotune_time:.1f}s\n")

    # --- Phase 2: report best configs ---
    print(f"{'='*100}")
    print(f"  Best autotuned configs per shape (2d-key: keyed on num_tokens × hidden_size)")
    print(f"{'='*100}")
    print(f"{'num_tokens':>12} {'hidden_size':>12} │ {'BLOCK_SIZE':>10} {'num_warps':>10} {'num_stages':>10}")
    print(f"{'─'*12} {'─'*12} ┼ {'─'*10} {'─'*10} {'─'*10}")

    best_configs_2d = _get_best_config(_rmsnorm_kernel_2d_key)
    for key in sorted(best_configs_2d.keys()):
        cfg = best_configs_2d[key]
        num_tokens, hidden_size = key
        bs = cfg.kwargs.get("BLOCK_SIZE", "?")
        nw = cfg.num_warps
        ns = cfg.num_stages
        print(f"{num_tokens:>12} {hidden_size:>12} │ {bs:>10} {nw:>10} {ns:>10}")

    # --- Phase 3: also trigger 1d-key autotune ---
    print(f"\n{'='*100}")
    print(f"  Best autotuned configs (1d-key: keyed on hidden_size only)")
    print(f"{'='*100}")

    for hidden_size in HIDDEN_SIZES:
        x = torch.randn(256, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)
        _ = rmsnorm_autotune(x, w, eps)

    best_configs_1d = _get_best_config(_rmsnorm_kernel)
    print(f"{'hidden_size':>12} │ {'BLOCK_SIZE':>10} {'num_warps':>10} {'num_stages':>10}")
    print(f"{'─'*12} ┼ {'─'*10} {'─'*10} {'─'*10}")
    for key in sorted(best_configs_1d.keys()):
        cfg = best_configs_1d[key]
        (hidden_size,) = key
        bs = cfg.kwargs.get("BLOCK_SIZE", "?")
        nw = cfg.num_warps
        ns = cfg.num_stages
        print(f"{hidden_size:>12} │ {bs:>10} {nw:>10} {ns:>10}")

    # --- Phase 4: actual latency comparison vs PyTorch ---
    print(f"\n{'='*100}")
    print(f"  Latency comparison: Triton autotuned (2d-key) vs PyTorch")
    print(f"{'='*100}")
    print(f"{'num_tokens':>12} {'hidden_size':>12} │ {'Triton (us)':>12} {'PyTorch (us)':>13} {'Speedup':>8}")
    print(f"{'─'*12} {'─'*12} ┼ {'─'*12} {'─'*13} {'─'*8}")

    for num_tokens, hidden_size in ALL_SHAPES:
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)

        triton_ms = triton.testing.do_bench(
            lambda: rmsnorm_autotune_2d(x, w, eps), warmup=warmup, rep=rep
        )
        pytorch_ms = triton.testing.do_bench(
            lambda: rmsnorm_pytorch(x, w, eps), warmup=warmup, rep=rep
        )

        triton_us = triton_ms * 1000
        pytorch_us = pytorch_ms * 1000
        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float("inf")

        print(f"{num_tokens:>12} {hidden_size:>12} │ {triton_us:>12.1f} {pytorch_us:>13.1f} {speedup:>7.2f}x")

    # --- Phase 5: compare 1d-key vs 2d-key ---
    print(f"\n{'='*100}")
    print(f"  1d-key vs 2d-key autotune comparison")
    print(f"{'='*100}")
    print(f"{'num_tokens':>12} {'hidden_size':>12} │ {'1d-key (us)':>12} {'2d-key (us)':>12} {'Winner':>8}")
    print(f"{'─'*12} {'─'*12} ┼ {'─'*12} {'─'*12} {'─'*8}")

    for num_tokens, hidden_size in ALL_SHAPES:
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)

        ms_1d = triton.testing.do_bench(
            lambda: rmsnorm_autotune(x, w, eps), warmup=warmup, rep=rep
        )
        ms_2d = triton.testing.do_bench(
            lambda: rmsnorm_autotune_2d(x, w, eps), warmup=warmup, rep=rep
        )

        us_1d = ms_1d * 1000
        us_2d = ms_2d * 1000
        winner = "2d" if us_2d <= us_1d else "1d"
        print(f"{num_tokens:>12} {hidden_size:>12} │ {us_1d:>12.1f} {us_2d:>12.1f} {winner:>8}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal RMSNorm Triton kernel with exhaustive autotune"
    )
    parser.add_argument("--test-only", action="store_true", help="Only run correctness tests")
    parser.add_argument("--bench-only", action="store_true", help="Only run benchmark")
    args = parser.parse_args()

    if args.test_only:
        test_correctness()
    elif args.bench_only:
        run_benchmark()
    else:
        ok = test_correctness()
        print()
        if ok:
            run_benchmark()
        else:
            print("Skipping benchmark due to correctness failures.")
