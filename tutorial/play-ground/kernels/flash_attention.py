"""FlashAttention-2 forward pass in Triton.

A simplified, self-contained implementation of the FlashAttention-2 forward
kernel using online softmax for O(1) extra memory.  Based on:

    Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning", 2023.

Supports causal and non-causal modes.  FP16 input only (no FP8 / backward).
"""

import os

import pytest
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
    for bm in [64, 128]
    for bn in [32, 64]
    for s in [2, 3]
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},
                       num_stages=2, num_warps=4),
    ]


def _prune_configs(configs, named_args, **kwargs):
    """Remove configs where BLOCK_M > SEQ_LEN or BLOCK_M < BLOCK_N (causal)."""
    seq_len = kwargs["SEQ_LEN"]
    stage = kwargs["STAGE"]
    return [
        c for c in configs
        if c.kwargs["BLOCK_M"] <= seq_len
        and (c.kwargs["BLOCK_M"] >= c.kwargs["BLOCK_N"] or stage == 1)
    ]


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_inner(
    acc, l_i, m_i, q,
    K, V,
    qkv_offset, start_m, qk_scale,
    stride_seq, stride_dim,
    offs_m,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Inner loop: iterate over K/V blocks and update online-softmax state.

    Args:
        STAGE: 1 = off-band (no mask), 2 = on-band (causal mask),
               3 = full range (non-causal).
    """
    # Determine the K/V range for this stage.
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, SEQ_LEN

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    k_ptrs = K + qkv_offset + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_dim
    v_ptrs = V + qkv_offset + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_dim

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # QK^T: [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        k = tl.load(k_ptrs + start_n * stride_seq)
        qk = tl.dot(q, tl.trans(k))

        # Scale + optional causal mask, then subtract row-max for numerical stability.
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # Online softmax: exp, rescale, accumulate P @ V  (all in FP32).
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)                       # FP32 sum before cast
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_n * stride_seq)
        p = p.to(tl.float16)                      # cast for tensor-core dot
        acc = tl.dot(p, v, acc)

        # Update running softmax stats (placed last to reduce register pressure).
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["SEQ_LEN", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _flash_attn_fwd(
    Q, K, V, O, LSE,
    sm_scale,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash-attention forward kernel.

    Grid: (ceil(SEQ_LEN / BLOCK_M),  BATCH * NUM_HEADS).
    Each program handles one [BLOCK_M, HEAD_DIM] tile of Q.
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # Locate (batch, head, query-block).
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // NUM_HEADS
    off_h = off_bh % NUM_HEADS
    qkv_offset = (off_b.to(tl.int64) * stride_batch
                  + off_h.to(tl.int64) * stride_head)

    # Load Q tile [BLOCK_M, HEAD_DIM] — stays in SRAM.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = (Q + qkv_offset
              + offs_m[:, None] * stride_seq
              + offs_d[None, :] * stride_dim)
    q = tl.load(q_ptrs)

    # Online-softmax accumulators (all FP32).
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)

    # Causal stage dispatch (bit-trick):
    #   STAGE=3 (causal):     bit0 -> off-band (inner 1), bit1 -> on-band (inner 2)
    #   STAGE=1 (non-causal): bit0 -> full range (inner 3)
    if STAGE & 1:
        acc, l_i, m_i = _flash_attn_inner(
            acc, l_i, m_i, q, K, V,
            qkv_offset, start_m, qk_scale,
            stride_seq, stride_dim, offs_m,
            HEAD_DIM, BLOCK_M, BLOCK_N, SEQ_LEN,
            STAGE=4 - STAGE,
        )
    if STAGE & 2:
        acc, l_i, m_i = _flash_attn_inner(
            acc, l_i, m_i, q, K, V,
            qkv_offset, start_m, qk_scale,
            stride_seq, stride_dim, offs_m,
            HEAD_DIM, BLOCK_M, BLOCK_N, SEQ_LEN,
            STAGE=2,
        )

    # Epilogue: normalize and store.
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    lse_ptrs = LSE + off_bh * SEQ_LEN + offs_m
    tl.store(lse_ptrs, m_i)

    o_ptrs = (O + qkv_offset
              + offs_m[:, None] * stride_seq
              + offs_d[None, :] * stride_dim)
    tl.store(o_ptrs, acc.to(tl.float16))


# ---------------------------------------------------------------------------
# Python wrapper (autograd Function)
# ---------------------------------------------------------------------------
class _FlashAttention(torch.autograd.Function):
    """Autograd wrapper for the FlashAttention-2 forward pass."""

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=True):
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor  [B, H, N, D], float16.
            k: Key tensor    [B, H, N, D], float16.
            v: Value tensor  [B, H, N, D], float16.
            sm_scale: Softmax scaling factor (typically 1/sqrt(D)).
            causal: Whether to apply causal masking.

        Returns:
            Output tensor [B, H, N, D], same dtype as q.
        """
        batch, num_heads, seq_len, head_dim = q.shape
        assert head_dim in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        lse = torch.empty(
            (batch, num_heads, seq_len), device=q.device, dtype=torch.float32,
        )

        grid = lambda META: (
            triton.cdiv(seq_len, META["BLOCK_M"]),
            batch * num_heads,
        )

        _flash_attn_fwd[grid](
            q, k, v, o, lse,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            STAGE=3 if causal else 1,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = head_dim
        ctx.causal = causal
        return o


attention = _FlashAttention.apply


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [2, 32])
def test_op(batch, seq_len, head_dim, num_heads,
            dtype=torch.float16, device=DEVICE):
    """Compare Triton FlashAttention output against a naive PyTorch reference."""
    torch.manual_seed(20)
    q = torch.empty(batch, num_heads, seq_len, head_dim,
                    device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
    k = torch.empty(batch, num_heads, seq_len, head_dim,
                    device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
    v = torch.empty(batch, num_heads, seq_len, head_dim,
                    device=device, dtype=dtype).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    # Reference: standard matmul attention with FP32 softmax.
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v).half()

    tri_out = attention(q, k, v, sm_scale).half()
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _naive_attention(q, k, v, sm_scale, causal=True):
    """Naive PyTorch attention for benchmarking (O(N^2) memory)."""
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        n = q.shape[2]
        mask = torch.tril(torch.ones(n, n, device=q.device))
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


_BENCH_BATCH, _BENCH_HEADS = 4, 32
_bench_configs = []
for _hd in [64, 128]:
    _bench_configs.append(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton FlashAttn", "PyTorch Naive"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=f"flash-attn-B{_BENCH_BATCH}-H{_BENCH_HEADS}-D{_hd}",
            args={
                "num_heads": _BENCH_HEADS,
                "batch": _BENCH_BATCH,
                "head_dim": _hd,
            },
        ))


@triton.testing.perf_report(_bench_configs)
def bench_flash_attn(batch, seq_len, head_dim, num_heads,
                     provider, device=DEVICE):
    """Benchmark FlashAttention vs naive PyTorch."""
    dtype = torch.float16
    q = torch.randn(batch, num_heads, seq_len, head_dim,
                    dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim,
                    dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim,
                    dtype=dtype, device=device)
    sm_scale = 1.3

    if provider == "triton":
        fn = lambda: attention(q, k, v, sm_scale)
    elif provider == "pytorch":
        mem_bytes = batch * num_heads * seq_len * seq_len * 2
        free_mem = torch.cuda.mem_get_info(device)[0]
        if mem_bytes * 3 > free_mem:
            return float("nan")
        fn = lambda: _naive_attention(q, k, v, sm_scale)

    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * batch * num_heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul * 0.5  # Q@K^T + P@V, causal halves
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_flash_attn.run(save_path=".", print_data=True)
