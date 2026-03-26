import torch
import triton
import triton.language as tl
import pytest

"""
    The simplest FlashAttention implementation in Triton language.
    The implementation is based on the paper "FlashAttention: Fast and Memory-Efficient Attention with IO-Awareness" by Narayan et al.
    This implementatin is v2 version, and support both foward and backward pass.
"""

DEVICE = triton.runtime.driver.active.get_active_torch_device()

BENCH_BATCH, BENCH_HEADS = 4, 32
configs = []
for HEAD_DIM in [64, 128]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton FlashAttn", "PyTorch Naive"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=
            f"fused-attention-batch{BENCH_BATCH}-head{BENCH_HEADS}-d{HEAD_DIM}",
            args={
                "num_heads": BENCH_HEADS,
                "batch": BENCH_BATCH,
                "head_dim": HEAD_DIM,
            },
        ))

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
    # STAGE 1: off-band (before diagonal), no mask
    # STAGE 2: on-band (diagonal block), causal mask
    # STAGE 3: non-causal full range
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, SEQ_LEN

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + qkv_offset + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_dim
    v_ptrs = V + qkv_offset + offs_n[:, None] * stride_seq + offs_d[None, :] * stride_dim

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # (1) Load K [BLOCK_N, HEAD_DIM], compute QK^T -> [BLOCK_M, BLOCK_N]
        k = tl.load(k_ptrs + start_n * stride_seq)
        qk = tl.dot(q, tl.trans(k))

        # (2) Scale + optional causal mask
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # (3) P = exp2(qk)  — stays in FP32
        p = tl.math.exp2(qk)

        # (4) Correction factor + softmax denominator in FP32
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # (5) Rescale acc, load V, accumulate P @ V
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * stride_seq)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        # (6) Update running softmax stats (l_ij was computed in FP32 above)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _flash_attn_fwd(
    Q, K, V, O, LSE,
    sm_scale,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # Step A: compute offsets for this (batch, head, query-block)
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)             # 这个是Batch * num_heads
    off_b = off_bh // NUM_HEADS
    off_h = off_bh % NUM_HEADS
    qkv_offset = off_b.to(tl.int64) * stride_batch + off_h.to(tl.int64) * stride_head

    # Step B: load Q block [BLOCK_M, HEAD_DIM] — stays in registers
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_seq + offs_d[None, :] * stride_dim
    q = tl.load(q_ptrs)

    # Initialize online-softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)

    # Step C: causal stage dispatch
    # STAGE=3 (causal):     bit0=1 → off-band (inner STAGE=1), bit1=1 → on-band (inner STAGE=2)
    # STAGE=1 (non-causal): bit0=1 → full scan (inner STAGE=3), bit1=0 → skip
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

    # Step D: epilogue — normalize and store
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    lse_ptrs = LSE + off_bh * SEQ_LEN + offs_m
    tl.store(lse_ptrs, m_i)

    o_ptrs = O + qkv_offset + offs_m[:, None] * stride_seq + offs_d[None, :] * stride_dim
    tl.store(o_ptrs, acc.to(tl.float16))

class _flash_attention_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=True):
        BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = q.shape
        assert HEAD_DIM in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        M = torch.empty((BATCH, NUM_HEADS, SEQ_LEN), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(SEQ_LEN, BLOCK_M), BATCH * NUM_HEADS)

        _flash_attn_fwd[grid](
            Q=q, K=k, V=v, O=o, LSE=M,
            sm_scale=sm_scale,
            stride_batch=q.stride(0), stride_head=q.stride(1),
            stride_seq=q.stride(2), stride_dim=q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=3 if causal else 1,
            num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        return o

attention = _flash_attention_impl.apply

@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [2, 32])
def test_op(
    batch, seq_len, head_dim, num_heads, detype=torch.float16, device=DEVICE
):
    """
    Unit test for the flash attention implementation.

    Args:
        batch (int): The batch size of the input tensors.
        seq_len (int): The sequence length.
        head_dim (int): The dimension of each attention head.
        num_heads (int): The number of attention heads.
        detype(torch.dtype): The data type of the input tensors.
        device (str, optional): The target device to run the benchmark. 
            Defaults to "cuda".
    """

    torch.manual_seed(20)
    q = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE, dtype=detype).normal_(mean=0.0, std=0.5)
    k = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE, dtype=detype).normal_(mean=0.0, std=0.5)
    v = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE, dtype=detype).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5
    mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(detype)
    ref_out = torch.matmul(p, v).half()
    tri_out = attention(q, k, v, sm_scale).half()

    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)

def naive_pytorch_attention(q, k, v, sm_scale, causal=True):
    """Naive PyTorch attention for benchmarking — O(N^2) memory."""
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        seq_len = q.shape[2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


@triton.testing.perf_report(configs)
def bench_flash_attn(batch, seq_len, head_dim, num_heads, provider, device=DEVICE):
    dtype = torch.float16
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    sm_scale = 1.3

    if provider == "triton":
        fn = lambda: attention(q, k, v, sm_scale)
    elif provider == "pytorch":
        # Naive attention materializes [B, H, N, N] — skip if it would OOM
        attn_bytes = batch * num_heads * seq_len * seq_len * 2  # fp16
        free_mem = torch.cuda.mem_get_info(device)[0]
        if attn_bytes * 3 > free_mem:  # 3x for qk, softmax, pv intermediates
            return float("nan")
        fn = lambda: naive_pytorch_attention(q, k, v, sm_scale)

    ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * batch * num_heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul  # Q@K^T + P@V
    total_flops *= 0.5  # causal
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    bench_flash_attn.run(save_path=".", print_data=True)
