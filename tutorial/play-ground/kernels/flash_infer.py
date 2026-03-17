import torch
import flashinfer

def test_flashinfer_topk_topp():
    # 1. 初始化
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    
    batch_size = 4
    vocab_size = 5
    top_k = 3
    top_p = 0.5
    
    # 2. 准备数据
    logits = torch.rand(batch_size, vocab_size, device=device).contiguous()
    
    # 3. 执行 FlashInfer 融合采样
    # deterministic=True 确保在基准测试或调试时结果可控
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, 
        top_k, 
        top_p, 
        deterministic=True
    )
    
    # 4. 计算概率分布用于对照
    probs = torch.softmax(logits, dim=-1)
    
    # 5. 输出对照信息
    print(f"{'Batch':<8} | {'Token ID':<10} | {'Prob':<10}")
    print("-" * 35)
    for i in range(batch_size):
        token_id = samples[i].item()
        prob_val = probs[i, token_id].item()
        print(f"{i:<8} | {token_id:<10} | {prob_val:<10.4f}")

    # 6. 简单的逻辑校验
    # 确保采样的 ID 是在 [0, vocab_size-1] 范围内
    assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
    print("\n测试通过: 采样结果有效。")

if __name__ == "__main__":
    test_flashinfer_topk_topp()