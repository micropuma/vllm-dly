import torch
import os

from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "cache")


if __name__ == "__main__":
    # 按照Llama-7B 配置测试
    num_tokens = 128    
    num_heads = 32     
    head_size = 128   

    output = torch.zeros((num_tokens, num_heads, head_size), device="cuda", dtype=torch.bfloat16)
    output_lse = torch.empty((num_heads, num_tokens), device="cuda", dtype=torch.float32)
    prefix_output = torch.randn((num_tokens, num_heads, head_size), device="cuda", dtype=torch.bfloat16)
    prefix_lse = torch.randn((num_heads, num_tokens), device="cuda", dtype=torch.float32)
    suffix_output = torch.randn((num_tokens, num_heads, head_size), device="cuda", dtype=torch.bfloat16)
    suffix_lse = torch.randn((num_heads, num_tokens), device="cuda", dtype=torch.float32)

    merge_attn_states(
        output, 
        prefix_output, 
        prefix_lse, 
        suffix_output, 
        suffix_lse, 
        output_lse
    )

    print("Output shape:", output.shape)
    print("Output_LSE shape:", output_lse.shape)