import time
import torch
from vllm import LLM, SamplingParams
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

# 设置采样参数
prompts = ["The future of AI is", "What is the capital of France?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B",
        tensor_parallel_size=2,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile_llama8b",
        },
    )

    print("Starting profile...")
    llm.start_profile()

    # 执行推理
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()
    print("Profiling finished. Waiting for flush...")
    
    # 给足够的时间让多进程将 Trace 数据写入磁盘
    time.sleep(15)

if __name__ == "__main__":
    main()