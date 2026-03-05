from vllm import LLM, SamplingParams

def main():
    # 初始化 LLM 放在函数里，避免多进程初始化冲突
    llm = LLM(
        model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
        tokenizer="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,
    )

    # 限制生成长度，避免 KV cache 溢出
    sampling_params = SamplingParams(max_tokens=128)

    # prompt 必须是 list[str]，避免模板渲染出错
    prompts = ["Explain mixture of experts in simple terms."]

    # 使用关键字参数传 sampling_params
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # 输出每条结果
    for i, output in enumerate(outputs):
        # output.outputs[0] 是第一个生成序列
        print(f"Prompt {i}:")
        print(output.outputs[0].text)

    # # 显式释放资源，避免多次运行时 EngineCore 崩溃
    # del llm
    # import gc
    # gc.collect()

if __name__ == "__main__":
    main()