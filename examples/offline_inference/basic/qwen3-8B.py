from vllm import LLM, SamplingParams

def main():
    # 注意：初始化 LLM 放在函数里，避免多进程初始化冲突
    llm = LLM(
        model="Qwen/Qwen3-8B",     
        tokenizer="Qwen/Qwen3-8B",  
        tensor_parallel_size=4,        
    )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    prompts = ["What is the difference between a compiler and an interpreter?",
               "Hello, how are you?",
    ]

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == "__main__":
    main()