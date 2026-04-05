vllm serve /home/douliyang/large/mlsys/vllm-dly/models/Qwen3-0.6B --trust-remote-code

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --backend vllm \
  --model /home/douliyang/large/mlsys/vllm-dly/models/Qwen3-0.6B \
  --trust-remote-code \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /home/douliyang/large/mlsys/vllm-dly/benchmarks/online_benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
