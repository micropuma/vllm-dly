# use gernerate.py and chat.py to do qwen3-8B inference
python generate.py --model Qwen/Qwen3-8B --cpu-offload-gb 10 --max-tokens 512 --temperature 0.8         # 单卡放不下，使用cpu offload技术
python generate.py --model Qwen/Qwen3-8B --tensor-parallel-size 2 --max-tokens 1024 --temperature 0.8   # 使用tensor parallel技术，分布在两张卡上

export VLLM_TRACE_FUNCTION=1

# try a llama-8B model  
CUDA_VISIBLE_DEVICES=2,3 python generate.py --model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 2 --max-tokens 1024 --temperature 0.8

# unset environment variables after debugging
unset VLLM_LOGGING_LEVEL
unset CUDA_LAUNCH_BLOCKING
unset NCCL_DEBUG
unset VLLM_TRACE_FUNCTION  

