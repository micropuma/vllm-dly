export TORCH_LOGS=output_code 
export CUDA_LAUNCH_BLOCKING=1
export TORCH_TRACE=/home/douliyang/large/mlsys/vllm-dly/examples/offline_inference/basic/trace_dir 
python generate.py --model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 2 --max-tokens 1024 --temperature 0.8 2>&1 | tee llama3-8B-debug.log

# 解析日志
tlparse trace_dir/dedicated_log_torch_trace_rank_0_hbg_vvrj.log    