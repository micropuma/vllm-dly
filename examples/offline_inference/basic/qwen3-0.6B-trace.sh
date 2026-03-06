TORCH_LOGS=output_code  TORCH_TRACE=/home/douliyang/large/mlsys/vllm-dly/examples/offline_inference/basic/trace_dir_2 python qwen3-0.6B.py

# # 解析日志
# tlparse trace_dir/dedicated_log_torch_trace_rank_0_hbg_vvrj.log  