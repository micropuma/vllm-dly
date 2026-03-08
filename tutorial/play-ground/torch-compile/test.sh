export TORCH_LOGS=output_code 
export CUDA_LAUNCH_BLOCKING=1
export TORCH_TRACE=/home/douliyang/large/mlsys/vllm-dly/tutorial/play-ground/torch-compile/trace-dir

python rms_norm.py

tlparse trace-dir/dedicated_log_torch_trace_wykhcb9a.log