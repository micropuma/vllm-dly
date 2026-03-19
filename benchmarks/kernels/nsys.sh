CUDA_VISIBLE_DEVICES=1 nsys profile --cuda-graph-trace=node \
    python3 benchmark_rmsnorm_v2.py \
    --hidden-sizes 7168 \
    --calls-per-run 3 \
    --use-cuda-graph