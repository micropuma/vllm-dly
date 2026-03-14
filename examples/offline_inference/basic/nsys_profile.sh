# 直接在命令行里回车运行
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    vllm bench latency --model Qwen/Qwen3-0.6B \
        --num-iters-warmup 5 --num-iters 1 --batch-size 16 --input-len 512 --output-len 8