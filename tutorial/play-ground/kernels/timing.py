""""
    CUDA 内核时间分析脚本  
    主要参考：https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
"""

import torch
import subprocess
import os
from typing import Callable

def set_clock_speed(clock_speed: int = 1350):
    """锁定 GPU 时钟频率，消除测试波动 (需要 sudo 权限)"""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"[*] Locking GPU clock to {clock_speed}MHz...")
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {device}", shell=True, check=True)
    subprocess.run(f"sudo nvidia-smi -lgc {clock_speed} -i {device}", shell=True, check=True)

def reset_clock_speed():
    """恢复 GPU 默认频率"""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print("[*] Resetting GPU clocks...")
    subprocess.run(f"sudo nvidia-smi -rgc -i {device}", shell=True, check=True)

CACHE_FLUSH_BUFFER = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')

def flush_cache():
    """通过写入全零数据覆盖 L2 Cache"""
    CACHE_FLUSH_BUFFER.zero_()

def benchmark_cuda_kernel(
    kernel_func: Callable, 
    *args, 
    warmup_iters: int = 20, 
    benchmark_iters: int = 100,
    use_cuda_graph: bool = False
):
    """
    一个完整的 CUDA 算子性能评测模板
    """
    # A. 预热 (Warm-up)
    # 消除 JIT 编译、内存分配、CUDA Context 初始化带来的开销
    for _ in range(warmup_iters):
        kernel_func(*args)
    torch.cuda.synchronize()

    # B. 设置 CUDA Events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    # C. 使用 CUDA Graphs (可选优化，针对超轻量算子)
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel_func(*args)
        
        for i in range(benchmark_iters):
            start_events[i].record()
            graph.replay()
            end_events[i].record()
    else:
        # D. 执行循环 (带缓存冲刷与防空转)
        for i in range(benchmark_iters):
            flush_cache()  # 冲刷缓存
            torch.cuda._sleep(100_000) # 防止 CPU "outrun" GPU
            
            start_events[i].record()
            kernel_func(*args)
            end_events[i].record()

    # E. 同步并统计
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    avg_time = sum(times) / benchmark_iters
    return avg_time

# ==========================================
# 4. 使用示例
# ==========================================
if __name__ == "__main__":
    # 模拟一个算子
    def my_kernel(x):
        return torch.matmul(x, x)

    x = torch.randn(1024, 1024, device='cuda')

    try:
        # 实际操作建议：手动在终端运行 sudo 权限操作，或者在脚本中按需调用
        # set_clock_speed() 
        
        latency = benchmark_cuda_kernel(my_kernel, x)
        print(f"Average Kernel Latency: {latency:.4f} ms")
        
    finally:
        # reset_clock_speed()
        pass