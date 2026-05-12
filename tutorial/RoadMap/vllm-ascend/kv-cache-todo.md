# KV Cache TODO

这份文档不是实现方案，而是我给自己准备的 `KV cache dump / 阅读 / 验证路线`。

目标分两层：

- 先达到：能支持和调试 `vllm-ascend` 的 `kv_cache_memory_bytes`
- 再达到：真正吃透 vLLM KV cache 的初始化和运行时机制

---

## Phase 1: 先打通主线

目标：

- 搞清楚 `kv spec -> KVCacheConfig -> worker 初始化` 这条主线
- 明确 `kv_cache_memory_bytes` 改的是哪一段，不改哪一段

要回答的问题：

- 每个 worker 的 `kv spec` 从哪里来
- `available_memory` 从哪里来
- `KVCacheConfig` 是谁生成的
- 每个 worker 怎么拿到自己的那份 config
- `num_blocks` 在哪一层被统一

重点文件：

- `vllm/v1/engine/core.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/worker_base.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`

建议 dump：

- `get_kv_cache_spec()` 的返回结果
- `determine_available_memory()` 的返回值
- `get_kv_cache_configs(...)` 生成的每个 worker 的：
  - `num_blocks`
  - `kv_cache_tensors`
  - `kv_cache_groups`
- `min_num_blocks` 对齐前后变化

完成标准：

- 能用一句话讲清楚：`kv_cache_memory_bytes` 影响的是 `available_memory -> num_blocks -> KVCacheConfig`

---

## Phase 2: 吃透大小计算

目标：

- 搞清楚 KV cache 大小到底怎么算
- 能从模型参数估算 page size / num_blocks

要回答的问题：

- `KVCacheSpec.page_size_bytes` 是怎么定义的
- 不同 spec 为什么 page size 不一样
- uniform spec / uniform type / hybrid group 的差别是什么
- `KVCacheTensor.size` 为什么有时是一层一块，有时是一组共享

重点文件：

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`

建议 dump：

- 每层 spec 的：
  - `block_size`
  - `num_kv_heads`
  - `head_size`
  - `dtype`
  - `page_size_bytes`
- 每个 group 的：
  - `layer_names`
  - `group_size`
  - `page_size`
- 每个 worker 的：
  - `available_memory`
  - `num_blocks`
  - `kv_cache_tensor.size`

完成标准：

- 能解释为什么同样的 `kv_cache_memory_bytes` 最终会变成某个 `num_blocks`

---

## Phase 3: 看懂每个 GPU 上的真实分配

目标：

- 搞清楚 worker 上到底在哪里真正开辟 KV cache
- 搞清楚 raw bytes 和逻辑 KV layout 的关系

要回答的问题：

- `initialize_kv_cache()` 里每一步分别在做什么
- `initialize_attn_backend()` 为什么先于分配
- `kernel_block_size` 和 `kv manager block_size` 为什么可能不同
- `torch.zeros(...)` 分配出来的 raw tensor 后面怎么被 `view/permute/as_strided`

重点文件：

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`

重点函数：

- `initialize_kv_cache`
- `initialize_attn_backend`
- `_prepare_kernel_block_sizes`
- `initialize_kv_cache_tensors`
- `_allocate_kv_cache_tensors`
- `_reshape_kv_cache_tensors`
- `allocate_uniform_kv_caches`

建议 dump：

- `kv_cache_config` 进入 `initialize_kv_cache()` 前后的变化
- 每个 group 的：
  - attention backend
  - kv manager block size
  - kernel block size
- 每个 `KVCacheTensor` 的：
  - `size`
  - `shared_by`
- 每个 layer 最终绑定到的 tensor：
  - shape
  - dtype
  - stride

完成标准：

- 能回答“每张卡上 KV cache 真正在哪一行代码开辟”

---

## Phase 4: 看懂 block table 和 slot mapping

目标：

- 不再把 KV tensor、block table、slot mapping 混在一起
- 搞清楚 request 的 block ids 是谁分配、谁消费

要回答的问题：

- block table 初始化时到底有什么，没有什么
- scheduler 分配的 block ids 怎么传到 worker
- slot mapping 为什么是 token 级别
- hybrid block size 下 block id 怎么映射到 kernel block id

重点文件：

- `vllm/v1/worker/block_table.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`

重点函数：

- `KVCacheManager.allocate_slots`
- `Scheduler.schedule`
- `GPUModelRunner._update_states`
- `GPUModelRunner._prepare_inputs`
- `BlockTable.append_row`
- `BlockTable.compute_slot_mapping`

建议 dump：

- scheduler 侧：
  - `new_block_ids`
  - `req_to_new_blocks`
- worker 侧：
  - `req_state.block_ids`
  - `input_batch.block_table`
  - `slot_mapping`

完成标准：

- 能解释：
  - block table 管什么
  - slot mapping 管什么
  - 它们在每一步怎么更新

---

## Phase 5: 反推 Ascend 落点

目标：

- 把 CUDA v1 主线映射到 `vllm-ascend`
- 找到支持 `kv_cache_memory_bytes` 最小需要改动的入口

要回答的问题：

- `vllm-ascend` 对应的 worker / model runner / executor 在哪
- Ascend 有没有自己的 `determine_available_memory()` 逻辑
- Ascend 有没有自己的 profile/warmup 逻辑
- Ascend 最终是不是也复用 `get_kv_cache_configs(...)`
- 哪些路径是共用 vLLM 主干，哪些路径是 Ascend 自己实现

建议动作：

1. 对照 CUDA v1 路径列一份 Ascend 映射表
2. 标出“共用代码”和“Ascend 特化代码”
3. 先只做 `kv_cache_memory_bytes` 的最小闭环
4. 暂时不要碰 KV layout 细节，除非 Ascend 已经特化了这部分

建议 dump：

- Ascend worker 的可用显存计算入口
- Ascend profile 后用于 KV sizing 的返回值
- Ascend 初始化后最终得到的：
  - `num_blocks`
  - `kv_cache_tensors`
  - `kv_cache_groups`

完成标准：

- 能明确写出：为了支持 `kv_cache_memory_bytes`，Ascend 需要改哪几个函数

---

## Phase 6: 进入运行时和复杂特性

目标：

- 不只懂初始化，还能看懂运行时为什么会分配、回收、复用 block

要回答的问题：

- prefix caching 怎么影响 block 分配
- preemption / resume 怎么影响 block ids
- sliding window / chunked local attention 怎么影响 memory usage
- Mamba / MLA / hybrid attention 为什么让 KV cache 更复杂
- kv transfer / kv sharing / offload 在初始化和运行时插在哪

重点文件：

- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/distributed/kv_transfer/...`

完成标准：

- 能 debug：
  - 为什么 block 数够却 still fail
  - 为什么某些 request 的 block ids 变化异常
  - 为什么某类 hybrid model 的 KV cache 初始化和普通模型不同

---

## 最小优先级

如果目标只是支持 `vllm-ascend` 的 `kv_cache_memory_bytes`，优先级按这个顺序：

1. Phase 1
2. Phase 2
3. Phase 5
4. Phase 3
5. Phase 4
6. Phase 6

如果目标是彻底吃透 KV cache，顺序按这个走：

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

---

## 最后一句话

先别把问题定义成“我要一口气弄懂 KV cache 全机制”。

更实际的路线是：

- 先把 `kv_cache_memory_bytes` 这条 sizing 主线吃透
- 再把 worker 分配看懂
- 再把 block table / slot mapping 接上
- 最后再扩到 prefix cache、preemption、hybrid model、kv transfer
