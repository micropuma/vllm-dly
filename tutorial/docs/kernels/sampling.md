# vLLM v1 Sampling 流程和算子实现

在使用pytorch profiler等工具对vLLM做性能分析的过程中，发现sampler过程是一大瓶颈。早期的sampler是在cpu上实现，vLLM开发者已经将其移到GPU上加速运行，但依旧耗时较长。

本文档详细解读 vLLM v1 架构中的 sampling 流程、torch.compile 对调试sampler流程的影响，以及各个 sampling 算子的实现细节，从而辅助理解sampler的性能瓶颈和可能优化点。

## 目录

- [1. 完整的 Sampling 流程](#1-完整的-sampling-流程)
- [2. 详细调用链：从 Model Runner 到 Triton Kernels](#2-详细调用链从-model-runner-到-triton-kernels)
- [3. torch.compile 与调试问题](#3-torchcompile-与调试问题)
- [4. Sample 算子详解](#4-sample-算子详解)
- [参考资料](#参考资料)

---

## 1. 完整的 Sampling 流程

### 1.1 调用链路

整个 sampling 流程从 `gpu_model_runner.py` 的 `_sample()` 方法触发，调用 `self.sampler(logits, ...)` 进入 [`sampler.py:57`](../../../vllm/v1/worker/gpu/sample/sampler.py#L57) 的 `__call__` 方法，然后执行 `sample()` 方法：

```
LLM forward pass → hidden_states → compute_logits → logits [num_reqs, vocab_size]
                                                          ↓
                                              Sampler.__call__()
                                                          ↓
                                              Sampler.sample()
```

### 1.2 八步处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: copy logits to FP32                                                │
│          将 logits 复制到 FP32 tensor，确保数值精度                            │
│                                                                             │
│  Step 2: apply_logit_bias                                                   │
│          - allowed_token_ids: 只允许特定 token（其余置 -inf）                  │
│          - logit_bias: 对特定 token 加偏置                                    │
│          - min_tokens: 在生成足够 token 前屏蔽 stop token                      │
│                                                                             │
│  Step 3: apply_penalties                                                    │
│          - repetition_penalty: 对已出现 token 施加重复惩罚                     │
│          - frequency_penalty: 根据出现频次线性惩罚                             │
│          - presence_penalty: 对已出现 token 施加固定惩罚                       │
│                                                                             │
│  Step 4: apply_bad_words                                                    │
│          检查 context 末尾是否匹配 bad word prefix，屏蔽对应 token              │
│                                                                             │
│  Step 5: apply_temperature                                                  │
│          logits /= temperature (temperature=0 时跳过，greedy decoding)        │
│                                                                             │
│  Step 6: apply_min_p                                                        │
│          屏蔽概率低于 max_prob * min_p 的 token                                │
│                                                                             │
│  Step 7: apply_top_k_top_p                                                  │
│          - top_k: 只保留概率最高的 k 个 token                                  │
│          - top_p: 保留累积概率达到 p 的最小 token 集合                          │
│                                                                             │
│  Step 8: gumbel_sample                                                      │
│          加 Gumbel 噪声 + argmax → 采样得到 token                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                          ↓
                                         (可选) compute_topk_logprobs
                                         计算 top-k logprobs 用于返回给用户
                                                          ↓
                                              SamplerOutput
                                         sampled_token_ids [num_reqs, 1]
```

### 1.3 Fast-path 优化

每一步都有 **fast-path 短路机制**：在 Python 层用 numpy 检查是否所有 request 都不需要该操作，如果是则直接跳过 kernel launch。例如：

- [`states.py:71`](../../../vllm/v1/worker/gpu/sample/states.py#L71): 检查是否所有 temperature 都是 0 或 1
- [`penalties.py:91`](../../../vllm/v1/worker/gpu/sample/penalties.py#L91): 检查是否有 request 使用 penalty
- [`min_p.py:83`](../../../vllm/v1/worker/gpu/sample/min_p.py#L83): 检查是否所有 min_p 都是 0

这种设计避免了不必要的 GPU kernel 启动开销。

---

## 2. 详细调用链：从 Model Runner 到 Triton Kernels

### 2.1 顶层入口

```
gpu_model_runner.py:_sample()          [L2896]
  └─ self.sampler(logits, ...)
       └─ sampler.py:Sampler.__call__() [L57]
            └─ self.sample(logits, ...) [L102]
```

`_sample()` 在 model forward pass 结束后被调用，拿到 `logits [num_reqs, vocab_size]` 后交给 `Sampler`。

### 2.2 Step 1: Copy to FP32

```python
# sampler.py:112
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
```

纯 PyTorch 操作，无 kernel 调用，确保后续计算精度。

### 2.3 Step 2: Logit Bias → `_bias_kernel`

```
sampler.py:115
  self.logit_bias_state.apply_logit_bias(logits, idx_mapping, idx_mapping_np, pos)
    └─ logit_bias.py:LogitBiasState.apply_logit_bias()
         │  [fast-path] if not np.any(self.use_logit_bias[idx_mapping_np]): return
         └─ logit_bias.py:apply_logit_bias()          [L238]
              └─ _bias_kernel[(num_reqs,)](...)        [Triton kernel, L260]
                   grid = (num_reqs,)  # 每个 req 一个 program
```

`_bias_kernel` 在一个 program 内串行处理三件事：allowed_token_ids → logit_bias → min_tokens。

### 2.4 Step 3: Penalties → `_penalties_kernel`

```
sampler.py:118
  self.penalties_state.apply_penalties(logits, idx_mapping, idx_mapping_np, ...)
    └─ penalties.py:PenaltiesState.apply_penalties()  [L82]
         │  [fast-path] if not np.any(self.use_penalty[idx_mapping_np]): return
         └─ penalties.py:apply_penalties()
              └─ _penalties_kernel[(num_reqs,)](...)   [Triton kernel, L109]
                   grid = (num_reqs,)
```

注意：`PenaltiesState.apply_staged_writes()` 在每个 step 开始前调用 `bincount()` 初始化 `prompt_bin_mask` 和 `output_bin_counts`：

```
apply_staged_writes()
  └─ penalties.py:bincount()
       └─ _bincount_kernel[(num_new_reqs, num_blocks)](...)  [Triton kernel]
            grid = (num_new_reqs, cdiv(max_prefill_len, BLOCK_SIZE))
```

### 2.5 Step 4: Bad Words → `_bad_words_kernel`

```
sampler.py:128
  self.bad_words_state.apply_bad_words(logits, idx_mapping, idx_mapping_np, ...)
    └─ bad_words.py:BadWordsState.apply_bad_words()   [L72]
         │  [fast-path] if max_num_bad_words == 0: return
         └─ bad_words.py:apply_bad_words()
              └─ _bad_words_kernel[(num_reqs, max_num_bad_words)](...) [Triton kernel, L100]
                   grid = (num_reqs, max_num_bad_words)  # 每个 bad word 一个 program
```

### 2.6 Step 5: Temperature → `_temperature_kernel`

```
sampler.py:137
  self.sampling_states.apply_temperature(logits, idx_mapping, idx_mapping_np)
    └─ states.py:SamplingStates.apply_temperature()   [L64]
         │  [fast-path] if np.all((temp == 0) | (temp == 1)): return
         └─ gumbel.py:apply_temperature()             [L34]
              └─ _temperature_kernel[(num_reqs, num_blocks)](...) [Triton kernel, L8]
                   grid = (num_reqs, cdiv(vocab_size, 8192))
                   BLOCK_SIZE = 8192  # 每个 block 处理 8192 个 vocab token
```

### 2.7 Step 6: Min-p → `_min_p_kernel`

```
sampler.py:140
  self.sampling_states.apply_min_p(logits, idx_mapping, idx_mapping_np)
    └─ states.py:SamplingStates.apply_min_p()         [L77]
         │  [fast-path] if np.all(min_p == 0): return
         └─ min_p.py:apply_min_p()                    [L44]
              └─ _min_p_kernel[(num_reqs,)](...)       [Triton kernel, L8]
                   grid = (num_reqs,)
                   # 每个 program 内两次遍历 vocab：第一次找 max，第二次 mask
```

### 2.8 Step 7: Top-k / Top-p → Triton 或 PyTorch

```
sampler.py:143
  self.sampling_states.apply_top_k_top_p(logits, idx_mapping, idx_mapping_np)
    └─ states.py:SamplingStates.apply_top_k_top_p()   [L88]
         │  [fast-path] if not (do_top_k or do_top_p): return logits
         └─ topk_topp_sampler.py:apply_top_k_top_p()  [L245]
              ├─ [batch >= 8 且有 Triton]
              │    └─ topk_topp_triton.py:apply_top_k_top_p_triton()
              │         └─ Triton kernel (并行 sort + mask)
              └─ [batch < 8 或无 Triton]
                   └─ topk_topp_sampler.py:apply_top_k_top_p_pytorch() [L258]
                        └─ PyTorch sort + cumsum + mask (纯 PyTorch)
```

### 2.9 Step 8: Gumbel Sample → `_gumbel_sample_kernel`

```
sampler.py:148
  gumbel_sample(logits, idx_mapping, temperature, seeds, pos, apply_temperature=False)
    └─ gumbel.py:gumbel_sample()                      [L108]
         └─ _gumbel_sample_kernel[(num_reqs, num_blocks)](...) [Triton kernel, L52]
              grid = (num_reqs, cdiv(vocab_size, 1024))
              BLOCK_SIZE = 1024
              # 每个 block 输出 local_argmax 和 local_max
         └─ [host-side reduce]
              max_block_idx = local_max.argmax(dim=-1)   # PyTorch
              sampled = local_argmax.gather(max_block_idx)
```

### 2.10 完整调用链总览

```
gpu_model_runner.py:_sample()
  └─ Sampler.__call__()
       └─ Sampler.sample()
            ├─ [PyTorch] copy to FP32
            │
            ├─ LogitBiasState.apply_logit_bias()
            │    └─ [Triton] _bias_kernel
            │         grid=(num_reqs,), BLOCK_SIZE=next_pow2(max_ids)
            │
            ├─ PenaltiesState.apply_penalties()
            │    └─ [Triton] _penalties_kernel
            │         grid=(num_reqs,)
            │
            ├─ BadWordsState.apply_bad_words()
            │    └─ [Triton] _bad_words_kernel
            │         grid=(num_reqs, max_num_bad_words)
            │
            ├─ SamplingStates.apply_temperature()
            │    └─ [Triton] _temperature_kernel
            │         grid=(num_reqs, cdiv(vocab_size, 8192))
            │
            ├─ SamplingStates.apply_min_p()
            │    └─ [Triton] _min_p_kernel
            │         grid=(num_reqs,)
            │
            ├─ SamplingStates.apply_top_k_top_p()
            │    ├─ [Triton] apply_top_k_top_p_triton  (batch >= 8)
            │    └─ [PyTorch] apply_top_k_top_p_pytorch (batch < 8)
            │
            └─ gumbel_sample()
                 ├─ [Triton] _gumbel_sample_kernel
                 │    grid=(num_reqs, cdiv(vocab_size, 1024))
                 └─ [PyTorch] local_max.argmax() → global argmax
```

每个 Triton kernel 的 grid 维度决定了并行粒度：
- `grid=(num_reqs,)`: 每个 request 一个 program，program 内串行遍历 vocab
- `grid=(num_reqs, num_blocks)`: 每个 request 的每个 vocab block 一个 program，最大并行度

---

## 3. torch.compile 与调试问题

### 3.1 为什么无法 debug sampler.py？

**直接原因**：`sampler.py` 本身没有 `torch.compile`，但它调用的 `apply_top_k_top_p` 最终会走到 [`topk_topp_sampler.py:237`](../../../vllm/v1/sample/ops/topk_topp_sampler.py#L237)：

```python
@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)
```

这个函数在 CPU path 下会被调用。

**更根本的原因**：整个 model 在 `load_model` 后会被 `torch.compile` 或 CUDA Graph 包裹：

```python
# gpu_model_runner.py:4327
if self.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
    backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
    self.model.compile(fullgraph=True, backend=backend)

# 或者 gpu_model_runner.py:4339
if cudagraph_mode.has_full_cudagraphs():
    self.model = CUDAGraphWrapper(self.model, self.vllm_config, ...)
```

### 4.2 torch.compile 的影响

当 `fullgraph=True` 时，整个 forward pass（包括 sampler 的调用链）都被编译进一个 graph，Python 断点完全失效，因为：

1. **FX Graph Tracing**: `torch.compile` 会把 Python 代码 trace 成 FX graph，实际执行的是编译后的 kernel，不再逐行执行 Python
2. **CUDA Graph 录制**: CUDA Graph 模式下，GPU 操作被录制后重放，也绕过了 Python 执行路径
3. **内联优化**: 编译器会内联函数调用，消除 Python 层的调用栈

### 4.3 解决方案

**方法 1**: 禁用 torch.compile
```bash
export VLLM_TORCH_COMPILE_LEVEL=0
# 或者启动时加参数
--enforce-eager
```

**方法 2**: 使用 `torch.compiler.disable()` 装饰器
```python
import torch.compiler

@torch.compiler.disable()
def sample(self, logits, ...):
    # 这个函数不会被 compile
    ...
```

**方法 3**: 使用 print/logging 调试
```python
# 在 Triton kernel 外部打印
print(f"[DEBUG] logits shape: {logits.shape}, temperature: {temperature}")
```

---

## 4. Sample 算子详解

### 4.1 算子总览

| 算子 | 文件 | 实现方式 | 核心逻辑 |
|------|------|----------|----------|
| **logit_bias** | [`logit_bias.py`](../../../vllm/v1/worker/gpu/sample/logit_bias.py) | Triton kernel `_bias_kernel` | 处理 allowed_token_ids（其余置 -inf）、logit_bias（加偏置）、min_tokens（屏蔽 stop token） |
| **penalties** | [`penalties.py`](../../../vllm/v1/worker/gpu/sample/penalties.py) | Triton kernel `_penalties_kernel` | repetition penalty（正 logit 除以 penalty，负 logit 乘以 penalty）；frequency penalty（减去 count × freq_penalty）；presence penalty（减去 pres_penalty） |
| **bad_words** | [`bad_words.py`](../../../vllm/v1/worker/gpu/sample/bad_words.py) | Triton kernel `_bad_words_kernel` | 检查当前 context 末尾是否匹配 bad word 的 prefix，若匹配则把 bad word 最后一个 token 的 logit 置 -inf |
| **temperature** | [`gumbel.py`](../../../vllm/v1/worker/gpu/sample/gumbel.py) | Triton kernel `_temperature_kernel` | `logits /= temperature`，temperature=0 或 1 时 early return |
| **min_p** | [`min_p.py`](../../../vllm/v1/worker/gpu/sample/min_p.py) | Triton kernel `_min_p_kernel` | 先找 max logit，threshold = max + log(min_p)，低于 threshold 的置 -inf |
| **top_k / top_p** | [`topk_topp_sampler.py`](../../../vllm/v1/sample/ops/topk_topp_sampler.py) | Triton（batch≥8）/ PyTorch sort / FlashInfer | top_k：sort 后 mask 掉排名靠后的；top_p：cumsum softmax 后 mask 掉累积概率超出的 |
| **gumbel_sample** | [`gumbel.py`](../../../vllm/v1/worker/gpu/sample/gumbel.py) | Triton kernel `_gumbel_sample_kernel` | 对每个 block 生成 Gumbel 噪声（`-log(-log(u))`），加到 logits 上，做 block-level argmax，最后 reduce 得到全局 argmax |
| **bincount** | [`penalties.py`](../../../vllm/v1/worker/gpu/sample/penalties.py) | Triton kernel `_bincount_kernel` | 初始化时统计 prompt/output token 的出现次数，用 atomic_or/atomic_add 写入 bin_mask 和 bin_counts |

### 4.2 Logit Bias

**文件**: [`vllm/v1/worker/gpu/sample/logit_bias.py`](../../../vllm/v1/worker/gpu/sample/logit_bias.py)

**功能**:
1. **allowed_token_ids**: 只允许特定 token 被采样（白名单机制）
2. **logit_bias**: 对特定 token 的 logit 加偏置（正偏置增加概率，负偏置降低概率）
3. **min_tokens**: 在生成足够 token 前屏蔽 stop token（确保最小生成长度）

**实现细节**:
```python
@triton.jit
def _bias_kernel(...):
    # 1. 处理 allowed_token_ids
    if num_allowed_token_ids > 0:
        # 保存 allowed token 的 logits
        allowed_logits = load(logits[allowed_token_ids])
        # 所有 token 置 -inf
        logits[:] = -inf
        # 恢复 allowed token 的 logits
        logits[allowed_token_ids] = allowed_logits

    # 2. 处理 logit_bias
    if num_logit_bias > 0:
        logits[bias_token_ids] += bias_values

    # 3. 处理 min_tokens
    if pos < min_len and num_stop_token_ids > 0:
        logits[stop_token_ids] = -inf
```

### 4.3 Penalties

**文件**: [`vllm/v1/worker/gpu/sample/penalties.py`](../../../vllm/v1/worker/gpu/sample/penalties.py)

**三种 penalty 机制**:

1. **Repetition Penalty** (默认 1.0，>1 惩罚重复):
   ```python
   if token in (prompt + output):
       if logit > 0:
           logit /= repetition_penalty  # 降低高概率 token
       else:
           logit *= repetition_penalty  # 进一步降低低概率 token
   ```

2. **Frequency Penalty** (默认 0.0):
   ```python
   logit -= frequency_penalty * token_count
   # 线性惩罚，出现次数越多惩罚越重
   ```

3. **Presence Penalty** (默认 0.0):
   ```python
   if token in output:
       logit -= presence_penalty
   # 固定惩罚，只要出现过就惩罚
   ```

**数据结构**:
- `prompt_bin_mask`: [max_num_reqs, vocab_size/32]，用 bitmap 记录 prompt 中出现的 token
- `output_bin_counts`: [max_num_reqs, vocab_size]，记录 output 中每个 token 的出现次数

**Speculative Decoding 支持**:
```python
# penalties_kernel 中处理 draft tokens
for prev_pos in range(MAX_SPEC_LEN):
    if prev_pos < pos:
        prev_token = input_ids[start_idx + prev_pos + 1]
        draft_counts[prev_token] += 1
output_bin_counts = base_output_counts + draft_counts
```

### 4.4 Bad Words

**文件**: [`vllm/v1/worker/gpu/sample/bad_words.py`](../../../vllm/v1/worker/gpu/sample/bad_words.py)

**功能**: 屏蔽会导致生成 bad word 的 token

**算法**:
```python
@triton.jit
def _bad_words_kernel(...):
    for bad_word in bad_words_list:
        prefix = bad_word[:-1]  # 前缀
        last_token = bad_word[-1]  # 最后一个 token

        # 检查 context 末尾是否匹配 prefix
        match = True
        for i in range(len(prefix)):
            expected = prefix[i]
            actual = context[-(len(prefix)-i)]
            match &= (expected == actual)

        # 如果匹配，屏蔽 last_token
        if match:
            logits[last_token] = -inf
```

**示例**:
- Bad word: ["I", "hate", "you"]
- Context: [..., "I", "hate"]
- 操作: `logits["you"] = -inf`

### 4.5 Temperature

**文件**: [`vllm/v1/worker/gpu/sample/gumbel.py`](../../../vllm/v1/worker/gpu/sample/gumbel.py)

**公式**: `logits /= temperature`

**特殊情况**:
- `temperature = 0`: greedy decoding（直接 argmax，不加噪声）
- `temperature = 1`: 不改变 logits 分布
- `temperature > 1`: 分布更平滑（增加随机性）
- `temperature < 1`: 分布更尖锐（降低随机性）

**实现**:
```python
@triton.jit
def _temperature_kernel(...):
    temperature = load(temperature_ptr[req_idx])
    if temperature == 0.0 or temperature == 1.0:
        return  # Early return

    logits = load(logits_ptr)
    logits = logits / temperature
    store(logits_ptr, logits)
```

### 4.6 Min-p Sampling

**文件**: [`vllm/v1/worker/gpu/sample/min_p.py`](../../../vllm/v1/worker/gpu/sample/min_p.py)

**算法**:
```python
max_logit = max(logits)
threshold = max_logit + log(min_p)
logits[logits < threshold] = -inf
```

**与 top-p 的区别**:
- **top-p**: 累积概率达到 p 的最小 token 集合（动态大小）
- **min-p**: 概率 ≥ max_prob × min_p 的所有 token（相对阈值）

**优势**: min-p 更稳定，不会因为分布形状变化而剧烈改变候选集大小。

### 4.7 Top-k / Top-p

**文件**: [`vllm/v1/sample/ops/topk_topp_sampler.py`](../../../vllm/v1/sample/ops/topk_topp_sampler.py)

**实现策略**:
1. **Triton 实现** (batch_size ≥ 8): 使用 Triton kernel 并行处理
2. **PyTorch sort** (batch_size < 8): 使用 PyTorch 原生 sort + mask
3. **FlashInfer** (可选): 使用 rejection sampling 避免 sort

**Top-k 算法**:
```python
# 方法 1: 完整 sort
logits_sorted, indices = logits.sort(descending=True)
logits_sorted[k:] = -inf
logits = logits.scatter(indices, logits_sorted)

# 方法 2: topk (更快)
top_k_values = logits.topk(k).values
threshold = top_k_values[:, -1:]
logits[logits < threshold] = -inf
```

**Top-p 算法**:
```python
logits_sorted, indices = logits.sort(descending=True)
probs_sorted = logits_sorted.softmax(dim=-1)
cumsum_probs = probs_sorted.cumsum(dim=-1)
mask = cumsum_probs > p
mask[:, 0] = False  # 至少保留 1 个 token
logits_sorted[mask] = -inf
logits = logits.scatter(indices, logits_sorted)
```

### 4.8 Gumbel Sampling

**文件**: [`vllm/v1/worker/gpu/sample/gumbel.py`](../../../vllm/v1/worker/gpu/sample/gumbel.py)

**核心思想**: Gumbel-Max trick 将采样转化为 argmax 操作

**数学原理**:
```
从分布 p(x) ∝ exp(logit_x) 中采样
等价于
argmax_x (logit_x + Gumbel_noise_x)

其中 Gumbel_noise = -log(-log(U)), U ~ Uniform(0,1)
```

**实现**:
```python
@triton.jit
def _gumbel_sample_kernel(...):
    # 1. 生成 Gumbel 噪声
    u = tl.rand(seed, block)
    u = tl.maximum(u, 1e-7)  # 避免 log(0)
    gumbel_noise = -tl.log(-tl.log(u))

    # 2. 应用 temperature (可选)
    if APPLY_TEMPERATURE:
        logits = logits / temperature

    # 3. 加噪声
    logits = logits + gumbel_noise

    # 4. Block-level argmax
    local_max, local_argmax = tl.max(logits, return_indices=True)

    # 5. 全局 reduce (在 host 端)
    # sampled = local_argmax[local_max.argmax()]
```

**优势**:
1. **避免 softmax**: 不需要计算 exp(logit) 和归一化
2. **避免 multinomial**: 不需要 CPU-GPU 同步
3. **支持 temperature=0**: 自动退化为 greedy decoding

**Block-level 优化**:
- 将 vocab_size 分成多个 block (BLOCK_SIZE=1024)
- 每个 block 独立计算 local argmax
- 最后在 host 端用 `local_max.argmax()` 找到全局 argmax

### 4.9 Bincount (Penalty 初始化)

**文件**: [`vllm/v1/worker/gpu/sample/penalties.py`](../../../vllm/v1/worker/gpu/sample/penalties.py)

**功能**: 统计 prompt 和 output 中每个 token 的出现次数

**数据结构**:
- `prompt_bin_mask`: bitmap，记录 prompt 中出现过的 token
- `output_bin_counts`: 计数器，记录 output 中每个 token 的出现次数

**实现**:
```python
@triton.jit
def _bincount_kernel(...):
    # 1. 统计 prompt tokens (用 bitmap)
    if block_idx * BLOCK_SIZE < prompt_len:
        prompt_tokens = load(all_token_ids[block])
        for token in prompt_tokens:
            idx = token // 32
            bit_idx = token % 32
            bit = 1 << bit_idx
            tl.atomic_or(prompt_bin_mask[idx], bit)

    # 2. 统计 output tokens (用 counter)
    if block_idx * BLOCK_SIZE >= prompt_len:
        output_tokens = load(all_token_ids[block])
        for token in output_tokens:
            tl.atomic_add(output_bin_counts[token], 1)
```

**为什么用 bitmap 存 prompt**:
- Prompt 通常很长，但只需要知道"是否出现"（repetition penalty）
- Bitmap 节省内存：vocab_size=128k → 128k/32 = 4k int32

**为什么用 counter 存 output**:
- Output 通常较短，需要精确计数（frequency penalty）
- 直接用 int32 数组存储每个 token 的出现次数

---

## 参考资料

1. [LLM Sampling blog](https://huyenchip.com/2024/01/16/sampling.html)
2. [FlashInfer sampling 文档](https://flashinfer.ai/2025/03/10/sampling.html)
3. [Gumbel-Max Trick 论文](https://arxiv.org/abs/1411.0030)
4. [vLLM v1 架构文档](../arch/)
5. [Triton 编程指南](https://triton-lang.org/main/programming-guide/index.html)