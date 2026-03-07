# vLLM LlaMA 模型源码分析

> 源文件：`vllm/model_executor/models/llama.py`（622 行）

---

## 文件总体结构

```
llama.py (622行)
│
├── LlamaMLP                              # FFN 前馈网络
├── LlamaAttention                        # 自注意力（含 RoPE、KV Cache）
├── LlamaDecoderLayer                     # 单层 Transformer Block
├── llama_model_invariants()              # Torch compile 形状断言
├── LlamaModel                            # 多层堆叠 + embedding + norm（支持 PP）
├── LlamaForCausalLM                      # 顶层推理入口（含 lm_head）
├── LlamaBidirectionalForSequenceClassification  # 序列分类适配
└── LlamaBidirectionalModel               # Embedding 模型适配
```

---

## 1. `LlamaMLP` — 前馈网络

**位置：** `vllm/model_executor/models/llama.py` L76–L122

```
                hidden_size
                    │
         ┌──────────┴──────────┐
     gate_proj              up_proj
    (MergedColumnParallel)  (合并在一起)
         │                     │
         └──────────┬──────────┘
               SiluAndMul()       ← silu(gate) * up 融合算子
                    │
               down_proj (RowParallel)
                    │
                hidden_size
```

**关键设计：**

| 组件 | 类型 | 说明 |
|---|---|---|
| `gate_up_proj` | `MergedColumnParallelLinear` | 将 `gate_proj` 和 `up_proj` 合并为一个矩阵乘法，节省 kernel launch |
| `act_fn` | `SiluAndMul` | 融合 CUDA kernel，一次 pass 完成 silu 激活 + element-wise 乘 |
| `down_proj` | `RowParallelLinear` | 张量并行 reduce，`reduce_results` 控制是否 all-reduce |

- 仅支持 `silu` 激活，其他会抛出 `ValueError`
- `disable_tp=True` 可关闭张量并行（PP 分割边界层使用）

---

## 2. `LlamaAttention` — 自注意力

**位置：** `vllm/model_executor/models/llama.py` L124–L261

```
hidden_states
    │
qkv_proj (QKVParallelLinear)
    │ split
  q / k / v
    │
rotary_emb(positions, q, k)    ← RoPE 位置编码
    │
Attention(q, k, v)             ← 核心：PagedAttention / FlashAttention
    │
o_proj (RowParallelLinear)
    │
output
```

**初始化参数：**

| 字段 | 说明 |
|---|---|
| `QKVParallelLinear` | Q/K/V 合并投影，支持 GQA（num_kv_heads < num_heads） |
| `get_rope()` | 动态获取 RoPE 实现（支持 `rope_parameters` 自定义） |
| `attn_type` | 支持 `DECODER`（因果）和 `ENCODER_ONLY`（双向）两种模式 |
| `sliding_window` | 通过 `config.layer_types` 按层配置滑动窗口注意力 |
| GGUF 兼容 | GGUF 量化格式下 `is_neox_style=False`，影响 RoPE 索引方式 |

**张量并行 KV head 分区规则：**

```python
if total_num_kv_heads >= tp_size:
    # KV head 数 >= TP 进程数 → 切分 KV heads
    assert total_num_kv_heads % tp_size == 0
else:
    # KV head 数 < TP 进程数 → 复制 KV heads
    assert tp_size % total_num_kv_heads == 0
```

**Eagle3 草稿模型兼容（L193–L210）：**

草稿模型的 `layer_idx` 需要减去 `target_layer_count` 才能正确索引 `layer_types` 数组：

```python
if hasattr(config, "target_layer_count"):
    effective_layer_idx = layer_idx - config.target_layer_count
else:
    effective_layer_idx = layer_idx
```

---

## 3. `LlamaDecoderLayer` — Transformer Block

**位置：** `vllm/model_executor/models/llama.py` L262–L337

```
                     positions
                         │
hidden_states ──→ input_layernorm(RMSNorm)  ──→ self_attn ──┐
      │                                                      │
      └─────────────────── residual ──────────────────── add ┘
                                                             │
                               post_attention_layernorm(RMSNorm)
                                                             │
                                                           mlp
                                                             │
                                                   (hidden, residual)
```

**Pre-norm + Fused Residual 设计：**

```python
# 第一层（residual=None）：分开执行
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)

# 后续层（residual 非 None）：RMSNorm 内部融合 residual add
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

`RMSNorm` 的双参数 `forward` 是 **fused add + norm CUDA kernel**，避免单独的 residual add 操作，减少显存读写。

**子模块：**

| 子模块 | 类型 | 说明 |
|---|---|---|
| `self_attn` | `LlamaAttention`（可替换） | 通过 `attn_layer_type` 参数支持自定义 Attention 实现 |
| `mlp` | `LlamaMLP` | 前馈网络 |
| `input_layernorm` | `RMSNorm` | pre-norm（Attention 前） |
| `post_attention_layernorm` | `RMSNorm` | pre-norm（MLP 前） |

---

## 4. `llama_model_invariants` + `@support_torch_compile`

**位置：** `vllm/model_executor/models/llama.py` L339–L356

```python
@support_torch_compile(shape_invariants=llama_model_invariants)
class LlamaModel(nn.Module):
```

`shape_invariants` 函数在编译期插入形状断言：

```python
torch._check(positions.size()[0] == input_ids.size()[0])
```

约束 unbacked dynamic shape，防止 token 数量变化触发不必要的 recompilation。

---

## 5. `LlamaModel` — 核心堆叠模型

**位置：** `vllm/model_executor/models/llama.py` L356–L505

### 5.1 Pipeline Parallel 分区

```
PP rank 0:  embed_tokens → layers[0..N/2]  → return IntermediateTensors →
PP rank 1:  layers[N/2..N] → final_norm → return hidden_states
```

```python
# 仅 first rank 有 embed_tokens
if get_pp_group().is_first_rank:
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()  # 占位，不参与计算

# 仅 last rank 有 final norm
if get_pp_group().is_last_rank:
    self.norm = RMSNorm(...)
else:
    self.norm = PPMissingLayer()
```

`make_layers()` 根据 PP rank 自动决定本 rank 负责的 layer 范围 `[start_layer, end_layer)`。

### 5.2 `forward` 控制流

```
is_first_rank?
    YES → embed(input_ids)  →  residual = None
    NO  → 从 IntermediateTensors 恢复 hidden_states, residual
        │
        ↓
for idx, layer in enumerate(layers[start:end]):
    if idx in aux_hidden_state_layers:        ← Eagle3：收集中间层输出
        aux_hidden_states.append(h + r)
    hidden_states, residual = layer(positions, hidden_states, residual)
        │
is_last_rank?
    NO  → return IntermediateTensors({"hidden_states", "residual"})
    YES → hidden_states = norm(hidden_states, residual)
         → if aux_hidden_states:
               return (hidden_states, aux_hidden_states)
           else:
               return hidden_states
```

### 5.3 `load_weights` — 权重名映射

将 HuggingFace checkpoint 中**分散的权重**合并映射到 vLLM **堆叠的参数**：

| HF checkpoint 名 | vLLM 内部名 | shard_id |
|---|---|---|
| `.q_proj` | `.qkv_proj` | `"q"` |
| `.k_proj` | `.qkv_proj` | `"k"` |
| `.v_proj` | `.qkv_proj` | `"v"` |
| `.gate_proj` | `.gate_up_proj` | `0` |
| `.up_proj` | `.gate_up_proj` | `1` |

此外处理：
- 跳过 `rotary_emb.inv_freq`（不需要，运行时计算）
- 跳过 `rotary_emb.cos_cached` / `sin_cached`（ColossalAI ckpt 残留）
- KV cache 量化 scale 加载（`quant_config.get_cache_scale`）
- FP8 KV scale/zero_point 名称重映射（`maybe_remap_kv_scale_name`）

---

## 6. `LlamaForCausalLM` — 推理顶层入口

**位置：** `vllm/model_executor/models/llama.py` L508–L611

### 6.1 继承接口

| 接口 | 作用 |
|---|---|
| `SupportsLoRA` | 启用 LoRA 适配（`embedding_modules`、`packed_modules_mapping`） |
| `SupportsPP` | 支持 Pipeline Parallel |
| `SupportsEagle` | 支持 Eagle 投机解码 |
| `SupportsEagle3` | 支持 Eagle3（多层中间状态辅助） |

### 6.2 结构

```
LlamaForCausalLM
    ├── model: LlamaModel            # 主干
    ├── lm_head: ParallelLMHead      # 词表投影（仅 last PP rank）
    └── logits_processor: LogitsProcessor
```

### 6.3 调用链

```
forward(input_ids, positions, intermediate_tensors, inputs_embeds)
    │
    └─ self.model(...)
           │
           └─ returns hidden_states  (or IntermediateTensors for middle PP ranks)

[由 GPU model runner 调用]
compute_logits(hidden_states)
    └─ logits_processor(lm_head, hidden_states)
           └─ vocab parallel softmax → logits
```

### 6.4 `tie_word_embeddings`

当 `config.tie_word_embeddings=True` 时：
- `lm_head` 权重和 `embed_tokens` 共享（`lm_head.tie_weights(embed_tokens)`）
- `load_weights` 时自动跳过 `lm_head.` 前缀，避免重复加载

### 6.5 Eagle3 辅助层

```python
def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)
```

默认收集第 2 层、中间层、倒数第 3 层的隐状态，作为草稿模型的辅助输入。

---

## 7. 双向模型适配

**位置：** `vllm/model_executor/models/llama.py` L614–L621

```python
class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    pass

class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    pass
```

通过 `adapters.py` 中的装饰器将 Causal LM 改造为：
- **序列分类模型**：`as_seq_cls_model`，用 pooling 输出 logits
- **Embedding 模型**：`as_embedding_model`，输出句向量

实际的双向注意力通过 `LlamaBidirectionalConfig` 中的 `is_causal=False` 注入，在 `LlamaDecoderLayer.__init__` 中切换为 `attn_type = ENCODER_ONLY`。

---

## 整体调用层次图

```
LlamaForCausalLM.forward(input_ids, positions, intermediate_tensors)
  │
  └─ LlamaModel.forward
       │
       ├─ [first PP rank] embed_tokens(input_ids)      # VocabParallelEmbedding
       │
       └─ for layer in layers[start_layer:end_layer]:
            LlamaDecoderLayer.forward(positions, hidden_states, residual)
              │
              ├─ RMSNorm(hidden_states, residual)       # fused add + norm
              │
              ├─ LlamaAttention.forward(positions, hidden_states)
              │    ├─ QKVParallelLinear                 # q/k/v 合并投影
              │    ├─ RoPE(positions, q, k)             # 旋转位置编码
              │    ├─ Attention(q, k, v)                # PagedAttention / FlashAttn
              │    └─ RowParallelLinear (o_proj)
              │
              ├─ RMSNorm(hidden_states, residual)       # fused
              │
              └─ LlamaMLP.forward(hidden_states)
                   ├─ MergedColumnParallelLinear        # gate + up 合并
                   ├─ SiluAndMul()                      # fused silu * mul
                   └─ RowParallelLinear (down_proj)
       │
       └─ [last PP rank] RMSNorm(final) → hidden_states

[GPU model runner]
  └─ LlamaForCausalLM.compute_logits(hidden_states)
       └─ LogitsProcessor(lm_head, hidden_states)      # vocab parallel logits
```

---

## 关键设计总结

| 设计点 | 实现方式 | 收益 |
|---|---|---|
| QKV 合并投影 | `QKVParallelLinear` | 减少 kernel launch，提升 MFU |
| Gate+Up 合并 | `MergedColumnParallelLinear` | 同上 |
| Fused SiluAndMul | CUDA fused kernel | 减少显存读写 |
| Fused RMSNorm+Add | `RMSNorm(h, residual)` 双参数 | 减少显存读写 |
| Torch Compile | `@support_torch_compile` + shape invariants | CUDA graph 友好 |
| Pipeline Parallel | `PPMissingLayer` + `IntermediateTensors` | 多机流水线并行 |
| Tensor Parallel | Column/Row Parallel Linear | 单节点多卡并行 |
| Eagle3 中间层 | `aux_hidden_state_layers` | 投机解码草稿模型辅助输入 |
