# 第 9 章 MoE 推理优化：Expert Offloading、缓存与投机解码

## 9.1 推理 MoE 的根本困境

MoE 在训练阶段有 batch 平均化的优势——大量 token 摊销到所有专家。**但在推理阶段，特别是单 batch、单 token 解码时**：

- 每 step 只产生 1 个 token；
- 这 1 个 token 只激活 K 个专家；
- 但你必须把**所有 N 个专家的权重**装在显存里以备路由结果不可知；
- 即使 99% 的专家这一步用不到，权重也得驻留。

**这是 MoE 推理的根本困境**：参数量大，但单步只用一小部分，**显存利用率极低**。

> Mixtral 8x7B 推理时显存占用 ≈ 47B 参数 × 2 bytes (bf16) ≈ 94 GB。
> 单步实际计算只用 13B × 2 bytes = 26 GB。
> 显存"浪费"73%。

本章讨论怎么对付这个问题。

---

## 9.2 推理场景分类

不同推理场景对 MoE 的优化偏好不同：

| 场景 | 主要关注 | 优化方向 |
|------|----------|----------|
| 在线对话（低 batch） | 延迟 | Expert Offloading, 缓存 |
| 离线批量（高 batch） | 吞吐 | EP, 路由批处理 |
| 边缘部署（小显存） | 显存 | Quantization, Pruning |
| 长上下文 | KV cache | MLA, 共享 KV |

---

## 9.3 Expert Offloading

### 9.3.1 思路

把不常用的专家**放在 CPU 内存或 NVMe**，需要时再 swap 到 GPU。

```
GPU memory:  active experts + attention + KV cache
CPU memory:  all experts (full copy)
解码时:      根据路由结果 prefetch 所需 expert 到 GPU
```

### 9.3.2 实际效果

挑战：

- PCIe 4 带宽 ~32 GB/s，单 expert ~5 GB → 加载 150ms+；
- 单 step 解码本身只要 30ms；
- 完全 offload **比纯 GPU 慢 5 倍以上**。

可行的折中方案：

- **Hot/Cold Expert 分离**：把常用的 K 个 expert 留 GPU，其余 offload；
- **Predictive Prefetching**：路由器预测下一步可能需要哪些 expert，提前 swap；
- **CPU 卸载 KV cache，GPU 全留专家**：对于长上下文 + 小 batch 场景。

### 9.3.3 代表项目

- **Mixtral-Offloading**（dvmazur）：把 Mixtral 推理装进 16GB GPU，速度可接受；
- **DeepSpeed-Inference MoE**：原生支持 expert offload；
- **fastmoe**：早期开源 MoE 推理框架。

---

## 9.4 Expert 量化

### 9.4.1 为什么 MoE 特别适合量化

MoE 的专家数多 → 每个专家的"重要性"差异大 → 不重要的专家可以更激进地量化。

经验观察：

- **共享专家**：保留 bf16；
- **频繁激活专家**：bf16 / fp8；
- **罕见激活专家**：int4 / int8；
- **路由器**：保留 bf16/fp32，量化太狠会破坏路由质量。

### 9.4.2 工具

- **GPTQ**：经典 LLM 量化方法，对 MoE 也适用，但要"专家分组"做 calibration；
- **AWQ (Activation-aware Weight Quantization)**：对 MoE 改进版叫 **MoE-AWQ**；
- **GGUF** (llama.cpp 格式)：支持 mixtral / dbrx 等 MoE 的 int4 / int5 部署；
- **bitsandbytes 4bit**：常被用于 Mixtral-4bit 部署。

实测 Mixtral-8x7B-4bit：显存约 24GB，质量损失 1–2 perplexity，可接受。

---

## 9.5 KV Cache 与 MLA

### 9.5.1 问题

MoE 只稀疏化 FFN，**Attention 仍然稠密**。这意味着 **KV cache 大小与 Dense 一样**：

```
KV cache = 2 × n_layers × n_heads × d_head × seq_len × batch × 2 bytes
```

对长上下文（128k）来说，KV cache 反而成为瓶颈，与是不是 MoE 无关。

### 9.5.2 Multi-head Latent Attention (MLA)

DeepSeek-V2 的关键发明：把 KV 投影到一个**低维 latent 空间**：

```
传统 Attention:
  K, V ∈ R^{seq × n_heads × d_head}    (cache size 大)

MLA:
  z ∈ R^{seq × d_latent}                (cache 这个就够)
  推理时再把 z 解码回 K, V
```

效果：DeepSeek-V2 的 KV cache 比 Llama2-70B 同上下文长度**小 8–10x**。这让 MoE + 长上下文成为可能。

### 9.5.3 GQA / MQA

更早的思路：

- **MQA (Multi-Query Attention)**：所有 head 共享同一对 K/V → KV cache 缩小 head 倍；
- **GQA (Grouped-Query Attention)**：K/V 分组共享 → 介于 MHA 和 MQA 之间。

Mixtral 用 GQA，DeepSeek 用 MLA，都是对 KV cache 的优化。

---

## 9.6 路由批处理 (Routing Batching)

### 9.6.1 问题

在 continuous batching 下，每个 step 不同 sequence 的 token 被一起送进模型。但路由结果不同，**专家计算无法一次性 batch**：

```
step t:
  seq 0 token → expert 3
  seq 1 token → expert 7
  seq 2 token → expert 3
  seq 3 token → expert 1
  ...
```

朴素实现：每个专家独立 GEMM，效率低。

### 9.6.2 优化思路

- **按 expert 分组**：把同一 expert 的 token 凑在一起 → 一次 GEMM；
- **Padding 到统一形状**：所有专家 GEMM 形状 padding 到 max → 统一 batch GEMM；
- **MegaBlocks fused kernel**（第 8 章）：直接处理不规则形状。

vLLM 的 fused MoE kernel 实现的就是这一类优化。

---

## 9.7 投机解码 (Speculative Decoding) 与 MoE

### 9.7.1 投机解码回顾

用一个小的 draft 模型快速预测 N 个 token，再让大模型一次 verify。如果都对就接受，否则回退。在 Dense 模型上能 2–3x 加速。

### 9.7.2 MoE 上的额外优势

MoE verify N 个 token 的成本只略高于 verify 1 个（因为大部分时间花在 attention 上，FFN 部分本来就在 batch 上摊销）。所以 **MoE + 投机解码的加速比 Dense 更大**。

### 9.7.3 MTP (Multi-Token Prediction)

DeepSeek-V3 训练时就让模型一次预测多个 token，作为辅助任务。推理时可以直接复用作为 self-speculative decoding 的 draft。

```
训练目标:
  L = L_main(token_t+1) + λ * L_aux(token_t+2)
```

V3 报告：MTP 训练让最终模型质量略升，且推理时 self-speculation 加速 1.8x 左右。

---

## 9.8 EP 推理 (Expert-Parallel Inference)

### 9.8.1 多卡推理的天然适配

MoE 在多卡推理时，把不同专家放到不同 GPU 上是天然的：

- attention：通常 TP 切分；
- 专家：EP 分布；
- 一次解码：路由器决定 token 去哪些 GPU，All-to-All 后并行计算。

DeepSeek-V3 推理参考配置：8 张 H100，EP=8，TP=1（attention 用 MLA 自身已小，无需 TP）。

### 9.8.2 单卡推理 vs 多卡推理 trade-off

| 部署 | 优点 | 缺点 |
|------|------|------|
| 单卡 (offload + 量化) | 成本低 | 慢，依赖好的 prefetch |
| 多卡 EP | 快，吞吐高 | 显存浪费多（每卡都装 attention/embedding） |
| 多卡 TP+EP | 最快 | 通信复杂 |

---

## 9.9 上下文专家 (Context Experts) 的实验

2024–2025 年有论文提出：让专家也在 **prompt 阶段被"激活/选择"**——长 prompt 内不同 token 选不同专家，在 prompt 处理结束后**只保留被路由次数高的专家**做后续解码。

这种 "Context-Aware Expert Pruning" 在 long context 部署中能省 30%+ 显存，但还在研究阶段，没有大规模产品化。

---

## 9.10 实际部署 checklist

部署 Mixtral / DeepSeek 这样的 MoE 模型，按顺序考虑：

1. **显存够吗？**
   - 不够 → 量化（4bit / 8bit）；
   - 仍不够 → Expert offloading；
2. **延迟敏感吗？**
   - 是 → 多卡 EP + fused kernel；
   - 否 → 单卡 + 大 batch；
3. **长上下文吗？**
   - 是 → 优先选支持 MLA / GQA 的模型；
4. **吞吐还是延迟？**
   - 吞吐 → continuous batching + EP；
   - 延迟 → 投机解码 + MTP（如果模型支持）；
5. **冷启动吗？**
   - 频繁切模型 → 选支持 expert offload + lazy load；
6. **质量底线？**
   - 不能掉 perplexity 太多 → 4bit 之后必须 benchmark。

---

## 9.11 一些数字

帮你建立"MoE 部署的尺度感"（粗略估算，bf16）：

| 模型 | 单卡部署 | 4×80GB 部署 | 量化后单卡 |
|------|----------|-------------|-----------|
| Mixtral 8x7B (47B) | 一张 H100 不够 | 充裕 | 4bit ≈ 24GB → 单卡可 |
| Mixtral 8x22B (141B) | 不行 | 紧张 | 4bit ≈ 70GB → 单 H100 边缘 |
| DeepSeek-V3 (671B) | 不行 | 不够 | 8bit ≈ 670GB → 8×H100 |
| Qwen3-MoE 235B | 不行 | 紧张 | 4bit ≈ 120GB → 2×H100 |

> 这就是为什么 MoE 推理基础设施远比 MoE 训练复杂——成本曲线与延迟约束都更严苛。

---

## 9.12 本章要点

- MoE 推理的根本困境：**参数大但单步用得少，显存利用率低**；
- Expert Offloading 在小显存下可行，但有带宽代价；
- 量化（特别是 4bit）是 MoE 部署的标配；
- KV cache 是长上下文 MoE 的真瓶颈，**MLA / GQA / MQA** 是关键技术；
- Fused MoE kernel 是 continuous batching 必需；
- **投机解码 + MTP** 与 MoE 是天作之合；
- 部署决策要看：显存、延迟、上下文长度、批处理、质量底线。

→ [第 10 章 MoE vs Dense](10-moe-vs-dense.md)
