# 第 7 章 经典 MoE 模型解析

本章把目前最重要的几个 MoE 模型拆开讲。每个模型按统一模板：**关键创新 → 配置参数 → 路由细节 → 重要发现**。

---

## 7.1 GShard (Google, 2020)

> Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"

**关键创新**

- 第一次把 MoE 大规模塞进 Transformer；
- 提出 **Expert Parallelism** + 自动分片框架；
- 引入 Auxiliary Loss + Capacity Factor。

**配置**

| 参数 | 值 |
|------|------|
| 任务 | 多语言机器翻译 (M4) |
| 总参数 | 600B |
| 专家数 | 2048 |
| 路由 | Top-2 |
| 层数 | 36 |
| 训练设备 | 2048 TPU v3 |

**路由细节**

```
Token-level Top-2
Capacity Factor = 1.25 (训练) / 2.0 (推理)
Auxiliary Loss = N * Σ f_i * P_i
```

**重要发现**

- **Sparse 比 Dense 更高效**：相同 FLOPs 下，BLEU 高 13.5 点；
- **专家越多越好**（直到 2048）；
- **Capacity Factor** 对最终性能影响显著。

> GShard 是 MoE-Transformer 的"奠基论文"。后续工作几乎都引用了它。

---

## 7.2 Switch Transformer (Google, 2021)

> Fedus, Zoph & Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

**关键创新**

- 把 K=2 简化到 K=1；
- 万亿参数（1.6T）；
- 极简代码，开源 Mesh-TF / T5x 实现。

**配置**

| 参数 | 值 |
|------|------|
| 总参数 | 7B / 26B / 1.6T |
| 专家数 | 128 / 64 / 2048 |
| 路由 | Top-1 |
| 层数 | 24+ (encoder + decoder) |
| 训练数据 | C4 |

**为什么 Top-1 也能 work**

```
传统观点：K=1 信息太少，容易"赌错专家"
Switch 反驳：
  1. 反正每个 token 进入一个 FFN，本身就只学一种变换
  2. 配合 capacity_factor 和 aux loss，路由错误的代价不大
  3. 通信减半，可以训更大模型 → 总质量胜出
```

**重要发现**

- 同样 token 数下，Switch-Base 比 T5-Base **训练速度快 2.5x**；
- 同样训练时间下，效果显著高于稠密 baseline；
- **Distillation**：能把 1.6T MoE 蒸馏到 0.4× 大小的 Dense 模型，保留 30% 收益。

---

## 7.3 GLaM (Google, 2021)

> Du et al., "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"

**关键创新**

- 1.2T 参数，**只激活 97B**；
- **训练能耗仅为 GPT-3 的 1/3**，推理 FLOPs 仅为 GPT-3 的 1/2；
- 性能在 29 个 zero/one-shot 任务上**全面超过 GPT-3**。

**配置**

| 参数 | 值 |
|------|------|
| 总参数 | 1.2T |
| 激活参数 | 97B |
| 专家数 | 64 |
| 路由 | Top-2 |
| 层数 | 64 |
| 训练 token | 1.6T |

**重要发现**

GLaM 第一次定量论证了 MoE 的"能耗优势"：

```
            GPT-3      GLaM
训练能耗:    1287 MWh   456 MWh    (-65%)
推理FLOPs:   175B       97B        (-45%)
零样本平均:  56.9       62.7       (+5.8)
```

> 这是 MoE "省钱省电" 第一次被业界严肃量化。

---

## 7.4 ST-MoE (Google, 2022)

> Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models"

**关键贡献**：把 MoE 从"难训"推向"可靠训练"。

**核心提案**：

1. **Router Z-Loss**（详见 [第 5 章](05-load-balancing.md)）；
2. **更大的 dropout**（0.1 → 0.2）防 finetune 过拟合；
3. **router weights 用 fp32**：路由器不要轻易降到 bf16；
4. **数据并行 vs 专家并行的混合**：32B 模型在 1024 TPU 上可训。

**重要发现**

- ST-MoE-32B 在 SuperGLUE 上**首次 MoE 超越同规模 Dense**；
- Z-Loss 是从 "fp16 训不动 MoE" 走向 "bf16 也能训 MoE" 的关键。

---

## 7.5 Mixtral 8x7B (Mistral, 2023)

> Jiang et al., "Mixtral of Experts"

**关键创新**

- **第一个真正高质量、可商用、开源的 MoE LLM**；
- 8 个专家，每个 7B；
- Top-2 路由；
- 总参数 47B（不是 56B —— 共享 attention，仅 FFN MoE）；
- 性能持平/超过 Llama 2-70B 与 GPT-3.5。

**为什么是 47B 不是 56B**

Mixtral 共享 attention、embedding 等参数，只把 FFN 做 MoE：

```
共享部分（attention, embedding 等）:    ~2B
每个专家 FFN 大小:                       5.6B
总专家:                                  8 × 5.6B = 44.8B
总参数:                                  ~47B
```

每 token 激活：2B 共享 + 2 × 5.6B = ~13B。

**架构细节（每层 MoE 块）**

```python
class MixtralMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2):
        ...
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MixtralExpert(hidden_size, intermediate_size)  # SwiGLU
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [batch, seq, hidden]
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_idx = routing_weights.topk(self.top_k, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        # ... dispatch and combine ...
```

**重要发现 / 业界贡献**

- 把 MoE 从 "Google 内部专属" 拉到 "全球开发者人手一台"；
- 开启了开源 MoE 生态（vLLM、SGLang 等都从 Mixtral 开始 MoE 支持）；
- 启发了大量开源 fork：Mixtral 8x22B、Wizard-Mixtral、Dolphin-Mixtral 等。

---

## 7.6 DeepSeek-MoE (DeepSeek, 2024)

> DeepSeek-AI, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"

**关键创新**

1. **Fine-grained Expert Segmentation**：64 个细粒度专家替代 8 个粗粒度专家；
2. **Shared Expert Isolation**：1–2 个共享专家分担"通用知识"；
3. **稀疏-激活比** 大幅提高：每 token 激活 6/64 专家（不是 2/8）。

**配置**（DeepSeek-MoE 16B / 145B）

| 参数 | 16B | 145B |
|------|-----|------|
| 总参数 | 16.4B | 145B |
| 激活参数 | 2.8B | 22B |
| 路由专家 | 64 | 128 |
| 共享专家 | 2 | 4 |
| Top-K | 6 | 12 |

**为什么细粒度有用**

直觉：8 选 2 → $C_8^2 = 28$ 种组合；64 选 6 → $C_{64}^6 \approx 7$ 千万。组合多 250 万倍。

实测：DeepSeek-MoE 16B 与 LLaMA2-7B 持平甚至胜出，**激活参数仅 40%**。

---

## 7.7 DeepSeek-V2 / V3 (DeepSeek, 2024)

> DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
> DeepSeek-AI, "DeepSeek-V3 Technical Report"

**架构组合拳**

| 组件 | 用途 |
|------|------|
| **DeepSeekMoE** (细粒度 + 共享) | 知识容量 |
| **MLA (Multi-head Latent Attention)** | KV cache 缩减 8x |
| **Auxiliary-Loss-Free 负载均衡** | 不牺牲质量的均衡 |
| **MTP (Multi-Token Prediction)** | 训练时一次预测多个 token |
| **节点限制路由** (V3) | 减少跨节点通信 |
| **fp8 训练** (V3) | 训练吞吐 2x |

**配置**

| 参数 | V2 | V3 |
|------|----|----|
| 总参数 | 236B | 671B |
| 激活参数 | 21B | 37B |
| 路由专家 | 160 | 256 |
| 共享专家 | 2 | 1 |
| Top-K | 6 | 8 |
| 层数 | 60 | 61 |
| 训练 token | 8.1T | 14.8T |

**重要发现**

- V3 训练成本 ~$5.6M，相比 GPT-4（估算 $100M+）便宜近 20x；
- V3 在多项 benchmark 持平 Claude 3.5 Sonnet、GPT-4o；
- 完全开源（权重、技术报告、推理代码），重塑了开源 LLM 格局。

---

## 7.8 Qwen-MoE 系列 (Alibaba, 2024–2025)

> Qwen Team, "Qwen1.5-MoE-A2.7B" / "Qwen3-MoE"

**Qwen1.5-MoE-A2.7B（2024 早期）**

| 参数 | 值 |
|------|------|
| 总参数 | 14B |
| 激活参数 | 2.7B |
| 专家数 | 60 |
| 共享专家 | 4 |
| Top-K | 4 |

特色：**Up-cycling** —— 用一个已训好的稠密 7B 模型初始化，复制 FFN 形成 8 个专家，再继续训练。这种"暖启动"省了一半 token。

**Qwen3-MoE（2025）**

更大规模，进一步采用 DeepSeek 风格的细粒度 + 共享 + Aux-Loss-Free。性能在开源 MoE 中第一梯队。

---

## 7.9 其他值得了解的 MoE

| 模型 | 团队 | 特色 |
|------|------|------|
| **Yi-MoE** | 01.AI | 早期开源 MoE 尝试 |
| **JetMoE** | MIT | 8B 模型，Attention 也做 MoE |
| **OLMoE** | Allen AI | 完全开源（含训练数据） |
| **Llama 4** | Meta | 多种规模 MoE，2025 主流开源 |
| **Grok-1** | xAI | 314B MoE，开源 |
| **Arctic** | Snowflake | 480B MoE + Dense 混合架构 |
| **Phi-3.5-MoE** | Microsoft | 小模型 MoE 尝试 (16x3.8B) |

---

## 7.10 GPT-4 / Claude / Gemini 是 MoE 吗？

**GPT-4**：根据 SemiAnalysis 等第三方报道，GPT-4 是 **8-expert MoE，每个约 220B**，总 1.8T，每 token Top-2。OpenAI 未官方确认。

**Claude**：Anthropic 没有公开架构。从 Claude 3/4 推理速度推测，可能采用了 MoE 或稀疏化。

**Gemini Pro/Ultra**：Google 在 Gemini 1.5 报告中**明确说采用了 MoE**，配合 long context 训练。

> 即使没有官方确认，业界共识是：**至 2025 年起，主流闭源前沿模型几乎都是 MoE**。

---

## 7.11 模型对比表

| 模型 | 年份 | 总参 | 激活 | 路由 | 共享专家 | 备注 |
|------|------|------|------|------|----------|------|
| GShard | 2020 | 600B | ~? | Top-2 | × | 翻译 |
| Switch | 2021 | 1.6T | ~26B | Top-1 | × | 通用 LM |
| GLaM | 2021 | 1.2T | 97B | Top-2 | × | 节能 |
| ST-MoE | 2022 | 32B | 6B | Top-2 | × | 稳定性 |
| Mixtral 8x7B | 2023 | 47B | 13B | Top-2 | × | 开源里程碑 |
| Mixtral 8x22B | 2024 | 141B | 39B | Top-2 | × | 商用强力 |
| DeepSeek-MoE 16B | 2024 | 16B | 2.8B | Top-6 | 2 | 细粒度首推 |
| DeepSeek-V2 | 2024 | 236B | 21B | Top-6 | 2 | + MLA |
| DeepSeek-V3 | 2024 | 671B | 37B | Top-8 | 1 | Aux-loss-free + fp8 |
| Qwen3-MoE | 2025 | 235B | 22B | Top-8 | 1 | 阿里旗舰 |

---

## 7.12 本章要点

- **GShard → Switch → GLaM** 三件套奠定 MoE-Transformer 的工程基础；
- **ST-MoE** 解决稳定性问题，让 MoE 进入"可靠训练"时代；
- **Mixtral** 是开源里程碑，让全社区都能玩 MoE；
- **DeepSeek 系列** 是 2024 年开源 MoE 的工程巅峰：细粒度 + 共享 + Aux-Loss-Free + MLA + fp8；
- **Qwen-MoE** 等紧随其后，国内开源生态完整成型；
- 闭源前沿模型（GPT-4、Gemini、Claude 4）大概率都是 MoE。

→ [第 8 章 MoE 的工程实现](08-implementation.md)
