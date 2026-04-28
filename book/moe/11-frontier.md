# 第 11 章 前沿与未来

## 11.1 Mixture of Depths (MoD)

> Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (2024)

### 11.1.1 核心思想

MoE 把"FFN 替换成专家组"。MoD 走得更激进：**不同 token 走不同深度的网络**——某些 token 直接 skip 某层！

```
Token A: Layer 1 → Layer 2 → Layer 3 → Layer 4 (走全部 4 层)
Token B: Layer 1 →           Layer 3           (跳过 2、4)
Token C: Layer 1 → Layer 2 →                   (跳过 3、4)
```

每层用一个路由器决定该层"接收多少 token"。简单 token 浅层就能处理完，复杂 token 多层细化。

### 11.1.2 与 MoE 的关系

- MoE：在**宽度**上稀疏化（多专家选 K 个）；
- MoD：在**深度**上稀疏化（每层选 K% 的 token）；
- 二者可组合：**MoD-MoE** 同时做深度 + 专家路由。

实验：MoD 在等 FLOPs 下质量持平 / 略胜 Dense，是一个有潜力的方向。

---

## 11.2 Mixture of Attention (MoA)

把 Attention 头本身做 MoE：每个 attention "head" 是一个专家，路由器决定每 token 用哪些 heads。

代表工作：
- "Mixture of Attention Heads" (Zhang et al., 2022)
- JetMoE 等

效果：在小模型上有微弱收益。Attention 头之间的"特化"不如 FFN 显著，所以 MoA 不如 MoE-FFN 流行。

---

## 11.3 Soft MoE 复兴 (Vision)

第 4 章提到 Soft MoE 在 ViT 上效果好。2024 年起，多模态大模型（VLM）越来越多采用 Soft MoE 或类 Soft MoE：

- **Vision Encoder MoE**：CLIP / SigLIP 风格 encoder 做专家化；
- **多模态 Decoder**：处理图像 token 与文本 token 的不同专家；
- 代表：Apple 的 MM1.5、Mistral 的 Pixtral 部分采用。

---

## 11.4 Up-Cycling：把 Dense 转成 MoE

> Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints" (2022)

### 11.4.1 思路

不从头训 MoE，而是用一个已经训好的 Dense 模型，把它的 FFN 复制成 N 个专家（加噪声打破对称性），再继续训。

```
Dense Llama 7B  →  克隆 FFN ×8  →  Mixtral-style 8x7B  →  继续训 0.5T token
```

Qwen1.5-MoE 就是这么做的。优点：

- 省 50%+ pretraining 算力；
- 利用已有 Dense 模型的 capability 起点；
- 方法简单可复现。

缺点：

- 专家初始相同，分工需要"破对称性"训练；
- 容量利用率不一定达到 from-scratch 的 MoE 水平。

---

## 11.5 Parameter Efficient MoE Fine-tuning

LoRA + MoE 的结合是研究热点：

- **LoRA-MoE**：每个专家只训 LoRA delta，不训完整专家；
- **MoLE / MoLoRA**：路由器选择 LoRA 适配器，而不是选专家；
- **LongLoRA-MoE**：长上下文 finetune。

效果：可在小数据集上 finetune MoE 而不易过拟合。

---

## 11.6 路由器架构的演进

到目前为止 router 一直是简单的 `Linear(d, N)`。前沿尝试：

- **MLP Router**：路由器多一层，能学更复杂分配；
- **Attention-based Router**：用 attention 做路由（Gating Network 与 self-attention 融合）；
- **Hierarchical Router**：先选"专家组"，再在组内选专家；
- **Stochastic Differentiable Router**：用 Gumbel-Softmax 让"硬选择"可微。

这些都还在 ablation 阶段，没有大规模应用。

---

## 11.7 共享专家与持续学习

DeepSeek 的"共享专家"是一个被低估的想法。它对**持续学习 / 领域适应**特别有用：

- 共享专家保留通用知识 → finetune 时不动；
- 稀疏专家专注新领域 → finetune 时大改；
- 类似"主体不变，外挂可换"。

预期 2025–2026 年会有大量基于共享 + 稀疏分离的 MoE 持续学习方案。

---

## 11.8 推理时的"专家剪枝"

如果某些专家在你的部署场景下几乎从不被路由（比如你的业务全是中文，但有个"日文专家"），可以直接**剪掉**它。

- 静态剪枝：根据 calibration 集统计每专家被路由频次，剪掉尾部；
- 动态剪枝：运行时根据负载动态卸载/加载；
- 带 finetune 的剪枝：剪完再微调路由器。

这种 "Domain-Specific Expert Pruning" 在企业部署中越来越受关注。

---

## 11.9 与状态空间模型 (Mamba) 的结合

Mamba 等 SSM 模型挑战了 Transformer 的主导地位。MoE 的思想可以无缝迁移到 Mamba：

- **Jamba** (AI21, 2024)：Mamba + Transformer + MoE 混合；
- **Mamba-MoE**：纯 Mamba 块的 MoE 化；
- **Hybrid MoE Architectures**：每层独立选用 Attn/Mamba/MoE 块。

预期 SSM-MoE 是 2026 年长上下文的强力候选。

---

## 11.10 多智能体角度的 MoE：Mixture of Agents

最新方向：把"专家"从 FFN 升级为**完整的小语言模型**。

- 每个"专家"是一个 7B 小模型；
- 路由器决定 query 让哪个小模型处理；
- 类似多智能体系统，但路由可学。

代表工作：Together AI 的 "Mixture of Agents"，把多个开源 LLM 联合起来超越单个大模型。

虽然名字叫 MoA / MoE，但跨度更大——已经从架构层面延伸到系统层面。

---

## 11.11 一些被讨论但未必落地的方向

- **Continuous MoE**：让"激活专家数"也是连续的，不再 Top-K；
- **Generative Routing**：路由器本身用一个小 LM 生成；
- **Memory-Augmented MoE**：专家访问外部记忆库；
- **Federated MoE**：分布式联邦学习中的 MoE，不同专家在不同节点。

---

## 11.12 三个长期趋势

预测未来 2–5 年 MoE 的发展：

1. **MoE 成为大模型默认架构**：开源 / 闭源大模型 ≥30B 几乎都会是 MoE。Dense 70B 量级会在 2026 后逐渐边缘化；
2. **MoE 的"细粒度化"继续**：专家数从几十涨到几百甚至几千，DeepSeek 风格的"细粒度 + 共享 + 无辅助损失"成为标准配方；
3. **MoE 推理基础设施成熟**：vLLM、SGLang、TensorRT-LLM 完善后，MoE 部署门槛会接近 Dense；可能出现专门的 MoE 加速芯片（类似 TPU 之于 Transformer）。

---

## 11.13 结语

MoE 不是一个新概念——它已经 30 多年。但它真正的"黄金时代"才刚开始。

它解决的核心问题——**让模型容量与单次计算解耦**——在 LLM 走向 AGI 的路上越来越关键。当我们追求更大模型、更长上下文、更便宜推理时，稀疏化是绕不开的答案。

MoE 也不是万能。它把工程复杂度从"训练一个大 Dense"转移到"训练一个稀疏专家集合"。这种复杂度的代价，是 LLM 工程师未来必须掌握的"基本功"。

希望这本书能让你：

- 理解 MoE 不只看到一面；
- 不会被任何一篇 MoE 论文吓住；
- 能在面试中讲清楚每一个细节；
- 能在工程上做出"用还是不用"的有理由的决策。

如果有一天，你做出了一个 MoE 模型，或者基于 MoE 做了一个产品，请告诉我——这就是这本书最想看到的事。

→ 接下来：
- [附录 A 面试题精选（30 题）](appendix-interview.md)
- [附录 B 参考文献与延伸阅读](appendix-references.md)
