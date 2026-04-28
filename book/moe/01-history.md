# 第 1 章 MoE 的前世今生：从集成学习到稀疏大模型

## 1.1 1991：Jacobs、Jordan 与"自适应专家"的诞生

MoE 的故事不是从大模型时代开始的。它的根扎在 1991 年的一篇论文：

> **Adaptive Mixtures of Local Experts**
> Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton
> Neural Computation, 1991

这篇论文回答了一个非常朴素的问题：

> **如果一个数据集天然有多个"模式" (modes)，让一个网络去学全部，会不会反而更糟？**

举个例子：你要训练一个网络识别"水果"。但水果其实分成两类：红色圆形（苹果、樱桃）和黄色长条（香蕉、芒果）。让一个网络学全部，它的中间层不得不在两类特征之间做平衡，最终两边都学不好。

Jacobs 等人的方案：**让多个"专家"网络分别学习子任务，由一个"门控网络"决定每个样本由谁来处理**。

数学上：

$$
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

- $E_i(x)$：第 $i$ 个专家网络的输出；
- $g_i(x)$：门控网络对该专家的权重，$\sum g_i = 1$，$g_i \ge 0$。

这就是后来所有 MoE 论文都要写的那个公式。**所有现代 MoE 都是这个公式的特例与扩展**。

> 直觉：MoE 把"一个大模型学所有"换成"多个小模型分工 + 一个调度员"，让模型容量提高的同时不增加单样本计算。

---

## 1.2 1994：Hierarchical Mixtures of Experts (HME)

三年后，Jordan 与 Jacobs 又提出了 **HME**：把 MoE 嵌套，专家自己还是 MoE。

```
        gate (level 0)
       /             \
    gate            gate
   /    \          /    \
  E      E        E      E
```

HME 在 90 年代曾被广泛用于回归与分类，并发展出基于 EM 算法的训练方法。但因为：

- 90 年代算力有限；
- 当时没有"超大数据集"驱动稀疏化的需求；

HME 后来主要活跃在统计机器学习圈，没能成为深度学习主流。

---

## 1.3 2013–2017：深度学习时代的"试水"

进入深度学习时代，MoE 的复兴有几个零散尝试：

- **Eigen, Ranzato & Sutskever (2013)**：Conditional Computation 的早期探索，把 MoE 用作 RNN 的一部分；
- **Bengio et al. (2015)**：Conditional Computation 综述，提出"按需激活子网络"的思想，但工程实现仍困难；
- **Outrageously Large Neural Networks (Shazeer et al., 2017)**：MoE 真正"出圈"的转折点。

**Shazeer 2017 这篇论文做了三件事：**

1. 把 MoE 嵌入到 LSTM 中，单层放 **128–131,072 个专家**（是的，13 万个！）；
2. 提出 **Top-K Gating + Noisy Gating**，第一次让稀疏路由变得可训练；
3. 在机器翻译和语言建模上，把模型参数推到了 **137B**（2017 年这是天文数字）。

> 这是 MoE 第一次从"特例技巧"变成"可扩展架构"。

可惜 LSTM 时代很快被 Transformer 取代，这条路线没有立刻铺开。但 Shazeer 团队在 Google 内部继续推进，三年后他们带着 Transformer 版的 MoE 回来了——那就是 GShard。

---

## 1.4 2020：GShard 把 MoE 装进 Transformer

> **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**
> Lepikhin et al., Google, 2020

GShard 解决了三个工程问题：

1. **专家放在哪里**：Transformer 的 FFN 层最适合替换为 MoE；
2. **怎么分布式训练**：发明 **Expert Parallelism** —— 不同专家放在不同设备上，token 通过 All-to-All 通信路由到对应设备；
3. **怎么稳定训练**：引入 **Auxiliary Loss** 防止门控坍塌。

GShard 训练了一个 **600B 参数的多语言翻译模型**，覆盖 100 种语言，效果远超稠密 baseline。这是第一个真正"工业级"的稀疏 Transformer。

```
             (Transformer Block)
             ┌─────────────────┐
input ──────►│  Self-Attention │
             ├─────────────────┤
             │     MoE Layer   │  ← 这里替换原来的 FFN
             │  ┌──┐ ┌──┐ ┌──┐ │
             │  │E1│ │E2│ │EN│ │
             │  └──┘ └──┘ └──┘ │
             │   Gate (Top-2)  │
             └─────────────────┘
```

---

## 1.5 2021：Switch Transformer —— "Top-1 也能 work"

GShard 用 Top-2，Switch Transformer (Fedus et al., Google, 2021) 把它简化到 **Top-1**：每个 token 只去 1 个专家。这个看起来"暴力"的简化带来三个好处：

1. **通信减半**：每个 token 只发到 1 个 device；
2. **路由复杂度降低**：实现简单；
3. **训练更稳定**：配合容量因子 (Capacity Factor)，相对 GShard 收敛更快。

Switch Transformer 第一次把模型推到 **1.6 万亿参数**，引爆了 MoE 在大模型时代的关注。

---

## 1.6 2021–2022：GLaM、ST-MoE 与稳定性的攻坚

**GLaM** (Du et al., Google, 2021)：1.2T MoE，每个 token 只激活 97B，**训练能耗仅为 GPT-3 的 1/3**。这第一次让人意识到：MoE 可以**省电**。

**ST-MoE** (Zoph et al., 2022)：专门研究 MoE 训练的稳定性，提出：

- **Router Z-Loss**：限制路由 logits 的数值范围，防止溢出；
- 一系列 fine-tuning 经验：MoE 在下游任务的 finetune 容易过拟合，需要更大正则。

ST-MoE 把 MoE 从"难训"推向"可训"。

---

## 1.7 2023：Mixtral —— 开源 MoE 的引爆点

Mistral AI 2023 年底放出 **Mixtral 8x7B**：

- 8 个专家，每个 7B；
- 总参数 47B（不是 56B —— 共享部分参数，仅 FFN 是专家化的）；
- 每个 token Top-2 激活，约 13B 活跃参数；
- 性能持平/超过 Llama2-70B、GPT-3.5。

Mixtral 是**开源生态第一次真正拥抱 MoE**。它的影响：

- 全球 LLM 工程师第一次能在自己机器上跑起一个完整 MoE；
- vLLM、llama.cpp、SGLang 等推理框架开始为 MoE 做专项优化；
- 启发了 Qwen-MoE、Yi-MoE、DeepSeek-MoE 等一系列开源 MoE。

---

## 1.8 2024：DeepSeek-MoE —— 中国选手的方法论贡献

> **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**
> DeepSeek-AI, 2024

DeepSeek 团队提出两个关键改进：

**1. 细粒度专家划分 (Fine-grained Expert Segmentation)**

不是 8 个大专家，而是 **64 个甚至 256 个小专家**，每个专家更窄，激活更多个（如 Top-6、Top-8）。这让"专家"真正能学到细粒度知识，而不是变成 8 个"全能选手"。

**2. 共享专家隔离 (Shared Expert Isolation)**

把"通用知识"放到 1–2 个**所有 token 都激活**的共享专家里，剩下的稀疏专家只负责"差异化知识"。这避免了所有稀疏专家都在重复学习"the、a、句号"这种通用 pattern。

DeepSeek-V2、V3 沿用并强化了这些设计。V3 是 671B 总参数 / 37B 激活的开源最强 MoE 模型之一。

---

## 1.9 2024–2025：辅助无损负载均衡 (Auxiliary-Loss-Free)

历史上 MoE 训练都依赖 Auxiliary Loss 强制专家被均匀使用，但这会引入"为了均衡而牺牲性能"的副作用。

DeepSeek-V3 提出**直接给每个专家加一个动态 bias**：

- 哪个专家被路由太多 → bias 减小；
- 哪个专家被路由太少 → bias 增大；
- 路由 logits 中只用于路由的部分加 bias，**不影响最终的 gating 权重**。

这个设计在 V3 训练中被证明：**既保持负载均衡，又不损失模型能力**。Qwen3-MoE 也采用了类似方案。

---

## 1.10 时间线总结

```
1991  Adaptive Mixtures of Local Experts        ← 起源
1994  Hierarchical MoE                          ← 经典扩展
2013  Conditional Computation                   ← 深度学习探索
2017  Outrageously Large NN (Top-K Gating)      ← LSTM-MoE 出圈
2020  GShard                                    ← Transformer-MoE
2021  Switch Transformer / GLaM                 ← 万亿参数时代
2022  ST-MoE                                    ← 稳定性
2023  Mixtral 8x7B                              ← 开源引爆
2024  DeepSeek-MoE / V2                         ← 细粒度+共享
2024  DeepSeek-V3 (Aux-Loss-Free)               ← 辅助无损均衡
2025+  Qwen3-MoE / Llama4-MoE / GPT-5...        ← 全面 MoE 化
```

---

## 1.11 一句话总结

> MoE 的发展史，就是一部**不断回答"如何让条件计算 (Conditional Computation) 可扩展、可训练、可部署"的工程史**。

- 1991–2017 解决了**怎么定义专家与门控**；
- 2020–2022 解决了**怎么训得稳**；
- 2023–2025 解决了**怎么用得好、用得开放**。

下一章，我们把这段历史"压缩"成一组数学公式，看看 MoE 真正在做什么。

→ [第 2 章 MoE 的数学基础](02-math-foundations.md)
