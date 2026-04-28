# 第 2 章 MoE 的数学基础：从概率视角看专家混合

## 2.1 一个最朴素的定义

设有 $N$ 个专家网络 $E_1, E_2, \ldots, E_N$，每个专家是从输入空间到输出空间的函数 $E_i: \mathbb{R}^d \to \mathbb{R}^{d'}$。

再设一个**门控函数** $g: \mathbb{R}^d \to \Delta^{N-1}$，把输入映射到一个 $N$ 维的概率单纯形上（$\sum g_i = 1, g_i \ge 0$）。

MoE 的输出是：

$$
\boxed{y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)}
$$

这就是 1991 年的原始定义。所有现代 MoE 都是它的特例。

---

## 2.2 概率视角：MoE 是一个隐变量模型

把 MoE 看作隐变量模型：

$$
p(y \mid x) = \sum_{i=1}^{N} p(z = i \mid x) \cdot p(y \mid x, z = i)
$$

- 隐变量 $z \in \{1, \ldots, N\}$ 表示**该样本属于哪个专家**；
- $p(z=i \mid x) = g_i(x)$：门控网络是后验；
- $p(y \mid x, z=i) = E_i(x)$：每个专家是一个条件分布。

这就是 "Mixture" 这个名字的来源——它本质是一个**条件混合分布**。

> 这个视角的好处：可以直接套用 EM 算法（早年 HME 就是这样训练的）。
> 但深度 MoE 时代，端到端 SGD 已经成为标配，EM 退到了背景里。

---

## 2.3 稠密 MoE vs 稀疏 MoE

### 稠密 MoE (Dense MoE)

$g_i(x) > 0$ 对所有 $i$ 成立。每个专家都参与计算，只是权重不同。

$$
y = \sum_{i=1}^{N} \text{softmax}(W_g x)_i \cdot E_i(x)
$$

**问题**：参数容量大了，但**计算量也大了 $N$ 倍**。这没解决任何问题。

### 稀疏 MoE (Sparse MoE)

只让 Top-K 个专家激活，其他权重置零：

$$
g(x) = \text{TopK}(\text{softmax}(W_g x), K)
$$

具体地：

$$
g_i(x) = \begin{cases}
\text{softmax}(W_g x)_i & \text{if } i \in \text{TopK indices} \\
0 & \text{otherwise}
\end{cases}
$$

这样：

- **参数量** = $N$ 倍专家 + 门控；
- **激活量** ≈ $K$ 倍专家 + 门控；
- 当 $K \ll N$ 时，**计算量与稠密单专家网络相当，但容量提高了 $N/K$ 倍**。

这就是稀疏 MoE 真正的魔法。

---

## 2.4 门控函数的几种典型形式

### 2.4.1 简单 Softmax 门控 (GShard 风格)

$$
g(x) = \text{softmax}(W_g x)
$$

然后取 Top-K，再对 Top-K 部分**重新归一化**（保证权重和为 1）：

```python
logits = x @ W_g                       # [B, T, N]
probs = softmax(logits, dim=-1)
topk_vals, topk_idx = probs.topk(k, dim=-1)
gates = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
```

### 2.4.2 Noisy Top-K Gating (Shazeer 2017)

为了鼓励探索 + 提高负载均衡，给 logits 加可学习的噪声：

$$
\text{logits}(x) = W_g x + \epsilon \cdot \text{softplus}(W_n x), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

训练时引入随机性，推理时关闭（或保留小幅度，看实现）。

### 2.4.3 Switch Routing (Top-1)

直接取 argmax：

$$
i^* = \arg\max_i (W_g x)_i, \quad y = g_{i^*}(x) \cdot E_{i^*}(x)
$$

最简单也最高效。代价是路由"硬"，需要更强的负载均衡机制。

### 2.4.4 Expert Choice (Zhou et al., 2022)

反过来：**让专家选 token，而不是 token 选专家**。

每个专家有固定容量 $C$，从所有 token 中按 logits 取 Top-C：

$$
\text{Expert}_i \text{ 选择 logits}_i \text{ 最高的 } C \text{ 个 token}
$$

好处是负载天然均衡（每个专家恰好处理 $C$ 个 token），代价是同一 token 可能被 0 个或多个专家选中。

---

## 2.5 为什么 Top-K Gating 不可导，但仍能训练？

这是面试常考的"陷阱题"。

**问题**：TopK 算子返回的是排序索引，**对于 logits 不可导**。

**关键观察**：我们要导的不是 "TopK 选了谁"，而是 **"被选中的那些 logits 进入 softmax 后的权重"**。

具体地：

```python
logits = x @ W_g                          # 可导
topk_vals, topk_idx = logits.topk(k, dim=-1)  # 索引不可导，但 topk_vals 可导
gates = softmax(topk_vals, dim=-1)        # 可导

y = sum_{i in topk_idx} gates[i] * E_i(x)
```

- `topk_idx` 决定哪些专家参与（**离散选择**），梯度不传过它；
- `topk_vals` 是被选中专家的 logits，softmax 后形成 gate 权重，**梯度正常回传**到 $W_g$；
- 每个被选中的专家本身可导。

**结论**：Top-K 把"是否被选中"和"选中后的权重"解耦，前者离散，后者连续。SGD 通过后者的梯度学到该提升哪些专家的 logits。

> 缺点：未被 Top-K 选中的专家**完全没有梯度**，可能永远学不到东西。这就是 MoE 训练的核心难题之一——**Expert Collapse（专家坍塌）**。第 5 章会详细讨论。

---

## 2.6 与集成学习的对比

很多同学第一次看到 MoE 会问："这不就是 ensemble 吗？"

| 维度 | Ensemble (Bagging/Boosting) | MoE |
|------|-----------------------------|-----|
| 子模型何时训练 | 独立训练 | 端到端联合训练 |
| 输入到子模型的分配 | **全部数据**送给所有子模型 | **门控**决定哪些数据送哪个子模型 |
| 子模型分工 | 隐式（数据采样 / 残差） | 显式（门控学习） |
| 推理是否激活全部 | 全部激活 | 稀疏 MoE 只激活 K 个 |
| 主要收益 | 降方差 / 提稳定性 | 提容量 / 控成本 |

**记忆点**：Ensemble 是"民主投票"，MoE 是"专家会诊"。

---

## 2.7 与 Attention 的微妙联系

Attention 与 MoE 在数学上有一定相似：

$$
\text{Attention}(q, K, V) = \sum_i \text{softmax}(q^\top k_i) \cdot v_i
$$

$$
\text{MoE}(x) = \sum_i \text{softmax}(W_g x)_i \cdot E_i(x)
$$

都是"加权求和"。差异在于：

- **Attention** 的 $v_i$ 是**输入相关的（每个 token 不同）**，权重也是输入对；
- **MoE** 的 $E_i$ 是**全局共享的（参数）**，权重是门控对每个 token 计算。

某种程度上，MoE 可以看作"专家是 Key/Value 缓冲区"的特殊 Attention。事实上，2024–2025 年有研究 (Mixture of Attention Heads, MoA) 把 Attention 头本身做成 MoE。

---

## 2.8 容量、负载与吞吐：三个绕不开的量

设：

- $T$：每个 batch 中 token 总数；
- $N$：专家数；
- $K$：每个 token 激活的专家数；
- $C$：每个专家的**容量** —— 最多接收多少个 token。

理想情况下，每个专家接收 $T \cdot K / N$ 个 token。但门控不会完美均衡，所以实际定义：

$$
C = \text{Capacity Factor} \times \frac{T \cdot K}{N}
$$

`Capacity Factor` 通常取 1.0 到 2.0：

- 太小：超过容量的 token 被 **drop**（直接跳过 MoE 层），影响效果；
- 太大：很多专家槽位空闲，浪费显存与算力。

> Capacity 是 MoE 工程的核心 trade-off 之一，第 3、5 章会反复出现。

---

## 2.9 一个最小可运行的 MoE 公式包

把上面所有内容凝结成一个最小代码骨架：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x):                                # x: [B, T, D]
        logits = self.gate(x)                            # [B, T, N]
        topk_logits, topk_idx = logits.topk(self.k, -1)  # [B, T, K]
        gates = F.softmax(topk_logits, dim=-1)           # [B, T, K]

        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                       # [B, T, K] bool
            if not mask.any(): continue
            # 该专家被选中的 token 加权求和
            for k in range(self.k):
                sel = mask[..., k]                       # [B, T]
                if not sel.any(): continue
                y[sel] += gates[..., k][sel].unsqueeze(-1) * expert(x[sel])
        return y
```

这段代码**会跑但不快**——它没有 batch 化专家计算、没有 capacity、没有负载均衡。第 4、5、8 章会一步步把它升级成生产级实现。

---

## 2.10 本章要点

- MoE = 多个专家 + 一个门控的加权混合，公式是 $y = \sum g_i(x) E_i(x)$；
- 稀疏 MoE 把权重稀疏化，让**容量与计算解耦**；
- Top-K 路由通过"离散选择 + 连续权重"实现可微训练；
- MoE ≠ Ensemble，前者端到端学习分工，后者各自独立；
- Capacity Factor 是工程上的核心调节量。

→ [第 3 章 稀疏 MoE 架构：把 FFN 替换成专家组](03-sparse-architecture.md)
