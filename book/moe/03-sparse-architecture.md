# 第 3 章 稀疏 MoE 架构：把 FFN 替换成专家组

## 3.1 为什么是 FFN？

Transformer 一个 block 里两个主要计算：

```
x → LayerNorm → MultiHeadAttention → +residual
  → LayerNorm → FFN(W1, W2)         → +residual
```

实测中：

- **Attention**：参数约 $4 d^2$，FLOPs 与序列长度 $T$ 平方相关；
- **FFN**：参数约 $8 d^2$（隐层维度 $4d$），FLOPs 与 $T$ 一次相关。

> 在大模型里，FFN 占总参数的 **2/3 左右**，但每个 token 是独立计算的。

这两点让 FFN 成为 MoE 化的**完美靶子**：

1. **参数集中**：把 FFN 改成稀疏，能直接砍掉 60%+ 的激活参数；
2. **逐 token 独立**：天然适合按 token 路由到不同专家；
3. **不影响 Attention**：保留全局信息流。

所以 GShard 之后所有主流 MoE Transformer 都遵循同一个范式：

> **Attention 保持稠密 + 把 FFN 替换成 MoE 层**

---

## 3.2 一个 MoE-Transformer Block 的结构

```
                ┌──────────────────────────────────────┐
                │              Block                   │
                │  ┌──────────────────────────────────┐│
input  ───────► │  │   LayerNorm + Self-Attention     ││
                │  └──────────────┬───────────────────┘│
                │                 │                    │
                │  ┌──────────────▼───────────────────┐│
                │  │   LayerNorm + MoE-FFN            ││
                │  │    ┌──────────────────────────┐  ││
                │  │    │  Gate(x) → Top-K Experts │  ││
                │  │    └──────────────────────────┘  ││
                │  │    E1   E2   E3   ...   EN      ││
                │  └──────────────┬───────────────────┘│
                │                 │                    │
                └─────────────────▼────────────────────┘
                              output
```

通常**不是每一层都做 MoE**。Switch、GLaM、Mixtral 都采用 **隔层 MoE**（Interleaved）：奇数层用稠密 FFN，偶数层用 MoE。原因有二：

- 减少 All-to-All 通信次数；
- 让稠密层吸收"通用计算"，MoE 层吸收"特化计算"。

---

## 3.3 token-level 还是 sequence-level？

路由的"粒度"有几种选择：

| 粒度 | 描述 | 代表模型 |
|------|------|----------|
| **Token-level** | 每个 token 独立选专家 | GShard, Switch, Mixtral, DeepSeek |
| **Sequence-level** | 同一个 sequence 里所有 token 走同一个专家 | 早期实验，已不主流 |
| **Sentence-level** | 跨 sentence 切换 | 多语言任务的特例 |

**Token-level 是现代 MoE 的默认选择**。原因：

- 同一 sentence 内不同 token 也可能需要不同知识（"Newton" 走物理专家，"Newton 这条街道" 走地理专家）；
- 训练数据混合更彻底，专家分工更细。

---

## 3.4 Capacity Factor 与 Token Drop

### 3.4.1 容量公式

每个专家在每个 batch 内，最多接收：

$$
C = \text{capacity\_factor} \cdot \left\lceil \frac{T \cdot K}{N} \right\rceil
$$

- 训练时通常 `capacity_factor = 1.25`；
- 推理时常用 `capacity_factor = 2.0` 或更高，保证不丢 token。

### 3.4.2 超过容量怎么办？

两种主流策略：

**策略 A：Drop**
- 超过容量的 token 直接**跳过 MoE 层**；
- 通过残差连接传到下一层；
- Switch Transformer、GShard 默认行为。

**策略 B：No-Drop（递补）**
- 超过容量的 token 路由到第二选择专家；
- 实现复杂，吞吐受影响；
- DeepSeek 等部分实现采用。

> 训练时少量 drop 是可接受的，但**推理时 drop 会破坏一致性**。生产部署中通常会：
> 1. 把 capacity_factor 拉大；
> 2. 或用 Expert Choice 等"逆向路由"避免 drop。

---

## 3.5 共享专家 (Shared Experts)

DeepSeek-MoE 提出的设计：在 $N$ 个稀疏专家之外，保留 $N_s$ 个**所有 token 都激活**的共享专家。最终输出：

$$
y = \underbrace{\sum_{i=1}^{N_s} S_i(x)}_{\text{共享部分}} + \underbrace{\sum_{i \in \text{TopK}} g_i(x) E_i(x)}_{\text{稀疏部分}}
$$

```
        ┌──────────┐
        │   Gate   │
        └────┬─────┘
             ▼
       ┌──── Top-K ────┐
   E1  E2  ...  En     │
       │               │
       ▼               │
       │   ┌───────────▼───────────┐
       └──►│  Shared Expert (S1)   │──► +
           └───────────────────────┘
```

**为什么共享专家有用？**

1. **吸收通用知识**：标点、连词、句法这类"所有 token 都需要的"知识不应该让稀疏专家重复学习；
2. **减少专家干扰**：稀疏专家可以把容量留给真正"特化"的知识；
3. **训练更稳**：共享专家保证每个 token 至少有一条稳定的梯度通路。

DeepSeek-V3 配置：1 个共享专家 + 256 个路由专家，每 token Top-8。

---

## 3.6 细粒度专家 (Fine-grained Experts)

传统 MoE：8 个大专家，每个 hidden size 是稠密 FFN 的 1×。

DeepSeek-MoE 思路：把 8 个大专家**切成 $m$ 倍数量、$1/m$ 大小**的细粒度专家，激活数也×$m$。

| 配置 | 激活专家数 | 单专家 hidden | 总激活参数 |
|------|------------|---------------|------------|
| 粗粒度 | 8 中选 2 | $4d$ | 等价 $2 \times 4d = 8d$ |
| 细粒度 (m=4) | 32 中选 8 | $d$ | 等价 $8 \times d = 8d$ |

**激活参数完全一样，但组合数从 $C_8^2 = 28$ 涨到 $C_{32}^8 = 10518300$**。

> 直觉：让组合空间指数级扩大，专家可以学到更细粒度的"概念组合"。

代价是：

- 通信：每个 token 要发到更多 device；
- 实现：需要更精细的 batched expert 计算（Megablocks）。

---

## 3.7 Expert 内部结构

每个专家通常就是一个标准的 FFN：

```python
class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

GLU 变种 (SwiGLU) 在现代 MoE 也很常见：

```python
class SwiGLUExpert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)  # gate proj
        self.w2 = nn.Linear(d, d_ff, bias=False)  # up proj
        self.w3 = nn.Linear(d_ff, d, bias=False)  # down proj

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

> 注意：SwiGLU 专家的参数量是标准 FFN 的 1.5 倍，做 MoE 化时要相应调整 hidden 维度。

---

## 3.8 一个完整的稀疏 MoE Block (PyTorch)

把上面所有概念组装起来：

```python
class SparseMoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k,
                 capacity_factor=1.25, n_shared=0):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)])
        self.shared = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)             # [B*T, D]
        logits = self.gate(x_flat)            # [B*T, N]
        topk_vals, topk_idx = logits.topk(self.k, -1)
        gates = F.softmax(topk_vals, dim=-1)

        y = torch.zeros_like(x_flat)

        # capacity per expert
        capacity = int(self.capacity_factor * (B * T * self.k) / self.n_experts)

        for i, expert in enumerate(self.experts):
            # 选中专家 i 的所有 (token, slot) 对
            idx = (topk_idx == i).nonzero(as_tuple=False)  # [M, 2]
            if idx.numel() == 0: continue
            if idx.shape[0] > capacity:                     # 超容量则丢弃
                idx = idx[:capacity]
            tok = idx[:, 0]                                 # token id
            slot = idx[:, 1]                                # 第几个 topk
            out = expert(x_flat[tok])                       # [M, D]
            y.index_add_(0, tok, out * gates[tok, slot].unsqueeze(-1))

        # shared experts (所有 token 都过一遍)
        for s in self.shared:
            y = y + s(x_flat)

        return y.reshape(B, T, D)
```

这段代码包含：
- Top-K + Softmax 门控；
- Capacity 限制（drop 策略 A）；
- 可选共享专家。

**仍然没有的：**
- 负载均衡损失（第 5 章）；
- 分布式专家并行（第 6 章）；
- 高效 batched 计算（第 8 章 Megablocks）。

---

## 3.9 本章要点

- MoE 化的最佳位置是 **FFN**，Attention 保持稠密；
- 通常 **隔层 MoE**，不是每层都做；
- 路由粒度选 **Token-level**；
- **Capacity Factor** 控制 drop 概率，是关键超参；
- **共享专家 + 细粒度专家** 是 DeepSeek 的关键贡献；
- 一个稀疏 MoE Block 的最小实现 ≈ Gate + Expert List + index_add。

→ [第 4 章 路由机制全解](04-routing.md)
