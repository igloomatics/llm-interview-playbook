# 第 4 章 路由机制全解：Top-K、Switch、Expert Choice、Soft MoE

路由 (Routing) 是 MoE 的"心脏"。它决定每个 token 进入哪个专家、专家如何被使用。一个好的路由器要同时满足：

1. **可微**（能用 SGD 训练）；
2. **均衡**（不让某些专家过载/饿死）；
3. **稳定**（训练不发散）；
4. **高效**（路由本身不能拖慢推理）。

这四个目标常常打架。本章把所有主流路由机制摊开看，讨论它们如何取舍。

---

## 4.1 Top-K Gating（GShard 风格）

### 4.1.1 算法

```
logits = W_g @ x                          # [N]
topk_logits, topk_idx = logits.topk(K)
gates = softmax(topk_logits)              # 仅在 Top-K 上做归一化
y = sum(gates[k] * E[topk_idx[k]](x) for k in range(K))
```

注意"仅在 Top-K 上做 Softmax"是一个微妙但关键的选择：

- **A 方案（Top-K Softmax，主流）**：先 TopK，再对 K 个 logits Softmax；
- **B 方案（Softmax-Then-Mask）**：先对所有 N 个 Softmax，再保留 Top-K，再重归一化。

两者梯度行为有细微差别。Mixtral 用 A 方案，Switch 用 B 方案的简化形式。

### 4.1.2 直觉

Top-K 把"软选择"硬截断到 K 个，等于做了一个**离散瓶颈**。它的好处是：

- **稀疏**：每个 token 只激活 K 个专家；
- **可微**：连续梯度仍能通过被选中的 K 条路径流回；

代价是：**未选中的专家完全没梯度**。这就需要负载均衡机制（第 5 章）。

### 4.1.3 K 怎么选

| K | 优点 | 缺点 |
|---|------|------|
| 1 | 通信、计算最小 | 路由波动大，需要更强 capacity |
| 2 | 单 token 看到两条专家路径，鲁棒 | 通信×2 |
| 4–8 | 细粒度专家场景 | 通信开销显著 |

DeepSeek-V2/V3 用 K=6/8（细粒度），Mixtral 用 K=2（粗粒度），Switch 用 K=1。

---

## 4.2 Noisy Top-K Gating（Shazeer 2017）

为了"鼓励所有专家被尝试"，给 logits 加可学习幅度的高斯噪声：

$$
H(x)_i = (W_g x)_i + \epsilon_i \cdot \text{softplus}((W_n x)_i), \quad \epsilon_i \sim \mathcal{N}(0, 1)
$$

```python
clean = x @ W_g
noise = F.softplus(x @ W_n) * torch.randn_like(clean)
logits = clean + noise          # 训练时
# logits = clean                # 推理时（或保留小 noise 看实现）
```

**为什么有用？**

- 训练初期所有专家都很烂，门控容易"死磕"少数几个；
- 加噪声让被选中的概率有随机性，更多专家能拿到训练信号；
- 噪声幅度自己学，模型可以决定何时该探索、何时该确定。

**缺点：**

- 推理时关闭噪声会带来训练-推理不一致；
- 后续模型（Switch、GLaM、Mixtral）大多直接放弃噪声，靠 auxiliary loss + capacity 解决均衡问题。

---

## 4.3 Switch Routing（Top-1）

Switch Transformer 把 K 砍到 1：

$$
i^* = \arg\max_i \text{softmax}(W_g x)_i, \quad y = p_{i^*}(x) \cdot E_{i^*}(x)
$$

注意：**虽然只选一个专家，仍然乘以它的 softmax 概率 $p_{i^*}$**。这是为了让 $W_g$ 仍然有梯度（如果直接乘 1，门控就退化成纯分类问题）。

**Switch 的关键工程贡献：**

1. 简化路由 → 单 device 只接收自己专家的 token；
2. 更小的通信量（Top-1 比 Top-2 通信减半）；
3. 配合 capacity_factor 1.0–1.25 + auxiliary loss，效果不差于 GShard。

> 一句话：**"K=1 + 容量因子 + 辅助损失" 是稀疏 MoE 的"最小可行配方"。**

---

## 4.4 Expert Choice Routing（Zhou et al., 2022）

### 4.4.1 思路反转

前面所有路由都是"**token 选专家**"。Expert Choice 反过来：**专家选 token**。

每个专家 $E_i$ 有固定容量 $C$，从所有 token 的 logits 中挑出**对自己得分最高的 C 个**：

```python
scores = softmax(x @ W_g, dim=-1)        # [T, N]，token 对每个专家的得分
# 对每个专家 i，取 score[:, i] 的 Top-C
for i in range(N):
    top_tokens = scores[:, i].topk(C).indices
    y[top_tokens] += scores[top_tokens, i].unsqueeze(-1) * E_i(x[top_tokens])
```

### 4.4.2 优点

- **天然均衡**：每个专家恰好处理 C 个 token，不需要 auxiliary loss；
- **不丢 token**：但同一 token 可能被 0 个专家选中（同样问题反向出现）；
- **训练更稳**：负载固定，路由器更容易学到稳定模式。

### 4.4.3 缺点

- **需要全 token 的得分矩阵**：和因果（causal）注意力的训练流程不太兼容；
- **推理时的"flow"反向**：在线推理逐 token 来，专家无法"等齐 C 个再选"，所以 Expert Choice 主要用在**训练**或**离线推理**。

---

## 4.5 Hash Routing

更极端的简化：完全不学 router，用 hash 函数把 token 映射到专家：

$$
i^* = \text{hash}(\text{token\_id}) \bmod N
$$

```python
expert_idx = (token_ids * 2654435761) % N
```

**优点**：零参数、零计算、负载完美均衡（hash 均匀）。
**缺点**：完全没用上"哪个专家擅长哪种 token"，专家分工随机，性能上限较低。

实践中 Hash Routing 多作为 baseline 出现，证明"学到的路由确实有用"。它的成功反而暗示一个重要事实：

> **路由质量没那么重要——只要不太差**。

后续很多研究（如 Router-Free MoE）就是基于这个观察。

---

## 4.6 Soft MoE（Puigcerver et al., 2023）

### 4.6.1 概念

Soft MoE 抛弃"硬路由"：每个专家不接收 token，而是接收所有 token 的**加权混合**（soft slot）。

设有 $N$ 个专家，每个专家有 $S$ 个 slot。

1. 计算分配矩阵 $D \in \mathbb{R}^{T \times (N \cdot S)}$，softmax over 列（每个 slot 是 token 的混合）；
2. 每个 slot 的输入：$\tilde{x}_{i,j} = \sum_t D_{t, i \cdot S + j} \cdot x_t$；
3. 专家计算：$\tilde{y}_{i,j} = E_i(\tilde{x}_{i,j})$；
4. 输出：$y_t = \sum_{i,j} D'_{t, i \cdot S + j} \cdot \tilde{y}_{i,j}$（再 softmax over 行回传）。

### 4.6.2 优点

- 完全可微，**无 TopK 截断**，梯度通到所有专家；
- 负载完美均衡（slot 数固定）；
- 训练稳。

### 4.6.3 缺点

- 不是真正的"稀疏"——每个专家其实看到了所有 token 的加权和；
- 在自回归 LLM 中应用受限（slot 混合会破坏 causal）；
- 主要在 **ViT 视觉 MoE** 中流行。

---

## 4.7 路由的"再思考"：DeepSeek-V3 的 Auxiliary-Loss-Free

DeepSeek-V3 在路由上做了三件值得记住的事：

**1. Sigmoid 替代 Softmax**

$$
g_i(x) = \sigma((W_g x)_i + b_i)
$$

每个专家的得分独立，不互斥。Top-K 还是从这些得分里取，但归一化只对 Top-K 部分做。

**2. 动态 Bias 实现负载均衡**

每个专家有一个**只用于路由的 bias** $b_i$：

- 该专家被路由的 token 数高于平均值 → $b_i$ 减小；
- 低于平均值 → $b_i$ 增大；
- 这个 bias **不进入最终的 gate 权重**（gate 只用纯 logits 做 softmax/sigmoid）。

详细讨论留到第 5 章。

**3. 节点限制路由 (Node-Limited Routing)**

为了减少跨节点通信，限制每个 token 最多发到 $M$ 个节点（不是 $K$ 个专家）。专家分布在多个节点上，token 优先选择"已经发往的节点上的专家"。

```
token logits over 256 experts (across 8 nodes, 32 experts/node)
       ↓
取得分最高的 4 个 node
       ↓
在这 4 个 node 内取 Top-K=8 专家
```

这一点对千卡集群训练至关重要，第 6 章会展开。

---

## 4.8 路由器的几种"病"

| 症状 | 描述 | 应对 |
|------|------|------|
| **专家坍塌 (Expert Collapse)** | 99% 的 token 都进同一个专家 | Auxiliary Loss / 加 noise / capacity drop |
| **死专家 (Dead Experts)** | 某些专家从来不被选中 | Re-init / dropout 路由 / aux loss |
| **路由抖动 (Routing Instability)** | logits 数值发散，softmax 输出近 one-hot | Z-Loss / logits 截断 |
| **训-推不一致** | 训练有 noise，推理无 noise，行为不同 | 推理保留小幅噪声 / 训练后期降噪 |
| **Drop 高** | capacity 不够，大量 token 跳过 MoE | 提升 capacity_factor / Expert Choice |

这些"病"大多都在第 5 章会有专门处方。

---

## 4.9 路由策略对比表

| 策略 | 可微 | 通信 | 均衡 | LLM 友好 | 代表模型 |
|------|------|------|------|----------|----------|
| Top-K Softmax | ✓ | K× | 需 aux loss | ✓ | GShard, Mixtral |
| Top-1 (Switch) | ✓ | 1× | 需 aux loss + capacity | ✓ | Switch Transformer |
| Noisy Top-K | ✓ | K× | 略好 | ✓ | Shazeer 2017 |
| Expert Choice | ✓ | C×N/T× | 天然均衡 | △（causal 不友好） | LIMoE, EC-MoE |
| Hash | ✗ | K× | 完美 | △ | baseline |
| Soft MoE | ✓ | NS× | 完美 | ✗ | ViT |
| Sigmoid + 动态 Bias | ✓ | K× | 完美（动态） | ✓ | DeepSeek-V3, Qwen3 |

---

## 4.10 一段"接近生产级"的 Top-K 路由代码

```python
def topk_route(x_flat, gate_w, k, capacity_factor, n_experts):
    """
    x_flat:  [T, D]    扁平化后的输入
    gate_w:  [D, N]    门控权重
    return:  topk_idx [T, K], gates [T, K], dispatch_mask [T, N, C], pos [T, K]
    """
    T = x_flat.shape[0]
    logits = x_flat @ gate_w                       # [T, N]
    probs = F.softmax(logits, dim=-1)              # [T, N]
    topk_vals, topk_idx = probs.topk(k, dim=-1)    # [T, K]
    gates = topk_vals / topk_vals.sum(-1, True)    # 重新归一化

    capacity = int(capacity_factor * T * k / n_experts)

    # 给每个 (token, expert) 对算一个"序号" pos
    # 同一个 expert 内 token 顺序编号，超过 capacity 的丢弃
    one_hot = F.one_hot(topk_idx, n_experts)       # [T, K, N]
    pos_in_expert = one_hot.cumsum(0) * one_hot - one_hot   # [T, K, N]
    pos = pos_in_expert.sum(-1)                    # [T, K]
    keep = (pos < capacity)                         # [T, K]
    gates = gates * keep
    return topk_idx, gates, pos, keep
```

实际生产代码会更复杂（处理跨设备 dispatch、bf16 数值稳定、causal 兼容），但骨架就是这样。

---

## 4.11 本章要点

- 路由的四个目标：**可微、均衡、稳定、高效**；
- Top-K + Softmax 是最常见路由，K=1/2/8 各有适用场景；
- Switch (K=1) 简单但需配合 capacity 与 aux loss；
- Expert Choice 用"专家选 token"换天然均衡，但不适合自回归推理；
- DeepSeek-V3 的 Sigmoid + 动态 bias 路由是新一代主流；
- 路由器的常见病：专家坍塌、死专家、抖动、drop 高 —— 都需要负载均衡机制。

→ [第 5 章 负载均衡](05-load-balancing.md)
