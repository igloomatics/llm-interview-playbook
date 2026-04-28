# 第 5 章 负载均衡：Auxiliary Loss、Z-Loss 与无辅助损失方案

## 5.1 为什么"不均衡"会要命

设你训练一个 8 专家 MoE，理想情况下每个专家接收 12.5% 的 token。但实际训练前几百步，可能出现：

```
expert 0:  87%
expert 1:   8%
expert 2:   3%
expert 3:   1%
expert 4-7: 0% （死专家）
```

这会导致：

1. **专家容量浪费**：80% 的算力闲置；
2. **梯度饥饿**：死专家永远学不到东西，越学越死；
3. **路由器只学到"无脑选 0 号"**：学习信号被"占主导"的专家固化；
4. **超容量 token 被 drop**：模型质量崩坏；
5. **分布式集群上 1 张卡过载，其余空转**。

> 一个不均衡的 MoE，本质是**伪装成 MoE 的 Dense 模型**——总参数大却没用到。

所以负载均衡不是"锦上添花"，而是"生死攸关"。

---

## 5.2 GShard 的 Auxiliary Loss（经典方案）

### 5.2.1 公式

GShard 与 Switch 引入辅助损失：

$$
\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

其中：

- $f_i = \frac{1}{T} \sum_t \mathbb{1}[i \in \text{TopK}(t)]$：专家 $i$ 在该 batch 中被选中的**频率**（每个 token 最多 K 次）；
- $P_i = \frac{1}{T} \sum_t \text{softmax}(\text{logits}_t)_i$：专家 $i$ 在该 batch 中被分配的**平均概率**。

### 5.2.2 直觉

- $f_i$ **不可导**（来自 TopK 离散决策），但仍然是个量；
- $P_i$ **可导**，是 logits 的 softmax 平均；
- 乘积 $f_i \cdot P_i$ 很大，意味着"既被选很多次，路由器也很想选它"——这就是失衡；
- 总和 $\sum f_i P_i$ 在均匀时最小（柯西-施瓦兹），不均时变大；
- 用它做正则项，会**推 logits 让分布趋于均衡**。

### 5.2.3 工程上的写法

```python
# logits: [T, N]
probs = F.softmax(logits, dim=-1)          # [T, N]
topk_idx = probs.topk(k, dim=-1).indices    # [T, K]

# 每个 token 在每个专家上的"is selected" mask
one_hot = F.one_hot(topk_idx, num_classes=n_experts).sum(1).float()  # [T, N]
f_i = one_hot.mean(0)                       # [N]，被选频率
P_i = probs.mean(0)                         # [N]，平均路由概率
aux_loss = (f_i * P_i).sum() * n_experts
```

通常以**很小的权重**加到主 loss 上：`total_loss = ce_loss + 0.01 * aux_loss`。系数过大会牺牲 perplexity。

---

## 5.3 Switch Transformer 的简化版

Switch 用的是 GShard 公式的特例（K=1），并把它解释为"importance × utilization"。

实际效果：**在 K=1 + capacity_factor=1.25 + aux_loss=0.01** 这一套配置下，已经能把负载均衡做得相当好。这是工业界长期默认的 baseline。

---

## 5.4 Router Z-Loss（ST-MoE, 2022）

### 5.4.1 它解决什么问题

辅助损失保证均衡，但**不保证数值稳定**。MoE 训练中常见的 bug：

- logits 越变越大；
- softmax 输出趋于 one-hot；
- 微小的扰动让 token 在专家间剧烈跳变；
- **bf16 下数值溢出**导致 NaN。

ST-MoE 论文发现这是因为 logits 的"绝对幅度"在不受约束地增长。

### 5.4.2 Z-Loss 公式

$$
\mathcal{L}_z = \frac{1}{T} \sum_t \left( \log \sum_i \exp(\text{logits}_{t,i}) \right)^2
$$

`logsumexp(logits)` 就是 softmax 的"分母对数"。它直接刻画了 logits 的最大幅度。把它平方做正则，可以**温和地把 logits 拉回小数值范围**。

```python
log_z = torch.logsumexp(logits, dim=-1)    # [T]
z_loss = (log_z ** 2).mean()
total_loss = ce_loss + 0.01 * aux_loss + 0.001 * z_loss
```

### 5.4.3 工程效果

ST-MoE 报告：加上 Z-Loss 后，**MoE 训练中的数值发散基本消失**，bf16 下可以放心训。后来 GLaM、PaLM-MoE、Mixtral 都把 Z-Loss 列为标配。

> Aux Loss 管"分布"，Z-Loss 管"幅度"。这两个加在一起就是 MoE 训练的"双安全带"。

---

## 5.5 DeepSeek-MoE：Auxiliary-Loss-Free Load Balancing

### 5.5.1 动机

辅助损失虽然有用，但有副作用：

> "为了均衡而牺牲性能"。

直觉：辅助损失会**惩罚那些确实只有少数 token 适合的专家**——比如某个专家专门处理代码，但训练 batch 里只有 5% 代码 token，aux loss 会强行把它推向接收 12.5%，反而损害分工。

DeepSeek-V3 干脆抛弃辅助损失，提出**直接给路由 logits 加可调 bias**。

### 5.5.2 算法

每个专家 $i$ 有两个参数：

- 路由 weight $W_g$（参与梯度训练）；
- 路由 bias $b_i$（**不参与梯度训练**，由控制器更新）。

路由时：

$$
s_i^{\text{rout}} = \sigma(W_g x)_i + b_i \quad \text{（用于 TopK 选择）}
$$

$$
g_i = \frac{\sigma(W_g x)_i}{\sum_{j \in \text{TopK}} \sigma(W_g x)_j} \quad \text{（用于加权专家输出）}
$$

注意：**bias 进入路由决策，但不进入最终 gate 权重**。

控制器更新规则（每个 step / 每若干 step）：

```
for i in experts:
    if load[i] > target:        # 该专家过载
        b[i] -= γ
    elif load[i] < target:      # 该专家欠载
        b[i] += γ
```

`γ` 是一个小常数（例如 0.001）。

### 5.5.3 为什么有效

- **优雅地**改变路由概率：不通过损失 → 不影响梯度；
- **完全均衡**可达到（足够步数后）；
- 不损害模型本身的学习目标。

### 5.5.4 实现细节

DeepSeek-V3 实际还保留了一个**非常小**的辅助损失（系数 0.0001 量级）作为兜底，主要靠 bias 控制器。Qwen3-MoE、近期开源模型都开始采纳此方案。

---

## 5.6 Expert Choice：用算法天然均衡

第 4 章已经讲过 Expert Choice。它的负载均衡是**结构性**的：

- 每个专家容量为 $C$；
- 专家从 token 中选 $C$ 个；
- **每个专家恰好处理 $C$ 个 token，永不超载也永不欠载**。

这是最"硬"的均衡方案。代价是不能用于自回归推理。

---

## 5.7 一些"小招数"

工程实践中还有一些不那么显眼但很有效的技巧：

### 5.7.1 Random Reroute

如果第一选择专家容量满了，把 token 路由到第二选择，而不是直接 drop。

### 5.7.2 Dropout on Router

类似 dropout，让路由器在训练时**随机屏蔽部分专家**，强迫它探索更多专家。

### 5.7.3 Warmup Capacity

训练初期用大容量（capacity_factor=2.0），让路由器有空间犯错；后期降到 1.25。

### 5.7.4 Expert Re-init

定期检查"死专家"，重新初始化它们的参数（罕见但确实有效）。

---

## 5.8 实战配方推荐

不同规模的 MoE 推荐不同配置（基于公开论文与社区经验）：

| 规模 | 路由 | Capacity Factor | Aux Loss | Z-Loss | 备注 |
|------|------|-----------------|----------|--------|------|
| 小 (≤8 专家) | Top-2 | 1.25 | 0.01 | 0.001 | Switch/Mixtral 风格 |
| 中 (16–64 专家) | Top-2 ~ Top-4 | 1.5 | 0.01 | 0.001 | 加共享专家更稳 |
| 大 (≥128 专家，细粒度) | Top-6 ~ Top-8 + 共享 | 2.0 | 几乎可关 | 0.001 | DeepSeek 风格 + bias 均衡 |

---

## 5.9 一个调试 checklist

如果你训 MoE 发现效果差或不稳，按顺序检查：

1. **看专家利用率**：每个专家 token 占比，理想是均匀；
2. **看 logits 幅度**：`max(|logits|)` 是否爆炸（应 < 50）；
3. **看 drop 率**：是否 > 5%（应 < 1%）；
4. **看死专家**：是否有专家长期 0 token（>1000 步零分配）；
5. **看 bf16 下是否 NaN**：先升 fp32 验证算法，再降回 bf16；
6. **看 expert capacity 是否成为瓶颈**：单专家显存是否爆。

---

## 5.10 本章要点

- **负载均衡决定 MoE 训练能不能 work**；
- Aux Loss = $N \sum f_i P_i$，靠 logits 梯度推动均衡；
- Z-Loss = $(\log\sum e^{\text{logits}})^2$，控制 logits 数值幅度；
- Aux Loss + Z-Loss 是稠密 baseline；
- DeepSeek-V3 的 **动态 bias** 是 2024 年起的新主流；
- Expert Choice 是结构性均衡，但只适合训练；
- 训练时多看专家利用率分布，比看 perplexity 更早发现问题。

→ [第 6 章 训练技巧与稳定性](06-training.md)
