# 第 6 章 训练技巧与稳定性：从专家并行到通信优化

## 6.1 训练 MoE 的"七宗罪"

千卡训 MoE 比训 Dense 难得多。常见崩溃模式：

1. **专家坍塌**（路由问题）；
2. **数值发散**（logits 爆炸）；
3. **All-to-All 通信瓶颈**；
4. **专家显存不均**（某些专家 token 多，OOM）；
5. **梯度同步开销**（专家梯度跨设备）；
6. **bf16 / fp16 精度不够**；
7. **Fine-tune 时过拟合**。

第 5 章解决了 1、2，本章解决 3–7。

---

## 6.2 专家并行 (Expert Parallelism)

### 6.2.1 为什么需要它

$N=64$ 专家、每个专家 hidden $4d$ ≈ 1B 参数 → 总专家参数 64B。一张 H100 只有 80GB 显存，**装不下**。

解决：把不同专家放在不同 device 上。

### 6.2.2 EP=N 的简单情况

```
GPU 0: Expert 0
GPU 1: Expert 1
  ...
GPU 7: Expert 7
```

每个 GPU 持有一个专家的全部参数，token 通过 All-to-All 通信被发送到对应 GPU。

### 6.2.3 EP < N 的情况

每张卡上放多个专家：

```
GPU 0: Expert 0, 8, 16, ...
GPU 1: Expert 1, 9, 17, ...
  ...
```

DeepSeek-V3：256 专家，64 张 H100，**EP=64**，每卡 4 专家。

---

## 6.3 All-to-All 通信

### 6.3.1 流程

```
[step 1] 每个 device 路由 token，计算每个 token 要发到哪个 device
[step 2] All-to-All: 把 token 重新分配到目标 device
[step 3] 各 device 上专家并行计算
[step 4] All-to-All: 把结果发回原 device
[step 5] 加权求和并继续 forward
```

每个 MoE 层有**两次 All-to-All**——前向一次发出去，一次收回来；反向再来一次。这是 MoE 训练的最大开销。

### 6.3.2 通信量分析

设 batch token 数 $T$，激活专家数 $K$，hidden $d$，bf16 (2 bytes)：

$$
\text{每次 All-to-All 通信量} = 2 \cdot T \cdot K \cdot d
$$

对于 $T=4096$、$K=2$、$d=4096$：

$$
2 \cdot 4096 \cdot 2 \cdot 4096 = 64 \text{ MB / device / All-to-All}
$$

每层 4 次 All-to-All（前向 2 + 反向 2），如果 32 层，每 step 通信量 ≈ 8 GB / device。InfiniBand 200Gbps 也要花十几毫秒，与计算重叠不上时是大瓶颈。

### 6.3.3 优化方向

- **拓扑感知**：节点内 NVLink 优先，节点间 IB 用得越少越好；
- **节点限制路由**（DeepSeek-V3）：限制每 token 只能跨少数节点；
- **激活重计算 vs 缓存** trade-off；
- **重叠通信与计算**（DeepSpeed-MoE 的 Expert-Parallel + Tensor-Parallel 重叠）。

---

## 6.4 节点限制路由 (Node-Limited Routing)

DeepSeek-V3 的关键工程优化：

```
专家分布：256 专家 / 8 节点 = 32 专家/节点
路由约束：每个 token 最多到 M=4 个节点
```

实现方式：

1. 计算 token 在所有 256 专家上的得分；
2. 按节点把得分聚合（sum 或 mean），取前 4 个节点；
3. 在这 4 个节点的 32×4=128 专家上取 Top-K=8。

**好处**：跨节点 All-to-All 流量减少 50%（原本可到 8 节点 → 现在只到 4 节点）。

---

## 6.5 梯度的"每专家"特性

MoE 的梯度有两个反常之处：

1. **专家梯度只来自被路由到的 token**：未被选中专家梯度为 0；
2. **专家间梯度规模差异大**：被频繁选中的专家梯度大，反之小。

这导致：

- **AdamW 的 momentum / variance** 对不同专家累积速度不同；
- 学习率需要小心；
- 梯度裁剪建议**按专家分组**而不是全局。

### 6.5.1 实践建议

- 学习率比同等 Dense 小 1.5–2x；
- 使用 AdamW 加 weight decay（专家容易过拟合）；
- Grad clip 设 1.0 即可，无需特别针对专家；
- 不要给专家做 EMA（会冲淡专家分工）。

---

## 6.6 bf16 / fp8 下的稳定性

### 6.6.1 bf16

- bf16 范围与 fp32 相同（exp 8-bit），精度低（mantissa 7-bit）；
- **Z-Loss 是 bf16 训 MoE 的必需品**（防止 logsumexp 溢出）；
- 路由器的 logits 用 fp32 算更稳（DeepSeek-V3、Mixtral 都这样）。

### 6.6.2 fp8

DeepSeek-V3 训练全程用 fp8（GEMM 部分），是公开报道中第一个大规模 fp8 MoE：

- 专家 GEMM 用 fp8；
- 路由器、aux loss、attention 的 K/V cache 等保留 bf16/fp32；
- 关键技巧：**block-wise scaling**——每 128 个元素一个 scaling factor，限制 fp8 量化误差。

> fp8 让 V3 训练吞吐相比 bf16 提高约 **2x**，是开源最强训练效率之一。

---

## 6.7 容量与 Drop 的训练曲线

经验性现象：

- **训练前 1k 步**：drop 率高达 10–20%（路由器还没学好）；
- **1k–10k 步**：drop 率快速下降到 1% 以内；
- **后期**：drop 率稳定，但具体数值取决于 capacity_factor。

这意味着：

- **不要在前 1k 步就根据 loss 判断模型不行**；
- 看专家利用率比看 loss 更早能反映训练健康度；
- 如果训了 5k 步专家利用率仍极不均衡 → bug。

---

## 6.8 微调 (Fine-tuning) 的特殊问题

ST-MoE 论文专门研究了 MoE 的下游 finetune，发现：

1. **MoE 比 Dense 更容易过拟合**：参数多但每次只激活一部分，等效于"高容量 + 低正则"；
2. **路由器在小数据集上学不好**：直接 freeze 路由器、只 finetune 专家，效果反而更好；
3. **MoE 在小任务上未必比 Dense 好**：finetune 数据少时，全部参数用上的 Dense 更不易过拟合。

### 6.8.1 finetune 配方

- **小数据集 (< 100k 样本)**：freeze 路由器；
- **大数据集 (> 1M 样本)**：全参数 finetune；
- 学习率 5e-6 ~ 5e-5（比 Dense 略小）；
- weight decay 0.1–0.3（比 Dense 略大）；
- 增加 dropout 0.1。

---

## 6.9 Megablocks：把 batched expert 计算变快

### 6.9.1 朴素实现的问题

第 3 章那个 PyTorch demo，每个专家是个 for loop。在 GPU 上：

- 每次 expert 调用启动 kernel 开销大；
- 不同专家 token 数差异大，难以并行；
- GEMM 形状不规则。

### 6.9.2 Megablocks 思路

把 MoE 的多个不同形状 GEMM 合并成一个**block-sparse GEMM**：

- 所有 token × hidden 矩阵，被组织成 block-diagonal 形状；
- 不同专家的权重矩阵也排成 block；
- 一次 cuSparse / 自研 kernel 处理所有专家计算。

吞吐相比 PyTorch 朴素 for-loop 提高 **2–3x**。Mistral、Databricks 都在用。

---

## 6.10 训练超参数 cheat sheet

| 项目 | 值 | 说明 |
|------|------|------|
| Capacity Factor (训练) | 1.25 | drop 控制在 1% 以内 |
| Capacity Factor (推理) | 2.0+ | 不丢任何 token |
| Aux Loss 系数 | 0.01 | Switch/Mixtral 经验值 |
| Z-Loss 系数 | 0.001 | 数值稳定 |
| 路由器 lr 倍数 | 1.0 | 与主网络相同 |
| 路由器 weight decay | 0 | 路由器不应被正则压扁 |
| 主网络 weight decay | 0.1 | 与 Dense 一致 |
| Grad clip | 1.0 | 全局 |
| 学习率 (相比 Dense) | 0.5–0.8x | 略小 |
| Warmup steps | 2000+ | 比 Dense 长 |

---

## 6.11 调试 / 排错指南

| 现象 | 可能原因 | 处方 |
|------|----------|------|
| Loss 在 1k 步内崩 | logits 爆炸 | 加 Z-Loss / fp32 router |
| 死专家持续存在 | aux loss 太弱 / 路由初始化差 | 调大 aux 系数 / re-init 路由 |
| All-to-All 占比 > 40% step | 通信瓶颈 | EP 优化 / 节点限制路由 |
| 单卡 OOM | capacity 过大 / 专家太胖 | 降 capacity / 降 hidden |
| Loss 收敛慢 | drop 太多 | 升 capacity_factor |
| Finetune 过拟合 | 全参 finetune | freeze router / 加 dropout |

---

## 6.12 本章要点

- **专家并行 + All-to-All** 是 MoE 训练的工程核心；
- **节点限制路由** 是千卡级 MoE 的标配优化；
- **bf16 + Z-Loss + fp32 router** 是数值稳定的三件套；
- **fp8 训练** 在 DeepSeek-V3 后成为新前沿；
- Finetune MoE 比 Dense 更容易过拟合，建议 freeze router 在小数据上；
- **Megablocks** 是单卡上把多专家 GEMM 合并的关键 kernel 技术。

→ [第 7 章 经典 MoE 模型解析](07-classic-models.md)
