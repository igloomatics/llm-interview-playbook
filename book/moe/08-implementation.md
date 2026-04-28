# 第 8 章 MoE 的工程实现：从一个 PyTorch demo 到 Megablocks

本章把"会跑"和"跑得快"的 MoE 实现一步步拆开。

---

## 8.1 单卡 MoE：教学版本

我们先实现一个能在单卡上跑起来的 MoE 块（与第 3 章那段类似，但更完整）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k=2,
                 capacity_factor=1.25, aux_loss_weight=0.01,
                 z_loss_weight=1e-3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_w = aux_loss_weight
        self.z_w = z_loss_weight

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        n_tok = x_flat.size(0)

        # ---- 路由 ----
        # router 用 fp32 算更稳
        logits = self.gate(x_flat.to(torch.float32))            # [Tok, N]
        log_z = torch.logsumexp(logits, dim=-1)                 # [Tok]
        probs = F.softmax(logits, dim=-1)                        # [Tok, N]
        topk_p, topk_i = probs.topk(self.top_k, dim=-1)          # [Tok, K]
        topk_p = topk_p / (topk_p.sum(-1, keepdim=True) + 1e-9)
        topk_p = topk_p.to(x.dtype)

        # ---- 容量 + dispatch ----
        cap = int(self.capacity_factor * n_tok * self.top_k / self.n_experts)
        y = torch.zeros_like(x_flat)
        # 简单实现：按专家循环
        for e in range(self.n_experts):
            mask = (topk_i == e)                                  # [Tok, K]
            if not mask.any(): continue
            # token 索引、对应的 slot
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() > cap:
                tok = tok[:cap]; slot = slot[:cap]
            out = self.experts[e](x_flat[tok])                    # [m, D]
            y.index_add_(0, tok, out * topk_p[tok, slot].unsqueeze(-1))

        # ---- 辅助损失（aux + z）----
        with torch.no_grad():
            f_i = F.one_hot(topk_i, self.n_experts).sum(1).float().mean(0)  # [N]
        P_i = probs.mean(0)                                       # [N]
        aux_loss = self.n_experts * (f_i * P_i).sum()
        z_loss = (log_z ** 2).mean()
        self.last_aux = self.aux_w * aux_loss + self.z_w * z_loss

        return y.view(B, T, D)
```

> 这段代码在单卡上能跑，速度可接受。但它在 8 卡及以上必须改成专家并行版本。

---

## 8.2 专家并行：分布式版本

### 8.2.1 总思路

```
设有 N 个专家、E 张 GPU 做 EP（N 是 E 的整数倍）。
每张 GPU 持有 N/E 个专家的参数。
forward:
  1. 本地路由：每张 GPU 自己计算 [本地 token, top_k_idx, gate]
  2. All-to-All:  把每个 token 按目标专家所在 GPU 发送
  3. 各 GPU 跑本地专家
  4. All-to-All: 把结果发回原 GPU
  5. 加权求和写入输出
```

### 8.2.2 用 NCCL All-to-All 的伪代码

```python
def moe_forward_ep(x, gate, experts, ep_group):
    """
    x:        [tok_local, D]
    experts:  本地专家 list (n_experts // ep_size 个)
    """
    # 1. 路由
    logits = gate(x.float())
    topk_p, topk_i = logits.topk(top_k, dim=-1)

    # 2. 把 token 按目标 GPU 排序、分组
    target_rank = topk_i // experts_per_rank   # 哪张卡
    sorted_x, perm = sort_by(x, topk_i, target_rank)

    # 3. All-to-All: 发送 + 接收
    counts_send = compute_send_counts(target_rank)
    counts_recv = all2all_counts(counts_send, ep_group)
    recv_x = all_to_all(sorted_x, counts_send, counts_recv, ep_group)
    recv_idx = all_to_all(local_topk_i, ...)   # 同时把 idx 发过去

    # 4. 本地专家计算（按本地专家分组）
    out = torch.zeros_like(recv_x)
    for local_e_id, expert in enumerate(experts):
        m = (recv_idx % experts_per_rank == local_e_id)
        out[m] = expert(recv_x[m])

    # 5. All-to-All 反向
    sorted_out = all_to_all(out, counts_recv, counts_send, ep_group)
    out = unpermute(sorted_out, perm)
    out = (out * topk_p.unsqueeze(-1)).sum(dim=-2)  # 合并 K 个 expert 输出
    return out
```

完整、生产级的实现见 DeepSpeed-MoE、ColossalAI、Megatron-Core。

---

## 8.3 DeepSpeed-MoE

DeepSpeed 是 Microsoft 的分布式训练框架，对 MoE 提供：

- **Expert Parallelism + Data Parallelism + Tensor Parallelism** 任意组合；
- 自动 All-to-All 的拓扑感知；
- **PR-MoE (Pyramid-Residual MoE)**：浅层用 Dense，深层用 MoE，省通信；
- **MoE 推理引擎**：把 expert 离线放在 CPU，用时再 swap 到 GPU。

典型配置（8 卡训练 64 专家）：

```yaml
moe:
  num_experts: 64
  ep_size: 8           # 每卡 8 专家
  top_k: 2
  capacity_factor: 1.25
  use_residual: false
```

---

## 8.4 Megablocks

> Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" (2022)

### 8.4.1 解决的问题

朴素 MoE forward 的 GEMM 形状极不规则：

```
expert 0: shape (24, 4096) × (4096, 11008)
expert 1: shape ( 7, 4096) × (4096, 11008)
expert 2: shape (51, 4096) × (4096, 11008)
... 不同 expert token 数差异大 ...
```

8 个独立 cuBLAS 调用，kernel launch 开销大、SM 利用率低。

### 8.4.2 思路：block-sparse GEMM

把所有 token 排成一个大矩阵，按 expert 块组织成 block-diagonal：

```
[tok_0..23, expert 0 block]
[tok_24..30, expert 1 block]
[tok_31..81, expert 2 block]
...
```

- 用一个**自定义 CUDA kernel** 一次完成所有 expert 的乘法；
- 利用 NVIDIA 的 sparse GEMM 原语（cuSparseLt、Triton 实现）；
- 吞吐相比朴素 PyTorch 提高 **2–3x**。

### 8.4.3 在哪里用

- Mistral、Databricks DBRX 等的训练栈；
- 常作为 Megatron-Core MoE 的 backend；
- Triton 上有完整开源实现。

---

## 8.5 vLLM 中的 MoE 推理

### 8.5.1 困难

推理 MoE 比训练 MoE 复杂：

- 解码阶段每 step 只有 **1 个 token / sequence**；
- 路由分布稀疏到极致（一个 token 只去 K 个专家）；
- 大量 batch 并发，路由分布更复杂。

### 8.5.2 vLLM 的处理

vLLM (≥ 0.3) 对 MoE 做了：

- **Fused MoE Kernel**：把 gate + dispatch + expert + combine 融合成一个 CUDA kernel；
- **TP + EP 混合并行**：attention 走 TP，MoE 走 EP；
- **Continuous Batching** 兼容：MoE 路由在 batch 级别动态计算。

实测：
- Mixtral 8x7B 在 4×A100 上吞吐 vs Llama2-70B：吞吐高 30–50%，激活参数仅 13B vs 70B。

---

## 8.6 SGLang / TensorRT-LLM

**SGLang**：UCB / Stanford 的高性能推理框架，对 MoE 也提供 fused kernel 与 EP；DeepSeek 推理参考实现就是 SGLang。

**TensorRT-LLM**：NVIDIA 官方，MoE 支持成熟，是闭源/商用部署的主流选择之一。

---

## 8.7 几个工程坑（实战经验）

### 8.7.1 路由器初始化

路由器权重初始化太小 → softmax 输出趋于均匀，所有专家都拿到差不多 token，看起来"很均衡"但其实"没分工"。

**经验**：路由器用与主网络相同的初始化（typical std=0.02），不要做特殊收敛。

### 8.7.2 专家初始化

8 个专家用相同种子初始化 → 训练初期所有专家**完全一样**，路由器学不到分工，专家保持相同。

**经验**：每个专家用**不同种子**初始化，或干脆从 dense FFN up-cycle（复制 + 加噪声）。

### 8.7.3 Capacity Factor 调试

如果训练时观察 drop > 5%，**先升 capacity，不要先动 aux loss**。drop 是显性问题，aux loss 是隐性问题，先解决显性。

### 8.7.4 fp8 / bf16 混合

DeepSeek-V3 经验：

- 路由器：bf16/fp32；
- Expert GEMM：fp8（block-wise scaling）；
- Aux/Z loss 计算：fp32；
- 不要全 fp8，会出 NaN。

### 8.7.5 推理时的 KV cache

**Attention 是稠密的**，所以 KV cache 与 Dense 一样。MoE 的稀疏只在 FFN 部分体现。这意味着**长上下文 MoE 推理仍然受 KV cache 显存限制**——这就是为什么 DeepSeek-V2 引入了 MLA 来压缩 KV cache。

---

## 8.8 一段最简的 EP + Megablocks 组合代码骨架

```python
# 用 megablocks（开源 Triton 实现）加速本地 expert 计算
from megablocks import ops as mb_ops

def moe_forward_with_mb(x, gate_w, expert_ws, top_k, ep_group):
    # 1. 路由
    logits = x @ gate_w
    topk_p, topk_i = logits.topk(top_k, dim=-1)

    # 2. 按 expert 排序
    sorted_idx, sort_map = mb_ops.sort_tokens(topk_i)
    sorted_x = x[sort_map]

    # 3. 跨设备 dispatch (All-to-All)
    sorted_x, recv_meta = ep_dispatch(sorted_x, sort_map, ep_group)

    # 4. 本地 megablocks GEMM
    sorted_out = mb_ops.grouped_gemm(sorted_x, expert_ws, recv_meta.expert_offsets)

    # 5. 跨设备 combine (All-to-All)
    sorted_out = ep_combine(sorted_out, recv_meta, ep_group)

    # 6. 反排序 + 加权求和
    out = sorted_out[unsort_map] * topk_p.view(-1, 1)
    return out.view(*x.shape[:-1], top_k, -1).sum(dim=-2)
```

实际项目里 ep_dispatch / ep_combine 由框架（DeepSpeed-MoE、Megatron-Core）封装，业务代码不直接写。

---

## 8.9 推理实测：什么决定 MoE 推理速度

按影响从大到小：

1. **激活参数大小**（决定 GEMM FLOPs）；
2. **Top-K**（决定通信量）；
3. **Expert 是否能 batch**（同 GPU 上的 expert 应能融合）；
4. **KV cache 大小**（attention 仍稠密，长上下文受限）；
5. **Capacity Factor**（推理时建议 ≥2）；
6. **路由分布的尾部**（极不均衡时拖慢整个 batch）。

经验：在 batch_size = 32, seq_len = 2048 下，Mixtral 8x7B 的吞吐**接近** Llama2-13B（激活相同），但**质量接近** Llama2-70B。这就是 MoE 的工程价值。

---

## 8.10 本章要点

- 单卡 demo → 专家并行 → Megablocks 是三层递进；
- **All-to-All 是 MoE 训练 / 推理的核心瓶颈**；
- **Megablocks** 把多 expert GEMM 合并，是单卡加速关键；
- **vLLM / SGLang** 在推理侧提供 fused kernel；
- 工程坑：路由器/专家初始化、capacity 调试、fp8 混合精度；
- MoE 推理快慢主要由 **激活参数 + Top-K + 路由分布** 决定。

→ [第 9 章 MoE 推理优化](09-inference-optimization.md)
