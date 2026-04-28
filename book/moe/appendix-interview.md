# 附录 A 面试题精选（30 题）

> 这些题来自真实大厂 LLM 岗位面经整理。每题先给"问题"和"考点"，再给"参考答案要点"。建议先盖住答案自己想，再对照。

---

## 一、概念与原理（10 题）

### Q1. 用一句话讲清楚 MoE 的核心思想。

**考点**：能不能不用术语讲清楚 MoE。

**要点**：
- 把一个大模型的某些层换成"多个小专家 + 一个路由器"；
- 每个 token 只走 K 个专家而非全部 → **总参数大但每次计算量小**；
- 直觉：图书馆有 10 万册书，但你每次只读 2 本。

---

### Q2. MoE 与 Ensemble (Bagging/Boosting) 的本质区别？

**考点**：检验对"端到端学习分工"的理解。

**要点**：
- Ensemble：子模型独立训练，推理时全部投票；
- MoE：路由器与专家**联合训练**，由路由器学到"哪类输入交给哪个专家"；
- 推理时 Ensemble 全部激活，MoE 仅 Top-K；
- 收益：Ensemble 降方差，MoE 提容量 + 控成本。

---

### Q3. 为什么 MoE 替换的是 FFN 而不是 Attention？

**考点**：检验对 Transformer 计算结构的理解。

**要点**：
- FFN 占大模型 ~2/3 参数；
- FFN 是 token-wise 的 → 天然适合按 token 路由；
- Attention 涉及 token 间相互作用，路由会破坏全局信息流；
- 实践中 Attention 也有 MoE 化尝试（MoA），但效果不显著。

---

### Q4. Top-K Gating 中 TopK 算子不可导，为什么还能训练？

**考点**：经典陷阱题。

**要点**：
- 不可导的是"哪些 K 被选中"这个**离散决策**；
- 可导的是"被选中的 K 个 logits 经 softmax 后的权重"；
- 通过这条路径，路由器学习到"应该提升哪些专家的 logits"；
- 副作用：未被选中的专家完全没梯度 → 需要负载均衡机制。

---

### Q5. K=1 (Switch) 与 K=2 (GShard/Mixtral) 各有什么优劣？

**要点**：

| 维度 | K=1 | K=2 |
|------|-----|-----|
| 通信量 | 1× | 2× |
| 计算量 | 1× | 2× |
| 路由抗噪 | 弱 | 较强 |
| 实现复杂度 | 简单 | 中等 |
| 训练稳定性 | 需 capacity & aux loss | 略宽松 |

K=1 + capacity_factor + aux loss = "稀疏 MoE 的最小可行配方"。

---

### Q6. 什么是 Capacity Factor？怎么调？

**要点**：
- 每个专家在每 batch 接收的最多 token 数 = $\text{capacity\_factor} \times T \cdot K / N$；
- 训练 1.25，推理 ≥2.0；
- 太小 → token drop → 质量下降；
- 太大 → 显存浪费；
- 调试经验：drop 率 < 1% 是健康线。

---

### Q7. 什么是专家坍塌 (Expert Collapse)？怎么解？

**要点**：
- 路由器学到"无脑选某个专家"，导致 99% token 进同一专家，其他专家死亡；
- 原因：训练初期所有专家烂，路由器哪个先优势就赢家通吃；
- 解决：**Auxiliary Loss + Capacity drop + 加 noise gating + Z-Loss**；
- 进阶：DeepSeek-V3 的动态 bias 控制器；
- 兜底：定期 re-init 死专家。

---

### Q8. 什么是 Auxiliary Loss？写出公式。

**要点**：

$$\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i$：专家 $i$ 的被选频率（不可导）；
- $P_i$：专家 $i$ 的平均 softmax 概率（可导）；
- 均匀时取最小，不均时变大；
- 系数通常 0.01；
- 通过梯度推动 logits 让分布趋于均衡。

---

### Q9. Router Z-Loss 是什么？为什么需要它？

**要点**：

$$\mathcal{L}_z = \frac{1}{T}\sum_t (\text{logsumexp}(\text{logits}_t))^2$$

- 控制 logits 数值幅度，防止溢出；
- 在 bf16 下尤其重要；
- 系数通常 0.001；
- "Aux Loss 管分布，Z-Loss 管幅度"。

---

### Q10. DeepSeek 的"细粒度专家 + 共享专家"分别解决什么问题？

**要点**：
- **细粒度**：把 8 大专家拆成 64 小专家，组合空间从 28 涨到千万 → 专家可以学到更细的"概念组合"；
- **共享专家**：所有 token 都激活 1–2 个专家 → 吸收"通用知识"，避免稀疏专家重复学习；
- 二者结合：稀疏专家专注差异化，共享专家保底通用，路由更稳定。

---

## 二、训练与系统（10 题）

### Q11. 什么是专家并行 (Expert Parallelism)？为什么需要它？

**要点**：
- 不同专家放在不同 GPU 上；
- token 通过 All-to-All 通信路由到目标 GPU；
- 单 GPU 装不下所有专家 → 必须分布式；
- 与数据并行、张量并行、流水并行可组合（4D 并行）。

---

### Q12. All-to-All 通信在 MoE 中怎么用？是不是瓶颈？

**要点**：
- 每层 MoE 前向 2 次（dispatch + combine），反向再 2 次；
- 通信量 = $2 \cdot T \cdot K \cdot d$ per device per call；
- 在千卡集群是主要瓶颈，可达 step 时间的 30–50%；
- 优化：节点限制路由、拓扑感知、计算-通信重叠。

---

### Q13. 节点限制路由 (Node-Limited Routing) 是什么？

**要点**：
- DeepSeek-V3 创新；
- 限制每 token 最多发往 M 个节点（不是 K 个专家）；
- 先对节点打分，选 top-M 节点；
- 在所选节点的专家上 TopK；
- 跨节点通信减半，质量基本不损。

---

### Q14. fp8 训练 MoE 有什么注意点？

**要点**：
- 路由器：bf16/fp32（不要 fp8）；
- Expert GEMM：fp8（block-wise scaling）；
- Aux/Z loss：fp32；
- KV cache：bf16；
- DeepSeek-V3 实测吞吐 vs bf16 提高 ~2x。

---

### Q15. Megablocks 是什么？解决什么问题？

**要点**：
- 不同专家 token 数不同 → GEMM 形状不规则 → 朴素实现慢；
- Megablocks 用 block-sparse GEMM 一次完成所有专家计算；
- Triton/CUDA kernel 实现；
- 单卡 MoE 计算吞吐相比朴素 PyTorch 提高 2–3x。

---

### Q16. MoE finetune 容易过拟合吗？怎么办？

**要点**：
- 容易，因为参数多 + 单次激活少 = 高容量低正则；
- 小数据集 (<100k)：freeze router + 加 dropout；
- 大数据集：全参 finetune，学习率比 dense 小 1.5–2x；
- weight decay 0.1–0.3 偏大；
- LoRA-MoE 是新兴方向。

---

### Q17. MoE 的学习率应该怎么调？

**要点**：
- 比同等 dense 模型小 0.5–0.8x；
- 路由器学习率与主网络一致；
- weight decay 路由器设 0；
- warmup 步数比 dense 长（路由器需要更多步学到分工）。

---

### Q18. MoE 训练前 1k 步 drop 率高是不是 bug？

**要点**：
- 不是；
- 路由器还没学好，前期不均衡正常；
- 1k–10k 步 drop 率应快速降到 1% 内；
- 如果 5k 步后仍 > 5%，才是问题（aux loss 太弱 / capacity 太低）。

---

### Q19. 怎么判断 MoE 训练是否健康？

**要点**：
- 看每个专家的 token 占比（应近均匀）；
- 看 logits 最大幅度（应 < 50）；
- 看 drop 率（应 < 1%）；
- 看是否有死专家（持续 0 token）；
- 看 loss 曲线（应平滑下降）。

---

### Q20. 为什么 MoE 推理显存利用率低？

**要点**：
- 总参数都得驻留（路由结果不可预知）；
- 单步只用 K/N 比例的参数计算；
- 显存利用率 ≈ K/N（Mixtral ≈ 25%，DeepSeek-V3 ≈ 14%）；
- 这是 MoE 部署的核心痛点。

---

## 三、模型与决策（10 题）

### Q21. 描述 Mixtral 8x7B 的架构。

**要点**：
- 8 个专家，每专家是 SwiGLU FFN，约 5.6B 参数；
- 共享 attention/embedding；
- 总参数 47B（不是 56B）；
- Top-2 路由；
- 每 token 激活约 13B 参数；
- 性能持平/超 Llama2-70B、GPT-3.5。

---

### Q22. DeepSeek-V3 在 MoE 上做了哪些关键创新？

**要点**：
- 细粒度专家 (256 个) + 共享专家 (1 个)；
- 节点限制路由 (M=4)；
- Auxiliary-Loss-Free 负载均衡（动态 bias）；
- MLA (KV cache 压缩 8x)；
- MTP (训练辅助多 token 预测)；
- fp8 训练（吞吐 2x）；
- 总参 671B / 激活 37B。

---

### Q23. MLA (Multi-head Latent Attention) 是 MoE 的一部分吗？

**要点**：
- 不是 MoE 本身，是 KV cache 压缩技术；
- 把 KV 投影到低维 latent，cache 缩小 8–10x；
- 长上下文 MoE 必备配套；
- DeepSeek-V2/V3 引入。

---

### Q24. GPT-4 是 MoE 吗？

**要点**：
- OpenAI 未官方确认；
- 第三方（如 SemiAnalysis）报道：8 个 expert × 220B，总 1.8T，Top-2；
- 推理速度与 cost 模式与 MoE 吻合；
- Gemini 1.5 报告中**明确**说自己是 MoE。

---

### Q25. 选 MoE 还是 Dense？给出决策框架。

**要点**：

按约束：
- 算力受限 + 想 pretraining 大模型 → MoE；
- 推理成本敏感（API 服务）→ MoE；
- 显存有限 + 私有部署 → Dense；
- 小数据集 finetune → Dense；
- 边缘部署 → Dense + 量化；
- 长上下文 → MoE+MLA 或 Dense+GQA。

---

### Q26. MoE 做 LoRA 微调有什么坑？

**要点**：
- 路由器 LoRA 不稳定 → 通常 freeze 路由器；
- 每个专家加 LoRA → 参数增多但 finetune 数据少容易过拟合；
- 建议 LoRA 只加共享专家或部分专家；
- 学术上 MoLE / MoLoRA / LoRA-MoE 仍在探索。

---

### Q27. 怎么对 MoE 做量化部署？

**要点**：
- 主流：4bit (bnb / GPTQ / AWQ)；
- 路由器保留 bf16/fp32（量化它会破坏路由）；
- 共享专家保留高精度，稀疏专家可激进量化；
- llama.cpp 的 GGUF 支持 Mixtral / DBRX；
- 实测 Mixtral-4bit 显存 ~24 GB，质量损失 1–2 perplexity。

---

### Q28. Expert Choice 与 Token Choice 的根本差别？

**要点**：
- Token Choice：每 token 选 K 专家 → 专家可能超载；
- Expert Choice：每专家选 C 个 token → token 可能 0 或多次被选；
- Expert Choice 天然均衡，但因果（causal）解码不友好；
- 主要用于训练，推理少。

---

### Q29. MoE 与稀疏化（Pruning）的关系？

**要点**：
- 都是"减少激活参数"的思路；
- Pruning 是**事后删除**部分权重；
- MoE 是**结构上**让权重按路由稀疏激活；
- MoE 是"动态稀疏"（每 token 不同），Pruning 是"静态稀疏"（所有 token 相同）；
- 二者可结合：先 MoE pretraining，再剪不常用专家。

---

### Q30. 你怎么看 MoE 的未来？

**要点**（可发挥）：
- 大模型默认架构（≥30B 量级几乎都会 MoE）；
- 细粒度化继续：百乃至千个专家；
- 与 SSM (Mamba) 结合（Jamba、Hybrid）；
- 推理基础设施成熟（vLLM/SGLang 完善）；
- 持续学习中"共享 + 稀疏"分离的天然适配；
- MoD (深度方向稀疏) 是新方向；
- 边缘部署仍是 MoE 的痛点。

---

## 答题策略 tips

1. **先讲直觉，再上公式**：面试官想看你"理解"，不只"背诵"；
2. **公式不要写太大**：写出关键变量与含义即可，例如 $\sum f_i \cdot P_i$ 配上"频次乘概率"的解释；
3. **代码要写"对的伪代码"**：handle topk, capacity, dispatch 三段即可；
4. **答完一个点，主动延伸**：例如答完 aux loss 后说"和 z-loss 配合用更稳"，体现你"知识网"完整；
5. **承认边界**：被追问到不确定时直说"这部分我没深入做过，根据论文我理解是 X，但实际效果可能要看 ablation"，比硬扛强。

→ [附录 B 参考文献](appendix-references.md)
