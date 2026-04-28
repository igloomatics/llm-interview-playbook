# 《MoE 混合专家模型完全指南》

> 一本面向 LLM 学习者与面试备考者的 Mixture of Experts 系统化手册。
> 从 1991 年的经典 MoE 起步，到 GShard、Switch Transformer、Mixtral、DeepSeek-V3 的稀疏化革命，
> 用一本书的篇幅，把 MoE 从原理、数学、训练、推理到工程一次讲透。

---

## 写作目的

近两年最重要的两个大模型架构事件：

1. **稠密模型在 70B–400B 量级遇到了显存与计算的天花板**；
2. **DeepSeek-V3、Mixtral、Qwen-MoE、GPT-4 等顶尖模型不约而同地走向了稀疏 MoE**。

理解 MoE，不再是一个"加分项"，而是 LLM 架构能力的**基础项**。但中文世界关于 MoE 的资料有两个问题：

- 论文导读型多，缺乏体系；
- 入门博客多，缺乏面试与实战所需的"细节穿透"。

本书希望补齐这一空白：**用一本书的篇幅，把 MoE 从 1991 年讲到 2026 年**。

---

## 适合谁读

- 准备 **LLM / 大模型方向** 面试，希望系统拿下 MoE 章节的同学；
- 在 **训练 / 推理 / 系统优化** 岗位上，需要理解稀疏专家通信、负载均衡的工程师；
- 对 **DeepSeek-MoE、Mixtral、GPT-4** 内部机制感兴趣的研究者；
- 已经看过若干 MoE 博客，希望把"零散知识点"串成"完整心智模型"的进阶学习者。

---

## 目录

### 第一部分：基础

- [第 0 章 序言：你为什么需要懂 MoE](00-preface.md)
- [第 1 章 MoE 的前世今生：从集成学习到稀疏大模型](01-history.md)
- [第 2 章 MoE 的数学基础：从概率视角看专家混合](02-math-foundations.md)

### 第二部分：架构与机制

- [第 3 章 稀疏 MoE 架构：把 FFN 替换成专家组](03-sparse-architecture.md)
- [第 4 章 路由机制全解：Top-K、Switch、Expert Choice、Soft MoE](04-routing.md)
- [第 5 章 负载均衡：Auxiliary Loss、Z-Loss 与无辅助损失方案](05-load-balancing.md)

### 第三部分：训练与系统

- [第 6 章 训练技巧与稳定性：从专家并行到通信优化](06-training.md)
- [第 7 章 经典 MoE 模型解析：GShard / Switch / GLaM / Mixtral / DeepSeek-MoE](07-classic-models.md)
- [第 8 章 MoE 的工程实现：从一个 PyTorch demo 到 Megablocks](08-implementation.md)

### 第四部分：推理与未来

- [第 9 章 MoE 推理优化：Expert Offloading、缓存与投机解码](09-inference-optimization.md)
- [第 10 章 MoE vs Dense：什么时候用，什么时候别用](10-moe-vs-dense.md)
- [第 11 章 前沿与未来：MoD、MoA、上下文专家、共享专家](11-frontier.md)

### 附录

- [附录 A 面试题精选（30 题）](appendix-interview.md)
- [附录 B 参考文献与延伸阅读](appendix-references.md)

---

## 阅读建议

| 你是谁 | 推荐路径 |
|--------|----------|
| 第一次接触 MoE | 0 → 1 → 3 → 7（先看故事，后看模型） |
| 准备面试 | 2 → 4 → 5 → 附录 A |
| 训练 / 系统工程师 | 5 → 6 → 8 → 9 |
| 研究者 / 架构师 | 全本，重点 4、5、7、11 |

---

## 关于本书

- 风格：以**问题驱动**写作，每章先抛"为什么需要它"，再讲机制，最后给"陷阱与坑"；
- 数学：保留核心公式，不堆砌；公式后必给直觉解释；
- 代码：以 PyTorch 伪代码为主，关键函数（gating、auxiliary loss、capacity）给出可运行片段；
- 引用：所有关键结论均回溯到原始论文，详见 [附录 B](appendix-references.md)。

> 如果你只能读一本中文 MoE 资料，我希望它是这本。
