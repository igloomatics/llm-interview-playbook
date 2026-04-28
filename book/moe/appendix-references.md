# 附录 B 参考文献与延伸阅读

## B.1 原始论文（按时间倒序）

### 2024–2025

- **DeepSeek-V3 Technical Report** — DeepSeek-AI, 2024.
  arXiv:2412.19437
  *671B/37B 激活的开源 MoE 旗舰，引入 Aux-Loss-Free、节点限制路由、fp8 训练。*

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** — DeepSeek-AI, 2024.
  arXiv:2405.04434
  *236B/21B 激活，引入 MLA。*

- **DeepSeekMoE: Towards Ultimate Expert Specialization** — DeepSeek-AI, 2024.
  arXiv:2401.06066
  *细粒度 + 共享专家的开山之作。*

- **Mixtral of Experts** — Jiang et al. (Mistral AI), 2024.
  arXiv:2401.04088
  *8x7B 开源 MoE 里程碑。*

- **OLMoE: Open Mixture-of-Experts Language Models** — Muennighoff et al. (Allen AI), 2024.
  arXiv:2409.02060
  *完全开源（含训练数据）的 MoE。*

- **Mixture-of-Depths** — Raposo et al. (DeepMind), 2024.
  arXiv:2404.02258
  *深度方向的稀疏化，MoE 的"垂直"对应物。*

### 2022–2023

- **ST-MoE: Designing Stable and Transferable Sparse Expert Models** — Zoph et al. (Google), 2022.
  arXiv:2202.08906
  *Z-Loss、稳定性、finetune 的系统研究。*

- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts** — Gale et al., 2022.
  arXiv:2211.15841
  *block-sparse GEMM kernel，MoE 训练加速关键。*

- **Mixture-of-Experts with Expert Choice Routing** — Zhou et al. (Google), 2022.
  arXiv:2202.09368
  *专家选 token 的反向路由。*

- **From Sparse to Soft Mixtures of Experts** — Puigcerver et al., 2023.
  arXiv:2308.00951
  *Soft MoE，ViT 上效果好。*

- **Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints** — Komatsuzaki et al., 2022.
  arXiv:2212.05055
  *dense → MoE 的 up-cycling 方法。*

### 2020–2021

- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** — Fedus, Zoph & Shazeer (Google), 2021.
  arXiv:2101.03961
  *K=1 路由，1.6T 参数。*

- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** — Lepikhin et al. (Google), 2020.
  arXiv:2006.16668
  *MoE-Transformer 工程化奠基。*

- **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts** — Du et al. (Google), 2021.
  arXiv:2112.06905
  *1.2T MoE，能耗仅为 GPT-3 的 1/3。*

### 2017 与更早

- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** — Shazeer et al. (Google Brain), 2017.
  arXiv:1701.06538
  *Top-K Gating + Noisy Gating，MoE 复兴的开山之作。*

- **Adaptive Mixtures of Local Experts** — Jacobs, Jordan, Nowlan, Hinton, 1991.
  Neural Computation 3(1):79–87.
  *MoE 的祖师爷论文。*

- **Hierarchical Mixtures of Experts and the EM Algorithm** — Jordan & Jacobs, 1994.
  Neural Computation 6(2):181–214.

---

## B.2 综述与教程

- **A Survey on Mixture of Experts in Large Language Models** — Cai et al., 2024.
  arXiv:2407.06204
  *系统梳理 LLM 中的 MoE 进展，推荐入门后通读。*

- **HuggingFace Blog: Mixture of Experts Explained** — Sanseviero et al., 2023.
  https://huggingface.co/blog/moe
  *英文社区入门最广为流传的解析。*

- **The Annotated DeepSeek-V3** — DeepSeek 社区，2024–2025.
  *逐节解读 V3 技术报告的中英文资料。*

- **Awesome Mixture of Experts** — GitHub 社区列表（多个仓库以此名）。
  *综合资源汇总。*

---

## B.3 工程项目与代码库

- **Mixtral 官方实现** — Mistral AI / HuggingFace Transformers
  *从这里开始读 MoE 源码最容易。*

- **DeepSeek-V3 官方仓库** — github.com/deepseek-ai
  *推理参考实现，工程细节丰富。*

- **MegaBlocks** — github.com/databricks/megablocks
  *block-sparse GEMM 实现。*

- **DeepSpeed-MoE** — github.com/microsoft/DeepSpeed
  *Microsoft 的分布式 MoE 训练框架。*

- **Megatron-Core MoE** — github.com/NVIDIA/Megatron-LM
  *NVIDIA 的官方 MoE 训练实现。*

- **vLLM** — github.com/vllm-project/vllm
  *Mixtral / DeepSeek MoE 推理首选框架。*

- **SGLang** — github.com/sgl-project/sglang
  *DeepSeek 推理参考框架。*

- **Mixtral-Offloading** — github.com/dvmazur/mixtral-offloading
  *把 Mixtral 装进 16GB GPU 的工程例子。*

- **fastmoe** — github.com/laekov/fastmoe
  *早期开源 MoE 训练框架，pyTorch + CUDA。*

---

## B.4 中文资料

- **DeepSeek-V3 技术报告（中文翻译）** — 多家自媒体翻译版本。
- **知乎"MoE 专题"** — 多位作者撰写的入门到进阶系列。
  *本书的最初参考线索来自该社区，特别是关于稀疏路由直觉的讲解。*
- **知乎专栏：大模型架构系列** — 包含 MoE / MLA / GQA 等专题。
- **机器之心、量子位** 等媒体的 MoE 报道。

> 注：本书写作过程中曾尝试访问 Zhihu 文章 `p/81886457827`（用户提供的参考链接），但因访问受限未能成功抓取。
> 写作主要回到原始论文（见 B.1）与公开技术报告。希望未来能补充更多优质中文一手资料。

---

## B.5 推荐阅读路径

### 第一周：入门

1. HuggingFace Blog: Mixture of Experts Explained；
2. Mixtral 论文；
3. 本书 [第 0–3 章](README.md)。

### 第二周：原理深化

1. Switch Transformer 论文；
2. GShard 论文；
3. ST-MoE 论文（Z-Loss）；
4. 本书 [第 4–6 章](04-routing.md)。

### 第三周：开源前沿

1. DeepSeekMoE 论文；
2. DeepSeek-V2 论文（重点看 MLA + Aux-Loss-Free）；
3. DeepSeek-V3 技术报告；
4. 本书 [第 7 章](07-classic-models.md)。

### 第四周：工程与部署

1. MegaBlocks 论文；
2. vLLM / SGLang MoE kernel 源码；
3. Mixtral-Offloading 项目阅读；
4. 本书 [第 8–9 章](08-implementation.md)。

### 第五周：决策与前沿

1. Mixture of Depths；
2. Cai et al. 综述；
3. 本书 [第 10–11 章](10-moe-vs-dense.md)。

---

## B.6 致谢

本书写作受益于：

- 1991–2025 年所有 MoE 相关论文的作者；
- 中英文社区无数撰写解析、源码注释的志愿者；
- 开源 MoE 模型的发布者（特别是 Mistral、DeepSeek、Qwen、Allen AI 等团队）；
- LLM Interview Playbook 项目的所有用户与提问者。

---

## 全书完

回到 [《MoE 混合专家模型完全指南》目录](README.md)。

> 如果这本书帮你看懂了一篇 MoE 论文，或在面试中答出了一个原本会卡壳的问题——那它就值得被写出来。
> 任何错误、补充与建议，欢迎通过 issue 或 PR 反馈。
