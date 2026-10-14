<div align="center">

# 🧠 LLM From Scratch

### **不只理解数学，更要亲手打通系统。**

**从 Tokenizer 到现代 Transformer，再到完整训练与研究实验的一条工程化学习路线。**  
受 Stanford **CS336: Language Modeling from Scratch** 启发，以模块化架构重新组织学习成果，目标是逐步搭建可理解、可验证、可扩展的语言模型代码库。

<p>
  <img alt="开发状态" src="https://img.shields.io/badge/status-active%20development-F59E0B?style=for-the-badge" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-based-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img alt="CS336" src="https://img.shields.io/badge/Inspired%20by-Stanford%20CS336-8C1515?style=for-the-badge" />
</p>

**[✨ 项目亮点](#highlights)** · **[✅ 当前进展](#status)** · **[🗺️ 工程架构](#architecture)** · **[🚀 快速开始](#quick-start)** · **[📍 开发路线](#roadmap)** · **[🇬🇧 English](README.md)**

</div>

---

> **🚧 持续开发中**

<a id="highlights"></a>

## 🌟 为什么要做这个仓库？

- 🧩 **从数学组件走向系统闭环**
- 🧠 **先理解，再优化**
- 🔬 **面向科研实验设计**

**适合谁？** 从零构建语言模型的学生、PyTorch 开发者与研究者。

<a id="status"></a>

## ✅ 目前已经完成了什么？

| 架构模块 | ✅ 已实现 | 🛠️ 下一步 | 🔭 有待扩展 |
|:--|:--|:--|:--|
| 📝 **Tokenizer** | Character / UTF-8 Byte Tokenizer；Python Byte-level BPE；Rust 后端 BPE 训练与编码；词表、合并规则、特殊 token 的精确保存加载；参数指纹 | 扩充分词后端一致性与语料覆盖 | 其他分词算法 |
| 📦 **DataPipeline** | TXT / JSONL / Parquet 读取；文档内 S+1 窗口采样；TextBatch 输入与目标对齐；短文档右侧 Padding 与监督 loss_mask | 通用 Packing 与独立可见性契约 | 数据分片、偏好与多模态批次 |
| 🧠 **ModernLM** | Linear、Embedding、RMSNorm、SiLU、RoPE；Reference 因果 MHA、SwiGLU、Pre-RMSNorm Decoder、TransformerModel / CausalLM；外部初始化、稳定逐 token CE、结构化输出、激活重计算；Temperature / Top-k / Greedy / EOS 生成 | KV Cache、Prefill / Decode、GQA / MQA | 可配置组件变体、混合模型、MoE、多模态组合 |
| 🎯 **Tasks** | Next-token 预训练；mean / sum 目标；监督筛选；累积窗口 token 计数；分离梯度的 CE 指标分子与分母 | SFT 数据与任务接线、辅助目标组合 | 偏好优化与在线 RL 目标 |
| ⚙️ **TrainingSystems / runner** | BaseRunner / LMRunner；共享配置驱动 optimizer / scheduler；梯度累积、尾窗、裁剪、fit / eval；模型、优化器、调度器、计数与 RNG 状态保存恢复 | 混合精度与运行优化 | 分布式执行、多模型角色更新 |
| 🚀 **scripts** | Python BPE 训练与导出；run_pretrained.py 自动训练或恢复 Rust BPE、按文档添加 EOS 编码、使用临时磁盘映射 CPU token 流，并记录训练、固定验证、生成与 Checkpoint | 将预训练回归验证同步到当前数据和配置接口 | run_sft / run_dpo 等任务入口 |

<a id="architecture"></a>

## 🗺️ 工程架构

![当前预训练逻辑管线：实验配置、BPE 与文档准备、模型和任务计算、优化更新、验证、生成及训练状态恢复](assets/training-pipeline.svg)

### 📁 工程目录

```text
LLM-From-Scratch/
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── assets/
│   ├── training-pipeline.svg
│   └── training-pipeline.png      # 同图的位图预览
├── DataPipeline/
│   ├── batch.py                  # TextBatch
│   ├── sampling.py               # 连续流 / 文档内 S+1 窗口
│   └── text.py                   # TXT / JSONL / Parquet 读取
├── ModernLM/
│   ├── layers/                   # Linear / Embedding / RMSNorm / SiLU
│   ├── position/rope.py
│   ├── attention/
│   │   ├── backends/reference.py # Reference SDPA
│   │   └── self_attention.py     # 因果 MHA 与 RoPE
│   ├── feedforward/dense.py      # SwiGLU
│   ├── blocks/decoder.py         # Pre-RMSNorm Decoder
│   ├── objectives/ce.py          # 逐 token Cross-Entropy
│   ├── generation/sampling.py    # 完整上下文生成，尚无 KV Cache
│   ├── config.py                 # ModelConfig / RuntimeConfig / TrainConfig
│   ├── initialization.py
│   ├── model.py                  # TransformerModel / CausalLM
│   └── outputs.py
├── Tasks/
│   └── pretrained.py             # 目标函数与窗口分母
├── Tokenizer/
│   ├── base.py
│   ├── bpe_utils.py
│   ├── common.py                 # 参数保存 / 加载 / 指纹
│   └── tokenizer.py              # Character / Byte / Python BPE / Rust BPE
├── TrainingSystems/
│   └── runner/                   # TrainingSystems 本阶段仅上传此软件包
│       ├── base.py
│       ├── lm_runner.py
│       └── checkpoint.py
└── scripts/
    ├── common.py                 # ExperimentConfig 与文档编码
    ├── run_bpe_train.py
    ├── run_pretrained.py
    ├── test_pretrained.py
    ├── test_runner.py
    └── test_tokenizer.py
```

<a id="quick-start"></a>

## 🚀 快速开始

### 1. 安装依赖

```powershell
conda create -n llm-scratch python=3.12 -y
conda activate llm-scratch
python -m pip install -r requirements.txt
```

### 2. 样例配置

```python
if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "quickstart"
    output_dir = PROJECT_ROOT / "outputs" / "quickstart"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_text = "".join(
        f"= Training article {i} =\n"
        "The fox reads a small book. The book is blue. "
        "A language model predicts the next token.\n"
        for i in range(24)
    )
    val_text = "".join(
        f"= Validation article {i} =\n"
        "A blue book is on the table. A small fox reads a book "
        "and predicts the next token.\n"
        for i in range(8)
    )
    (data_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (data_dir / "val.txt").write_text(val_text, encoding="utf-8")

    experiment = ExperimentConfig(
        model_cfg=ModelConfig(
            vocab_size=512, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_kv_heads=4,
            head_dim=16, value_head_dim=16,
            intermediate_size=128, max_seq_len=64,
        ),
        train_cfg=TrainConfig(
            optimizer={"name": "adamw", "params": {"lr": 1e-3}},
            batch_size=2, gradient_accumulation_steps=2,
        ),
        runtime_cfg=RuntimeConfig(
            device="cpu", dtype=torch.float32, grad_max_norm=1.0,
        ),
        tokenizer_path=output_dir / "tokenizer" / "bpe_params.json",
        train_paths=[data_dir / "train.txt"],
        val_paths=[data_dir / "val.txt"],
        output_dir=output_dir,
        max_train_tokens=None, max_val_tokens=None,
        context_length=64, num_updates=20,
        eval_interval=5, eval_batches=2, generate_interval=10,
        generate_prompts=("A small fox",), max_new_tokens=16,
    )
    run_pretrained(experiment)
```

### 3. 运行 run_pretrained.py

```powershell
python scripts/run_pretrained.py
```

### 4. 恢复训练

```python
experiment = replace(
    experiment,
    resume_from=Path(experiment.output_dir) / "last.pt",
    num_updates=2,
)
run_pretrained(experiment)
```

<a id="roadmap"></a>

## 📍 Roadmap：从基础组件到科研工具箱

| 阶段 | 目标 | 进展 |
|:--|:--|:--|
| **v0.1 · 核心基础** | Tokenizer、基础层、RoPE、Reference MHA、SwiGLU、Decoder、CausalLM、初始化与稳定 CE | ✅ 核心组件已实现 |
| **v0.2 · Dense LM 训练闭环** | 文档级数据、训练 / 验证、可恢复 Checkpoint、基础生成与真实语料实验 | ✅ 闭环已打通 |
| **v0.3 · 推理机制** | KV Cache、Prefill / Decode、GQA / MQA、缓存生成一致性 | 🧭 规划中；Temperature / Top-k / Greedy / EOS 采样已实现 |
| **v0.4 · 架构消融** | 显式选择 Norm / FFN / Attention / Block，开展可控对照实验 | 🧭 规划中；当前构造仅选择一种 Dense 架构 |
| **v0.5 · 稀疏与混合模型** | 单设备 MoE 路由、专家与辅助损失，再推进混合序列实验 | 🧭 规划中 |
| **v0.6 · 运行扩展** | AMP 与性能优化，再按需要接入分布式执行 | 🧭 规划中；累积、裁剪与激活重计算已实现 |
| **v0.7+ · 科研工具箱** | SFT / 偏好优化 → 多模态建模 → 在线 RL；可复用实验配置与评测流程 | 🔭 长期方向 |

## 🤝 一起学习、一起改进

欢迎通过 GitHub Issue 或 Pull Request 交流问题、改进代码与文档。

## 📚 致谢与声明

- **Stanford CS336:** [课程官网](https://cs336.stanford.edu/) · [官方 GitHub](https://github.com/stanford-cs336)
- **PyTorch:** [官方文档](https://pytorch.org/docs/stable/index.html)
- **einops:** [官方文档](https://einops.rocks/)
- **Hugging Face Tokenizers:** [官方文档](https://huggingface.co/docs/tokenizers/index)
- **WikiText:** [数据集与说明](https://huggingface.co/datasets/Salesforce/wikitext)

本项目是**独立开展的学习与工程实践**，在知识内容和学习路线方面受到 CS336 启发。

---

<div align="center">

**从第一个 Byte 到下一个 Token。** 🚀

*保持好奇 · 深入原理 · 尊重工程 · 用实验说话*

</div>
