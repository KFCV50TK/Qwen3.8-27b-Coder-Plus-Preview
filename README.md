---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- qwen
- code
- lora
- preview
---

# Qwen3.8-27b-Coder-Plus-Preview

基于 **Qwen/Qwen3.8-27B** 的社区代码微调预览。当前发布的是本轮原始模型训练的 **第 300 个优化器步**，提供 BF16 合并权重和独立 LoRA adapter。完整训练计划为 1,742 步，发布时训练仍在继续。此项目与 Qwen 官方无隶属关系。

This community preview merges checkpoint 300 of a fresh code LoRA into Qwen3.8-27B. It includes full BF16 safetensors weights and the standalone adapter. The ongoing training run is not finished. This is not an official Qwen release.

## 下载完整模型

```bash
git clone https://github.com/KFCV50TK/Qwen3.8-27b-Coder-Plus-Preview.git
cd Qwen3.8-27b-Coder-Plus-Preview
python download_model.py --output ./model
```

权重约 55.6 GB，放在 GitHub Release 的 `checkpoint-300-preview` 中。为满足单文件限制，大于 1.9 GB 的原始 shard 拆为 `.part001` 等文件，脚本会自动重组，并验证每个分块与最终文件的 SHA256。下载目录建议预留至少 65 GB。不要直接把分块当 safetensors 加载。

仅下载 adapter（约 467 MB）：

```bash
python download_model.py --adapter-only --output ./adapter-download
```

Adapter 位于 `adapter-download/adapter/`，需要原始 Qwen3.8-27B。合并模型则可直接加载，不需要再叠加 adapter。原始快照的 adapter SHA256 为 `72401094408c7ffa326e0ef20c94265e86f54931009c2c753bedee8b7f3d2459`。

## 加载示例

使用支持 Qwen3.8 对应 `qwen3_5` 架构的 Transformers 版本。此次环境记录为 Transformers 5.8.0.dev0、PEFT 0.20.0。具体后端需要与部署设备匹配。

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

path = './model'
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(
    path, dtype=torch.bfloat16, device_map='auto'
)
messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Write a Python binary search function.'}]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors='pt', enable_thinking=True
).to(model.device)
output = model.generate(**inputs, max_new_tokens=2048)
print(processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```

本示例未在此预览权重上进行推理验证。视觉权重及处理器文件保留自原模型；本轮训练数据为文本代码任务，没有进行视觉能力评测。

## 训练记录

| 项目 | 值 |
|---|---|
| 起点 | 原始 Qwen3.8-27B，新建 LoRA；启动时 496 个 B 矩阵为零 |
| 发布检查点 | 300 / 1742 优化器步 |
| 数据 | 12000 条公开 CodeFeedback 代码样本 + 645 条有效蒸馏样本 |
| 蒸馏样本 | 38 + 200 + 407，全部保留，各呈现 3 次 |
| 有效呈现 | 13935 行，1 个混合数据 epoch |
| 思考格式 | 621 条非空摘要与代码合并为一条 assistant；按样本设 thinking 开关；屏蔽空 think 标签损失 |
| LoRA | rank 16、alpha 32、dropout 0.05，496 对 A/B 张量 |
| 优化 | LR 5e-5、cosine、warmup 0.03、batch 1 × accumulation 8 |
| 精度及长度 | BF16，max length 4608；全量编码最长 4457，无长度丢题 |

新蒸馏快照原有 408 行，其中 1 行超时且无答案，已记录排除。本仓库不重新分发训练题目、蒸馏答案或验证集。

## 校验与局限

合并在 CPU 上按原始 18 个 shard 执行；覆盖全部 496 个 LoRA 权重更新，检查源/目标张量有限性，3 个更新与 PEFT `merge_and_unload(safe_merge=True)` 精确一致。`merge-manifest.json` 和 `release-manifest.json` 记录来源、权重更新和 SHA256。

**本预览尚未进行能力对照测试。** loss 下降、权重有限、合并成功都不等同于代码能力改善。100 题独立验证集与盲测工具仅已准备，没有对第 300 步启动推理。也未证明思考、通用、视觉或长上下文能力得以保留。请将此版本视作实验预览。

## 许可与来源

沿用原模型随附的 **Apache License 2.0**，见 `LICENSE`、`NOTICE` 和 `MODIFICATIONS.md`。原始模型说明保留在 `UPSTREAM_README.md`，其中官方 benchmark 不属于本微调预览的测试结果。训练记录见 `training_config.json`，合并代码见 `scripts/`。
