# 模型训练与重标注

本文档介绍模型训练、重标注流程以及相关脚本使用。

---

## 数据生成

### 生成多任务数据

```bash
python data/new_v2.py --input_dir data/dataset --output_dir data/dataset
```

输出文件：
- `train_v2.json` / `dev_v2.json` / `test_v2.json`
- `multitask_v2.json`
- `dataset_info_v2.json`

---

## 重标注流程（7步 CoT）

使用 7 步 Chain-of-Thought 推理对数据集进行全量 AI 重标注。

### 运行重标注脚本

```bash
python data/relabel_v2_with_model.py \
  --input_dir data/dataset \
  --output_dir data/dataset \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --infer_backend vllm \
  --max_model_len 8192 \
  --max_batch_size 8 \
  --batch_size 8 \
  --max_tokens 1024 \
  --temperature 0.2 \
  --top_p 0.9 \
  --retries 1
```

**显存紧张时**：降低 `--max_batch_size` 和 `--batch_size`（如 4/2）

**调试模式**（流式输出）：
```bash
--stream
```

### 输出文件

重标注后生成：
- `train_v2_relabel.json`
- `dev_v2_relabel.json`
- `test_v2_relabel.json`
- `dataset_info_v2_relabel.json`
- `relabel_failed.jsonl`（失败样本）

每个 `*_v2_relabel.json` 包含 4 类任务样本：
- `task_emotion_reg` - 情感回归
- `task_primary_cls` - 主情感分类
- `task_mbti_pred` - MBTI 预测
- `task_cot_gen` - CoT 生成

---

## 训练流程

### 方式一：LLaMA-Factory（推荐）

确保 `dataset_info_v2_relabel.json` 包含：
- `cmacd_multitask_train_v2`
- `cmacd_multitask_dev_v2`

```bash
llamafactory-cli train train_lora_v2.yaml
```

**AutoDL 配置调整**：
```yaml
model_name_or_path: /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct
dataset_dir: /root/autodl-tmp/data/dataset
```

### 方式二：ms-swift

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct \
  --dataset /root/autodl-tmp/data/dataset/train_v2_relabel.json \
  --val_dataset /root/autodl-tmp/data/dataset/dev_v2_relabel.json \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 4096 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --report_to swanlab \
  --swanlab_project cmacd-lora-v2 \
  --swanlab_exp_name cmacd-v2-relabel-run1 \
  --output_dir /root/autodl-tmp/output/cmacd_lora_v2_relabel
```

---

## 模型合并与量化

### 1. 合并 LoRA

将训练好的 LoRA adapter 合并到基座模型。

### 2. 导出 GGUF

1. 先导出 F16 GGUF
2. 再量化到 Q4_K_M

---

## 建议

1. **先小规模测试**：先用 `dev/test` 数据集试跑，确认输出格式稳定后再跑全量 `train`
2. **监控失败率**：若失败率较高，降低 `--batch_size` 并适当提高 `--max_tokens`
3. **检查中间结果**：定期检查 `relabel_failed.jsonl`，分析失败原因
