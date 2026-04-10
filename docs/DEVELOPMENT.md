# 开发指南

本文档涵盖数据格式、评估脚本使用、前端数据对接等内容。

---

## 项目结构

```
Sentiment-Analysis-System/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置（含推理预设）
│   ├── api/
│   │   ├── routes.py        # API 路由
│   │   └── schemas.py        # 数据模型
│   ├── database/
│   │   ├── db.py            # 数据库连接
│   │   ├── models.py        # SQLAlchemy 模型
│   │   └── init_mysql.sql    # MySQL 初始化
│   └── services/
│       ├── lmstudio_client.py # 推理客户端
│       ├── parser.py         # 结果解析
│       └── evaluator.py      # 指标计算
│
├── frontend/
│   └── src/
│       ├── pages/           # 6 个页面
│       └── services/api.ts  # API 调用
│
├── outputs/                  # 评估输出
│   ├── metrics_*.json       # 评估指标
│   └── compare_metrics.*    # 对比报告
│
├── eval_v2.py                # 评估脚本
└── compare_to_swanlab.py    # 模型对比脚本
```

---

## 评估脚本使用

### 生成推理结果

```bash
python infer_async_for_eval.py
```

### 运行评估

```bash
# 评估基座模型
python eval_v2.py \
  --gold data/dataset/test_v2_relabel.json \
  --pred outputs/predictions_base.jsonl \
  --out outputs/metrics_base.json \
  --model_variant base

# 评估 LoRA 微调模型
python eval_v2.py \
  --gold data/dataset/test_v2_relabel.json \
  --pred outputs/predictions_lora.jsonl \
  --out outputs/metrics_lora_merged.json \
  --model_variant lora_merged

# 评估 GGUF 量化模型
python eval_v2.py \
  --gold data/dataset/test_v2_relabel.json \
  --pred outputs/predictions_gguf.jsonl \
  --out outputs/metrics_gguf4bit.json \
  --model_variant gguf4bit
```

### 生成对比报告

```bash
python compare_to_swanlab.py \
  --items base=./outputs/metrics_base.json \
          lora_merged=./outputs/metrics_lora_merged.json \
          gguf4bit=./outputs/metrics_gguf4bit.json \
  --out_csv ./outputs/compare_metrics.csv \
  --out_md ./outputs/compare_metrics.md \
  --out_dir ./outputs/charts \
  --swanlab_project cmacd-eval \
  --swanlab_run_name compare_3models
```

### 复制到前端目录

```bash
cp outputs/metrics_*.json frontend/public/output/
cp outputs/compare_metrics.json frontend/public/output/
```

---

## 前端数据文件

### 文件清单

| 文件 | 来源 | 用途 |
|------|------|------|
| `metrics.json` | `eval_v2.py` | 单模型评估指标 |
| `compare_metrics.json` | `compare_to_swanlab.py` | 多模型对比 |
| `training_history.json` | SwanLab 导出 | 训练历史曲线 |

### 导出训练历史

```bash
python export_training_history.py \
  --project cmacd-lora-v2 \
  --experiment cmacd-v2-relabel-run1 \
  --output frontend/public/output/training_history.json
```

---

## metrics.json 字段说明

| 字段 | 说明 |
|------|------|
| `primary_cls_accuracy` | 主情感分类准确率 |
| `primary_cls_macro_f1` | 宏平均 F1 |
| `primary_cls_macro_auc` | 宏平均 AUC-ROC |
| `primary_cls_macro_ap` | 宏平均 Average Precision |
| `primary_cls_per_class_metrics` | 每类 precision/recall/f1/support |
| `primary_cls_pr_curves` | PR 曲线数据 |
| `primary_cls_roc_curves` | ROC 曲线数据 |
| `primary_cls_confusion_matrix` | 混淆矩阵 |
| `emotion_macro_mae/mse` | 情感强度 MAE/MSE |
| `emotion_per_dim_mae/mse` | 每维情感 MAE/MSE |
| `mbti_accuracy/f1` | MBTI 预测指标 |
| `json_parse_rate` | JSON 解析成功率 |
| `cot7_complete_rate` | CoT 完整率 |

---

## 推理配置（config.py）

### InferencePresets

| 预设 | Temperature | 适用场景 |
|------|-------------|---------|
| `quick` | 0.03 | 快速测试 |
| `standard` | 0.05 | 标准推理 |
| `deep` | 0.07 | 深度推理 |

---

## CoT 推理特性

### 7 步推理链

1. 提取显式情感信号
2. 识别隐式情感信号
3. 确定情感强度
4. 识别情感原因
5. 主情感判断
6. MBTI 推断
7. 最终总结

### 红线规则（防幻觉）

- 无证据的情绪分数必须 ≤ 0.03
- happy=0.00 当且仅当没有 happy 证据
- 模型禁止在 check_items 中撒谎

### 工程层验证

- 后端对模型输出进行验证
- 违反红线规则的分数会被自动修正
- 支持自动重试机制

---

## 数据格式

### 推理请求

```typescript
interface InferRequest {
  text: string;
  inference_mode: 'quick' | 'standard' | 'deep';
  enable_cot: boolean;
  custom_params?: {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
  };
}
```

### 推理响应

```typescript
interface InferResponse {
  success: boolean;
  data: {
    text: string;
    raw_intensity_scores: {
      happy: number;
      sad: number;
      angry: number;
      fear: number;
      surprise: number;
      neutral: number;
    };
    target_scores: {
      /* 归一化后的分数 */
    };
    primary_emotion: string;
    mbti_type: string;
    vad_scores: {
      valence: number;
      arousal: number;
      dominance: number;
    };
    uncertainty: number;
    cot_reasoning: string;
    evidence: Record<string, string[]>;
  };
  latency_ms: number;
  model_variant: string;
}
```

---

## 常见问题

### 前端显示"使用默认数据"

检查 JSON 文件是否在 `frontend/public/output/` 目录下

### PR/ROC 曲线不显示

重新运行 `eval_v2.py` 生成新字段

### SwanLab 导出失败

检查 API key 配置或手动从 SwanLab 网页导出

---

## 相关文档

- [部署指南](DEPLOYMENT.md) - 本地/AutoDL 部署、数据库配置
- [训练指南](TRAINING.md) - 模型训练与重标注
