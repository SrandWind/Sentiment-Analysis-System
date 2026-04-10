# 情感分析系统

基于 Qwen2.5-7B-Instruct 的中文多模态情感分析系统，支持 CoT 推理、VAD 评分、MBTI 推断。

## 快速开始

### 1. 启动 LMStudio

1. 下载 [LMStudio](https://lmstudio.ai/)
2. 下载模型 `Qwen2.5-7B-Instruct` (GGUF Q4_K_M)
3. 启动 Local Server，确保地址为 `http://localhost:1234/v1`

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000

---

## 项目文档

| 文档 | 说明 |
|------|------|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 完整部署指南（本地/AutoDL/数据库配置） |
| [docs/TRAINING.md](docs/TRAINING.md) | 模型训练与重标注流程 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发指南（数据格式、评估脚本、API） |

---

## 系统特性

- **CoT 推理**：7 步 Chain-of-Thought 推理链
- **VAD 评分**：情感效价、唤醒度、支配度
- **MBTI 推断**：基于文本推断 MBTI 人格类型
- **红线规则**：防幻觉，确保无证据情绪分数为 0
- **多模式推理**：quick / standard / deep 三种模式

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI, SQLAlchemy, Uvicorn |
| 前端 | React 18, Ant Design 5, ECharts 5, Vite 5 |
| 推理 | LMStudio (本地) / vLLM (云端) |
| 模型 | Qwen2.5-7B-Instruct + LoRA 微调 |

---

## 命令参考

| 操作 | 命令 |
|------|------|
| 启动后端 | `cd backend && python main.py` |
| 启动前端 | `cd frontend && npm run dev` |
| 运行评估 | `python eval_v2.py --gold data/dataset/test_v2.json --pred predictions.jsonl --out outputs/metrics.json` |
| AutoDL 部署 | `./deploy_autodl.sh` |

---

## API 文档

启动后端后访问：http://localhost:8000/docs

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/infer` | POST | 单句推理 |
| `/api/batch` | POST | 批量推理 |
| `/api/history` | GET | 历史记录 |
| `/api/metrics/{variant}` | GET | 模型指标 |
| `/health` | GET | 健康检查 |
