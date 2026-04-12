# 部署指南

本文档涵盖本地开发、AutoDL 云服务器部署以及数据库配置。

---

## 系统要求

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Python | 3.10+ | 后端运行 |
| Node.js | 20+ | 前端编译 |
| LMStudio/vLLM | 最新版 | 模型推理 |

---

## 本地部署（使用 LMStudio）

### 1. 启动 LMStudio

1. 下载并安装 [LMStudio](https://lmstudio.ai/)
2. 下载模型：`Qwen3-8B-Instruct`，GGUF Q4_K_M 量化版本
3. 点击 **Start Server**，确保地址为 `http://localhost:1234/v1`

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

后端运行在 http://localhost:8000，API 文档：http://localhost:8000/docs

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端运行在 http://localhost:3000

---

## AutoDL 云服务器部署

### 环境信息

- **GPU**: RTX 4090 24GB
- **CUDA**: 12.1+
- **推理引擎**: vLLM

### 1. 初始化环境

```bash
cd /root/autodl-tmp/Sentiment-Analysis-System

# 创建虚拟环境
conda create -n sentiment python=3.10 -y
conda activate sentiment

# 安装后端依赖
pip install -r backend/requirements.txt

# 安装 vLLM
pip install vllm

# 安装前端依赖
cd frontend && npm install
```

### 2. 下载模型

```bash
pip install huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen3-8B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct
```

### 3. 配置后端

在 `backend/` 目录创建 `.env` 文件：

```bash
cd /root/autodl-tmp/Sentiment-Analysis-System/backend
cat > .env << EOF
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
LMSTUDIO_BASE_URL=http://localhost:8001/v1
LMSTUDIO_MODEL=Qwen3-8B-Instruct
DATABASE_URL=sqlite:///./sentiment.db
DEPLOY_MODE=server
CORS_ORIGINS=http://localhost:3000,http://your-instance-ip:3000
EOF
```

### 4. 启动服务

```bash
# 启动 vLLM 推理服务
cd /root/autodl-tmp
nohup python -m vllm.entrypoints.api_server \
  --model /root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  > vllm.log 2>&1 &

# 启动后端
cd /root/autodl-tmp/Sentiment-Analysis-System/backend
nohup python main.py > backend.log 2>&1 &

# 启动前端
cd /root/autodl-tmp/Sentiment-Analysis-System/frontend
nohup npm run dev -- --host 0.0.0.0 --port 3000 > frontend.log 2>&1 &
```

### 5. 开放端口（AutoDL 控制台）

在 AutoDL 控制台 → 实例详情 → 自定义开放端口，添加：
- `3000` - 前端
- `8000` - 后端 API
- `8001` - vLLM（调试用）

### 6. 访问地址

- 前端：`http://<实例 IP>:3000`
- API 文档：`http://<实例 IP>:8000/docs`

---

## 数据库配置

### 三种模式对比

| 模式 | 配置 | 适用场景 | 数据持久化 |
|------|------|---------|-----------|
| 内存 SQLite | 无需配置 | 快速测试 | ❌ |
| SQLite 文件 | `DATABASE_URL=sqlite:///./sentiment.db` | 个人开发 | ✅ |
| MySQL | `DATABASE_URL=mysql+pymysql://...` | 生产部署 | ✅ |

### SQLite 文件存储

在 `backend/` 目录创建 `.env`：

```ini
DATABASE_URL=sqlite:///./sentiment.db
```

### MySQL 部署

#### 1. 安装 MySQL

**Linux:**
```bash
sudo apt update
sudo apt install mysql-server -y
sudo systemctl start mysql
```

**Docker (推荐):**
```bash
docker run -d \
  --name mysql-sentiment \
  -e MYSQL_ROOT_PASSWORD=your_password \
  -e MYSQL_DATABASE=sentiment_db \
  -p 3306:3306 \
  mysql:8.0
```

#### 2. 初始化数据库

```bash
mysql -u root -p < backend/database/init_mysql.sql
```

#### 3. 配置连接

在 `backend/.env` 中：

```ini
DATABASE_URL=mysql+pymysql://root:你的密码@localhost:3306/sentiment_db
```

#### 4. 安装驱动

```bash
pip install pymysql cryptography
```

#### 5. 验证连接

访问 http://localhost:8000/health，检查 `database_connected` 为 `true`。

---

## 服务管理脚本

### 一键部署

```bash
chmod +x deploy_autodl.sh manage_autodl.sh
./deploy_autodl.sh
```

### 服务控制

```bash
./manage_autodl.sh status   # 查看状态
./manage_autodl.sh stop     # 停止服务
./manage_autodl.sh restart  # 重启服务
./manage_autodl.sh logs     # 查看日志
```

---

## 常见问题

### LMStudio 连接失败

1. 确保 LMStudio 已启动 Local Server
2. 访问 http://localhost:1234/v1/models 验证
3. 检查 `LMSTUDIO_BASE_URL` 配置

### vLLM 显存不足

降低 `--gpu-memory-utilization` 到 0.8 或 0.7

### 端口无法访问

检查 AutoDL 控制台「自定义开放端口」是否已添加对应端口

### MySQL 连接失败

1. 检查 MySQL 服务是否启动
2. 验证用户名密码正确
3. 确认字符集为 `utf8mb4`

---

## 快速命令参考

| 操作 | 命令 |
|------|------|
| 启动后端（本地） | `cd backend && python main.py` |
| 启动前端（本地） | `cd frontend && npm run dev` |
| AutoDL 部署 | `./deploy_autodl.sh` |
| 查看后端日志 | `tail -f backend/backend.log` |
| 查看 vLLM 日志 | `tail -f /root/autodl-tmp/vllm.log` |
