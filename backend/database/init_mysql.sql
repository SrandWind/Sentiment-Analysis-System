-- ========================================
-- 情感分析系统 - MySQL 数据库初始化脚本
-- ========================================

-- 1. 创建数据库
CREATE DATABASE IF NOT EXISTS sentiment_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

-- 2. 创建用户（可选，生产环境建议单独创建用户）
-- CREATE USER IF NOT EXISTS 'sentiment'@'localhost' IDENTIFIED BY 'your_password';
-- GRANT ALL PRIVILEGES ON sentiment_db.* TO 'sentiment'@'localhost';
-- FLUSH PRIVILEGES;

-- 3. 使用数据库
USE sentiment_db;

-- 4. 创建推理历史表
CREATE TABLE IF NOT EXISTS inference_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL COMMENT '输入文本',
    output TEXT COMMENT '模型原始输出',
    parsed_result JSON COMMENT '解析后的 JSON 结果',

    -- 情绪分数
    emotion_angry FLOAT DEFAULT 0.0,
    emotion_fear FLOAT DEFAULT 0.0,
    emotion_happy FLOAT DEFAULT 0.0,
    emotion_neutral FLOAT DEFAULT 0.0,
    emotion_sad FLOAT DEFAULT 0.0,
    emotion_surprise FLOAT DEFAULT 0.0,

    -- 分类结果
    primary_emotion VARCHAR(50),
    mbti_type VARCHAR(10),

    -- CoT 推理链
    cot_reasoning JSON,

    -- 元数据
    model_variant VARCHAR(50) DEFAULT 'gguf4bit',
    latency_ms FLOAT,
    json_parse_ok BOOLEAN DEFAULT FALSE,
    cot_complete BOOLEAN DEFAULT FALSE,

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 索引
    INDEX idx_primary_emotion (primary_emotion),
    INDEX idx_mbti_type (mbti_type),
    INDEX idx_created_at (created_at),
    INDEX idx_model_variant (model_variant)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='推理历史记录表';

-- 5. 查看表结构
DESCRIBE inference_history;

-- 6. 查看创建语句
SHOW CREATE TABLE inference_history;
