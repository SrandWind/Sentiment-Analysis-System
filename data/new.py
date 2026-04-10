#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMACD 数据集预处理脚本 v4 —— vLLM 加速版
改动：推理引擎从 transformers.generate → vLLM LLM.generate
保留：流式写入 / 断点续传（进度与数据分离）/ 失败样本单独记录 / 重试工具
数据格式：posts, angry, fear, happy, neutral, sad, surprise, type
目标模型：Qwen3-8B-Instruct（AutoDL autodl-tmp 路径）
输出格式：LLaMA-Factory Alpaca JSONL

依赖安装（AutoDL 环境）：
    pip install vllm
注意：vLLM 与 load_in_8bit 不兼容，改用 gpu_memory_utilization 控制显存占用
"""

import pandas as pd
import json
import os
import random
import re
import gc
import sys
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

# vLLM 核心导入
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import torch

# ==================== 全局配置 ====================

CONFIG = {
    # ---- 模型路径 ----
    "model_name": "/root/autodl-tmp/models/Qwen3-8B-Instruct",

    # ---- vLLM 推理参数 ----
    # vLLM 不用 load_in_8bit，用 gpu_memory_utilization 控制显存
    # 24GB 显存建议 0.90；16GB 建议 0.85；OOM 时调低
    "gpu_memory_utilization": 0.90,
    "max_model_len": 10000,          # 输入+输出总 token 上限；含 few-shot prompt 约 2500 token
    "max_new_tokens": 5000,         # 单条最大输出 token
    "temperature": 0.3,
    "top_p": 0.9,

    # vLLM 一次性提交所有请求，内部自动调度，不需要手动 batch_size
    # vllm_batch_size 控制每次向 vLLM 提交多少条，避免一次性构建过大的 prompt 列表
    # 建议 200~500；显存越大可以越大
    "vllm_batch_size": 300,

    # ---- 数据路径 ----
    "input_csv": "/root/autodl-tmp/test/demo.csv",
    "output_path": "./dataset",
    "checkpoint_dir": "./checkpoints",
    "log_dir": "./logs",

    # ---- 处理控制 ----
    "sample_size": -1,          # -1 = 全量
    "task_name": "cmacd_cot",
    "flush_every": 20,          # 每 N 条强制 fsync

    # ---- 数据集划分 ----
    "train_ratio": 0.8,
    "dev_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 42,
}

EMOTION_COLUMNS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_CN_MAP = {
    'angry': '愤怒', 'fear': '恐惧', 'happy': '快乐',
    'neutral': '中性', 'sad': '悲伤', 'surprise': '惊讶'
}

# ==================== 日志 ====================

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f"preprocess_{ts}.log"), encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging(CONFIG["log_dir"])

# ==================== Prompt 模板 ====================

CMACD_COT_FEWSHOT = """\
# 角色定义
你是专业的情感分析专家，专注于中文社交媒体文本的多维度情绪识别与推理。

# 任务说明
基于以下示例，对目标文本进行完整的思维链推理分析，输出 JSON 格式。

# ==================== 示例 1 ====================
输入文本："我擦我才知道为什么每次我坐飞机都觉得难受如坐针毡，因为每次坐的都是厦航原来是因为厦航大多都是波音然后波音的座位又比较窄🤔下半年可能还要飞但是可能还是会选厦航，大概福建人对厦航有一种天然的信任（？）"
MBTI：ENTP
参考标签：angry=0.0, fear=0.06, happy=0.17, neutral=0.22, sad=0.42, surprise=0.18

输出：
{{
    "mbti_type": "ENTP",
    "emotion_analysis": {{
        "angry":    {{"intensity": 0.0,  "evidence": "无明显愤怒表达", "reasoning": "'我擦'为轻度感叹词，整体语气为理性分析，非愤怒"}},
        "fear":     {{"intensity": 0.06, "evidence": "下半年可能还要飞", "reasoning": "对再次飞行的轻微担忧，强度极低"}},
        "happy":    {{"intensity": 0.17, "evidence": "天然的信任", "reasoning": "找到原因后的释然感与信任感，强度中低"}},
        "neutral":  {{"intensity": 0.22, "evidence": "厦航大多都是波音、座位比较窄", "reasoning": "客观信息陈述占一定比例"}},
        "sad":      {{"intensity": 0.42, "evidence": "难受如坐针毡", "reasoning": "身体不适是核心情绪，比喻强化负面体验，强度中等"}},
        "surprise": {{"intensity": 0.18, "evidence": "我才知道、原来是因为", "reasoning": "发现真相的惊讶已被理性接受，强度中低"}}
    }},
    "cot_reasoning_chain": {{
        "step1_text_features":      "关键词：'我擦''难受如坐针毡''才知道'；触发事件：发现坐飞机难受的根本原因；语境：🤔思考表情、括号问号表不确定",
        "step2_emotion_analysis":   "sad(0.42)主导；neutral(0.22)、happy(0.17)、surprise(0.18)次要；angry/fear极低",
        "step3_intensity_reasoning":"'如坐针毡'强化不适感，但整体用理性分析表达而非情绪宣泄，故sad中等而非高",
        "step4_compound_emotion":   "身体不适(sad) + 发现真相(surprise) + 对厦航信任(happy) 构成复合情绪",
        "step5_mbti_factor":        "ENTP理性分析倾向明显，负面情绪也以逻辑方式呈现，符合'发现问题→分析原因→做出决策'模式"
    }},
    "final_conclusion": {{
        "primary_emotion": "sad",
        "secondary_emotions": ["neutral", "surprise", "happy"],
        "emotion_summary": "以身体不适的悲伤为主导，ENTP理性分析风格掩盖了部分情绪强度，整体呈'发现问题并释然'状态",
        "confidence_score": 0.86
    }}
}}

# ==================== 示例 2 ====================
输入文本："[address] 一行，太多想讲反而因为行程太满，回程后再好好写，本来吃食还想着放着，没想到也常常立马出发就忘了😂。就后面再好好整理吧 (拖延症患者)"
MBTI：INFJ
参考标签：angry=0.0, fear=0.07, happy=0.12, neutral=0.16, sad=0.68, surprise=0.0

输出：
{{
    "mbti_type": "INFJ",
    "emotion_analysis": {{
        "angry":    {{"intensity": 0.0,  "evidence": "无愤怒表达", "reasoning": "文本无任何愤怒词汇"}},
        "fear":     {{"intensity": 0.07, "evidence": "太多想讲反而...忘了", "reasoning": "对无法完成计划的轻微担忧，强度极低"}},
        "happy":    {{"intensity": 0.12, "evidence": "😂", "reasoning": "😂为自嘲用法而非真实喜悦，强度极低"}},
        "neutral":  {{"intensity": 0.16, "evidence": "回程后再好好写、再好好整理", "reasoning": "计划安排的中性陈述"}},
        "sad":      {{"intensity": 0.68, "evidence": "忘了、拖延症患者", "reasoning": "'拖延症患者'自我标签化表达强烈自我嫌弃与无奈，是核心情绪"}},
        "surprise": {{"intensity": 0.0,  "evidence": "无", "reasoning": "'没想到'为习惯用语，非真实惊讶"}}
    }},
    "cot_reasoning_chain": {{
        "step1_text_features":      "关键词：'忘了''拖延症患者'；触发事件：行程太满导致遗忘计划；语境：😂自嘲、括号补充说明",
        "step2_emotion_analysis":   "sad(0.68)绝对主导且指向自我，其余情绪均低",
        "step3_intensity_reasoning":"'拖延症患者'自我标签强烈表达内疚与无奈，😂自嘲强化而非掩盖sad",
        "step4_compound_emotion":   "自我指向的悲伤(sad)为核心，伴随轻微担忧(fear)和自嘲快乐(happy)",
        "step5_mbti_factor":        "INFJ内省倾向深，善于自我反思批评，自嘲是其表达内疚的典型方式"
    }},
    "final_conclusion": {{
        "primary_emotion": "sad",
        "secondary_emotions": ["neutral", "fear"],
        "emotion_summary": "对自身拖延行为的强烈无奈与自我嫌弃，INFJ内省特质使情绪向内聚焦",
        "confidence_score": 0.91
    }}
}}


# ==================== 目标文本 ====================
输入文本：{text}
MBTI 类型：{mbti}
参考标签：angry={angry}, fear={fear}, happy={happy}, neutral={neutral}, sad={sad}, surprise={surprise}

# 输出要求
- 严格参照示例 JSON 结构，字段顺序一致
- 只输出 JSON，无任何额外内容（无代码块标记、无说明文字）
- step1~step5 推理步骤必须完整
- 每个情绪的 intensity 必须有 evidence 和 reasoning 支撑
- intensity 数值应与参考标签基本对齐，但推理必须来自文本本身

# 开始分析
"""

# ==================== 断点管理（与 v3 完全一致）====================

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, task_name: str,
                 output_jsonl: str, failed_jsonl: str, flush_every: int = 10):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.progress_path = os.path.join(checkpoint_dir, f"{task_name}.progress.json")
        self.output_jsonl = output_jsonl
        self.failed_jsonl = failed_jsonl
        self.flush_every = flush_every
        self._write_count = 0
        self._stream = open(output_jsonl, "a", encoding="utf-8", buffering=1)
        self._fail_stream = open(failed_jsonl, "a", encoding="utf-8", buffering=1)

    def load_progress(self) -> Dict:
        default = {"completed": [], "failed": [], "total": 0,
                   "start_time": None, "last_update": None}
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"断点恢复 → 已完成 {len(data['completed'])} 条，失败 {len(data['failed'])} 条")
                return data
            except Exception as e:
                logger.warning(f"进度文件读取失败（{e}），重新开始")
        return default

    def save_progress(self, progress: Dict):
        progress["last_update"] = datetime.now().isoformat()
        tmp = self.progress_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        os.replace(tmp, self.progress_path)

    def write_sample(self, sample: dict):
        self._stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._write_count += 1
        if self._write_count % self.flush_every == 0:
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def write_failed(self, record: dict):
        self._fail_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fail_stream.flush()
        os.fsync(self._fail_stream.fileno())

    def flush(self):
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._fail_stream.flush()
        os.fsync(self._fail_stream.fileno())

    def close(self):
        self.flush()
        self._stream.close()
        self._fail_stream.close()

    @staticmethod
    def recover_written_indices(jsonl_path: str) -> set:
        indices = set()
        if not os.path.exists(jsonl_path):
            return indices
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get("metadata", {}).get("original_index")
                    if idx is not None:
                        indices.add(int(idx))
                except Exception:
                    continue
        return indices


# ==================== 主处理类（vLLM 版）====================

class CMACDPreprocessorVLLM:
    """
    推理引擎换成 vLLM，其余逻辑与 v3 保持一致。

    vLLM 与 transformers 的关键差异：
    1. 不支持 load_in_8bit，用 gpu_memory_utilization 代替
    2. 输入直接传 prompt 字符串列表，不需要 tokenizer padding/truncation
    3. generate() 返回 RequestOutput 列表，取 .outputs[0].text 获取生成文本
    4. vLLM 内部已做连续批处理，外部无需管 batch 大小（但分批提交避免内存溢出）
    5. Qwen3 需要在 chat_template 里关闭内置思维链（enable_thinking=False）
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._load_model()

    def _load_model(self):
        model_name = self.cfg["model_name"]
        logger.info(f"[vLLM] 加载模型：{model_name}")

        # vLLM 用 LLM 类加载，内部自动处理 tokenizer
        # enable_prefix_caching=True：few-shot prompt 前缀在所有请求中相同，开启后首次计算，后续复用 KV cache
        # 对于含大量 few-shot 的 prompt 有显著加速（约减少 40% 的 prefill 时间）
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=self.cfg["gpu_memory_utilization"],
            max_model_len=self.cfg["max_model_len"],
            dtype="float16",
            enable_prefix_caching=True,   # ★ few-shot 前缀复用
        )

        # 仍然需要 tokenizer 来构建 chat_template 格式的 prompt
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # vLLM 的采样参数（与 transformers generate 参数对应）
        self.sampling_params = SamplingParams(
            temperature=self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            max_tokens=self.cfg["max_new_tokens"],
        )

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"[vLLM] GPU 显存：{mem:.1f} GB")
        logger.info("[vLLM] 模型加载完成，prefix caching 已启用")

    # ---------- 数据加载 ----------

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df = df.rename(columns={"posts": "text", "type": "mbti"})
        required = ["text", "mbti"] + EMOTION_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少列：{missing}")
        for emo in EMOTION_COLUMNS:
            df[emo] = pd.to_numeric(df[emo], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        logger.info(f"数据加载：{len(df)} 条")
        return df

    # ---------- Prompt 构建 ----------

    def _build_prompt(self, row: pd.Series) -> str:
        return CMACD_COT_FEWSHOT.format(
            text=str(row["text"]).strip(),
            mbti=row["mbti"],
            angry=row["angry"], fear=row["fear"], happy=row["happy"],
            neutral=row["neutral"], sad=row["sad"], surprise=row["surprise"],
        )

    def _build_chat_text(self, prompt: str) -> str:
        """将 prompt 包装为 Qwen3 chat template 格式"""
        messages = [
            {"role": "system", "content": "你是专业的情感分析专家，专注于中文社交媒体情感分析。"},
            {"role": "user", "content": prompt},
        ]
        # enable_thinking=False：关闭 Qwen3 内置 <think> token，避免与自定义 CoT 冲突
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    # ---------- vLLM 批量推理（核心替换）----------

    def _vllm_generate(self, prompts: List[str]) -> List[str]:
        """
        将 prompt 列表转为 chat template 格式后，批量提交给 vLLM。
        vLLM 内部自动做连续批处理（continuous batching），无需手动分 batch。
        返回与输入顺序一致的文本列表。
        """
        chat_texts = [self._build_chat_text(p) for p in prompts]

        # vLLM generate：一次提交所有请求，内部异步调度
        outputs = self.llm.generate(
            prompts=chat_texts,
            sampling_params=self.sampling_params,
        )

        # outputs 顺序与 prompts 一致，取第一个候选输出的文本
        return [out.outputs[0].text for out in outputs]

    # ---------- JSON 解析 ----------

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            text = m.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start: end + 1]
        try:
            return json.loads(text)
        except Exception:
            return None

    # ---------- 转 Alpaca 格式 ----------

    def _to_alpaca(self, cot_data: dict, row: pd.Series, idx: int) -> dict:
        ea = cot_data.get("emotion_analysis", {})
        rc = cot_data.get("cot_reasoning_chain", {})
        fc = cot_data.get("final_conclusion", {})

        emo_lines = "\n".join(
            f"  {EMOTION_CN_MAP.get(e, e)}({e}): "
            f"{float(ea.get(e, {}).get('intensity', 0.0)):.2f}  "
            f"[证据: {ea.get(e, {}).get('evidence', '-')}]  "
            f"[推理: {ea.get(e, {}).get('reasoning', '-')}]"
            for e in EMOTION_COLUMNS
        )
        rc_lines = "\n".join(f"  {k}: {v}" for k, v in rc.items())
        primary = fc.get("primary_emotion", "unknown")
        secondary = "、".join(fc.get("secondary_emotions", []))
        summary = fc.get("emotion_summary", "无")
        confidence = fc.get("confidence_score", 0.0)

        output_text = (
            f"【情绪强度分析】\n{emo_lines}\n\n"
            f"【思维链推理过程】\n{rc_lines}\n\n"
            f"【主要情绪】{EMOTION_CN_MAP.get(primary, primary)}\n"
            f"【次要情绪】{secondary}\n"
            f"【情绪总结】{summary}\n"
            f"【置信度】{confidence:.2f}"
        )

        ground_truth = {
            e: float(row[e]) if e in row and pd.notnull(row[e]) else 0.0
            for e in EMOTION_COLUMNS
        }

        return {
            "id": f"CMACD_{idx:05d}",
            "instruction": "请分析以下中文社交媒体文本的情感状态，输出6种情绪维度的强度值（0~1）及完整的思维链推理过程。",
            "input": str(row["text"]).strip(),
            "output": output_text,
            "history": [],
            "metadata": {
                "mbti": row["mbti"],
                "ground_truth": ground_truth,
                "original_index": int(idx),
            },
        }

    # ---------- 构建失败记录 ----------

    def _build_failed_record(self, row: pd.Series, idx: int,
                              fail_type: str, reason: str, raw_output: str) -> dict:
        return {
            "metadata": {
                "original_index": int(idx),
                "mbti": row["mbti"],
                "ground_truth": {
                    e: float(row[e]) if e in row and pd.notnull(row[e]) else 0.0
                    for e in EMOTION_COLUMNS
                },
            },
            "original_text": str(row["text"]).strip(),
            "angry":    float(row.get("angry", 0.0)),
            "fear":     float(row.get("fear", 0.0)),
            "happy":    float(row.get("happy", 0.0)),
            "neutral":  float(row.get("neutral", 0.0)),
            "sad":      float(row.get("sad", 0.0)),
            "surprise": float(row.get("surprise", 0.0)),
            "fail_type":   fail_type,
            "fail_reason": reason,
            "raw_output":  raw_output[:500] if raw_output else "",
            "timestamp":   datetime.now().isoformat(),
        }

    # ==================== 核心处理流程 ====================

    def process(self, df: pd.DataFrame) -> str:
        cfg = self.cfg
        os.makedirs(cfg["output_path"], exist_ok=True)

        n = cfg["sample_size"]
        if 0 < n < len(df):
            df_work = df.sample(n=n, random_state=cfg["random_seed"]).reset_index(drop=True)
        else:
            df_work = df.reset_index(drop=True)

        output_jsonl = os.path.join(cfg["output_path"], "alpaca_format.jsonl")
        failed_jsonl = os.path.join(cfg["output_path"], "failed_samples.jsonl")

        ckpt_mgr = CheckpointManager(
            checkpoint_dir=cfg["checkpoint_dir"],
            task_name=cfg["task_name"],
            output_jsonl=output_jsonl,
            failed_jsonl=failed_jsonl,
            flush_every=cfg["flush_every"],
        )

        progress = ckpt_mgr.load_progress()

        done_from_progress = set(progress["completed"])
        fail_from_progress = set(progress["failed"])
        done_from_jsonl = CheckpointManager.recover_written_indices(output_jsonl)
        fail_from_jsonl = CheckpointManager.recover_written_indices(failed_jsonl)
        done_set = done_from_progress | done_from_jsonl
        fail_set = fail_from_progress | fail_from_jsonl

        extra_done = done_from_jsonl - done_from_progress
        extra_fail = fail_from_jsonl - fail_from_progress
        if extra_done or extra_fail:
            logger.info(f"从 JSONL 补录：成功 +{len(extra_done)} 条，失败 +{len(extra_fail)} 条")
            progress["completed"] = list(done_set)
            progress["failed"] = list(fail_set)

        pending = sorted(set(range(len(df_work))) - done_set - fail_set)
        logger.info(
            f"总计 {len(df_work)} | 已完成 {len(done_set)} | "
            f"失败 {len(fail_set)} | 待处理 {len(pending)}"
        )

        if not pending:
            logger.info("全部处理完毕")
            ckpt_mgr.close()
            self._save_dataset_config(cfg["output_path"])
            return output_jsonl

        if not progress["start_time"]:
            progress["total"] = len(df_work)
            progress["start_time"] = datetime.now().isoformat()

        # ── vLLM 分批提交 ──
        # 虽然 vLLM 内部自动调度，但仍然分批提交有两个好处：
        # 1. 每批处理完后及时保存断点，程序崩溃最多丢一批
        # 2. 避免一次性把全部 prompt 字符串加载进内存
        vbs = cfg["vllm_batch_size"]
        batches = [pending[i: i + vbs] for i in range(0, len(pending), vbs)]

        logger.info(f"分 {len(batches)} 批提交，每批最多 {vbs} 条")

        for batch_num, batch_indices in enumerate(tqdm(batches, desc="vLLM 推理", unit="batch")):
            rows_batch = df_work.iloc[batch_indices]
            prompts = [self._build_prompt(r) for _, r in rows_batch.iterrows()]

            # ── vLLM 批量推理 ──
            try:
                responses = self._vllm_generate(prompts)
            except Exception as e:
                # vLLM 批量失败（极少见），整批标记为 inference_error
                logger.error(f"[batch={batch_num}] vLLM 推理异常：{e}，整批标记失败")
                for idx in batch_indices:
                    row = df_work.iloc[idx]
                    fail_record = self._build_failed_record(
                        row, idx, "inference_error", str(e), ""
                    )
                    ckpt_mgr.write_failed(fail_record)
                    fail_set.add(idx)
                    progress["failed"].append(idx)
                ckpt_mgr.save_progress(progress)
                continue

            # ── 逐条解析、分流写入 ──
            for idx, raw in zip(batch_indices, responses):
                row = df_work.iloc[idx]

                parsed = self._parse_json(raw)

                if parsed is None:
                    # 细化失败原因
                    if not raw.strip():
                        reason = "模型输出为空"
                    elif "{" not in raw:
                        reason = f"输出不含JSON结构（可能被截断或拒绝）: {raw[:120]}"
                    else:
                        reason = f"JSON结构不完整或格式错误: {raw[:120]}"

                    logger.warning(f"[idx={idx}] JSON解析失败 | {reason}")
                    ckpt_mgr.write_failed(
                        self._build_failed_record(row, idx, "json_parse_error", reason, raw)
                    )
                    fail_set.add(idx)
                    progress["failed"].append(idx)
                else:
                    ckpt_mgr.write_sample(self._to_alpaca(parsed, row, idx))
                    done_set.add(idx)
                    progress["completed"].append(idx)

            # 每批结束保存进度
            ckpt_mgr.save_progress(progress)

            # 进度日志（每 5 批打印一次）
            if (batch_num + 1) % 5 == 0 or batch_num == len(batches) - 1:
                total_done = len(done_set) + len(fail_set)
                logger.info(
                    f"进度 {total_done}/{len(df_work)} | "
                    f"成功 {len(done_set)} | 失败 {len(fail_set)}"
                )

        ckpt_mgr.flush()
        ckpt_mgr.close()

        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info(f"  成功：{len(done_set)} 条 → {output_jsonl}")
        logger.info(f"  失败：{len(fail_set)} 条 → {failed_jsonl}")
        if fail_set:
            logger.info("  ⚠ 失败原因详见日志，可用 --retry 模式统一重试")

        self._save_dataset_config(cfg["output_path"])
        return output_jsonl

    # ==================== 数据集划分 ====================

    def split_dataset(self, input_jsonl: str):
        data = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        continue

        logger.info(f"读取 JSONL：{len(data)} 条有效样本")
        random.seed(self.cfg["random_seed"])
        random.shuffle(data)

        total = len(data)
        tr = int(total * self.cfg["train_ratio"])
        dv = int(total * (self.cfg["train_ratio"] + self.cfg["dev_ratio"]))

        splits = {
            "train.json": data[:tr],
            "dev.json":   data[tr:dv],
            "test.json":  data[dv:],
        }

        out_dir = os.path.dirname(input_jsonl)
        for fname, subset in splits.items():
            path = os.path.join(out_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(subset, f, ensure_ascii=False, indent=2)
            logger.info(f"  {fname}：{len(subset)} 条")

        self._save_dataset_config(out_dir, splits=splits)

    def _save_dataset_config(self, output_dir: str, splits: Optional[dict] = None):
        alpaca_base = {
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        }
        config = {"cmacd_emotion": {**alpaca_base, "file_name": "alpaca_format.jsonl"}}
        if splits:
            for key, fname in [("cmacd_train", "train.json"),
                                ("cmacd_dev",   "dev.json"),
                                ("cmacd_test",  "test.json")]:
                config[key] = {**alpaca_base, "file_name": fname}

        cfg_path = os.path.join(output_dir, "dataset_info.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"dataset_info.json → {cfg_path}")

    def cleanup(self):
        """释放 vLLM 占用的显存"""
        try:
            destroy_model_parallel()
        except Exception:
            pass
        del self.llm
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[vLLM] 显存已释放")


# ==================== 失败样本重试 ====================

def retry_failed(failed_jsonl: str, cfg: dict):
    """
    读取 failed_samples.jsonl，对其中所有条目重新推理一次（用 vLLM）。
    重试成功的追加写入 alpaca_format.jsonl；仍失败的更新 retry_count 后写回。
    """
    if not os.path.exists(failed_jsonl):
        logger.info("没有失败样本文件，跳过重试")
        return

    failed_records = []
    with open(failed_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    failed_records.append(json.loads(line))
                except Exception:
                    continue

    if not failed_records:
        logger.info("failed_samples.jsonl 为空，跳过重试")
        return

    from collections import Counter
    type_counts = Counter(r.get("fail_type", "unknown") for r in failed_records)
    logger.info(f"开始重试 {len(failed_records)} 条 | 类型分布：{dict(type_counts)}")

    preprocessor = CMACDPreprocessorVLLM(cfg)
    output_jsonl = os.path.join(cfg["output_path"], "alpaca_format.jsonl")
    still_failed = []

    # 将失败记录批量提交给 vLLM（一次性，效率更高）
    rows_list, idx_list = [], []
    for rec in failed_records:
        idx_list.append(rec["metadata"]["original_index"])
        rows_list.append(pd.Series({
            "text": rec["original_text"], "mbti": rec["metadata"]["mbti"],
            "angry": rec["angry"], "fear": rec["fear"], "happy": rec["happy"],
            "neutral": rec["neutral"], "sad": rec["sad"], "surprise": rec["surprise"],
        }))

    prompts = [preprocessor._build_prompt(r) for r in rows_list]

    try:
        responses = preprocessor._vllm_generate(prompts)
    except Exception as e:
        logger.error(f"重试 vLLM 推理整体失败：{e}")
        preprocessor.cleanup()
        return

    with open(output_jsonl, "a", encoding="utf-8", buffering=1) as success_stream:
        for rec, idx, row, raw in zip(failed_records, idx_list, rows_list, responses):
            parsed = preprocessor._parse_json(raw)
            if parsed:
                sample = preprocessor._to_alpaca(parsed, row, idx)
                success_stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                logger.info(f"[idx={idx}] 重试成功")
            else:
                reason = f"重试仍解析失败: {raw[:120]}"
                logger.warning(f"[idx={idx}] {reason}")
                rec["fail_reason"] = reason
                rec["retry_count"] = rec.get("retry_count", 0) + 1
                rec["timestamp"] = datetime.now().isoformat()
                still_failed.append(rec)

    # 覆盖写回仍然失败的条目
    with open(failed_jsonl, "w", encoding="utf-8") as f:
        for rec in still_failed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    preprocessor.cleanup()
    logger.info(
        f"重试完成 | 成功 {len(failed_records) - len(still_failed)} 条 | "
        f"仍失败 {len(still_failed)} 条"
    )


# ==================== 入口 ====================

if __name__ == "__main__":
    if "--retry" in sys.argv:
        failed_path = os.path.join(CONFIG["output_path"], "failed_samples.jsonl")
        retry_failed(failed_path, CONFIG)
    else:
        preprocessor = CMACDPreprocessorVLLM(CONFIG)
        df = preprocessor.load_data(CONFIG["input_csv"])
        output_file = preprocessor.process(df)
        preprocessor.split_dataset(output_file)
        preprocessor.cleanup()