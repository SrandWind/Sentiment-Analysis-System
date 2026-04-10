#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from swift.llm import InferRequest, PtEngine, RequestConfig
# 如果你要 vllm，把上面 PtEngine 保留，再在 create_engine 里按需 import VllmEngine


# ========= 配置 =========
MODEL = "/root/autodl-tmp/models/JunHowie/Qwen3-8B-Instruct"
INFER_BACKEND = "vllm"         # "pt" / "vllm"
MAX_BATCH_SIZE = 16
MAX_MODEL_LEN = 4096

INPUT_PATH = "/root/autodl-tmp/data/dataset/test_v2_relabel.json"
OUTPUT_PATH = "/root/autodl-tmp/forCompare/outputs/pred_base_async.jsonl"

MAX_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
CONCURRENCY = 16             # 异步并发数，4090可先16再试24


TASK_RULES = {
    "task_emotion_reg": (
        "你现在只做 task_emotion_reg。"
        "只输出严格JSON，不要markdown，不要解释，不要代码块，不要多余字段。"
        "输出必须包含且仅包含键：mbti_type, emotion_analysis, cot_reasoning_chain_v2, final_conclusion。"
        "emotion_analysis 必须包含 angry,fear,happy,neutral,sad,surprise 六个键。"
        "final_conclusion 必须是对象，包含 primary_emotion, secondary_emotions, emotion_summary, confidence_score。"
    ),
    "task_primary_cls": (
        "你现在只做 task_primary_cls。"
        "只输出严格JSON，不要markdown，不要解释，不要代码块，不要多余字段。"
        "输出必须包含且仅包含键：primary_emotion, secondary_emotions, distribution。"
        "primary_emotion 必须是 angry/fear/happy/neutral/sad/surprise 之一。"
        "distribution 必须包含 angry,fear,happy,neutral,sad,surprise 六个键，值在[0,1]。"
    ),
    "task_mbti_pred": (
        "你现在只做 task_mbti_pred。"
        "只输出严格JSON，不要markdown，不要解释，不要代码块，不要多余字段。"
        "输出格式必须严格为：{\"mbti_type\":\"XXXX\"}。"
        "XXXX 必须且只能是16种之一："
        "INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, "
        "ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP。"
        "顶层只允许 mbti_type 这1个键。"
    ),
    "task_cot_gen": (
        "你现在只做 task_cot_gen。"
        "只输出严格JSON，不要markdown，不要解释，不要代码块，不要多余字段。"
        "顶层必须且仅允许4个键：mbti_type, cot_reasoning_chain_v2, target_scores, primary_emotion。"
        "cot_reasoning_chain_v2 必须包含且仅包含："
        "step1_text_cues, step2_candidate_emotions, step3_intensity_grounding, "
        "step4_compound_relation, step5_mbti_modulation, step6_counterfactual_check, "
        "step7_uncertainty_and_final；每个值必须为非空字符串。"
        "target_scores 必须包含 angry,fear,happy,neutral,sad,surprise 六个键，值在[0,1]。"
        "primary_emotion 必须是 angry/fear/happy/neutral/sad/surprise 之一。"
        "若不确定，也必须补全所有字段，不得缺失。"
    ),
}



def create_engine():
    if INFER_BACKEND == "pt":
        return PtEngine(MODEL, max_batch_size=MAX_BATCH_SIZE)
    if INFER_BACKEND == "vllm":
        from swift.llm import VllmEngine
        return VllmEngine(MODEL, max_model_len=MAX_MODEL_LEN)
    raise ValueError(f"Unsupported backend: {INFER_BACKEND}")


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                i = str(obj.get("id", "")).strip()
                if i:
                    done.add(i)
            except Exception:
                pass
    return done


def build_prompt(row: Dict) -> str:
    task = str(row.get("metadata", {}).get("task", ""))
    rule = TASK_RULES.get(task, "只输出严格JSON。")
    return (
        f"{rule}\n\n"
        f"任务指令:\n{row.get('instruction','')}\n\n"
        f"输入文本:\n{row.get('input','')}"
    )


def extract_json_obj(text: str):
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def schema_ok(task: str, text: str) -> bool:
    obj = extract_json_obj(text)
    if not isinstance(obj, dict):
        return False
    if task == "task_emotion_reg":
        return isinstance(obj.get("emotion_analysis"), dict) and isinstance(obj.get("final_conclusion"), dict)
    if task == "task_primary_cls":
        return "primary_emotion" in obj
    if task == "task_mbti_pred":
        return "mbti_type" in obj
    if task == "task_cot_gen":
        return isinstance(obj.get("cot_reasoning_chain_v2"), dict)
    return True


async def main():
    rows = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(out_path)
    rows = [r for r in rows if str(r.get("id", "")) not in done_ids]
    print(f"pending={len(rows)}")

    engine = create_engine()
    req_cfg = RequestConfig(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

    sem = asyncio.Semaphore(CONCURRENCY)
    bad = 0
    bad_by_task = Counter()

    async def infer_one(row: Dict):
        nonlocal bad
        rid = str(row["id"])
        task = str(row.get("metadata", {}).get("task", ""))
        prompt = build_prompt(row)
        infer_req = InferRequest(messages=[{"role": "user", "content": prompt}])

        async with sem:
            try:
                resp = await engine.infer_async(infer_req, req_cfg)
                out_text = resp.choices[0].message.content
            except Exception as e:
                out_text = f'{{"error":"infer_failed","message":"{str(e).replace(chr(34), chr(39))}"}}'

        if not schema_ok(task, out_text):
            bad += 1
            bad_by_task[task] += 1

        return {"id": rid, "output": out_text}

    chunk_size = 200
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            results = await asyncio.gather(*[infer_one(r) for r in chunk])
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{min(i + chunk_size, len(rows))}/{len(rows)} done")

    total = len(rows)
    rate = bad / max(1, total)
    print(f"schema_bad={bad}, total={total}, schema_bad_rate={rate:.4f}, by_task={dict(bad_by_task)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
