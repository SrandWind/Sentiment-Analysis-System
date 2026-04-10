#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-label CMACD with swift infer engine, then expand to multitask format.

Compatible backends:
- pt
- vllm
- lmdeploy

Reference style:
  from swift.llm import InferRequest, RequestConfig, PtEngine, VllmEngine, LmdeployEngine
  from swift.plugin import InferStats
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from swift.llm import InferRequest, RequestConfig
from swift.plugin import InferStats

EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
STEP7 = [
    "step1_text_cues",
    "step2_candidate_emotions",
    "step3_intensity_grounding",
    "step4_compound_relation",
    "step5_mbti_modulation",
    "step6_counterfactual_check",
    "step7_uncertainty_and_final",
]
TASK_WEIGHTS = {
    "task_emotion_reg": 4,
    "task_primary_cls": 2,
    "task_mbti_pred": 2,
    "task_cot_gen": 2,
}
TASK_NAMES = set(TASK_WEIGHTS.keys())

SYSTEM_PROMPT = "你是专业的中文社交媒体情感分析专家，擅长微博文本多维情绪推理。"
INSTRUCTION = """
请基于输入文本与MBTI，完成六维情绪分析，并仅输出严格JSON（不要解释文字、不要markdown、不要代码块）。

情绪维度必须包含：
angry, fear, happy, neutral, sad, surprise

输出结构必须为：
{
  "mbti_type": "INFJ",
  "emotion_analysis": {
    "angry": {"intensity": 0.00, "evidence": "...", "reasoning": "..."},
    "fear": {"intensity": 0.00, "evidence": "...", "reasoning": "..."},
    "happy": {"intensity": 0.00, "evidence": "...", "reasoning": "..."},
    "neutral": {"intensity": 0.00, "evidence": "...", "reasoning": "..."},
    "sad": {"intensity": 0.00, "evidence": "...", "reasoning": "..."},
    "surprise": {"intensity": 0.00, "evidence": "...", "reasoning": "..."}
  },
  "cot_reasoning_chain_v2": {
    "step1_text_cues": "...",
    "step2_candidate_emotions": "...",
    "step3_intensity_grounding": "...",
    "step4_compound_relation": "...",
    "step5_mbti_modulation": "...",
    "step6_counterfactual_check": "...",
    "step7_uncertainty_and_final": "..."
  },
  "final_conclusion": {
    "primary_emotion": "neutral",
    "secondary_emotions": ["fear", "sad"],
    "emotion_summary": "...",
    "confidence_score": 0.85
  }
}

约束：
1) 所有 intensity 在 [0,1] 范围，保留两位小数。
2) step1~step7 必须完整。
3) 主情绪必须是六维之一。
4) 若证据不足，也必须输出合法JSON且字段不得缺失。
5) 强度值需尽量贴近参考标签，偏移需在 reasoning 中解释。
6) step6 至少给出两个反事实候选并说明证据不足。
7) 只输出 JSON。
""".strip()


def create_engine(model: str, infer_backend: str, max_batch_size: int, max_model_len: int):
    if infer_backend == "pt":
        from swift.llm import PtEngine
        return PtEngine(model, max_batch_size=max_batch_size)
    if infer_backend == "vllm":
        from swift.llm import VllmEngine
        return VllmEngine(model, max_model_len=max_model_len)
    if infer_backend == "lmdeploy":
        from swift.llm import LmdeployEngine
        return LmdeployEngine(model)
    raise ValueError(f"Unsupported infer_backend: {infer_backend}")


def clamp(v: float) -> float:
    return max(0.0, min(1.0, round(float(v), 2)))


def extract_json_obj(text: str) -> Dict:
    x = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", x, re.I)
    if m:
        x = m.group(1)
    s, e = x.find("{"), x.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("no_json_object")
    return json.loads(x[s : e + 1])


def build_default_cot7(text: str, primary: str, secondary: List[str]) -> Dict[str, str]:
    sec = ", ".join(secondary) if secondary else "无"
    return {
        "step1_text_cues": f"文本关键线索：{text[:80]}...",
        "step2_candidate_emotions": f"候选主情绪：{primary}；次级候选：{sec}。",
        "step3_intensity_grounding": "根据文本语气、程度词、否定结构进行强度映射。",
        "step4_compound_relation": f"复合情绪关系：以 {primary} 为主，{sec} 为辅。",
        "step5_mbti_modulation": "结合MBTI对解释做调制，但不覆盖文本证据。",
        "step6_counterfactual_check": "反事实：给出两个候选主情绪并说明证据不足。",
        "step7_uncertainty_and_final": "说明不确定来源并给出最终结论与置信度。",
    }


def normalize_relabel_obj(obj: Dict, fallback_mbti: str, gt_scores: Dict[str, float], text: str) -> Dict:
    mbti = str(obj.get("mbti_type", fallback_mbti)).upper()
    emo = obj.get("emotion_analysis", {})
    norm_emo: Dict[str, Dict[str, object]] = {}
    for e in EMOTIONS:
        item = emo.get(e, {})
        if isinstance(item, dict):
            intensity = item.get("intensity", gt_scores.get(e, 0.0))
            evidence = str(item.get("evidence", ""))
            reasoning = str(item.get("reasoning", ""))
        else:
            intensity = item
            evidence = ""
            reasoning = ""
        norm_emo[e] = {
            "intensity": clamp(float(intensity)),
            "evidence": evidence,
            "reasoning": reasoning,
        }

    if all(float(norm_emo[e]["intensity"]) == 0.0 for e in EMOTIONS):
        for e in EMOTIONS:
            norm_emo[e]["intensity"] = clamp(gt_scores.get(e, 0.0))
            if not norm_emo[e]["evidence"]:
                norm_emo[e]["evidence"] = "回退至参考标签"
            if not norm_emo[e]["reasoning"]:
                norm_emo[e]["reasoning"] = "模型输出异常，使用参考标签回退"

    final = obj.get("final_conclusion", {})
    primary = str(final.get("primary_emotion", "")).lower()
    if primary not in EMOTIONS:
        primary = max(EMOTIONS, key=lambda k: float(norm_emo[k]["intensity"]))
    secondary = final.get("secondary_emotions", [])
    if not isinstance(secondary, list):
        secondary = []
    secondary = [str(x) for x in secondary if str(x) in EMOTIONS and str(x) != primary]

    cot = obj.get("cot_reasoning_chain_v2", {})
    norm_cot = {k: str(cot.get(k, "")).strip() for k in STEP7}
    if any(not norm_cot[k] for k in STEP7):
        fallback = build_default_cot7(text=text, primary=primary, secondary=secondary)
        for k in STEP7:
            if not norm_cot[k]:
                norm_cot[k] = fallback[k]

    return {
        "mbti_type": mbti,
        "emotion_analysis": norm_emo,
        "cot_reasoning_chain_v2": norm_cot,
        "final_conclusion": {
            "primary_emotion": primary,
            "secondary_emotions": secondary,
            "emotion_summary": str(final.get("emotion_summary", "")),
            "confidence_score": clamp(float(final.get("confidence_score", 0.8))),
        },
    }


def build_user_prompt(text: str, mbti: str, gt_scores: Dict[str, float]) -> str:
    gt_line = ", ".join([f"{k}={gt_scores[k]:.2f}" for k in EMOTIONS])
    return f"{INSTRUCTION}\n\n文本: {text}\nMBTI: {mbti}\n参考强度标签: {gt_line}"


def make_multitask_rows(base_row: Dict, relabeled_obj: Dict, split: str) -> List[Dict]:
    base_id = str(base_row.get("id"))
    text = str(base_row.get("input", ""))
    mbti = str(relabeled_obj.get("mbti_type", "INFJ")).upper()
    emo = relabeled_obj.get("emotion_analysis", {})
    scores = {e: clamp(float(emo.get(e, {}).get("intensity", 0.0))) for e in EMOTIONS}

    primary = str(relabeled_obj.get("final_conclusion", {}).get("primary_emotion", "")).lower()
    if primary not in EMOTIONS:
        primary = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary = [e for e, s in ranked[1:3] if s > 0 and e != primary]
    cot7 = relabeled_obj.get("cot_reasoning_chain_v2", {})

    common_meta = {
        "split": split,
        "base_id": base_id,
        "mbti": mbti,
        "ground_truth": scores,
        "relabel_v2": True,
    }

    rows: List[Dict] = []
    out_reg = {
        "mbti_type": mbti,
        "emotion_analysis": {e: emo.get(e, {"intensity": scores[e], "evidence": "", "reasoning": ""}) for e in EMOTIONS},
        "cot_reasoning_chain_v2": {k: str(cot7.get(k, "")) for k in STEP7},
        "final_conclusion": {
            "primary_emotion": primary,
            "secondary_emotions": secondary,
            "emotion_summary": str(relabeled_obj.get("final_conclusion", {}).get("emotion_summary", "")),
            "confidence_score": clamp(float(relabeled_obj.get("final_conclusion", {}).get("confidence_score", 0.8))),
        },
    }
    rows.append(
        {
            "id": f"{base_id}::task_emotion_reg",
            "instruction": "[task_emotion_reg] 请对中文社交文本做六维情绪强度分析，输出严格 JSON。",
            "input": text,
            "output": json.dumps(out_reg, ensure_ascii=False),
            "history": [],
            "metadata": {**common_meta, "task": "task_emotion_reg", "task_weight": TASK_WEIGHTS["task_emotion_reg"]},
        }
    )

    out_cls = {"primary_emotion": primary, "secondary_emotions": secondary, "distribution": scores}
    rows.append(
        {
            "id": f"{base_id}::task_primary_cls",
            "instruction": "[task_primary_cls] 请在六类情绪 angry/fear/happy/neutral/sad/surprise 中进行主情绪分类，并输出严格 JSON。",
            "input": text,
            "output": json.dumps(out_cls, ensure_ascii=False),
            "history": [],
            "metadata": {**common_meta, "task": "task_primary_cls", "task_weight": TASK_WEIGHTS["task_primary_cls"]},
        }
    )

    rows.append(
        {
            "id": f"{base_id}::task_mbti_pred",
            "instruction": "[task_mbti_pred] 请根据文本预测 MBTI 类型，并输出严格 JSON。",
            "input": text,
            "output": json.dumps({"mbti_type": mbti}, ensure_ascii=False),
            "history": [],
            "metadata": {**common_meta, "task": "task_mbti_pred", "task_weight": TASK_WEIGHTS["task_mbti_pred"]},
        }
    )

    out_cot = {
        "mbti_type": mbti,
        "cot_reasoning_chain_v2": {k: str(cot7.get(k, "")) for k in STEP7},
        "target_scores": scores,
        "primary_emotion": primary,
    }
    rows.append(
        {
            "id": f"{base_id}::task_cot_gen",
            "instruction": "[task_cot_gen] 请生成包含 MBTI 调制信息的 7 步情绪推理链，并输出严格 JSON。",
            "input": text,
            "output": json.dumps(out_cot, ensure_ascii=False),
            "history": [],
            "metadata": {**common_meta, "task": "task_cot_gen", "task_weight": TASK_WEIGHTS["task_cot_gen"]},
        }
    )
    return rows


def infer_batch_texts(
    engine,
    infer_requests: List[InferRequest],
    request_config: RequestConfig,
    stream: bool,
    use_async: bool,
) -> List[str]:
    metric = InferStats()
    if use_async and not stream:
        async def _run_async_batch():
            tasks = [engine.infer_async(req, request_config) for req in infer_requests]
            return await asyncio.gather(*tasks)
        resp_list = asyncio.run(_run_async_batch())
    else:
        resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    texts: List[str] = []
    if not stream:
        for resp in resp_list:
            texts.append(resp.choices[0].message.content)
    else:
        for gen in resp_list:
            parts: List[str] = []
            for resp in gen:
                if resp is None:
                    continue
                delta = resp.choices[0].delta.content
                if delta:
                    parts.append(delta)
            texts.append("".join(parts))
    return texts


def _base_id_from_row_obj(row_obj: Dict) -> str:
    meta = row_obj.get("metadata", {}) if isinstance(row_obj, dict) else {}
    base_id = str(meta.get("base_id", "")).strip()
    if base_id:
        return base_id
    rid = str(row_obj.get("id", "")).strip()
    if "::" in rid:
        return rid.split("::", 1)[0]
    return ""


def _task_from_row_obj(row_obj: Dict) -> str:
    meta = row_obj.get("metadata", {}) if isinstance(row_obj, dict) else {}
    task = str(meta.get("task", "")).strip()
    if task:
        return task
    rid = str(row_obj.get("id", "")).strip()
    if "::" in rid:
        return rid.split("::", 1)[1]
    return ""


def load_completed_base_ids_from_jsonl(jsonl_path: str) -> Set[str]:
    if not os.path.exists(jsonl_path):
        return set()

    task_map: Dict[str, Set[str]] = defaultdict(set)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            base_id = _base_id_from_row_obj(obj)
            task = _task_from_row_obj(obj)
            if base_id and task in TASK_NAMES:
                task_map[base_id].add(task)

    return {bid for bid, tasks in task_map.items() if tasks == TASK_NAMES}


def compact_jsonl_keep_complete_bases(jsonl_path: str) -> Tuple[int, int]:
    """Keep only rows whose base_id already has all 4 tasks.
    Returns: (kept_lines, removed_lines)
    """
    if not os.path.exists(jsonl_path):
        return 0, 0

    task_map: Dict[str, Set[str]] = defaultdict(set)
    rows: List[Tuple[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            base_id = _base_id_from_row_obj(obj)
            task = _task_from_row_obj(obj)
            if not base_id or task not in TASK_NAMES:
                continue
            task_map[base_id].add(task)
            rows.append((base_id, line))

    complete_bases = {bid for bid, tasks in task_map.items() if tasks == TASK_NAMES}
    tmp_path = jsonl_path + ".tmp"
    kept = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for base_id, line in rows:
            if base_id in complete_bases:
                f.write(line + "\n")
                kept += 1
    os.replace(tmp_path, jsonl_path)
    return kept, len(rows) - kept


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def jsonl_to_json_array(jsonl_path: str, json_path: str) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(json_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if not first:
                fout.write(",\n")
            fout.write(line)
            first = False
        fout.write("\n]\n")


def relabel_split(
    engine,
    in_path: str,
    out_path: str,
    out_jsonl_path: str,
    failed_path: str,
    split: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    sleep_between_retries: float,
    stream: bool,
    use_async: bool,
    resume: bool,
    export_json: bool,
) -> Dict[str, int]:
    with open(in_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    total = len(items)
    if resume and os.path.exists(out_jsonl_path):
        kept, removed = compact_jsonl_keep_complete_bases(out_jsonl_path)
        if removed > 0:
            print(f"[{split}] resume compacted jsonl: kept={kept}, removed_incomplete={removed}")

    completed_base_ids = load_completed_base_ids_from_jsonl(out_jsonl_path) if resume else set()
    base_id_set = {str(x.get("id")) for x in items}
    resumed_done = len(base_id_set & completed_base_ids)
    pending_items = [x for x in items if str(x.get("id")) not in completed_base_ids]
    pending_total = len(pending_items)

    if resume and os.path.exists(out_jsonl_path):
        print(f"[{split}] resume enabled, completed={resumed_done}, pending={pending_total}")
    elif not resume and os.path.exists(out_jsonl_path):
        os.remove(out_jsonl_path)
        print(f"[{split}] resume disabled, removed existing: {out_jsonl_path}")

    relabel_success = 0
    failed_count = 0
    expanded_new = 0
    request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream)

    for start in range(0, pending_total, batch_size):
        batch = pending_items[start : start + batch_size]
        infer_requests: List[InferRequest] = []
        cache: List[Tuple[Dict, str, Dict[str, float], str]] = []

        for row in batch:
            meta = row.get("metadata", {})
            mbti = str(meta.get("mbti", "INFJ")).upper()
            gt = meta.get("ground_truth", {})
            gt_scores = {e: float(gt.get(e, 0.0)) for e in EMOTIONS}
            text = str(row.get("input", "")).strip()

            infer_requests.append(
                InferRequest(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(text, mbti, gt_scores)},
                    ]
                )
            )
            cache.append((row, mbti, gt_scores, text))

        raw_texts = None
        last_err = None
        for _ in range(retries + 1):
            try:
                raw_texts = infer_batch_texts(
                    engine,
                    infer_requests,
                    request_config,
                    stream=stream,
                    use_async=use_async,
                )
                break
            except Exception as exc:
                last_err = str(exc)
                time.sleep(sleep_between_retries)

        if raw_texts is None:
            batch_rows: List[Dict] = []
            for row, mbti, gt_scores, text in cache:
                failed_count += 1
                with open(failed_path, "a", encoding="utf-8") as ff:
                    ff.write(json.dumps({"id": row.get("id"), "error": last_err or "infer_failed"}, ensure_ascii=False) + "\n")
                relabeled_obj = normalize_relabel_obj({}, mbti, gt_scores, text)
                batch_rows.extend(make_multitask_rows(row, relabeled_obj, split))
            with open(out_jsonl_path, "a", encoding="utf-8") as fo:
                for x in batch_rows:
                    fo.write(json.dumps(x, ensure_ascii=False) + "\n")
            expanded_new += len(batch_rows)
            continue

        batch_rows: List[Dict] = []
        for (row, mbti, gt_scores, text), raw in zip(cache, raw_texts):
            try:
                obj = extract_json_obj(raw)
                relabeled_obj = normalize_relabel_obj(obj, mbti, gt_scores, text)
                relabel_success += 1
            except Exception as exc:
                failed_count += 1
                with open(failed_path, "a", encoding="utf-8") as ff:
                    ff.write(json.dumps({"id": row.get("id"), "error": str(exc)}, ensure_ascii=False) + "\n")
                relabeled_obj = normalize_relabel_obj({}, mbti, gt_scores, text)
            batch_rows.extend(make_multitask_rows(row, relabeled_obj, split))

        with open(out_jsonl_path, "a", encoding="utf-8") as fo:
            for x in batch_rows:
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
        expanded_new += len(batch_rows)

        if (start // batch_size + 1) % 20 == 0:
            print(f"[{split}] {min(start + batch_size, pending_total)}/{pending_total} (pending)")

    if export_json:
        jsonl_to_json_array(out_jsonl_path, out_path)

    expanded_total = count_jsonl_lines(out_jsonl_path)
    print(
        f"done: {in_path} -> {out_jsonl_path}, base_total={total}, pending={pending_total}, "
        f"expanded_new={expanded_new}, expanded_total={expanded_total}, "
        f"relabel_success={relabel_success}, failed={failed_count}"
    )
    return {
        "base_total": total,
        "base_skipped_resume": resumed_done,
        "base_pending": pending_total,
        "expanded_new": expanded_new,
        "expanded_total": expanded_total,
        "relabel_success": relabel_success,
        "failed": failed_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data/dataset")
    parser.add_argument("--output_dir", default="./data/dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--infer_backend", default="pt", choices=["pt", "vllm", "lmdeploy"])
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true", help="Enable stream mode infer. Slower for large relabel jobs.")
    parser.add_argument("--use_async", action="store_true", help="Use engine.infer_async for non-stream inference.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing *_v2_relabel.jsonl")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume and overwrite existing jsonl")
    parser.add_argument("--export_json", action="store_true", default=True, help="Export jsonl to json array after each split")
    parser.add_argument("--no_export_json", action="store_true", help="Only keep jsonl, do not export json")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep_between_retries", type=float, default=0.8)
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False
    if args.no_export_json:
        args.export_json = False

    os.makedirs(args.output_dir, exist_ok=True)
    failed_path = os.path.join(args.output_dir, "relabel_failed.jsonl")
    if os.path.exists(failed_path) and not args.resume:
        os.remove(failed_path)

    engine = create_engine(
        model=args.model,
        infer_backend=args.infer_backend,
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_model_len,
    )

    split_stats = {}
    for split in ["train", "dev", "test"]:
        out_json_path = os.path.join(args.output_dir, f"{split}_v2_relabel.json")
        out_jsonl_path = os.path.join(args.output_dir, f"{split}_v2_relabel.jsonl")
        split_stats[split] = relabel_split(
            engine=engine,
            in_path=os.path.join(args.input_dir, f"{split}.json"),
            out_path=out_json_path,
            out_jsonl_path=out_jsonl_path,
            failed_path=failed_path,
            split=split,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            retries=args.retries,
            sleep_between_retries=args.sleep_between_retries,
            stream=args.stream,
            use_async=args.use_async,
            resume=args.resume,
            export_json=args.export_json,
        )

    info = {
        "cmacd_multitask_train_v2_relabel": {
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
            "file_name": "train_v2_relabel.json",
        },
        "cmacd_multitask_dev_v2_relabel": {
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
            "file_name": "dev_v2_relabel.json",
        },
        "cmacd_multitask_test_v2_relabel": {
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
            "file_name": "test_v2_relabel.json",
        },
        "stats": {
            "infer_backend": args.infer_backend,
            "stream": args.stream,
            "use_async": args.use_async,
            "resume": args.resume,
            "export_json": args.export_json,
            "model": args.model,
            "batch_size": args.batch_size,
            "max_batch_size": args.max_batch_size,
            "task_weights": TASK_WEIGHTS,
            "split_stats": split_stats,
        },
    }
    with open(os.path.join(args.output_dir, "dataset_info_v2_relabel.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("All splits done.")


if __name__ == "__main__":
    main()
