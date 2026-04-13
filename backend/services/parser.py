# -*- coding: utf-8 -*-
"""
Parser for extracting structured data from model output
Multi-source confidence calculation with VAD, uncertainty, and evidence
"""
import re
import json
import math
from typing import Dict, Any, List, Optional, Tuple

try:
    import json5
except ImportError:
    json5 = None

EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

VAD_STEP_KEYS = [
    "step1_lexical_grounding",
    "step2_dimensional_analysis",
    "step3_negation_detection",
    "step4_cause_extraction",
    "step5_consistency_check",
    "step6_uncertainty_calibration",
    "step7_faithful_synthesis",
]

VAD_EMOTION_MAP = {
    "positive": {"happy": 1.0, "surprise": 0.3},
    "negative": {"sad": 1.0, "angry": 0.8, "fear": 0.7},
    "neutral": {"neutral": 1.0},
}

POSITIVE_WORDS = ["happy", "joy", "great", "good", "love", "excellent", "wonderful", "开心", "高兴", "快乐", "喜欢", "棒", "好"]
NEGATIVE_WORDS = ["sad", "angry", "fear", "hate", "bad", "terrible", "awful", "痛苦", "难过", "伤心", "生气", "害怕", "讨厌", "糟"]

NEGATION_WORDS = ["并未", "绝非", "何曾", "不再", "毫无", "从未", "别再", "未曾", "不买了", "不会再", "不想再", "不用了", "不再买"]
SARCASM_KEYWORDS = ["比XX还专业", "简直是个笑话", "谁买谁后悔", "绝了", "可真行", "简直绝了"]


def clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def parse_json_safely(raw_str: str) -> Optional[dict]:
    """安全解析JSON，兼容json5小瑕疵，支持正则兜底提取"""
    raw_str = raw_str.strip()
    
    if not raw_str:
        return None
    
    content = raw_str
    
    if "```" in content:
        start = content.find("```")
        end = content.rfind("```")
        if start != -1 and end != -1 and end > start:
            inner = content[start + 3:end]
            if inner.startswith("json"):
                inner = inner[4:].strip()
            content = inner.strip()
    
    if not content:
        return None
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    if json5:
        try:
            return json5.loads(content)
        except Exception:
            pass
    
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_candidate = content[brace_start:brace_end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
        if json5:
            try:
                return json5.loads(json_candidate)
            except Exception:
                pass
    
    return None
    return None


def force_no_evidence_zero_score(data: dict) -> dict:
    """无evidence的情绪强制0分"""
    if "cot_reasoning_chain" not in data or "step1_lexical_grounding" not in data["cot_reasoning_chain"]:
        return data
    
    evidence = data["cot_reasoning_chain"]["step1_lexical_grounding"].get("evidence", {})
    raw_scores = data.get("raw_intensity_scores", {})
    
    for emotion in EMOTIONS:
        emo_evidence = evidence.get(emotion, [])
        if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
            if raw_scores.get(emotion, 0.0) != 0.00:
                raw_scores[emotion] = 0.00
    
    data["raw_intensity_scores"] = raw_scores
    return data


def force_neutral_objective_text(data: dict) -> dict:
    """纯客观陈述兜底：cues全为neutral/空 + evidence全为neutral/空 → neutral=1.00，其余=0.00"""
    if "cot_reasoning_chain" not in data or "step1_lexical_grounding" not in data["cot_reasoning_chain"]:
        return data
    
    step1 = data["cot_reasoning_chain"]["step1_lexical_grounding"]
    cues = step1.get("cues", {})
    evidence = step1.get("evidence", {})
    
    is_cues_neutral = (
        not cues.get("strong_emotion", []) and
        not cues.get("sarcasm", []) and
        not cues.get("weak_emotion", [])
    )
    is_evidence_neutral = (
        not evidence.get("happy", []) and
        not evidence.get("sad", []) and
        not evidence.get("angry", []) and
        not evidence.get("fear", []) and
        not evidence.get("surprise", [])
    )
    
    if is_cues_neutral and is_evidence_neutral:
        data["raw_intensity_scores"] = {
            "happy": 0.00,
            "sad": 0.00,
            "angry": 0.00,
            "fear": 0.00,
            "surprise": 0.00,
            "neutral": 1.00
        }
        data["primary_emotion"] = "neutral"
        data["vad_dimensions"] = {
            "valence": 0.50,
            "arousal": 0.20,
            "dominance": 0.50
        }
    
    return data


def force_valid_score_format(data: dict) -> dict:
    """强制2位小数，禁止科学计数法"""
    if "raw_intensity_scores" in data:
        for emotion in EMOTIONS:
            if emotion in data["raw_intensity_scores"]:
                data["raw_intensity_scores"][emotion] = round(float(data["raw_intensity_scores"][emotion]), 2)
    
    if "vad_dimensions" in data:
        vad = data["vad_dimensions"]
        for key in ["valence", "arousal", "dominance"]:
            if key in vad:
                vad[key] = round(float(vad[key]), 2)
        data["vad_dimensions"] = vad
    
    return data


def force_primary_emotion_max(data: dict) -> dict:
    """primary_emotion必须为最高分情绪"""
    if "raw_intensity_scores" not in data or "primary_emotion" not in data:
        return data
    
    raw_scores = data["raw_intensity_scores"]
    if not raw_scores:
        return data
    
    max_score = max(raw_scores.values())
    max_emotions = [e for e, v in raw_scores.items() if abs(v - max_score) < 0.001]
    
    if data["primary_emotion"] not in max_emotions:
        data["primary_emotion"] = max_emotions[0] if max_emotions else "neutral"
    
    return data


def force_clamp_scores(data: dict) -> dict:
    """分数截断到0-1范围"""
    if "raw_intensity_scores" in data:
        for emotion in EMOTIONS:
            if emotion in data["raw_intensity_scores"]:
                data["raw_intensity_scores"][emotion] = clamp(data["raw_intensity_scores"][emotion])
    
    if "vad_dimensions" in data:
        vad = data["vad_dimensions"]
        for key in ["valence", "arousal", "dominance"]:
            if key in vad:
                vad[key] = clamp(vad[key])
        data["vad_dimensions"] = vad
    
    return data


def force_all_corrections(data: dict, risk_warning: str = "") -> dict:
    """执行所有强制修正"""
    data = force_clamp_scores(data)
    data = force_no_evidence_zero_score(data)
    data = force_neutral_objective_text(data)
    data = force_valid_score_format(data)
    data = force_primary_emotion_max(data)
    if risk_warning:
        data["risk_warning"] = risk_warning
    return data


def normalize_emotion_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """将raw_intensity_scores归一化为target_scores（概率分布）"""
    if not raw_scores or not any(raw_scores.values()):
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    
    total = sum(max(0.0, float(raw_scores.get(e, 0.0))) for e in EMOTIONS)
    if total <= 0:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    
    normalized = {}
    for e in EMOTIONS:
        raw_val = max(0.0, float(raw_scores.get(e, 0.0)))
        normalized[e] = round(raw_val / total, 4)
    
    current_sum = sum(normalized.values())
    if abs(current_sum - 1.0) > 0.01:
        max_emotion = max(normalized, key=normalized.get)
        normalized[max_emotion] = round(normalized[max_emotion] + (1.0 - current_sum), 4)
    
    return normalized


def calculate_statistical_confidence(scores: Dict[str, float]) -> float:
    probs = [scores.get(e, 0.0) for e in EMOTIONS]
    probs = [max(0.0, min(1.0, p)) for p in probs]
    
    max_prob = max(probs)
    sorted_probs = sorted(probs, reverse=True)
    top_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    
    safe_probs = [max(p, 1e-10) for p in probs]
    entropy = -sum(p * math.log(p) for p in safe_probs)
    max_entropy = math.log(len(probs)) if len(probs) > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    confidence = (1 - normalized_entropy) * 0.4 + top_gap * 0.4 + max_prob * 0.2
    return clamp(confidence)


def calculate_vad_consistency(vad_dimensions: Optional[Dict], target_scores: Dict[str, float]) -> float:
    if not vad_dimensions:
        return 0.5
    
    valence = vad_dimensions.get("valence", 0.5)
    if valence > 0.6:
        expected_high = ["happy"]
        expected_low = ["sad", "angry", "fear"]
    elif valence < 0.4:
        expected_high = ["sad", "angry", "fear"]
        expected_low = ["happy"]
    else:
        return 0.5
    
    max_high = max(target_scores.get(e, 0.0) for e in expected_high) if expected_high else 0.0
    max_low = max(target_scores.get(e, 0.0) for e in expected_low) if expected_low else 0.0
    
    if max_low > max_high:
        return 0.3
    
    consistency = max_high - max_low
    return clamp(0.5 + consistency * 0.5)


def calculate_evidence_strength(cot_data: Optional[Dict]) -> float:
    if not cot_data:
        return 0.5
    
    grounding = cot_data.get("step1_lexical_grounding", {})
    evidence = grounding.get("evidence", {}) if isinstance(grounding, dict) else {}
    
    if not evidence or not isinstance(evidence, dict):
        return 0.5
    
    non_empty_count = sum(1 for e in EMOTIONS if evidence.get(e) and len(evidence.get(e, [])) > 0)
    return clamp(non_empty_count / len(EMOTIONS))


def calculate_uncertainty_adjustment(uncertainty_level: str) -> float:
    mapping = {"low": 1.0, "medium": 0.7, "high": 0.4}
    return mapping.get(uncertainty_level.lower(), 0.7)


def calculate_confidence(
    scores: Dict[str, float],
    vad_dimensions: Optional[Dict] = None,
    uncertainty_level: str = "low",
    cot_data: Optional[Dict] = None
) -> float:
    w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.2
    
    stat_conf = calculate_statistical_confidence(scores)
    vad_conf = calculate_vad_consistency(vad_dimensions, scores)
    uncertainty_adj = calculate_uncertainty_adjustment(uncertainty_level)
    evidence_strength = calculate_evidence_strength(cot_data)
    
    confidence = (w1 * stat_conf + w2 * vad_conf + w3 * uncertainty_adj + w4 * evidence_strength)
    return clamp(confidence)


def parse_model_output(text: str) -> Dict[str, Any]:
    result = {
        "json_ok": False,
        "scores": {e: 0.0 for e in EMOTIONS},
        "target_scores": {e: 0.0 for e in EMOTIONS},
        "cot": {k: "" for k in VAD_STEP_KEYS},
        "primary_emotion": "",
        "vad_dimensions": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
        "emotion_cause": "",
        "uncertainty_level": "medium",
        "raw": text
    }
    
    obj = parse_json_safely(text)
    if obj is not None:
        _parse_json_object(obj, result)
        result["json_ok"] = True
        return result
    
    _parse_with_regex(text, result)
    return result


def _parse_json_object(obj: dict, result: Dict[str, Any]) -> None:
    raw = obj.get("raw_intensity_scores", {})
    if isinstance(raw, dict):
        for e in EMOTIONS:
            value = raw.get(e, 0.0)
            if isinstance(value, dict):
                value = value.get("intensity", 0.0)
            result["scores"][e] = clamp(value)
        
        result["target_scores"] = normalize_emotion_scores(result["scores"])
    
    cot = obj.get("cot_reasoning_chain", {})
    if not cot:
        cot = obj.get("cot_reasoning_chain_v2", {})
    
    if isinstance(cot, dict):
        for key in VAD_STEP_KEYS:
            val = cot.get(key, "")
            if isinstance(val, dict):
                result["cot"][key] = json.dumps(val, ensure_ascii=False)
            else:
                result["cot"][key] = val if isinstance(val, str) else str(val)
    
    if result["target_scores"] and any(result["target_scores"].values()):
        result["primary_emotion"] = max(result["target_scores"], key=result["target_scores"].get)
    
    primary = obj.get("primary_emotion", "")
    if primary and str(primary).lower() in EMOTIONS:
        result["primary_emotion"] = str(primary).lower()
    
    vad = obj.get("vad_dimensions", {})
    if isinstance(vad, dict):
        result["vad_dimensions"] = {
            "valence": float(vad.get("valence", 0.5)),
            "arousal": float(vad.get("arousal", 0.5)),
            "dominance": float(vad.get("dominance", 0.5)),
        }
    
    cause = obj.get("emotion_cause", "")
    if cause:
        result["emotion_cause"] = str(cause)
    
    uncertainty = obj.get("uncertainty_level", "medium")
    if uncertainty in ["low", "medium", "high"]:
        result["uncertainty_level"] = uncertainty.lower()


def _parse_with_regex(text: str, result: Dict[str, Any]) -> None:
    for e in EMOTIONS:
        r = re.search(rf'["\x27]?{e}["\x27]?\s*[:：]\s*([0-9.]+)', text, re.I)
        if r:
            result["target_scores"][e] = clamp(float(r.group(1)))
    
    p = re.search(r'["\x27]?primary_emotion["\x27]?\s*[:：]\s*["\x27]?([a-zA-Z_]+)["\x27]?', text, re.I)
    if p and p.group(1).lower() in EMOTIONS:
        result["primary_emotion"] = p.group(1).lower()
    elif result["target_scores"]:
        result["primary_emotion"] = max(result["target_scores"], key=result["target_scores"].get)
    
    v_match = re.search(r'"valence"\s*[:：]\s*([0-9.]+)', text)
    a_match = re.search(r'"arousal"\s*[:：]\s*([0-9.]+)', text)
    d_match = re.search(r'"dominance"\s*[:：]\s*([0-9.]+)', text)
    if v_match:
        result["vad_dimensions"]["valence"] = float(v_match.group(1))
    if a_match:
        result["vad_dimensions"]["arousal"] = float(a_match.group(1))
    if d_match:
        result["vad_dimensions"]["dominance"] = float(d_match.group(1))
    
    unc = re.search(r'"uncertainty_level"\s*[:：]\s*["\x27](low|medium|high)["\x27]', text, re.I)
    if unc:
        result["uncertainty_level"] = unc.group(1).lower()


def check_cot_complete(cot: Dict[str, str]) -> bool:
    return all(cot.get(k, "").strip() for k in VAD_STEP_KEYS)


def format_inference_result(
    output: str,
    latency_ms: float,
    model_variant: str = "gguf4bit"
) -> Dict[str, Any]:
    parsed = parse_model_output(output)
    
    final_scores = parsed["scores"]
    if not any(final_scores.values()):
        final_scores = parsed["target_scores"]
    
    target_scores = parsed["target_scores"]
    if not any(target_scores.values()):
        target_scores = normalize_emotion_scores(final_scores)
    
    confidence = calculate_confidence(
        scores=target_scores,
        vad_dimensions=parsed["vad_dimensions"],
        uncertainty_level=parsed["uncertainty_level"],
        cot_data=parsed["cot"]
    )
    
    return {
        "output": output,
        "latency_ms": latency_ms,
        "model_variant": model_variant,
        "json_parse_ok": parsed["json_ok"],
        "cot_complete": check_cot_complete(parsed["cot"]),
        "scores": final_scores,
        "raw_intensity_scores": final_scores,
        "target_scores": target_scores,
        "cot": parsed["cot"],
        "primary_emotion": parsed["primary_emotion"],
        "vad_dimensions": parsed["vad_dimensions"],
        "emotion_cause": parsed["emotion_cause"],
        "uncertainty_level": parsed["uncertainty_level"],
        "confidence": confidence
    }


def validate_emotion_output(model_output: str, input_text: str) -> Tuple[bool, dict, str]:
    """校验模型输出是否符合规则"""
    output_data = parse_json_safely(model_output)
    if output_data is None:
        return (False, {}, "JSON_PARSE_ERROR: JSON格式解析失败")
    
    required_top_fields = ["cot_reasoning_chain", "raw_intensity_scores", "primary_emotion", "vad_dimensions", "emotion_cause", "uncertainty_level"]
    missing_fields = [f for f in required_top_fields if f not in output_data]
    if missing_fields:
        return (False, output_data, f"MISSING_FIELD: {', '.join(missing_fields)}")
    
    vad = output_data.get("vad_dimensions", {})
    out_of_range_vad = []
    for k, v in vad.items():
        if not (0.00 <= float(v) <= 1.00):
            out_of_range_vad.append(f"{k}={v}")
    if out_of_range_vad:
        return (False, output_data, f"SCORE_OUT_OF_RANGE: VAD {', '.join(out_of_range_vad)}")
    
    raw_scores = output_data.get("raw_intensity_scores", {})
    out_of_range_scores = []
    for emo in EMOTIONS:
        if emo not in raw_scores:
            return (False, output_data, f"MISSING_FIELD: raw_intensity_scores.{emo}")
        if not (0.00 <= float(raw_scores[emo]) <= 1.00):
            out_of_range_scores.append(f"{emo}={raw_scores[emo]}")
    if out_of_range_scores:
        return (False, output_data, f"SCORE_OUT_OF_RANGE: {', '.join(out_of_range_scores)}")
    
    cot_cot = output_data.get("cot_reasoning_chain", {})
    if not cot_cot:
        cot_cot = output_data.get("cot_reasoning_chain_v2", {})
    
    evidence = {}
    cues_obj = {}
    if isinstance(cot_cot, dict):
        step1 = cot_cot.get("step1_lexical_grounding", {})
        if isinstance(step1, dict):
            evidence = step1.get("evidence", {})
            cues_obj = step1.get("cues", {})
    
    required_cue_keys = ["strong_emotion", "sarcasm", "weak_emotion", "neutral"]
    if isinstance(cues_obj, dict):
        for key in required_cue_keys:
            if key not in cues_obj:
                return (False, output_data, f"MISSING_FIELD: cues.{key}")
    else:
        return (False, output_data, "MISSING_FIELD: cues结构异常")
    
    no_evidence_scores = []
    if isinstance(evidence, dict):
        for emo in EMOTIONS:
            emo_evidence = evidence.get(emo, [])
            if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                if float(raw_scores.get(emo, 0.0)) != 0.00:
                    no_evidence_scores.append(f"{emo}={raw_scores.get(emo)}")
    
    if no_evidence_scores:
        return (False, output_data, f"NO_EVIDENCE_SCORE: {', '.join(no_evidence_scores)}")
    
    step2 = cot_cot.get("step2_dimensional_analysis", {}) if isinstance(cot_cot, dict) else {}
    initial_scores = step2.get("initial_scores", {}) if isinstance(step2, dict) else {}
    if isinstance(initial_scores, dict) and initial_scores:
        initial_no_evidence = []
        for emo, score in initial_scores.items():
            if emo in EMOTIONS and float(score) > 0:
                emo_evidence = evidence.get(emo, []) if isinstance(evidence, dict) else []
                if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                    initial_no_evidence.append(f"{emo}={score}")
        if initial_no_evidence:
            return (False, output_data, f"STEP2_INITIAL_NO_EVIDENCE: step2.initial_scores中{', '.join(initial_no_evidence)}在step1无evidence")
    
    max_score = max(float(raw_scores[emo]) for emo in EMOTIONS)
    max_emos = [emo for emo, v in raw_scores.items() if abs(float(v) - max_score) < 0.001]
    primary_emo = output_data.get("primary_emotion", "")
    if primary_emo not in max_emos:
        return (False, output_data, f"PRIMARY_EMOTION_ERROR: 当前{primary_emo}应为{max_emos[0] if max_emos else 'unknown'}")
    
    has_negation = any(word in input_text for word in NEGATION_WORDS)
    has_sarcasm = any(keyword in input_text for keyword in SARCASM_KEYWORDS)
    
    if isinstance(cot_cot, dict):
        step3 = cot_cot.get("step3_negation_detection", {})
        if isinstance(step3, dict):
            negation_found = step3.get("negations_found", False)
            sarcasm_found = step3.get("sarcasm_detected", False)
            
            if has_negation and not negation_found:
                return (False, output_data, "STEP3_NEGATION_MISSED: 否定词漏检")
            
            if has_sarcasm and not sarcasm_found:
                return (False, output_data, "STEP3_SARCASM_MISSED: 反讽漏检")
            
            if (negation_found or sarcasm_found):
                adjusted_scores = step3.get("adjusted_scores", {})
                step7 = cot_cot.get("step7_faithful_synthesis", {})
                adjustment_log = step7.get("adjustment_log", "") if isinstance(step7, dict) else ""
                
                if not adjusted_scores and adjustment_log == "":
                    return (False, output_data, "STEP3_NO_ADJUSTMENT: 检测到否定/反讽但无adjustment记录")
    
    step5_data = cot_cot.get("step5_consistency_check", {}) if isinstance(cot_cot, dict) else {}
    if isinstance(step5_data, dict):
        check_items = step5_data.get("check_items", [])
        check_text = " ".join(check_items) if isinstance(check_items, list) else str(check_items)
        
        if "无证据" in check_text and "=0.00" in check_text:
            for emo in EMOTIONS:
                emo_evidence = evidence.get(emo, []) if isinstance(evidence, dict) else []
                if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                    if float(raw_scores.get(emo, 0.0)) != 0.00:
                        return (False, output_data, f"STEP5_LIE: 声称{emo}无证据=0.00但实际={raw_scores.get(emo)}")
    
    return (True, output_data, "VALID")


def get_retry_prompt(original_prompt: str, error_msg: str) -> str:
    """分场景精准纠错重试提示词"""
    if error_msg.startswith("JSON_PARSE_ERROR"):
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 输出标准合法JSON，禁止多余逗号、单引号、尾随换行
【错误说明】
上一次输出JSON格式错误，无法解析。
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("MISSING_FIELD:"):
        missing = error_msg.replace("MISSING_FIELD:", "").strip()
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 必须包含字段：cot_reasoning_chain, raw_intensity_scores, primary_emotion, vad_dimensions, emotion_cause, uncertainty_level
【错误说明】
上一次输出缺少必填字段：{missing}
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("SCORE_OUT_OF_RANGE:"):
        items = error_msg.replace("SCORE_OUT_OF_RANGE:", "").strip()
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 所有VAD值、情绪分数必须在0.00-1.00之间，保留2位小数
【错误说明】
上一次输出存在分数越界：{items}
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("NO_EVIDENCE_SCORE:"):
        items = error_msg.replace("NO_EVIDENCE_SCORE:", "").strip()
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 无原文证据的情绪，分数强制为0.00
【错误说明】
上一次输出中，{items}，违反无证据=0分规则
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("STEP2_INITIAL_NO_EVIDENCE:"):
        items = error_msg.replace("STEP2_INITIAL_NO_EVIDENCE:", "").strip()
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 非零分情绪必须在step1有对应evidence（原文分句）
3. step2语义推导发现的新情绪，必须同步补全step1的evidence（原文分句）+ cues（线索词）
4. step1补全后才能在step2.initial_scores中给分
【错误说明】
上一次输出中，{items}，在step1无对应evidence
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("PRIMARY_EMOTION_ERROR:"):
        current = error_msg.split("当前")[1].split("应为")[0].strip() if "当前" in error_msg else "unknown"
        highest = error_msg.split("应为")[1].strip() if "应为" in error_msg else "unknown"
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. primary_emotion必须是raw_intensity_scores中的最高分情绪
【错误说明】
上一次输出的primary_emotion为{current}，但最高分是{highest}
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("STEP5_LIE:"):
        items = error_msg.replace("STEP5_LIE:", "").strip()
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 所有推理必须100%来自原文，严格遵守无evidence=0分规则
【错误说明】
上一次输出的step5自查声称合规，但实际存在违规：{items}
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("STEP3_NEGATION_MISSED") or error_msg.startswith("STEP3_SARCASM_MISSED"):
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. step3必须检测否定词和反讽，检测到时adjusted_scores必须有记录
【错误说明】
上一次输出漏检了否定词或反讽
【待分析文本】
{original_prompt}"""
    
    elif error_msg.startswith("STEP3_NO_ADJUSTMENT"):
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 检测到否定词或反讽时，adjusted_scores必须有对应调整记录
【错误说明】
上一次输出检测到否定/反讽但adjusted_scores为空
【待分析文本】
{original_prompt}"""
    
    else:
        return f"""【绝对规则】
1. 仅输出```json代码块，块外无任何文本
2. 所有分数0.00-1.00，无evidence=0分
【错误说明】
上一次输出校验失败：{error_msg}
【待分析文本】
{original_prompt}"""
