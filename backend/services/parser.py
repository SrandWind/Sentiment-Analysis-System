# -*- coding: utf-8 -*-
"""
Parser for extracting structured data from model output
Multi-source confidence calculation with VAD, uncertainty, and evidence
"""
import re
import json
import math
from typing import Dict, Any, List, Optional


EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]


def normalize_emotion_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """
    将raw_intensity_scores归一化为target_scores（概率分布）
    :param raw_scores: 模型输出的原始强度分，无总和限制
    :return: 归一化后的概率分布，总和为1.0
    """
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


def clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


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
    
    valence = vad_dimensions.get("valence", "neutral")
    if isinstance(valence, str):
        if valence == "positive":
            expected_high = ["happy"]
            expected_low = ["sad", "angry", "fear"]
        elif valence == "negative":
            expected_high = ["sad", "angry", "fear"]
            expected_low = ["happy"]
        else:
            return 0.5
    else:
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
    
    candidate = text.strip()
    
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", candidate, re.I)
    if m:
        candidate = m.group(1)
    
    s, e = candidate.find("{"), candidate.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(candidate[s: e + 1])
            _parse_json_object(obj, result)
            result["json_ok"] = True
            return result
        except json.JSONDecodeError:
            pass
    
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


NEGATION_WORDS = ["不", "没", "非", "无", "别", "并未", "绝非", "何曾", "不再", "毫无", "从未", "别再", "未曾"]
SARCASM_KEYWORDS = ["比XX还专业", "简直是个笑话", "谁买谁后悔", "绝了", "可真行", "简直绝了"]


def validate_emotion_output(model_output: str, input_text: str) -> tuple[bool, dict, str]:
    """
    校验模型输出是否符合规则
    :param model_output: 模型返回的原始输出字符串
    :param input_text: 原始输入文本
    :return: (校验是否通过, 解析后的JSON, 错误信息)
    """
    output_data = None
    
    # 1. 格式校验：提取JSON并解析
    try:
        json_str = model_output
        if "```json" in model_output:
            json_str = model_output.split("```json")[1].split("```")[0].strip()
        elif "```" in model_output:
            json_str = model_output.split("```")[1].split("```")[0].strip()
        
        s, e = json_str.find("{"), json_str.rfind("}")
        if s != -1 and e != -1 and e > s:
            json_str = json_str[s: e + 1]
        
        output_data = json.loads(json_str)
    except Exception as e:
        return False, {}, f"JSON格式解析失败：{str(e)}"

    # 2. 必填字段校验
    required_fields = ["cot_reasoning_chain_v2", "raw_intensity_scores", "primary_emotion", "vad_dimensions", "emotion_cause", "uncertainty_level"]
    for field in required_fields:
        if field not in output_data:
            return False, output_data, f"缺失必填字段：{field}"

    # 3. VAD数值范围校验
    vad = output_data.get("vad_dimensions", {})
    for k, v in vad.items():
        if not (0.00 <= float(v) <= 1.00):
            return False, output_data, f"VAD数值越界：{k}={v}，必须在0.00-1.00之间"

    # 4. raw_intensity_scores校验
    raw_scores = output_data.get("raw_intensity_scores", {})
    required_emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
    for emo in required_emotions:
        if emo not in raw_scores:
            return False, output_data, f"raw_intensity_scores缺失情绪：{emo}"
        if not (0.00 <= float(raw_scores[emo]) <= 1.00):
            return False, output_data, f"情绪分数越界：{emo}={raw_scores[emo]}"

    # 4.1 无证据情绪分数校验（raw_intensity_scores无总和限制）
    cot_cot = output_data.get("cot_reasoning_chain_v2", {})
    evidence = {}
    cues_obj = {}
    if isinstance(cot_cot, dict):
        step1 = cot_cot.get("step1_lexical_grounding", {})
        if isinstance(step1, dict):
            evidence = step1.get("evidence", {})
            cues_obj = step1.get("cues", {})
    
    # 4.2 step1 cues结构校验
    required_cue_keys = ["strong_emotion", "sarcasm", "weak_emotion", "neutral"]
    if isinstance(cues_obj, dict):
        for key in required_cue_keys:
            if key not in cues_obj:
                return False, output_data, f"cues缺少必填key：{key}，必须包含4个固定key：strong_emotion、sarcasm、weak_emotion、neutral"
            if not isinstance(cues_obj[key], list):
                return False, output_data, f"cues.{key}必须为数组类型"
    elif cues_obj:
        return False, output_data, "cues必须是对象类型，包含4个固定key：strong_emotion、sarcasm、weak_emotion、neutral"
    else:
        return False, output_data, "cues不能为空，必须包含4个固定key：strong_emotion、sarcasm、weak_emotion、neutral"
    
    if isinstance(evidence, dict):
        for emo in required_emotions:
            emo_evidence = evidence.get(emo, [])
            if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                if float(raw_scores[emo]) > 0.03:
                    return False, output_data, f"无证据情绪分数超标：{emo}={raw_scores[emo]}，无证据时不得超过0.03"
            if emo == "happy":
                if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                    if float(raw_scores[emo]) != 0.00:
                        return False, output_data, f"happy无证据时必须为0.00，当前为{raw_scores[emo]}"

    # 5. 主情绪校验（基于raw_intensity_scores）
    max_score = max(float(raw_scores[emo]) for emo in required_emotions)
    max_emos = [emo for emo, v in raw_scores.items() if abs(float(v) - max_score) < 0.001]
    if output_data.get("primary_emotion") not in max_emos:
        return False, output_data, f"主情绪错误：primary_emotion={output_data['primary_emotion']}，但最高分情绪为{max_emos}"

    # 6. 否定词漏检兜底校验
    has_negation = any(word in input_text for word in NEGATION_WORDS)
    negation_found = False
    if isinstance(cot_cot, dict):
        step3 = cot_cot.get("step3_negation_detection", {})
        if isinstance(step3, dict):
            negation_found = step3.get("negations_found", False)
    if has_negation and not negation_found:
        return False, output_data, "否定词漏检：原文包含否定词，但negations_found标记为false"

    # 7. 反讽漏检兜底校验
    has_sarcasm = any(keyword in input_text for keyword in SARCASM_KEYWORDS)
    sarcasm_found = False
    if isinstance(cot_cot, dict):
        step3 = cot_cot.get("step3_negation_detection", {})
        if isinstance(step3, dict):
            sarcasm_found = step3.get("sarcasm_detected", False)
    if has_sarcasm and not sarcasm_found:
        return False, output_data, "反讽漏检：原文包含反讽特征，但sarcasm_detected标记为false"

    # 8. 模型step5自检二次校验：防止模型撒保证书
    step5_data = output_data.get("cot_reasoning_chain_v2", {}).get("step5_consistency_check", {})
    if isinstance(step5_data, dict):
        check_items = step5_data.get("check_items", [])
        check_text = " ".join(check_items) if isinstance(check_items, list) else str(check_items)
        if "无证据的情绪分数未超过0.03" in check_text or "无证据的情绪分数均≤0.03" in check_text:
            for emo in ["sad", "angry", "fear", "surprise", "neutral"]:
                emo_evidence = evidence.get(emo, []) if isinstance(evidence, dict) else []
                if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                    if float(raw_scores.get(emo, 0.0)) > 0.03:
                        return False, output_data, f"step5自检撒谎：模型声称'{emo}'无证据≤0.03，但实际raw_scores.{emo}={raw_scores.get(emo)}，违反规则"

    # 所有校验通过
    return True, output_data, "校验通过，输出完全符合规则"


def force_happy_redline_rule(output_data: dict) -> dict:
    """
    红线规则强制执行：所有无证据情绪的分数规则，代码层面彻底锁死
    - happy无证据时必须为0.00（最高优先级）
    - 其他情绪(sad/angry/fear/surprise/neutral)无证据时必须<=0.03
    :param output_data: 模型解析后的完整输出JSON对象
    :return: 100%合规的修正后输出对象
    """
    try:
        step1 = output_data.get("cot_reasoning_chain_v2", {}).get("step1_lexical_grounding", {})
        evidence = step1.get("evidence", {})
        raw_scores = output_data.get("raw_intensity_scores", {})
        step5 = output_data.get("cot_reasoning_chain_v2", {}).get("step5_consistency_check", {})
        step7 = output_data.get("cot_reasoning_chain_v2", {}).get("step7_faithful_synthesis", {})
        
        adjustment_records = []
        
        # Happy: 无证据时必须为0.00（最高优先级红线）
        happy_evidence = evidence.get("happy", [])
        if isinstance(happy_evidence, list) and len(happy_evidence) == 0:
            original_happy = raw_scores.get("happy", 0.0)
            if original_happy != 0.00:
                raw_scores["happy"] = 0.00
                adjustment_records.append(f"happy: {original_happy:.2f}→0.00(红线规则：happy无证据必须为0.00)")
        
        # 其他情绪: 无证据时必须<=0.03
        other_emotions = ["sad", "angry", "fear", "surprise", "neutral"]
        for emo in other_emotions:
            emo_evidence = evidence.get(emo, [])
            if isinstance(emo_evidence, list) and len(emo_evidence) == 0:
                original_score = raw_scores.get(emo, 0.0)
                if original_score > 0.03:
                    raw_scores[emo] = 0.03
                    adjustment_records.append(f"{emo}: {original_score:.2f}→0.03(规则修正：{emo}无证据不得超过0.03)")
        
        if adjustment_records:
            output_data["raw_intensity_scores"] = raw_scores
            
            adjust_log = "；".join(adjustment_records)
            if step7.get("adjustment_log") == "无分数调整":
                step7["adjustment_log"] = adjust_log
            else:
                step7["adjustment_log"] = step7.get("adjustment_log", "") + "，" + adjust_log
            output_data["cot_reasoning_chain_v2"]["step7_faithful_synthesis"] = step7
            
            check_items = step5.get("check_items", [])
            rule5_correct = f"5. 分数与evidence对应：无证据情绪均已修正为合规分数：{adjust_log}"
            check_updated = False
            for i in range(len(check_items)):
                if "无证据的情绪分数" in check_items[i] or "证据一一对应" in check_items[i]:
                    check_items[i] = rule5_correct
                    check_updated = True
                    break
            if not check_updated:
                check_items.append(rule5_correct)
            
            if "最高优先级红线规则" not in " ".join(check_items):
                check_items.append("8. 红线规则校验：所有无证据情绪分数均已强制修正为合规值，符合规则")
            
            step5["check_items"] = check_items
            step5["vad_consistent"] = True
            output_data["cot_reasoning_chain_v2"]["step5_consistency_check"] = step5
    except Exception:
        pass
    
    return output_data


def get_retry_prompt(original_prompt: str, error_msg: str) -> str:
    """
    生成带错误信息注入的重试prompt，包含绝对强制修正指令
    """
    return f"""【绝对强制修正指令，必须100%执行，无任何例外】
上一次输出校验失败，失败原因：{error_msg}
你必须严格执行以下所有操作，禁止任何变通：
【红线规则（违反直接无效）】
1. happy无证据时必须为0.00，0.01、0.03等任何非零数值都属于严重违规
2. sad/angry/fear/surprise/neutral无证据时不得超过0.03
【修正操作】
3. 逐个检查evidence数组为空的情绪，修正其分数至合规值
4. 修正step5_consistency_check的check_items第5条，如实填写各情绪的修正情况
5. 修正step7的adjustment_log，记录所有分数调整
6. 其他推理内容保持不变
【禁止事项】
7. 禁止声称"无证据情绪分数≤0.03"同时输出超标分数
8. 禁止在step5中撒谎，必须如实反映实际分数

输入文本：{original_prompt}"""
