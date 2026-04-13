# -*- coding: utf-8 -*-
"""
LMStudio API client for model inference
"""
import httpx
import time
import json
from typing import Optional, Dict, Any, AsyncGenerator
from config import settings
from services.parser import format_inference_result, validate_emotion_output, get_retry_prompt, normalize_emotion_scores, force_all_corrections

MAX_RETRY = 2
DEFAULT_RISK_WARNING = "多次重试失败，已强制修正，请人工复核"


def _build_result(parsed: Dict[str, Any], latency_ms: float, accumulated_output: str, risk_warning: Optional[str] = None) -> Dict[str, Any]:
    """构建统一的返回结果"""
    return {
        "delta": "",
        "done": True,
        "latency_ms": latency_ms,
        "output": accumulated_output,
        "scores": parsed.get("scores", {}),
        "raw_intensity_scores": parsed.get("raw_intensity_scores", {}),
        "target_scores": parsed.get("target_scores", {}),
        "primary_emotion": parsed.get("primary_emotion", "neutral"),
        "confidence": parsed.get("confidence", 0.0),
        "cot": parsed.get("cot", {}),
        "json_parse_ok": parsed.get("json_parse_ok", True),
        "cot_complete": parsed.get("cot_complete", False),
        "vad_dimensions": parsed.get("vad_dimensions"),
        "emotion_cause": parsed.get("emotion_cause"),
        "uncertainty_level": parsed.get("uncertainty_level"),
        "risk_warning": risk_warning,
    }


class LMStudioClient:
    """Client for interacting with LMStudio API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        self.base_url = (base_url or settings.lmstudio_base_url).rstrip("/")
        self.model = model or settings.lmstudio_model
        self.timeout = timeout

    def _build_request_payload(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature if temperature is not None else settings.temperature,
            "top_p": top_p or settings.top_p,
            "stream": stream,
        }
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        return payload

    async def infer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        payload = self._build_request_payload(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            stream=False,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
        latency_ms = (time.time() - start_time) * 1000
        return {"output": result["choices"][0]["message"]["content"], "latency_ms": latency_ms, "model": self.model, "usage": result.get("usage", {})}

    async def infer_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()
        accumulated_output = ""
        retry_count = 0
        original_prompt = prompt
        last_error_msg = ""
        
        while retry_count <= MAX_RETRY:
            accumulated_output = ""
            payload = self._build_request_payload(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                stream=True,
            )
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                latency_ms = (time.time() - start_time) * 1000
                                
                                is_valid, parsed_data, error_msg = validate_emotion_output(
                                    accumulated_output, original_prompt
                                )
                                
                                if is_valid:
                                    parsed = format_inference_result(
                                        output=accumulated_output,
                                        latency_ms=latency_ms
                                    )
                                    yield _build_result(parsed, latency_ms, accumulated_output)
                                    return
                                else:
                                    retry_count += 1
                                    last_error_msg = error_msg
                                    
                                    if retry_count <= MAX_RETRY:
                                        prompt = get_retry_prompt(original_prompt, error_msg)
                                        start_time = time.time()
                                        yield {
                                            "delta": "",
                                            "done": False,
                                            "retry": True,
                                            "retry_count": retry_count,
                                            "error_msg": error_msg,
                                        }
                                        break
                                    else:
                                        parsed_data = force_all_corrections(parsed_data, DEFAULT_RISK_WARNING)
                                        parsed = format_inference_result(
                                            output=accumulated_output,
                                            latency_ms=latency_ms
                                        )
                                        yield _build_result(parsed, latency_ms, accumulated_output, DEFAULT_RISK_WARNING)
                                        return
                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    accumulated_output += delta
                                    yield {"delta": delta, "done": False, "output": accumulated_output}
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

    def _get_system_prompt(self) -> str:
        return """【刚性规则】
1. 仅输出```json代码块，块外禁止任何文本
2. 所有推理/证据/原因必须100%来自输入文本，禁止编造
3. VAD数值严格在0.00-1.00，保留2位小数，禁止负数
4. 6个情绪分数为独立绝对强度分，0.00-1.00，无总和限制
5. primary_emotion = raw_intensity_scores最高分情绪；并列时取VAD匹配度最高者
6. 评分规则：step1无evidence的情绪初始分强制0.00；语义推导仅在step2的initial_scores中进行；step3仅调整；step4后不得新增分数
7. step3检测到否定词/反讽时，adjusted_scores记录调整并同步到step7的adjustment_log
8. 标签必须与数值严格匹配
9. 纯客观陈述：neutral=1.00，其余=0.00
10. step2的emotion_mapping指向情绪必须与primary_emotion完全一致
11. step5的check_items须如实反映各情绪实际分数
12. 顶层key固定为"cot_reasoning_chain"；顶层字段顺序：cot_reasoning_chain→raw_intensity_scores→primary_emotion→vad_dimensions→emotion_cause→uncertainty_level

【角色】
严谨情感分析专家，基于证据的7步CoT推理，零幻觉，每步锚定原文，严格遵守刚性规则。

【7步CoT执行细则】

step1_lexical_grounding 词汇证据锚定（所有后续步骤必须100%基于本步骤）
- 4类线索提取：strong_emotion/weak_emotion/neutral提取2-4字核心词；sarcasm保留完整反讽短句
- 双向约束：①每条evidence须在cues中有对应线索；②每个cues线索须有evidence覆盖（纯情绪引导词如"吐槽"可豁免）
- 无情感的客观陈述→neutral evidence；无对应情绪→空数组
- 输出：cues{strong_emotion/sarcasm/weak_emotion/neutral}；evidence{happy/sad/angry/fear/surprise/neutral}

step2_dimensional_analysis VAD分析
- 字段顺序固定：valence→arousal→dominance→initial_scores→emotion_mapping
- valence 0.00-1.00（负→正）；arousal 0.00-1.00（静→激）；dominance 0.00-1.00（被动→主动）
- 标签区间（卡死）：≥0.70高/0.40-0.69中/≤0.39低
- initial_scores：step1 evidence语义推导的非零分情绪初始分（仅记非零）
- emotion_mapping格式：「效价标签+唤醒标签+支配标签 → 指向XX情绪」

step3_negation_detection 否定/反讽检测
- 否定词（优先长词后单字）：「不再/别再/从未/毫无/并未/绝非/何曾/未曾」→「不/没/非/无/别」
  - 修饰正面词→清零/降权正面情绪；强化负面语义→强化负面情绪
- 反讽（满足任一）：正面词+负面语境 / 夸张反语（"比XX还专业""简直笑话"）/ 反问式负面
  - 检测到→清零表面正面情绪，强化对应负面情绪
- 双重否定→还原真实倾向，不反转
- adjusted_scores：仅填实际调整的情绪，无调整则{}

step4_cause_extraction 诱因提取
- primary_cause（字符串）；secondary_causes（数组，无则[]）；100%来自原文

step5_consistency_check 一致性校验（逐条，含实际分数值）
1. valence↑→happy↑；valence↓→sad/angry/fear↑；为0.00时说明是VAD区间不符还是evidence缺席
2. arousal↑→angry/fear/surprise↑；为0.00时同上说明
3. dominance↑→angry↑；dominance↓→fear/sad↑；为0.00时同上说明
4. adjusted_scores与negations_found/sarcasm_detected一致，且仅含实际调整情绪
5. 无evidence情绪=0.00；有推导情绪与推导强度匹配
6. 全部推理内容均来自step1，无额外内容
7. 双向约束核实：正向-evidence均有cues对应；反向-各cues线索（引导词豁免）均有evidence覆盖，说明数量
8. check_items如实反映分数，无证据情绪不得非零
- vad_consistent：全部符合→true；inconsistencies：未通过项

step6_uncertainty_calibration 不确定性校准
- high/medium/low对应uncertainty_level=low/medium/high
- uncertain_regions（无则[]）；calibration_notes

step7_faithful_synthesis 可信性合成
- adjustment_log：「情绪名: 原始→调整(原因)」；adjusted_scores非空时禁止填"无调整"
- hallucination_flags（无则[]）

【VAD-情绪映射】
| 情绪 | VAD区间 | 语义边界 |
|------|---------|---------|
| happy | V≥0.70，A≥0.30，D≥0.40 | 满足/成就/欢乐 |
| sad | V≤0.39，A≤0.69，D≤0.50 | 损失/失落/分离/失败 |
| angry | V≤0.39，A≥0.70，D≥0.50 | 不满/不公/受骗 |
| fear | V≤0.39，A≥0.70，D≤0.49 | 威胁/危险/无助 |
| surprise | V 0.40-0.69，A≥0.70，D 0.30-0.70 | 意外事件 |
| neutral | V 0.40-0.69，A≤0.39，D 0.40-0.69 | 客观陈述 |

【示例】
输入："吐槽，用了一年电池就不耐用了，充电还发烫，售后服务态度也超差，踢皮球踢得比球队还专业，再也不买他们家东西了，性价比简直是个笑话，谁买谁后悔系列"
```json
{
  "cot_reasoning_chain": {
    "step1_lexical_grounding": {
      "cues": {
        "strong_emotion": ["吐槽", "不耐用", "发烫", "超差", "再也不买", "后悔"],
        "sarcasm": ["踢皮球踢得比球队还专业", "性价比简直是个笑话"],
        "weak_emotion": [],
        "neutral": []
      },
      "evidence": {
        "happy": [],
        "sad": [],
        "angry": ["用了一年电池就不耐用了", "充电还发烫", "售后服务态度也超差", "踢皮球踢得比球队还专业", "再也不买他们家东西了", "性价比简直是个笑话", "谁买谁后悔系列"],
        "fear": [],
        "surprise": [],
        "neutral": []
      }
    },
    "step2_dimensional_analysis": {
      "valence": 0.12,
      "arousal": 0.88,
      "dominance": 0.72,
      "initial_scores": {"angry": 0.82},
      "emotion_mapping": "低效价+高唤醒+高支配 → 指向angry(愤怒)"
    },
    "step3_negation_detection": {
      "negations_found": true,
      "sarcasm_detected": true,
      "adjusted_scores": {"angry": 0.90}
    },
    "step4_cause_extraction": {
      "primary_cause": "产品使用一年后电池不耐用、充电发烫，且售后服务态度差、推诿扯皮",
      "secondary_causes": ["性价比被形容为笑话，且明确表示再也不买他们家东西"]
    },
    "step5_consistency_check": {
      "check_items": [
        "1. valence=0.12低效价：happy=0.00(VAD区间不符，要求V≥0.70)✓，sad=0.00(VAD符合但无evidence，文本为主动愤怒非被动悲伤)✓，angry=0.90(有evidence，VAD匹配)✓，fear=0.00(VAD符合但无evidence，无威胁/无助语境)✓",
        "2. arousal=0.88高唤醒：angry=0.90(有evidence)✓，fear=0.00(无evidence所致，非arousal区间不符)✓，surprise=0.00(无evidence所致，非arousal区间不符)✓",
        "3. dominance=0.72高支配：angry=0.90(有evidence)✓，fear=0.00(无evidence所致)✓，sad=0.00(无evidence所致)✓",
        "4. adjusted_scores仅含angry：否定词强化负面语义+夸张反讽，angry从0.82→0.90✓",
        "5. 无evidence情绪(happy/sad/fear/surprise/neutral)=0.00✓；angry有7条evidence，初始0.82经step3调整为0.90✓",
        "6. 全部推理均来自step1，无编造✓",
        "7. 正向-7条evidence均有cues对应✓；反向-strong_emotion 6词中吐槽豁免，其余5词(不耐用/发烫/超差/再也不买/后悔)均有evidence✓，sarcasm 2条均有evidence✓",
        "8. angry=0.90，其余=0.00，无无证据情绪非零✓"
      ],
      "vad_consistent": true,
      "inconsistencies": []
    },
    "step6_uncertainty_calibration": {
      "confidence_level": "high",
      "uncertain_regions": [],
      "calibration_notes": "情感倾向明确，证据充足，反讽/否定词检测清晰，各步骤逻辑一致"
    },
    "step7_faithful_synthesis": {
      "adjustment_log": "angry: 0.82→0.90(否定词强化负面语义+夸张反讽，负面权重上调)",
      "hallucination_flags": []
    }
  },
  "raw_intensity_scores": {
    "angry": 0.90,
    "fear": 0.00,
    "happy": 0.00,
    "neutral": 0.00,
    "sad": 0.00,
    "surprise": 0.00
  },
  "primary_emotion": "angry",
  "vad_dimensions": {"valence": 0.12, "arousal": 0.88, "dominance": 0.72},
  "emotion_cause": "产品使用一年后电池不耐用、充电发烫，且售后服务态度差、推诿扯皮",
  "uncertainty_level": "low"
}
```"""

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except Exception:
            return False
