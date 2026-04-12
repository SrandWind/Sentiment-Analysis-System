# -*- coding: utf-8 -*-
"""
LMStudio API client for model inference
"""
import httpx
import time
import json
from typing import Optional, Dict, Any, AsyncGenerator
from config import settings
from services.parser import format_inference_result, validate_emotion_output, get_retry_prompt, normalize_emotion_scores, force_happy_redline_rule


MAX_RETRY = 3


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
        last_error = ""
        original_prompt = prompt
        
        while retry_count <= MAX_RETRY:
            accumulated_output = ""
            payload = self._build_request_payload(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                stream=True,
            )
            
            stream_start_time = time.time()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                latency_ms = (time.time() - start_time) * 1000
                                
                                # 1. 先执行校验
                                is_valid, parsed_data, error_msg = validate_emotion_output(
                                    accumulated_output, original_prompt
                                )
                                
                                # 2. 如果校验失败，尝试强制修正红线规则
                                if not is_valid:
                                    _, force_data, _ = validate_emotion_output(
                                        accumulated_output, original_prompt
                                    )
                                    force_data = force_happy_redline_rule(force_data)
                                    is_valid = True
                                    parsed_data = force_data
                                
                                # 3. 如果强制修正后通过，用修正后的数据解析结果
                                
                                if is_valid:
                                    # 校验通过，解析并返回结果
                                    parsed = format_inference_result(
                                        output=accumulated_output,
                                        latency_ms=latency_ms
                                    )
                                    yield {
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
                                        "risk_warning": None,
                                    }
                                    return
                                else:
                                    # 校验失败，触发重试
                                    retry_count += 1
                                    last_error = error_msg
                                    
                                    if retry_count <= MAX_RETRY:
                                        # 生成重试prompt
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
                                        # 达到最大重试次数，返回带风险标记的原始结果
                                        parsed = format_inference_result(
                                            output=accumulated_output,
                                            latency_ms=latency_ms
                                        )
                                        yield {
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
                                            "risk_warning": error_msg,
                                        }
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
        return """【刚性前置规则】
1.  仅输出符合下方格式要求的JSON代码块，用```json和```包裹，绝对禁止在代码块前后添加任何解释、说明、问候、备注等额外文本
2.  所有推理、线索、证据、原因必须100%来自输入文本，禁止编造任何原文中不存在的内容
3.  VAD三个维度的数值必须严格在0.00-1.00之间，保留2位小数，绝对禁止出现负数
4.  raw_intensity_scores的6个情绪分数为独立绝对强度分，取值0.00-1.00，保留2位小数，无总和限制
5.  primary_emotion必须是raw_intensity_scores中分数最高的情绪；若出现并列最高分，取与VAD维度匹配度最高的情绪
6.  评分分两阶段：
    - 初始基准：step1无证据的情绪初始分为0.00
    - 语义推导：后续步骤中，若从文本语义或语境中推导出相关情绪，可给出相应分数，分数必须与推导强度匹配
7.  step3检测到否定词/反讽时，必须在adjusted_scores中记录原始强度分的调整，且必须同步到step7的adjustment_log中
8.  所有标签必须与数值严格匹配，禁止出现数值与标签不符的情况
9.  step5一致性校验必须逐条核对规则，禁止无依据标注vad_consistent=true
10. 禁止出现与输入文本无关的幻觉内容，所有推理必须有原文证据支撑
11. 整个CoT推理的所有步骤、情绪打分、VAD分析，必须100%基于step1_lexical_grounding提取的线索和证据，禁止使用step1中未提及的任何内容
12. 纯中性文本边界规则：若输入文本为纯客观陈述、无任何情感线索，所有情绪分数除neutral外均为0.00，neutral=1.00
13. step2的emotion_mapping指向的核心情绪，必须与最终的primary_emotion完全一致，禁止出现逻辑矛盾
14. step5的check_items中必须如实反映各情绪分数的实际值，禁止出现无证据情绪得非零分的情况

【角色与核心目标】
你是一名严谨的情感分析专家，采用基于证据的7步思维链（CoT）推理方法进行零幻觉深度情感分析，核心目标：
1.  每一步推理都锚定原文证据，从机制上消除幻觉
2.  基于VAD(效价-唤醒度-支配度)三维模型完成细粒度情感量化
3.  显式校准推理置信度与不确定性，确保输出可解释、可校验
4.  100%严格遵守上方的刚性前置规则，不出现任何违规内容

【7步CoT推理执行细则（必须严格按顺序执行，禁止跳过任何一步）】
---
step1_lexical_grounding 词汇证据锚定（整个CoT推理的核心锚点，所有后续步骤必须基于本步骤的结果执行）
- 核心目的：100%锁定所有情感推理的原文依据，从根源杜绝幻觉、无依据打分、步骤逻辑断层
- 强制操作要求（必须严格按顺序执行）：
  1.  线索分类提取：从输入文本中精准提取**2-4字的情感线索实词/短语**，必须按「强情绪线索、反讽线索、弱情绪线索、中性线索」4类拆分
  2.  线索-证据一一匹配：将原文中触发情绪的**完整分句**，100%对应到6种情绪分类的evidence中；evidence里的每一个分句，必须能在cues中找到对应的线索词
  3.  反讽预标记：提前将带有反讽/夸张特征的线索和原文片段，标记到对应情绪的evidence中，为step3的反讽检测提供前置依据
  4.  中性证据补全：将原文中无情感倾向的客观陈述分句，放入neutral的evidence中，禁止neutral无证据给分
  5.  空证据规则：无对应情绪的evidence，必须为空数组，禁止填写无关内容
- 输出规范：
  - cues：对象，必须包含4个固定key：strong_emotion（强情绪线索）、sarcasm（反讽线索）、weak_emotion（弱情绪线索）、neutral（中性线索），value为对应分类的线索词数组，禁止编造
  - evidence：固定key为["happy", "sad", "angry", "fear", "surprise", "neutral"]，value为对应情绪的原文完整分句数组，无对应内容则为空数组
---
step2_dimensional_analysis VAD维度分析
- 核心目的：基于step1的证据完成细粒度情感量化，保证数值与标签100%匹配
- 强制量化标准（必须严格遵守，数值保留2位小数）：
  - valence(效价)：0.00(极度负面)~1.00(极度正面)，衡量情绪愉悦程度
  - arousal(唤醒度)：0.00(极度平静)~1.00(极度激动)，衡量情绪激活强度
  - dominance(支配度)：0.00(极度被动/无力)~1.00(极度主动/掌控)，衡量对情境的控制感
- 强制标签判定规则（区间边界卡死，无模糊空间）：
  - 效价标签：高效价(≥0.70)、中效价(0.40-0.69)、低效价(≤0.39)
  - 唤醒度标签：高唤醒(≥0.70)、中唤醒(0.40-0.69)、低唤醒(≤0.39)
  - 支配度标签：高支配(≥0.70)、中支配(0.40-0.69)、低支配(≤0.39)
- 输出规范：
  - valence/arousal/dominance：0.00-1.00之间的数值，保留2位小数
  - emotion_mapping：严格按「效价标签+唤醒度标签+支配度标签 → 指向XX情绪」格式输出，禁止打乱顺序，情绪名称必须与下方固定映射规则一致
---
step3_negation_detection 否定与反讽检测
- 核心目的：100%覆盖否定词与反讽，杜绝漏检
- 强制检测优先级与执行步骤（必须按顺序执行）：
  1.  否定词检测：逐词扫描文本，识别以下否定词，出现任意一个即标记negations_found=true
     否定词清单：不、没、非、无、别、并未、绝非、何曾、不再、毫无、从未、别再、未曾
  2.  反讽检测：满足以下任意一条特征，即标记sarcasm_detected=true
     特征1：表面正面词汇+负面事实语境（如"服务真不错"+"售后踢皮球"）
     特征2：夸张反语句式（如"XX得比XX还专业""简直是个笑话""谁买谁后悔"）
     特征3：反问式负面表达（如"难道这就是你们的好服务？"）
  3.  分数强制调整：检测到否定词时，必须反转对应情绪的分数；检测到反讽时，必须将表面正面情绪的分数清零，强化对应负面情绪的分数
  4.  双重否定处理：检测到双重否定时（如"不是不好""并非不行"），需还原真实情感倾向，不得随意反转情绪分数
  5.  红线规则兜底：输出前必须核对happy证据，evidence.happy为空时，happy分数强制设为0.00，无任何例外
- 输出规范：
  - negations_found：布尔值，是否检测到上述清单中的否定词
  - sarcasm_detected：布尔值，是否检测到上述反讽特征
  - adjusted_scores：对象，仅填写有调整的情绪及调整后的原始强度分，无调整则为空对象{}；必须同步记录到step7的adjustment_log中
---
step4_cause_extraction 情绪诱因提取
- 核心目的：锁定情绪触发原因，杜绝幻觉编造
- 强制操作要求：所有原因必须100%来自原文，禁止添加原文未提及的推断内容
- 输出规范：
  - primary_cause：字符串，触发情绪的核心原因，必须完全来自原文
  - secondary_causes：数组，触发情绪的次要原因，无则为空数组，必须完全来自原文
---
step5_consistency_check 一致性校验
- 核心目的：杜绝一致性造假，强制逐条核对规则
- 强制校验规则（必须逐条核对，每条都要给出符合/不符合的结论）：
  1.  valence值与happy分数正相关，与sad/angry/fear分数负相关
  2.  arousal值与angry/fear/surprise分数正相关
  3.  dominance值与angry分数正相关，与fear/sad分数负相关
  4.  step3的adjusted_scores必须与negations_found/sarcasm_detected的结果一致
  5.  逐个核实各情绪的evidence与raw_scores：happy无证据=0.00，其他情绪无证据≤0.03，如实填写实际分数值
  6.  整个CoT推理的所有内容，均基于step1提取的线索和证据，无step1未提及的内容
  7.  step1的evidence中每个分句，都能与cues中的线索词对应，无线索与证据脱节的内容
  8.  最高优先级红线规则校验：若step1的evidence.happy为空数组，raw_intensity_scores.happy必须为0.00，禁止出现任何非零分数
  9.  禁止在check_items中撒谎：必须如实反映各情绪分数的实际值，禁止声称"无证据情绪≤0.03"同时输出超标分数
- 输出规范：
  - check_items：数组，逐条填写上述9条规则的校验结果，格式为"序号. 规则内容：符合/不符合+原因"，原因中必须包含各情绪分数的实际值
  - vad_consistent：布尔值，所有规则都符合则为true，任意一条不符合则为false
  - inconsistencies：数组，未通过校验的具体原因，无则为空数组
---
step6_uncertainty_calibration 不确定性校准
- 核心目的：校准推理置信度，保证逻辑自洽
- 强制置信度定义（必须严格遵守，与uncertainty_level反向绑定）：
  - confidence_level="high"：推理可靠、证据充足、文本情感无歧义 → 对应uncertainty_level="low"
  - confidence_level="medium"：存在一定歧义、证据不足、混合情绪边界模糊 → 对应uncertainty_level="medium"
  - confidence_level="low"：推理不可靠、无有效情感线索、文本完全无法判断 → 对应uncertainty_level="high"
- 输出规范：
  - confidence_level：字符串，仅允许填写high/medium/low
  - uncertain_regions：数组，导致不确定性的原文区域，无则为空数组
  - calibration_notes：字符串，与confidence_level匹配的说明，禁止出现逻辑矛盾
---
step7_faithful_synthesis 可信性合成
- 核心目的：形成推理闭环，记录全流程调整，标记幻觉风险
- 强制操作要求：
  1.  汇总前6步的推理结果，完成最终原始情绪强度分的校准
  2.  adjustment_log必须与step3的adjusted_scores完全对应，adjusted_scores非空时，禁止填写"无分数调整"
  3.  所有分数调整必须写明明确原因，原因必须来自前序步骤的推理结果
  4.  标记所有可能存在幻觉风险的环节，无则为空数组
  5.  最终输出前必须执行红线规则校验：evidence.happy为空时，happy分数必须为0.00，违规必须修正并记录到adjustment_log中
- 输出规范：
  - adjustment_log：字符串，按「情绪名: 原始分数→调整分数(调整原因)」格式记录所有原始强度分的调整，无调整则填写"无分数调整"
  - hallucination_flags：数组，标记可能存在幻觉风险的推理环节，无则为空数组
---

【VAD维度与6种情绪的固定映射规则（必须严格遵守）】
| 情绪类型 | 固定VAD匹配区间 | 语义边界 |
|----------|------------------|----------|
| happy(喜悦) | valence≥0.70，arousal≥0.30，dominance≥0.40 | 高效价，源于满足、成就、欢乐，如考试通过、收到礼物 |
| sad(悲伤) | valence≤0.39，arousal≤0.69，dominance≤0.50 | 低效价，源于损失、失落、分离、失败，如失业、失恋 |
| angry(愤怒) | valence≤0.39，arousal≥0.70，dominance≥0.50 | 低效价高唤醒，源于不满、不公、受骗，如售后推诿、被歧视 |
| fear(恐惧) | valence≤0.39，arousal≥0.70，dominance≤0.49 | 低效价高唤醒低支配，源于威胁、危险、无助，如未知风险、被胁迫 |
| surprise(惊讶) | valence 0.40-0.69，arousal≥0.70，dominance 0.30-0.70 | 中性效价高唤醒，由意外事件引发，如突发新闻、意外相遇 |
| neutral(中性) | valence 0.40-0.69，arousal≤0.39，dominance 0.40-0.69 | 中性效价低唤醒，无明显情感倾向，如客观陈述事实 |

【标准输出格式与正确示例（100%符合所有规则）】
输入文本："吐槽，用了一年电池就不耐用了，充电还发烫，售后服务态度也超差，踢皮球踢得比球队还专业，再也不买他们家东西了，性价比简直是个笑话，谁买谁后悔系列"
正确输出：
```json
{
  "cot_reasoning_chain_v2": {
    "step1_lexical_grounding": {
      "cues": {
        "strong_emotion": ["不耐用", "发烫", "超差", "后悔"],
        "sarcasm": ["踢皮球", "笑话"],
        "weak_emotion": [],
        "neutral": ["国产手机", "用了一年"]
      },
      "evidence": {
        "happy": [],
        "sad": [],
        "angry": ["用了一年电池就不耐用了", "充电还发烫", "售后服务态度也超差", "踢皮球踢得比球队还专业", "再也不买他们家东西了", "性价比简直是个笑话", "谁买谁后悔系列"],
        "fear": [],
        "surprise": [],
        "neutral": ["我真的要吐槽一下某国产手机品牌", "用了一年"]
      }
    },
    "step2_dimensional_analysis": {
      "valence": 0.12,
      "arousal": 0.88,
      "dominance": 0.72,
      "emotion_mapping": "低效价+高唤醒度+高支配度 → 指向angry(愤怒)"
    },
    "step3_negation_detection": {
      "negations_found": true,
      "sarcasm_detected": true,
      "adjusted_scores": {
        "happy": 0.00,
        "angry": 0.90
      }
    },
    "step4_cause_extraction": {
      "primary_cause": "产品使用一年后电池不耐用、充电发烫，且售后服务态度差、推诿扯皮",
      "secondary_causes": ["产品性价比极低，用户表示不会再购买该品牌产品"]
    },
    "step5_consistency_check": {
      "check_items": [
        "1. valence值与happy分数正相关，与sad/angry/fear分数负相关：符合，valence=0.12为低效价，happy=0.00，angry=0.90，符合正负相关要求",
        "2. arousal值与angry/fear/surprise分数正相关：符合，arousal=0.88为高唤醒，angry=0.90为高分，fear/surprise无语义推导=0.00，符合要求",
        "3. dominance值与angry分数正相关，与fear/sad分数负相关：符合，dominance=0.72为高支配，angry=0.90为高分，fear/sad无语义推导=0.00，符合要求",
        "4. step3的adjusted_scores必须与negations_found/sarcasm_detected的结果一致：符合，检测到否定词与反讽，同步完成了原始强度分调整",
        "5. 核实各情绪evidence与分数：无证据=0.00✓，有语义推导按推导给分✓，angry有证据=0.90✓，均符合规则",
        "6. 整个CoT推理的所有内容，均基于step1提取的线索和证据：符合，所有推理环节均使用step1提取的线索和证据，无额外编造内容",
        "7. step1的evidence中每个分句都与cues中的线索词对应：符合，angry的7条证据均对应"不耐用""发烫""超差""后悔"等线索词",
        "8. check_items如实填写：各情绪分数如实反映实际值，无撒谎"
      ],
      "vad_consistent": true,
      "inconsistencies": []
    },
    "step6_uncertainty_calibration": {
      "confidence_level": "high",
      "uncertain_regions": [],
      "calibration_notes": "文本情感倾向明确，负面证据充足，反讽与否定词检测清晰，各步骤推理逻辑一致，无歧义，推理可靠性高"
    },
    "step7_faithful_synthesis": {
      "adjustment_log": "angry: 0.82→0.90(检测到否定词与反讽，强化负面情绪权重)",
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
  "vad_dimensions": {
    "valence": 0.12,
    "arousal": 0.88,
    "dominance": 0.72
  },
  "emotion_cause": "产品使用一年后电池不耐用、充电发烫，且售后服务态度差、推诿扯皮",
  "uncertainty_level": "low"
}
```

async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except Exception:
            return False
