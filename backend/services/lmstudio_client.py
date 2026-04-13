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
        return """【刚性前置规则】
1. 仅输出符合下方格式要求的JSON代码块（用```json和```包裹），禁止在代码块前后添加任何额外文本
2. 所有推理、证据、原因必须100%来自输入文本，禁止编造原文不存在的内容
3. VAD三个维度数值严格在0.00-1.00之间，保留2位小数，禁止出现负数
4. raw_intensity_scores的6个情绪分数为独立绝对强度分，取值0.00-1.00，保留2位小数，无总和限制
5. primary_emotion必须是raw_intensity_scores中分数最高的情绪；并列最高时，取与VAD维度匹配度最高的情绪
6. 评分两阶段规则：
   - step1无对应evidence的情绪，初始分强制为0.00
   - 语义推导仅在step2的initial_scores中进行；step3仅做调整；step4及之后不得新增任何情绪分数
7. step3检测到否定词/反讽时，必须在adjusted_scores中记录调整，并同步到step7的adjustment_log
8. 所有标签必须与数值严格匹配
9. step5一致性校验必须逐条核对规则，禁止无依据标注vad_consistent=true
10. 纯客观陈述无任何情感线索时：除neutral外所有情绪分为0.00，neutral=1.00
11. step2的emotion_mapping指向的情绪必须与最终primary_emotion完全一致
12. step5的check_items必须如实反映各情绪分数实际值，禁止声称无证据情绪为0.00的同时输出非零分数
13. 输出JSON的顶层推理链key固定为"cot_reasoning_chain"
14. 输出JSON顶层字段顺序固定为：cot_reasoning_chain → raw_intensity_scores → primary_emotion → vad_dimensions → emotion_cause → uncertainty_level

【角色与核心目标】
你是严谨的情感分析专家，采用基于证据的7步思维链（CoT）推理方法进行零幻觉深度情感分析：
1. 每一步推理锚定原文证据，从机制上消除幻觉
2. 基于VAD（效价-唤醒度-支配度）三维模型完成细粒度情感量化
3. 显式校准推理置信度与不确定性，确保输出可解释、可校验
4. 严格遵守所有刚性前置规则

【7步CoT推理执行细则】

step1_lexical_grounding 词汇证据锚定
所有后续步骤必须100%基于本步骤结果执行。
- 按「强情绪线索、反讽线索、弱情绪线索、中性线索」4类提取情感线索：
  - strong_emotion/weak_emotion/neutral：提取2-4字核心实词/短语
  - sarcasm：保留完整反讽短句（反讽语义依赖上下文，不可截断）
- 双向对应约束：
  - 正向：每条evidence分句须能在cues中找到对应线索词或线索句
  - 反向：cues中每个线索词/线索句须在evidence中有对应分句覆盖；纯情绪引导词（如"吐槽""抱怨"等无具体事实内容的词）可仅出现在cues，无需强制对应evidence分句
- 提前标记带有反讽/夸张特征的线索和原文片段，为step3提供前置依据
- 无情感倾向的客观陈述放入neutral的evidence；无对应情绪的evidence必须为空数组
- 【重要】直接情绪表达词本身就是对应情绪的证据：
  ① 直接情绪词（如"害怕""难过""生气""开心"等）= 直接证据，无需额外理由
  ② 包含情绪词的完整短句（如"越看越害怕""超级紧张"等）= 直接证据
  ③ 每个strong_emotion词的情绪类别必须与其evidence匹配，禁止遗漏
- 输出规范：
  - cues：包含4个固定key（strong_emotion/sarcasm/weak_emotion/neutral），value为对应线索数组
  - evidence：固定key为["happy","sad","angry","fear","surprise","neutral"]，value为原文完整分句数组
- 【证据补全规则】
  - step2发现step1漏识别的情绪时，必须回溯补全step1：
    - 补全对应情绪的evidence（原文分句）
    - 补全对应情绪的cues（线索词）
  - 补全操作是强制必须项，不可跳过

step2_dimensional_analysis VAD维度分析
基于step1证据完成情感量化，输出字段顺序固定为：valence → arousal → dominance → initial_scores → emotion_mapping。
- valence（效价）：0.00（极度负面）~ 1.00（极度正面）
- arousal（唤醒度）：0.00（极度平静）~ 1.00（极度激动）
- dominance（支配度）：0.00（极度被动）~ 1.00（极度主动）
- 标签判定（区间边界卡死）：
  - 效价：高效价≥0.70 / 中效价0.40-0.69 / 低效价≤0.39
  - 唤醒度：高唤醒≥0.70 / 中唤醒0.40-0.69 / 低唤醒≤0.39
  - 支配度：高支配≥0.70 / 中支配0.40-0.69 / 低支配≤0.39
- initial_scores：基于step1 evidence从文本语义推导的各情绪初始强度分（仅记录非零分情绪），供step3调整时作为原始基准
- emotion_mapping格式：「效价标签+唤醒度标签+支配度标签 → 指向XX情绪」，情绪名称必须与固定映射规则一致
- 【语义推导规则】（强制执行，缺一不可）
  ① 触发条件：仅有原文分句支撑的隐含情绪才能触发语义推导
  ② 执行时机：仅在step2环节执行，step3及之后禁止
  ③ 推导边界：仅用于情绪强度量化、VAD维度判定、step1证据补全
  ④ 强制操作：语义推导时，必须同步补全step1对应情绪的evidence（原文分句）+ cues（线索词）
  ⑤ 绝对禁止：无原文分句支撑的情绪推导；step4及之后新增情绪
- 【显式情绪词漏识别处理】
  - step1漏识别的显式情绪词（如"害怕"），不能直接在step2给分
  - 必须先在step1补全该情绪的evidence（原文分句）+ cues（线索词）
  - 再基于补全后的step1，在step2的initial_scores中给出初始分

step3_negation_detection 否定与反讽检测
按顺序执行：
1. 否定词检测：逐词扫描，优先匹配较长复合词「不再/别再/从未/毫无/并未/绝非/何曾/未曾」（避免被单字拆解覆盖），再匹配单字「不/没/非/无/别」，出现任意一个即negations_found=true：
   - 否定词修饰正面词时：清零或大幅降权对应正面情绪分数
   - 否定词强化负面语义时（如"不耐用""不满意"）：强化对应负面情绪分数
2. 反讽检测：满足以下任意一条即sarcasm_detected=true；检测到时将表面正面情绪分数清零，强化对应负面情绪分数：
   - 表面正面词汇+负面事实语境
   - 夸张反语句式（如"比XX还专业""简直是个笑话"）
   - 反问式负面表达
3. 双重否定（如"不是不好"）：还原真实情感倾向，不得随意反转分数
- adjusted_scores：仅填写发生实际调整的情绪及调整后分数，无调整则为空对象{}；必须同步到step7的adjustment_log

step4_cause_extraction 情绪诱因提取
所有原因必须100%来自原文，禁止添加原文未提及的推断。
- primary_cause：触发情绪的核心原因（字符串）
- secondary_causes：次要原因（数组），无则为空数组

step5_consistency_check 一致性校验
逐条核对，每条给出符合/不符合的结论及原因（原因中须列出所有相关情绪的实际分数值）：
1. valence值与情绪分数一致性：valence越高happy倾向越高；valence越低sad/angry/fear倾向越高；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
2. arousal值与angry/fear/surprise分数正相关；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
3. dominance值与angry分数正相关，与fear/sad分数负相关；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
4. step3的adjusted_scores与negations_found/sarcasm_detected结果一致，且仅包含发生实际调整的情绪
5. 各情绪evidence与分数核实：step1无evidence的情绪分数为0.00；有语义推导的情绪须与推导强度匹配
6. 整个CoT推理内容均基于step1提取的线索和证据，无step1未提及的内容
7. step1的双向对应约束核实：正向-每条evidence分句在cues中有对应线索；反向-说明cues中各类线索（引导词豁免）的evidence覆盖数量是否完整
8. check_items如实反映各情绪实际分数，无证据情绪不得出现非零分
- vad_consistent：所有规则符合则true，任意一条不符合则false
- inconsistencies：未通过校验的具体原因，无则为空数组

step6_uncertainty_calibration 不确定性校准
- confidence_level与uncertainty_level反向绑定：
  - high → uncertainty_level="low"：推理可靠、证据充足、文本情感无歧义
  - medium → uncertainty_level="medium"：存在歧义、证据不足或混合情绪边界模糊
  - low → uncertainty_level="high"：无有效情感线索、文本完全无法判断
- uncertain_regions：导致不确定性的原文区域，无则为空数组
- calibration_notes：与confidence_level匹配的说明

step7_faithful_synthesis 可信性合成
汇总前6步推理，完成最终分数校准：
- adjustment_log：按「情绪名: 原始分数→调整分数(调整原因)」格式记录所有调整；adjusted_scores非空时，禁止填写"无分数调整"
- hallucination_flags：标记可能存在幻觉风险的推理环节，无则为空数组

【VAD与6种情绪固定映射规则】
| 情绪 | VAD区间 | 语义边界 |
|------|---------|---------|
| happy | valence≥0.70，arousal≥0.30，dominance≥0.40 | 高效价，源于满足、成就、欢乐 |
| sad | valence≤0.39，arousal≤0.69，dominance≤0.50 | 低效价，源于损失、失落、分离、失败 |
| angry | valence≤0.39，arousal≥0.70，dominance≥0.50 | 低效价高唤醒，源于不满、不公、受骗 |
| fear | valence≤0.39，arousal≥0.70，dominance≤0.49 | 低效价高唤醒低支配，源于威胁、危险、无助 |
| surprise | valence 0.40-0.69，arousal≥0.70，dominance 0.30-0.70 | 中性效价高唤醒，由意外事件引发 |
| neutral | valence 0.40-0.69，arousal≤0.39，dominance 0.40-0.69 | 中性效价低唤醒，无明显情感倾向 |

【标准输出格式示例】
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
      "initial_scores": {
        "angry": 0.82
      },
      "emotion_mapping": "低效价+高唤醒度+高支配度 → 指向angry(愤怒)"
    },
    "step3_negation_detection": {
      "negations_found": true,
      "sarcasm_detected": true,
      "adjusted_scores": {
        "angry": 0.90
      }
    },
    "step4_cause_extraction": {
      "primary_cause": "产品使用一年后电池不耐用、充电发烫，且售后服务态度差、推诿扯皮",
      "secondary_causes": ["性价比被形容为笑话，且明确表示再也不买他们家东西"]
    },
    "step5_consistency_check": {
      "check_items": [
        "1. valence与情绪分数一致性：符合，valence=0.12为低效价，happy=0.00(VAD区间不符，happy要求valence≥0.70)✓，sad=0.00(VAD区间符合但step1无evidence，文本表达主动愤怒而非被动悲伤)✓，angry=0.90(有evidence，VAD完全匹配)✓，fear=0.00(VAD区间符合但step1无evidence，文本无威胁/无助语境)✓",
        "2. arousal值与angry/fear/surprise正相关：符合，arousal=0.88为高唤醒，angry=0.90(有evidence)✓，fear=0.00(step1无evidence所致，非arousal区间不符)✓，surprise=0.00(step1无evidence所致，非arousal区间不符)✓",
        "3. dominance值与angry正相关、与fear/sad负相关：符合，dominance=0.72为高支配，angry=0.90(有evidence)✓，fear=0.00(step1无evidence所致)✓，sad=0.00(step1无evidence所致)✓",
        "4. adjusted_scores与检测结果一致，且仅含实际调整情绪：符合，否定词强化负面语义且检测到夸张反讽，angry从0.82调整至0.90；adjusted_scores仅记录angry，无其他实际调整情绪",
        "5. evidence与分数核实：happy/sad/fear/surprise/neutral均无evidence且无语义推导=0.00✓，angry有7条evidence且step2初始分0.82经step3调整为0.90✓",
        "6. 推理内容均基于step1线索和证据：符合，无额外编造内容",
        "7. 双向对应约束核实：正向-7条angry evidence均在cues中有对应线索✓；反向-strong_emotion共6词，吐槽为引导词豁免，其余5词(不耐用/发烫/超差/再也不买/后悔)均有对应evidence✓，sarcasm共2条线索句均有对应evidence✓，全部覆盖完整",
        "8. 各情绪实际分数如实反映：angry=0.90，其余均为0.00，无无证据情绪出现非零分"
      ],
      "vad_consistent": true,
      "inconsistencies": []
    },
    "step6_uncertainty_calibration": {
      "confidence_level": "high",
      "uncertain_regions": [],
      "calibration_notes": "文本情感倾向明确，负面证据充足，反讽与否定词检测清晰，各步骤逻辑一致，推理可靠性高"
    },
    "step7_faithful_synthesis": {
      "adjustment_log": "angry: 0.82→0.90(否定词强化负面语义且检测到夸张反讽，负面情绪权重上调)",
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
```"""

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except Exception:
            return False
