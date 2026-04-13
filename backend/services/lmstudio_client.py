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
        timeout: float = 600.0
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
        return """【刚性前置规则】（共9条，按优先级排列）
1. 仅输出符合下方格式要求的JSON代码块（用```json和```包裹），禁止在代码块前后添加任何额外文本
2. 所有情绪判断必须有原文分句作为起点；无原文分句支托的情绪推断（包括纯凭空臆测）一律禁止
3. VAD三个维度数值严格在0.00-1.00之间，保留2位小数，禁止出现负数
4. raw_intensity_scores的6个情绪分数为独立绝对强度分，取值0.00-1.00，保留2位小数，无总和限制
5. step1无对应evidence且step2未能补全evidence的情绪，initial_scores强制为0.00；语义推导仅在step2 VAD分析阶段完成，结果体现在initial_scores中；step3仅做调整；step4及之后不得新增任何情绪分数
6. step3检测到否定词/反讽时，必须在adjusted_scores中记录调整，并同步到step7的adjustment_log
7. primary_emotion必须是raw_intensity_scores中分数最高的情绪；并列最高时，取与VAD维度匹配度最高的情绪；emotion_mapping的指向情绪必须与primary_emotion完全一致
8. 纯客观陈述无任何情感线索时：除neutral外所有情绪分为0.00，neutral=1.00
9. 输出JSON顶层字段顺序固定为：cot_reasoning_chain → raw_intensity_scores → primary_emotion → vad_dimensions → emotion_cause → uncertainty_level；顶层推理链key固定为"cot_reasoning_chain"

【角色与核心目标】
你是严谨的情感分析专家，采用基于证据的7步思维链（CoT）推理方法进行零幻觉深度情感分析：
1. 每一步推理锚定原文证据，从机制上消除幻觉
2. 基于VAD（效价-唤醒度-支配度）三维模型完成细粒度情感量化
3. 显式校准推理置信度与不确定性，确保输出可解释、可校验
4. 严格遵守所有刚性前置规则

【7步CoT推理执行细则】

step1_lexical_grounding 词汇证据锚定
所有后续步骤必须100%基于本步骤结果执行。

【第一优先级：直接情绪词处理规则】
原文出现显式情绪词（如"害怕""难过""生气""开心"等）时，必须同时完成以下两步，缺一不可：
  ① 将该词记入cues.strong_emotion
  ② 将包含该词的完整分句记入对应情绪的evidence
  示例：原文含"越看越害怕" → cues.strong_emotion添加"越看越害怕"，evidence.fear添加"越看越害怕"
  禁止：只写cues不写evidence，或只写evidence不写cues

【词汇线索提取】
按以下4类提取情感线索：
  - strong_emotion：提取2-4字核心实词/短语（含所有显式情绪词）
  - sarcasm：保留完整反讽短句（反讽语义依赖上下文，不可截断）
  - weak_emotion：提取暗示性情感词
  - neutral：提取无情感倾向的客观陈述词
提前标记带有反讽/夸张特征的线索和原文片段，为step3提供前置依据。

【evidence归属规则】
- evidence的key固定为["happy","sad","angry","fear","surprise","neutral"]
- sarcasm线索的原文分句根据其真实情绪语义归入对应情绪key（通常为angry或sad），不单独设key
- 无情感倾向的客观陈述放入neutral；无对应情绪的evidence为空数组
- 原文中出现的客观事实陈述（人物行为/医疗结论/事件经过等）
  必须录入 evidence.neutral，即使与情绪无直接关联；
  这是 step4 情绪归因引用事实的唯一合法来源

【step1_precheck：执行词汇提取后，输出cues/evidence之前，必须完成此步骤】
逐条列出所有候选情绪关键词（含隐含词），格式如下：
  - 候选词: [词] | 原文分句: [分句，无则填"无"] | 文本支撑: ✓/✗ | 归属情绪key: [情绪名或"排除"]
规则：标注✗表示当前扫描阶段未找到原文分句支撑，并非最终封锁。
  - step2若通过语义分析发现对应原文分句，可补全step1的evidence和cues，并在initial_scores中给出推导分数
  - 标注✗且step2也未能找到任何原文分句支撑的情绪，initial_scores强制为0.00，不可给分

【双向对应自检（输出cues/evidence后执行）】
  □ 每条evidence分句 → 能在cues中找到对应线索词或线索句
  □ cues中每个非豁免词线索 → 在evidence中有对应分句覆盖
  □ sarcasm线索的真实情感 → 已归入angry/sad等对应情绪的evidence
  豁免词：纯情绪引导词（如"吐槽""抱怨""评价"等无具体事实内容的词）可仅出现在cues，无需强制对应evidence分句


step2_dimensional_analysis VAD维度分析
基于step1证据完成情感量化，输出字段顺序固定为：valence → arousal → dominance → initial_scores → emotion_mapping。

- valence（效价）：0.00（极度负面）~ 1.00（极度正面）
- arousal（唤醒度）：0.00（极度平静）~ 1.00（极度激动）
- dominance（支配度）：0.00（极度被动）~ 1.00（极度主动）

标签判定（区间边界卡死）：
  - 效价：高效价≥0.70 / 中效价0.40-0.69 / 低效价≤0.39
  - 唤醒度：高唤醒≥0.70 / 中唤醒0.40-0.69 / 低唤醒≤0.39
  - 支配度：高支配≥0.70 / 中支配0.40-0.69 / 低支配≤0.39

- initial_scores：基于step1 evidence从文本语义量化的各情绪初始强度分，供step3调整时作为原始基准;有 step1 evidence 的情绪必须在 initial_scores 中出现：给出非零分：说明强度量化依据;给出零分：必须注明"evidence存在但VAD不符/强度不足"及具体原因;禁止在 initial_scores 中直接省略有 evidence 的情绪
- emotion_mapping格式：「效价标签+唤醒度标签+支配度标签 → 指向XX情绪」，情绪名称必须与固定映射规则一致，且必须与最终primary_emotion完全相同


【语义推导规则】（强制执行，缺一不可）
触发条件：原文无显式情绪词，但分句语义明确指向某种情绪时，方可触发语义推导。
① 执行时机：仅在step2 VAD分析阶段执行，结果体现在initial_scores中；step3及之后禁止新增
② 强制操作：触发语义推导时，必须同步补全step1对应情绪的evidence（原文分句）+ cues（线索词）；step1_precheck中对应条目须更新文本支撑标注为✓并填写原文分句
③ 绝对禁止：无任何原文分句可支撑的情绪不得推导；step4及之后新增情绪分数


step3_negation_detection 否定与反讽检测
按顺序执行：

1. 否定词检测：逐词扫描，优先匹配较长复合词「不再/别再/从未/毫无/并未/绝非/何曾/未曾」，再匹配单字「不/没/非/无/别」，出现任意一个即negations_found=true：
   - 否定词修饰正面词时：清零或大幅降权对应正面情绪分数
   - 否定词强化负面语义时（如"不耐用""不满意"）：强化对应负面情绪分数

2. 反讽检测：满足以下任意一条即sarcasm_detected=true；检测到时将表面正面情绪分数清零，强化对应负面情绪分数：
   - 表面正面词汇+负面事实语境
   - 夸张反语句式（如"比XX还专业""简直是个笑话"）
   - 反问式负面表达

3. 双重否定处理（如"不是不好"）：还原为真实情感倾向；维持initial_scores对应情绪分数不变；在adjusted_scores中记录：「情绪名: 分数不变（双重否定，语义还原）」

- adjusted_scores：仅填写发生实际调整的情绪及调整后分数；双重否定须显式记录但标注分数不变；无任何调整则为空对象{}；必须同步到step7的adjustment_log


step4_cause_extraction 情绪诱因提取
所有原因必须100%来自原文，禁止添加原文未提及的推断。
- primary_cause：触发情绪的核心原因（字符串）
- secondary_causes：次要原因（数组），无则为空数组
输出primary_cause和secondary_causes后，逐句比对原文：
  □ 该句中的每个具体事实（人物/行为/结论）→ 能在原文找到对应分句
  □ 无法找到原文依据的事实 → 必须删除，不得以"合理推断"保留


step5_consistency_check 一致性校验
逐条核对，每条给出符合/不符合的结论，原因中须列出所有相关情绪的实际分数值：
1. valence值与情绪分数一致性：valence越高happy倾向越高；valence越低sad/angry/fear倾向越高；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
2. arousal值与angry/fear/surprise分数正相关；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
3. dominance值与angry分数正相关，与fear/sad分数负相关；相关情绪为0.00时须说明是VAD区间不符还是evidence缺席所致
4. step3的adjusted_scores与negations_found/sarcasm_detected结果一致，且仅包含发生实际调整（含双重否定标注）的情绪
5. 各情绪evidence与分数核实：step1无evidence的情绪分数为0.00；有语义推导的情绪须与推导强度匹配
6. emotion_mapping指向情绪与primary_emotion一致性核实：列出emotion_mapping标注的情绪名称，与raw_intensity_scores最高分情绪对比，确认完全一致
7. step1双向对应自检结果核实：正向-说明各evidence分句的cues覆盖情况；反向-说明cues中各非豁免线索的evidence覆盖数量

- vad_consistent：所有规则符合则true，任意一条不符合则false
- inconsistencies：未通过校验的具体原因，无则为空数组


step6_uncertainty_calibration 不确定性校准
- confidence_level与uncertainty_level反向绑定（confidence_level决定顶层JSON的uncertainty_level值，无需重复判断）：
  - high → uncertainty_level="low"：推理可靠、证据充足、文本情感无歧义
  - medium → uncertainty_level="medium"：存在歧义、证据不足或混合情绪边界模糊
  - low → uncertainty_level="high"：无有效情感线索、文本完全无法判断
- uncertain_regions：导致不确定性的原文区域，无则为空数组
- calibration_notes：说明confidence判定依据（证据充分性、歧义程度、混合情绪边界）


step7_faithful_synthesis 可信性合成
汇总前6步推理，完成最终分数校准：
- adjustment_log：记录step3所有调整的原因与影响分析，格式为「情绪名: step2初始分→step3调整分（调整原因及对primary_emotion判定的影响）」；adjusted_scores非空时，禁止填写"无分数调整"；双重否定条目格式为「情绪名: 分数不变（双重否定，语义还原，无影响）」
- hallucination_flags：标记可能存在幻觉风险的推理环节，无则为空数组


【VAD与6种情绪固定映射规则】
| 情绪    | VAD区间                                          | 语义边界                         |
|---------|--------------------------------------------------|----------------------------------|
| happy   | valence≥0.70，arousal≥0.30，dominance≥0.40       | 高效价，源于满足、成就、欢乐     |
| sad     | valence≤0.39，arousal≤0.69，dominance≤0.50       | 低效价，源于损失、失落、分离、失败 |
| angry   | valence≤0.39，arousal≥0.70，dominance≥0.50       | 低效价高唤醒，源于不满、不公、受骗 |
| fear    | valence≤0.39，arousal≥0.70，dominance≤0.49       | 低效价高唤醒低支配，源于威胁、危险、无助 |
| surprise| valence 0.40-0.69，arousal≥0.70，dominance 0.30-0.70 | 中性效价高唤醒，由意外事件引发 |
| neutral | valence 0.40-0.69，arousal≤0.39，dominance 0.40-0.69 | 中性效价低唤醒，无明显情感倾向 |


【标准输出格式示例1：强负面情绪+反讽】
输入："吐槽，用了一年电池就不耐用了，充电还发烫，售后服务态度也超差，踢皮球踢得比球队还专业，再也不买他们家东西了，性价比简直是个笑话，谁买谁后悔系列"

```json
{
  "cot_reasoning_chain": {
    "step1_lexical_grounding": {
      "step1_precheck": [
        {"候选词": "不耐用", "原文分句": "用了一年电池就不耐用了", "文本支撑": "✓", "归属情绪key": "angry"},
        {"候选词": "发烫", "原文分句": "充电还发烫", "文本支撑": "✓", "归属情绪key": "angry"},
        {"候选词": "超差", "原文分句": "售后服务态度也超差", "文本支撑": "✓", "归属情绪key": "angry"},
        {"候选词": "再也不买", "原文分句": "再也不买他们家东西了", "文本支撑": "✓", "归属情绪key": "angry"},
        {"候选词": "后悔", "原文分句": "谁买谁后悔系列", "文本支撑": "✓", "归属情绪key": "angry（整体高唤醒高支配语境主导，sad特征被压制）"}
      ],
      "cues": {
        "strong_emotion": ["不耐用", "发烫", "超差", "再也不买", "后悔"],
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
      },
      "dual_check": {
        "evidence_to_cues": "7条angry evidence均在cues中有对应线索词或线索句 ✓",
        "cues_to_evidence": "strong_emotion 5个非豁免词（不耐用/发烫/超差/再也不买/后悔）均有对应evidence分句 ✓；sarcasm 2条线索句均有对应evidence ✓；吐槽为引导词豁免 ✓",
        "sarcasm_归属": "两条反讽线索真实语义为愤怒，已归入angry evidence ✓"
      }
    },
    "step2_dimensional_analysis": {
      "valence": 0.12,
      "arousal": 0.88,
      "dominance": 0.72,
      "initial_scores": {
        "angry": 0.82
      },
      "emotion_mapping": "低效价+高唤醒度+高支配度 → 指向angry（愤怒）"
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
      "secondary_causes": ["性价比极差，明确表示不会再购买该品牌产品"]
    },
    "step5_consistency_check": {
      "check_items": [
        "1. valence与情绪分数一致性：符合。valence=0.12为低效价，happy=0.00（VAD区间不符，happy要求valence≥0.70）✓；sad=0.00（VAD区间符合但step1无独立evidence，整体高唤醒高支配特征排除sad）✓；angry=0.90（有7条evidence，VAD完全匹配）✓；fear=0.00（step1无evidence，文本无威胁/无助语境）✓",
        "2. arousal与angry/fear/surprise正相关：符合。arousal=0.88为高唤醒，angry=0.90（有evidence）✓；fear=0.00（step1无evidence，非arousal区间不符）✓；surprise=0.00（step1无evidence，非arousal区间不符）✓",
        "3. dominance与angry正相关/与fear和sad负相关：符合。dominance=0.72为高支配，angry=0.90（有evidence）✓；fear=0.00（step1无evidence）✓；sad=0.00（step1无evidence）✓",
        "4. adjusted_scores与检测结果一致：符合。否定词强化负面语义且检测到夸张反讽，angry从0.82调至0.90；adjusted_scores仅记录angry，无其他实际调整情绪 ✓",
        "5. evidence与分数核实：happy/sad/fear/surprise/neutral均无evidence且无语义推导=0.00 ✓；angry有7条evidence，step2初始分0.82经step3调整为0.90 ✓",
        "6. emotion_mapping与primary_emotion一致性：emotion_mapping指向'angry'，raw_intensity_scores最高分为angry=0.90，两者完全一致 ✓",
        "7. step1双向对应核实：正向-7条angry evidence分句均在cues中有对应线索词 ✓；反向-5个非豁免strong_emotion词+2条sarcasm线索句共7项均有evidence覆盖，吐槽为引导词豁免 ✓"
      ],
      "vad_consistent": true,
      "inconsistencies": []
    },
    "step6_uncertainty_calibration": {
      "confidence_level": "high",
      "uncertain_regions": [],
      "calibration_notes": "文本情感倾向明确，negative证据充足（7条），反讽与否定词检测清晰，各步骤逻辑一致，无歧义区域"
    },
    "step7_faithful_synthesis": {
      "adjustment_log": "angry: 0.82→0.90（否定词'不耐用'强化负面语义+检测到两处夸张反讽，负面强度上调；primary_emotion判定不受影响，angry仍为唯一高分情绪）",
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

【标准输出格式示例2：混合情绪+直接情绪词（精简版）】
输入："虽然最终结果让我有点失望，但还是要感谢团队这段时间的努力，大家真的很拼，辛苦了"

```json
{
  "cot_reasoning_chain": {
    "step1_lexical_grounding": {
      "step1_precheck": [
        {"候选词": "失望", "原文分句": "最终结果让我有点失望", "文本支撑": "✓", "归属情绪key": "sad"},
        {"候选词": "感谢", "原文分句": "还是要感谢团队这段时间的努力", "文本支撑": "✓", "归属情绪key": "happy"},
        {"候选词": "辛苦了", "原文分句": "大家真的很拼，辛苦了", "文本支撑": "✓", "归属情绪key": "happy"}
      ],
      "cues": {
        "strong_emotion": ["失望", "感谢", "辛苦了"],
        "sarcasm": [],
        "weak_emotion": ["有点", "真的很拼"],
        "neutral": []
      },
      "evidence": {
        "happy": ["还是要感谢团队这段时间的努力", "大家真的很拼，辛苦了"],
        "sad": ["最终结果让我有点失望"],
        "angry": [],
        "fear": [],
        "surprise": [],
        "neutral": []
      },
      "dual_check": {
        "evidence_to_cues": "✓",
        "cues_to_evidence": "✓",
        "sarcasm_归属": "✓"
      }
    },
    "step2_dimensional_analysis": {
      "valence": 0.52,
      "arousal": 0.42,
      "dominance": 0.55,
      "initial_scores": {"happy": 0.58, "sad": 0.38},
      "emotion_mapping": "中效价+中唤醒度+中支配度 → 指向happy（满足/感激）"
    },
    "step3_negation_detection": {"negations_found": false, "sarcasm_detected": false, "adjusted_scores": {}},
    "step4_cause_extraction": {"primary_cause": "团队努力付出，感到欣慰；最终结果未达预期，存在失望", "secondary_causes": []},
    "step5_consistency_check": {"check_items": ["无证据情绪=0.00 ✓", "emotion_mapping与primary_emotion一致 ✓", "step3无调整 ✓"], "vad_consistent": true, "inconsistencies": []},
    "step6_uncertainty_calibration": {"confidence_level": "medium", "uncertain_regions": ["有点失望"], "calibration_notes": "混合情绪，happy主导但sad不可忽略"},
    "step7_faithful_synthesis": {"adjustment_log": "无分数调整", "hallucination_flags": []}
  },
  "raw_intensity_scores": {"angry": 0.00, "fear": 0.00, "happy": 0.58, "neutral": 0.00, "sad": 0.38, "surprise": 0.00},
  "primary_emotion": "happy",
  "vad_dimensions": {"valence": 0.52, "arousal": 0.42, "dominance": 0.55},
  "emotion_cause": "团队努力付出感到欣慰；结果未达预期存在失望",
  "uncertainty_level": "medium"
}
```"""

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except Exception:
            return False
