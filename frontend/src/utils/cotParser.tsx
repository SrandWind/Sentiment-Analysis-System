import React from 'react'
import { Tag, Progress } from 'antd'

export const formatLatency = (ms: number | undefined): string => {
  if (ms === undefined || ms === null || ms < 0) return '-'
  if (ms < 1000) {
    return `${Math.round(ms)}ms`
  }
  return `${(ms / 1000).toFixed(2)}s`
}

export const EMOTION_LABELS: Record<string, string> = {
  angry: '愤怒',
  fear: '恐惧',
  happy: '高兴',
  neutral: '中性',
  sad: '悲伤',
  surprise: '惊讶',
}

export const EMOTION_COLORS: Record<string, string> = {
  angry: '#dc8888',
  fear: '#a894c4',
  happy: '#88c4a8',
  neutral: '#b0a8a0',
  sad: '#88a4c4',
  surprise: '#dcc488',
}

export const EMOTION_ICONS: Record<string, string> = {
  angry: '😠',
  fear: '😨',
  happy: '😄',
  neutral: '😐',
  sad: '😢',
  surprise: '😮',
}

export const VAD_LABELS: Record<string, string> = {
  valence: '效价',
  arousal: '唤醒度',
  dominance: '支配度',
}

export const VAD_VALUE_LABELS: Record<string, (v: number) => string> = {
  valence: (v) => v > 0.65 ? '正' : v < 0.35 ? '负' : '中性',
  arousal: (v) => v > 0.65 ? '高' : v < 0.35 ? '低' : '中',
  dominance: (v) => v > 0.65 ? '高' : v < 0.35 ? '低' : '中',
}

export const UNCERTAINTY_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  low: { label: '低', color: '#52c41a', icon: '🟢' },
  medium: { label: '中', color: '#faad14', icon: '🟡' },
  high: { label: '高', color: '#ff4d4f', icon: '🔴' },
}

export const RELIABILITY_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  high: { label: '高', color: '#52c41a', icon: '🟢' },
  medium: { label: '中', color: '#faad14', icon: '🟡' },
  low: { label: '低', color: '#ff4d4f', icon: '🔴' },
}

export const STEP_TITLES: Record<string, string> = {
  step1_lexical_grounding: '📌 证据提取',
  step2_dimensional_analysis: '📊 VAD维度分析',
  step3_negation_detection: '🔍 否定/反讽检测',
  step4_cause_extraction: '💡 情感归因',
  step5_consistency_check: '✅ 一致性检验',
  step6_uncertainty_calibration: '📈 置信度校准',
  step7_faithful_synthesis: '🔄 可信性综合',
}

export const EMOTION_NAME_MAP: Record<string, string> = {
  angry: '愤怒',
  fear: '恐惧',
  happy: '高兴',
  neutral: '中性',
  sad: '悲伤',
  surprise: '惊讶',
}

export interface ParsedCotContent {
  type: string
  content: React.ReactNode
}

export const parseCotContent = (stepKey: string, content: string, primaryEmotion: string): ParsedCotContent => {
  if (!content) return { type: 'empty', content: '（未生成内容）' }

  let parsed: any = null
  try {
    parsed = JSON.parse(content)
  } catch {
    return { type: 'text', content }
  }

  switch (stepKey) {
    case 'step1_lexical_grounding':
      return parseStep1LexicalGrounding(parsed, primaryEmotion)
    case 'step2_dimensional_analysis':
      return parseStep2DimensionalAnalysis(parsed)
    case 'step3_negation_detection':
      return parseStep3NegationDetection(parsed)
    case 'step4_cause_extraction':
      return parseStep4CauseExtraction(parsed)
    case 'step5_consistency_check':
      return parseStep5ConsistencyCheck(parsed)
    case 'step6_uncertainty_calibration':
      return parseStep6UncertaintyCalibration(parsed)
    case 'step7_faithful_synthesis':
      return parseStep7FaithfulSynthesis(parsed)
    default:
      return { type: 'text', content }
  }
}

const parseStep1LexicalGrounding = (data: any, primaryEmotion: string): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const cuesObj = data.cues || {}
  const evidence = data.evidence || {}

  const CUE_CATEGORIES = [
    { key: 'strong_emotion', label: '🔥 强情绪线索', color: '#ff4d4f', className: 'cue-tag-strong' },
    { key: 'sarcasm', label: '🎭 反讽线索', color: '#fa8c16', className: 'cue-tag-sarcasm' },
    { key: 'weak_emotion', label: '🌫️ 弱情绪线索', color: '#1890ff', className: 'cue-tag-weak' },
    { key: 'neutral', label: '⚪ 中性线索', color: '#8c8c8c', className: 'cue-tag-neutral' },
  ]

  const hasEvidence = (emotion: string) => {
    const e = evidence[emotion]
    return Array.isArray(e) && e.length > 0
  }

  const sortedEmotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'].sort((a, b) => {
    const aHas = hasEvidence(a)
    const bHas = hasEvidence(b)
    if (aHas && !bHas) return -1
    if (!aHas && bHas) return 1
    if (a === primaryEmotion) return -1
    if (b === primaryEmotion) return 1
    return 0
  })

  const allCues = typeof cuesObj === 'object' && !Array.isArray(cuesObj)
    ? cuesObj
    : { strong_emotion: Array.isArray(cuesObj) ? cuesObj : [] }

  return {
    type: 'step1',
    content: (
      <div className="cot-step1-content">
        <div className="cot-section">
          <div className="cot-section-title">🔑 情感线索分类</div>
          <div className="cot-cues-grid">
            {CUE_CATEGORIES.map(cat => {
              const cues = allCues[cat.key] || []
              return (
                <div key={cat.key} className="cot-cue-category">
                  <div className="cot-cue-category-label" style={{ color: cat.color }}>
                    {cat.label}
                  </div>
                  <div className="cot-cues-list">
                    {cues.length > 0 ? (
                      cues.map((cue: string, i: number) => (
                        <Tag key={i} className={`cue-tag ${cat.className}`} style={{ borderColor: cat.color }}>
                          {cue}
                        </Tag>
                      ))
                    ) : (
                      <span className="no-cues-text">无</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <div className="cot-section">
          <div className="cot-section-title">📎 证据映射（6种情绪）</div>
          {sortedEmotions.map(emotion => {
            const emotionEvidence = evidence[emotion] || []
            const hasEv = emotionEvidence.length > 0
            return (
              <div key={emotion} className={`evidence-row ${hasEv ? 'has-evidence' : 'no-evidence'}`}>
                <span className="evidence-emotion">
                  {EMOTION_ICONS[emotion]} {EMOTION_NAME_MAP[emotion]}
                </span>
                <span className="evidence-dots">
                  {hasEv ? (
                    <span className="evidence-text">{emotionEvidence.slice(0, 3).join(' | ')}{emotionEvidence.length > 3 ? '...' : ''}</span>
                  ) : (
                    <span className="no-evidence-text">无证据</span>
                  )}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    )
  }
}

const parseStep2DimensionalAnalysis = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const vad = {
    valence: data.valence ?? 0.5,
    arousal: data.arousal ?? 0.5,
    dominance: data.dominance ?? 0.5,
  }
  const emotionMapping = data.emotion_mapping || ''

  const getVadLabelFromMapping = (mapping: string): Record<string, string> => {
    const result: Record<string, string> = { valence: '中', arousal: '中', dominance: '中' }
    if (!mapping) return result
    
    if (/高效价|高正效价|高效/.test(mapping) && !/低效价/.test(mapping)) result.valence = '高'
    else if (/低效价|低负效价/.test(mapping)) result.valence = '低'
    
    if (/高唤醒/.test(mapping) && !/低唤醒/.test(mapping)) result.arousal = '高'
    else if (/低唤醒/.test(mapping)) result.arousal = '低'
    
    if (/高支配/.test(mapping) && !/低支配/.test(mapping)) result.dominance = '高'
    else if (/低支配/.test(mapping)) result.dominance = '低'
    
    return result
  }

  const vadLabels = getVadLabelFromMapping(emotionMapping)

  return {
    type: 'step2',
    content: (
      <div className="cot-step2-content">
        <div className="vad-bars">
          {(Object.entries(vad) as [keyof typeof vad, number][]).map(([key, value]) => {
            const absValue = Math.abs(value)
            const label = vadLabels[key]
            const color = label === '高' ? '#52c41a' : label === '低' ? '#ff4d4f' : '#faad14'
            return (
              <div key={key} className="vad-bar-row">
                <span className="vad-bar-label">{VAD_LABELS[key]}</span>
                <Progress
                  percent={Math.round(absValue * 100)}
                  strokeColor={color}
                  trailColor="rgba(148, 163, 184, 0.2)"
                  size="small"
                  showInfo={false}
                />
                <span className="vad-bar-value">
                  {value.toFixed(2)} ({VAD_VALUE_LABELS[key](value)})
                </span>
              </div>
            )
          })}
        </div>
        {emotionMapping && (
          <div className="emotion-mapping">
            <span className="mapping-arrow">→</span>
            <span className="mapping-text">{emotionMapping}</span>
          </div>
        )}
      </div>
    )
  }
}

const parseStep3NegationDetection = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const negationsFound = data.negations_found ?? data.negationsFound ?? false
  const sarcasmDetected = data.sarcasm_detected ?? data.sarcasmDetected ?? false
  const adjustedScores = data.adjusted_scores || data.adjustedScores || {}

  const adjustments = Object.entries(adjustedScores)
  const hasAdjustments = adjustments.length > 0

  return {
    type: 'step3',
    content: (
      <div className="cot-step3-content">
        <div className="detection-results">
          <div className="detection-row">
            <span className="detection-icon">{negationsFound ? '⚠️' : '✅'}</span>
            <span className="detection-label">否定词检测</span>
            <Tag color={negationsFound ? 'orange' : 'green'}>
              {negationsFound ? '检测到' : '未检测到'}
            </Tag>
          </div>
          <div className="detection-row">
            <span className="detection-icon">{sarcasmDetected ? '⚠️' : '✅'}</span>
            <span className="detection-label">讽刺/反讽检测</span>
            <Tag color={sarcasmDetected ? 'orange' : 'green'}>
              {sarcasmDetected ? '检测到' : '未检测到'}
            </Tag>
          </div>
        </div>
        {hasAdjustments && (
          <div className="score-adjustments">
            <div className="adjustments-title">📝 分数调整</div>
            {adjustments.map(([emotion, adjustment]: [string, any]) => (
              <div key={emotion} className="adjustment-row">
                <Tag color={EMOTION_COLORS[emotion]}>{EMOTION_NAME_MAP[emotion] || emotion}</Tag>
                <span className="adjustment-text">{String(adjustment)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }
}

const parseStep4CauseExtraction = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const primaryCause = data.primary_cause || data.primaryCause || ''
  const secondaryCauses = data.secondary_causes || data.secondaryCauses || []

  return {
    type: 'step4',
    content: (
      <div className="cot-step4-content">
        <div className="cause-section">
          <div className="cause-label">🎯 主要原因</div>
          <div className="cause-text primary-cause">{primaryCause}</div>
        </div>
        {secondaryCauses.length > 0 && (
          <div className="cause-section">
            <div className="cause-label">📌 次要原因</div>
            <ul className="secondary-causes-list">
              {secondaryCauses.map((cause: string, i: number) => (
                <li key={i}>{cause}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    )
  }
}

const parseStep5ConsistencyCheck = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const vadConsistent = data.vad_consistent ?? data.vadConsistent ?? true
  const inconsistencies = data.inconsistencies || []

  return {
    type: 'step5',
    content: (
      <div className="cot-step5-content">
        <div className="consistency-result">
          <Tag color={vadConsistent ? 'green' : 'red'} className="consistency-tag">
            {vadConsistent ? '✅ 一致' : '❌ 不一致'}
          </Tag>
          <span className="consistency-label">VAD一致性检验</span>
        </div>
        {inconsistencies.length > 0 ? (
          <div className="inconsistencies">
            <div className="inconsistencies-title">⚠️ 发现的不一致项</div>
            <ul className="inconsistencies-list">
              {inconsistencies.map((item: string, i: number) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="no-inconsistencies">无不一致项</div>
        )}
      </div>
    )
  }
}

const parseStep6UncertaintyCalibration = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const confidenceLevel = data.confidence_level || data.confidenceLevel || 'medium'
  const uncertainRegions = data.uncertain_regions || data.uncertainRegions || []
  const calibrationNotes = data.calibration_notes || data.calibrationNotes || ''

  const reliabilityInfo = RELIABILITY_LABELS[confidenceLevel] || RELIABILITY_LABELS.medium

  return {
    type: 'step6',
    content: (
      <div className="cot-step6-content">
        <div className="confidence-level">
          <Tag
            style={{ background: reliabilityInfo.color, borderColor: reliabilityInfo.color, color: '#fff' }}
            className="confidence-tag"
          >
            🛡️ 推理可靠性：{reliabilityInfo.label}
          </Tag>
        </div>
        {uncertainRegions.length > 0 ? (
          <div className="uncertain-regions">
            <div className="regions-title">⚠️ 不确定区域</div>
            <ul className="regions-list">
              {uncertainRegions.map((region: string, i: number) => (
                <li key={i}>{region}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="no-uncertain-regions">无不确定区域</div>
        )}
        {calibrationNotes && (
          <div className="calibration-notes">
            <div className="notes-title">📋 校准说明</div>
            <div className="notes-text">{calibrationNotes}</div>
          </div>
        )}
      </div>
    )
  }
}

const parseStep7FaithfulSynthesis = (data: any): ParsedCotContent => {
  if (!data || typeof data !== 'object') return { type: 'text', content: JSON.stringify(data) }

  const adjustmentLog = data.adjustment_log || data.adjustmentLog || ''
  const hallucinationFlags = data.hallucination_flags || data.hallucinationFlags || []

  const parseAdjustmentLog = (log: string) => {
    const adjustments: { emotion: string; from: string; to: string }[] = []
    const matches = log.matchAll(/([a-zA-Z_]+):\s*([\d.]+)\s*→\s*([\d.]+)/g)
    for (const match of matches) {
      adjustments.push({
        emotion: match[1],
        from: match[2],
        to: match[3],
      })
    }
    return adjustments
  }

  const adjustments = parseAdjustmentLog(adjustmentLog)
  const hasHallucinations = hallucinationFlags.length > 0

  return {
    type: 'step7',
    content: (
      <div className="cot-step7-content">
        {adjustments.length > 0 && (
          <div className="adjustments-section">
            <div className="adjustments-title">📊 分数调整</div>
            <div className="adjustments-list">
              {adjustments.map((adj, i) => (
                <div key={i} className="adjustment-item">
                  <Tag color={EMOTION_COLORS[adj.emotion]}>{EMOTION_NAME_MAP[adj.emotion] || adj.emotion}</Tag>
                  <span className="adjustment-arrow">{adj.from} → {adj.to}</span>
                  <span className="adjustment-diff">
                    ({parseFloat(adj.to) > parseFloat(adj.from) ? '+' : ''}
                    {(parseFloat(adj.to) - parseFloat(adj.from)).toFixed(2)})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="hallucination-section">
          <div className="hallucination-title">
            {hasHallucinations ? '⚠️ 潜在幻觉标记' : '✅ 幻觉检测'}
          </div>
          {hasHallucinations ? (
            <ul className="hallucination-list">
              {hallucinationFlags.map((flag: string, i: number) => (
                <li key={i}>{flag}</li>
              ))}
            </ul>
          ) : (
            <div className="no-hallucinations">未检测到潜在幻觉</div>
          )}
        </div>
      </div>
    )
  }
}

export const formatEmotionLabel = (emotion: string): string => {
  return EMOTION_LABELS[emotion] || emotion
}

export const formatVadPreview = (vad: Record<string, number> | undefined): string => {
  if (!vad) return '-'
  const valence = vad.valence?.toFixed(2) || '-'
  const arousal = vad.arousal?.toFixed(2) || '-'
  const dominance = vad.dominance?.toFixed(2) || '-'
  return `V: ${valence}\nA: ${arousal}\nD: ${dominance}`
}

export const formatUncertaintyBadge = (level: string | undefined): React.ReactNode => {
  if (!level) return '-'
  const info = UNCERTAINTY_LABELS[level] || UNCERTAINTY_LABELS.medium
  return (
    <Tag
      style={{ background: info.color, borderColor: info.color, color: '#fff', margin: 0 }}
    >
      {info.icon} {info.label}
    </Tag>
  )
}
