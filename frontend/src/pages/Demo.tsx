import React, { useState, useEffect } from 'react'
import { Card, Input, Button, Spin, Alert, Tag, Space, Progress, Divider, Collapse, Row, Col, Typography, Select, Slider, Switch, Tooltip } from 'antd'
import { SendOutlined, StopOutlined, ThunderboltOutlined, SettingOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { sentimentApi, InferResponse, StreamChunk } from '@/services/api'
import NeuralBackground from '@/components/NeuralBackground'
import '../assets/styles/design-system.scss'
import './Demo.scss'

const { TextArea } = Input
const { Title, Paragraph } = Typography

const EMOTION_LABELS: Record<string, string> = {
  angry: '愤怒',
  fear: '恐惧',
  happy: '高兴',
  neutral: '中性',
  sad: '悲伤',
  surprise: '惊讶',
}

const EMOTION_COLORS: Record<string, string> = {
  angry: '#dc8888',
  fear: '#a894c4',
  happy: '#88c4a8',
  neutral: '#b0a8a0',
  sad: '#88a4c4',
  surprise: '#dcc488',
}

const EMOTION_ICONS: Record<string, string> = {
  angry: '😠',
  fear: '😨',
  happy: '😄',
  neutral: '😐',
  sad: '😢',
  surprise: '😮',
}

const VAD_LABELS: Record<string, string> = {
  valence: '效价',
  arousal: '唤醒度',
  dominance: '支配度',
}

const VAD_VALUE_LABELS: Record<string, (v: number) => string> = {
  valence: (v) => v > 0.65 ? '正' : v < 0.35 ? '负' : '中性',
  arousal: (v) => v > 0.65 ? '高' : v < 0.35 ? '低' : '中',
  dominance: (v) => v > 0.65 ? '高' : v < 0.35 ? '低' : '中',
}

const UNCERTAINTY_LABELS: Record<string, { label: string; color: string; icon: string }> = {
  low: { label: '低', color: '#52c41a', icon: '🟢' },
  medium: { label: '中', color: '#faad14', icon: '🟡' },
  high: { label: '高', color: '#ff4d4f', icon: '🔴' },
}

const RELIABILITY_LABELS: Record<string, { label: string; color: string }> = {
  high: { label: '高', color: '#52c41a' },
  medium: { label: '中', color: '#faad14' },
  low: { label: '低', color: '#ff4d4f' },
}

const STEP_TITLES: Record<string, string> = {
  step1_lexical_grounding: '📌 证据提取',
  step2_dimensional_analysis: '📊 VAD维度分析',
  step3_negation_detection: '🔍 否定/反讽检测',
  step4_cause_extraction: '💡 情感归因',
  step5_consistency_check: '✅ 一致性检验',
  step6_uncertainty_calibration: '📈 置信度校准',
  step7_faithful_synthesis: '🔄 忠诚综合',
}

const formatLatency = (ms: number | undefined): string => {
  if (ms === undefined || ms === null || ms < 0) return '-'
  if (ms < 1000) {
    return `${Math.round(ms)}ms`
  }
  return `${(ms / 1000).toFixed(2)}s`
}

const EMOTION_NAME_MAP: Record<string, string> = {
  angry: '愤怒',
  fear: '恐惧',
  happy: '高兴',
  neutral: '中性',
  sad: '悲伤',
  surprise: '惊讶',
}

interface ParsedCotContent {
  type: string
  content: React.ReactNode
}

const parseCotContent = (stepKey: string, content: string, primaryEmotion: string): ParsedCotContent => {
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
  const step1Precheck = data.step1_precheck || []
  const dualCheck = data.dual_check || {}

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

  const renderPrecheck = () => {
    if (!Array.isArray(step1Precheck) || step1Precheck.length === 0) return null
    return (
      <div className="cot-section">
        <div className="cot-section-title">📋 候选情绪词筛查</div>
        <div className="precheck-table">
          <table className="precheck-table-inner">
            <thead>
              <tr>
                <th>候选词</th>
                <th>原文分句</th>
                <th>文本支撑</th>
                <th>归属情绪</th>
              </tr>
            </thead>
            <tbody>
              {step1Precheck.map((item: any, idx: number) => {
                const hasSupport = item['文本支撑'] === '✓' || item.文本支撑 === true
                return (
                  <tr key={idx} className={hasSupport ? 'has-support' : 'no-support'}>
                    <td><Tag color={hasSupport ? 'green' : 'red'}>{item['候选词'] || item.候选词}</Tag></td>
                    <td className="precheck-sentence">{item['原文分句'] || item.原文分句}</td>
                    <td>
                      {hasSupport ? (
                        <span style={{ color: '#52c41a', fontWeight: 'bold' }}>✓</span>
                      ) : (
                        <span style={{ color: '#ff4d4f', fontWeight: 'bold' }}>✗</span>
                      )}
                    </td>
                    <td><Tag>{item['归属情绪key'] || item.归属情绪key || '-'}</Tag></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderDualCheck = () => {
    if (!dualCheck || Object.keys(dualCheck).length === 0) return null
    const checks = [
      { key: 'evidence_to_cues', label: '正向约束（evidence→cues）' },
      { key: 'cues_to_evidence', label: '反向约束（cues→evidence）' },
      { key: 'sarcasm_归属', label: '反讽归属' },
    ]
    return (
      <div className="cot-section">
        <div className="cot-section-title">🔄 双向对应自检</div>
        <div className="dual-check-list">
          {checks.map(({ key, label }) => {
            const value = dualCheck[key]
            if (!value) return null
            return (
              <div key={key} className="dual-check-item">
                <span className="dual-check-label">{label}：</span>
                <span className="dual-check-value">{value}</span>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  return {
    type: 'step1',
    content: (
      <div className="cot-step1-content">
        {renderPrecheck()}
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
            const hasEvidence = emotionEvidence.length > 0
            return (
              <div key={emotion} className={`evidence-row ${hasEvidence ? 'has-evidence' : 'no-evidence'}`}>
                <span className="evidence-emotion">
                  {EMOTION_ICONS[emotion]} {EMOTION_NAME_MAP[emotion]}
                </span>
                <span className="evidence-dots">
                  {hasEvidence ? (
                    <span className="evidence-text">{emotionEvidence.slice(0, 3).join(' | ')}{emotionEvidence.length > 3 ? '...' : ''}</span>
                  ) : (
                    <span className="no-evidence-text">无证据</span>
                  )}
                </span>
              </div>
            )
          })}
        </div>
        {renderDualCheck()}
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

const getDisplayScores = (result: InferResponse): Record<string, number> => {
  if (result.target_scores && Object.keys(result.target_scores).length > 0) {
    const hasValues = Object.values(result.target_scores).some(v => v > 0)
    if (hasValues) {
      return result.target_scores
    }
  }
  return result.scores
}

const getRawScores = (result: InferResponse): Record<string, number> => {
  if (result.raw_intensity_scores && Object.keys(result.raw_intensity_scores).length > 0) {
    const hasValues = Object.values(result.raw_intensity_scores).some(v => v > 0)
    if (hasValues) {
      return result.raw_intensity_scores
    }
  }
  return result.scores
}

const SAMPLE_TEXTS = [
  { text: '我真的要吐槽一下某国产手机品牌！！用了一年电池就不耐用了，充电还发烫？？？售后服务态度也超差，踢皮球踢得比球队还专业😤 以后再也不买他们家东西了，性价比简直是个笑话，谁买谁后悔系列！', label: '愤怒情绪', emotion: 'angry' },
  { text: '救命！体检报告上有一项肿瘤标志物超标了，虽然医生说可能是炎症引起的让复查，但我还是超级紧张😰 搜了一下越看越害怕，这几天估计都睡不好觉了...有没有懂的姐妹告诉我问题大不大啊？', label: '恐惧情绪', emotion: 'fear' },
  { text: '天呐天呐天呐！！！我上岸啦！！！等了三个月的编制终于公示了，笔面都是第一🎉🎉🎉 爸妈知道后高兴得眼眶都红了，我也终于可以告慰这两年拼命努力的自己了！！感谢自己没有放弃，未来的路还很长，继续加油吧💪', label: '开心情绪', emotion: 'happy' },
  { text: '#今日分享# 今天看了一部电影，讲述了一个人在职场中从基层员工成长为管理者的故事。剧情比较平淡，没有太多反转，但演员的演技还不错。整体来说是一部中规中矩的职业剧，适合周末放松时观看，不踩雷也不惊艳。', label: '中性情绪', emotion: 'neutral' },
  { text: '今天送走了陪伴我十二年的狗狗，它走得很安详...😢 从高中到工作，它见证了我人生中最重要的阶段，每次回家它都第一个冲过来摇尾巴。现在家里突然安静了，真的很不习惯...谢谢你小黄，来汪星球要快乐啊，我会永远记得你的🥺', label: '悲伤情绪', emotion: 'sad' },
  { text: '我的天！！！刷到高中暗恋三年的男神居然当明星了？？？他演的网剧刚上了热搜，我反复确认了八百遍真的是他😭😭 从前的土味少年现在居然这么帅！！评论区还有人说像某个顶流...我整个人都震惊了，姐妹们快帮我看看是不是我眼花了！！', label: '惊讶情绪', emotion: 'surprise' },
]

const Demo: React.FC = () => {
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<InferResponse | null>(null)
  const [error, setError] = useState('')
  const [activeKey, setActiveKey] = useState<string[]>(['scores'])
  const [streamingOutput, setStreamingOutput] = useState('')
  const [streamingCot, setStreamingCot] = useState<Record<string, string>>({})
  const [streamingLatency, setStreamingLatency] = useState<number>(0)
  const [useStreaming, setUseStreaming] = useState(true)
  const [retryInfo, setRetryInfo] = useState<{count: number; error: string} | null>(null)
  const [riskWarning, setRiskWarning] = useState<string | null>(null)
  
  // Inference settings
  const [inferencePreset, setInferencePreset] = useState<'quick' | 'standard' | 'deep'>('standard')
  const [showSettings, setShowSettings] = useState(false)
  const [customParams, setCustomParams] = useState({
    temperature: 0.10,
    maxTokens: 2560,
    topP: 0.80,
    repeatPenalty: 1.08,
  })
  const [useCustomParams, setUseCustomParams] = useState(false)

  useEffect(() => {
    const savedText = localStorage.getItem('demo_text')
    if (savedText) {
      setInputText(savedText)
      localStorage.removeItem('demo_text')
    }
  }, [])

  const parseJsonSections = (output: string) => {
    // Try to extract JSON from markdown code blocks first
    let jsonContent = null
    
    const jsonMatch = output.match(/```json\s*([\s\S]*?)(?:```|$)/)
    if (jsonMatch) {
      jsonContent = jsonMatch[1].trim()
    } else {
      // Try to find raw JSON (between { and })
      const rawJsonMatch = output.match(/\{[\s\S]*\}/)
      if (rawJsonMatch) {
        jsonContent = rawJsonMatch[0]
      }
    }
    
    if (!jsonContent) return null

    try {
      return JSON.parse(jsonContent)
    } catch {
      // JSON may be incomplete during streaming, try to fix common issues
      try {
        let fixed = jsonContent
          .replace(/,\s*}/g, '}')
          .replace(/,\s*]/g, ']')
        return JSON.parse(fixed)
      } catch {
        return null
      }
    }
  }

  const handleInfer = async () => {
    if (!inputText.trim()) {
      setError('请输入要分析的文本')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setStreamingOutput('')
    setStreamingCot({})
    setStreamingLatency(0)

    try {
      if (useStreaming) {
        // Streaming mode - progressive display
        const requestParams: any = {
          text: inputText,
          model_variant: 'gguf4bit',
          preset: useCustomParams ? undefined : inferencePreset,
        }
        
        if (useCustomParams) {
          requestParams.temperature = customParams.temperature
          requestParams.max_tokens = customParams.maxTokens
          requestParams.top_p = customParams.topP
          requestParams.repeat_penalty = customParams.repeatPenalty
        }
        
        const reader = await sentimentApi.inferStream(requestParams)

        let finalChunk: StreamChunk | null = null

        for await (const chunk of reader) {
          if (chunk.error) {
            throw new Error(chunk.error)
          }

          finalChunk = chunk
          setStreamingOutput(chunk.output || '')
          setStreamingLatency(chunk.latency_ms || 0)

          // Handle retry signal from backend
          if (chunk.retry && chunk.retry_count) {
            setRetryInfo({ count: chunk.retry_count, error: chunk.error_msg || '校验失败，正在重试...' })
          }

          // Try to parse partial JSON and update CoT for streaming display
          if (chunk.output) {
            const parsed = parseJsonSections(chunk.output)
            if (parsed?.cot_reasoning_chain) {
              setStreamingCot(parsed.cot_reasoning_chain)
            }
          }

          if (chunk.done) {
            break
          }
        }

        // Final parse - use backend-calculated values if available
        const finalOutput = finalChunk?.output || streamingOutput
        
        // Check if backend returned calculated values in the final chunk
        // Use target_scores (CoT-adjusted) for the condition check
        if (finalChunk?.done && (finalChunk.target_scores || finalChunk.scores) && finalChunk.primary_emotion !== undefined) {
          // Use backend-calculated values for consistency with history
          const displayScores = finalChunk.target_scores || finalChunk.scores || {}
          const rawScores = finalChunk.raw_intensity_scores || finalChunk.scores || {}
          const formatted: InferResponse = {
            output: finalOutput,
            scores: rawScores,
            raw_intensity_scores: rawScores,
            target_scores: finalChunk.target_scores || {},
            primary_emotion: finalChunk.primary_emotion || 'neutral',
            confidence: finalChunk.confidence !== undefined ? finalChunk.confidence : 0,
            cot: finalChunk.cot || {},
            json_parse_ok: finalChunk.json_parse_ok ?? true,
            cot_complete: finalChunk.cot_complete,
            latency_ms: finalChunk.latency_ms || streamingLatency,
            text: inputText,
            vad_dimensions: finalChunk.vad_dimensions,
            emotion_cause: finalChunk.emotion_cause,
            uncertainty_level: finalChunk.uncertainty_level,
          }
          setResult(formatted)
          setRiskWarning(finalChunk.risk_warning || null)
          setActiveKey(['result', 'scores', 'cot'])
        } else {
          // Fallback to frontend parsing if backend didn't return parsed values
          const finalParsed = parseJsonSections(finalOutput)
          
          if (finalParsed) {
            // Get primary emotion from target_scores
            let primaryEmotion = 'neutral'
            const finalScores = finalParsed.target_scores || {}
            if (Object.keys(finalScores).length > 0) {
              primaryEmotion = Object.entries(finalScores).reduce((max, [e, v]) => 
                (v as number) > (finalScores[max] as number || 0) ? e : max, 'neutral')
            }
            
            const formatted = {
              output: finalOutput,
              scores: finalScores,
              target_scores: finalParsed.target_scores || {},
              primary_emotion: primaryEmotion,
              confidence: 0,
              cot: finalParsed.cot_reasoning_chain || {},
              json_parse_ok: true,
              latency_ms: finalChunk?.latency_ms || streamingLatency,
              text: inputText,
              vad_dimensions: finalParsed.vad_dimensions,
              emotion_cause: finalParsed.emotion_cause,
              uncertainty_level: finalParsed.uncertainty_level,
            }
            setResult(formatted)
            setActiveKey(['result', 'scores', 'cot'])
          } else {
            // If no valid result, show raw output with error
            setError('推理完成但无法解析结果，请检查输出格式')
            setResult({
              output: finalOutput,
              scores: {},
              primary_emotion: 'unknown',
              confidence: 0,
              cot: {},
              json_parse_ok: false,
              latency_ms: finalChunk?.latency_ms || streamingLatency,
              text: inputText,
            } as InferResponse)
          }
        }
      } else {
        // Non-streaming mode - wait for complete response
        const requestParams: any = {
          text: inputText,
          model_variant: 'gguf4bit',
          preset: useCustomParams ? undefined : inferencePreset,
        }
        
        if (useCustomParams) {
          requestParams.temperature = customParams.temperature
          requestParams.max_tokens = customParams.maxTokens
          requestParams.top_p = customParams.topP
          requestParams.repeat_penalty = customParams.repeatPenalty
        }
        
        const response = await sentimentApi.infer(requestParams)
        // Use backend-calculated confidence for consistency
        setResult(response)
        setActiveKey(['result', 'scores', 'cot'])
      }
    } catch (err: any) {
      setError(err.message || '推理失败，请检查后端服务是否启动')
    } finally {
      setLoading(false)
    }
  }

  const handleLoadSample = (text: string) => {
    setInputText(text)
    setResult(null)
    setError('')
  }

  const getBarChartOption = (scores: Record<string, number>) => {
    const data = Object.entries(scores).map(([key, value]) => ({
      name: EMOTION_LABELS[key] || key,
      value: (value * 100).toFixed(1),
      itemStyle: { color: EMOTION_COLORS[key] || '#999' },
    }))

    return {
      animation: true,
      animationDuration: 800,
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: '{b}: {c}%',
        backgroundColor: 'rgba(36, 48, 67, 0.95)',
        borderColor: 'rgba(148, 163, 184, 0.2)',
        textStyle: { color: '#f1f5f9' },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: data.map((d) => d.name),
        axisLabel: { color: '#94a3b8' },
      },
      yAxis: {
        type: 'value',
        max: 100,
        axisLabel: { formatter: '{value}%', color: '#94a3b8' },
        splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.1)' } },
      },
      series: [
        {
          name: '情绪强度',
          type: 'bar',
          data,
          itemStyle: { borderRadius: [8, 8, 0, 0] },
          label: {
            show: true,
            position: 'top',
            formatter: '{c}%',
            fontSize: 11,
            color: '#94a3b8',
          },
          animationDelay: (idx: number) => idx * 80,
        },
      ],
    }
  }

  const getSpectrumOption = (scores: Record<string, number>) => {
    const data = Object.entries(scores).map(([key, value]) => ({
      name: EMOTION_LABELS[key] || key,
      value: value * 100,
      itemStyle: { color: EMOTION_COLORS[key] || '#999' },
    }))

    return {
      animation: true,
      animationDuration: 1000,
      tooltip: {
        formatter: '{b}: {c}%',
        backgroundColor: 'rgba(36, 48, 67, 0.95)',
        borderColor: 'rgba(148, 163, 184, 0.2)',
        textStyle: { color: '#f1f5f9' },
      },
      series: [
        {
          name: '情绪光谱',
          type: 'pie',
          radius: ['50%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: 'rgba(36, 48, 67, 0.8)',
            borderWidth: 2,
          },
          label: {
            show: true,
            formatter: '{b}\n{c}%',
            fontSize: 12,
            color: '#f1f5f9',
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 14,
              fontWeight: 'bold',
            },
          },
          data,
        },
      ],
    }
  }

  return (
    <div className="demo-page">
      <NeuralBackground density={0.3} speed={0.5} opacity={0.4} />

      <div className="page-content">
        {/* Page Header */}
        <div className="page-header">
          <Title level={1} className="page-title">
            <ThunderboltOutlined style={{ marginRight: 12, color: EMOTION_COLORS.surprise }} />
            在线演示
          </Title>
          <Paragraph className="page-subtitle">
            输入文本或使用示例，实时获取多任务情感分析结果
          </Paragraph>
        </div>

        {/* Sample Text Cards */}
        <div className="samples-section">
          <div className="section-header">
            <Title level={3} className="section-title">示例文本</Title>
            <Paragraph className="section-subtitle">点击下方卡片快速加载示例</Paragraph>
          </div>
          <Row gutter={[16, 16]}>
            {SAMPLE_TEXTS.map((sample, index) => (
              <Col xs={24} sm={12} md={8} key={index}>
                <div
                  className="sample-card"
                  onClick={() => handleLoadSample(sample.text)}
                  style={{ borderLeftColor: EMOTION_COLORS[sample.emotion] }}
                >
                  <div className="sample-header">
                    <Tag
                      className="sample-tag"
                      style={{
                        background: EMOTION_COLORS[sample.emotion],
                        borderColor: EMOTION_COLORS[sample.emotion],
                      }}
                    >
                      {sample.label}
                    </Tag>
                  </div>
                  <Paragraph className="sample-text">"{sample.text}"</Paragraph>
                  <div className="sample-footer">
                    <span>点击加载</span>
                    <SendOutlined style={{ fontSize: 14 }} />
                  </div>
                </div>
              </Col>
            ))}
          </Row>
        </div>

        {/* Inference Settings Card */}
        <Card className="demo-card">
          <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <span style={{ marginRight: 8 }}>⚙️</span>
              推理设置
            </div>
            <Button
              icon={<SettingOutlined />}
              onClick={() => setShowSettings(!showSettings)}
              type={showSettings ? 'primary' : 'default'}
              size="small"
            >
              {showSettings ? '收起' : '展开'}
            </Button>
          </div>
          
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: '#94a3b8' }}>预设模式:</span>
              <Select
                value={inferencePreset}
                onChange={setInferencePreset}
                style={{ width: 120 }}
                disabled={useCustomParams}
                options={[
                  { value: 'quick', label: '快速 (Quick)' },
                  { value: 'standard', label: '标准 (Standard)' },
                  { value: 'deep', label: '深度 (Deep)' },
                ]}
              />
            </div>
            
            <Tooltip
              title={<>Quick：短句/单情绪（≤50字），基础CoT，tokens=1536<br />Standard【推荐】：常规文本（50-300字），完整7步CoT，tokens=2560<br />Deep：长文本/混合情绪/反讽（≥300字），tokens=4096</>}
            >
              <Button type="link" size="small">?</Button>
            </Tooltip>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: '#94a3b8' }}>自定义参数:</span>
              <Switch
                checked={useCustomParams}
                onChange={(checked) => {
                  setUseCustomParams(checked)
                  if (checked) {
                    setShowSettings(true)
                  }
                }}
                size="small"
              />
            </div>
          </div>
          
          {showSettings && useCustomParams && (
            <div style={{
              marginTop: 16,
              padding: 16,
              background: 'var(--theme-bg-tertiary, rgba(35, 48, 68, 0.5))',
              borderRadius: 8,
              border: '1px solid var(--theme-border-color, rgba(148, 163, 184, 0.2))',
              color: 'var(--ant-text-color-secondary, #94a3b8)'
            }}>
              <Row gutter={[24, 16]}>
                <Col span={12}>
                  <div style={{ marginBottom: 8 }}>
                    Temperature (随机性): {customParams.temperature}
                  </div>
                  <Slider
                    min={0.01}
                    max={0.3}
                    step={0.01}
                    value={customParams.temperature}
                    onChange={(v) => setCustomParams({ ...customParams, temperature: v })}
                    marks={{ 0.01: '0.01', 0.1: '0.10', 0.2: '0.20', 0.3: '0.30' }}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 8 }}>
                    Max Tokens (最大长度): {customParams.maxTokens}
                  </div>
                  <Slider
                    min={512}
                    max={4096}
                    step={256}
                    value={customParams.maxTokens}
                    onChange={(v) => setCustomParams({ ...customParams, maxTokens: v })}
                    marks={{ 512: '512', 1536: '1536', 2560: '2560', 4096: '4096' }}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 8 }}>
                    Top P (采样范围): {customParams.topP}
                  </div>
                  <Slider
                    min={0.5}
                    max={0.95}
                    step={0.05}
                    value={customParams.topP}
                    onChange={(v) => setCustomParams({ ...customParams, topP: v })}
                    marks={{ 0.5: '0.50', 0.7: '0.70', 0.85: '0.85', 0.95: '0.95' }}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 8 }}>
                    Repeat Penalty (重复惩罚): {customParams.repeatPenalty}
                  </div>
                  <Slider
                    min={1.0}
                    max={1.2}
                    step={0.02}
                    value={customParams.repeatPenalty}
                    onChange={(v) => setCustomParams({ ...customParams, repeatPenalty: v })}
                    marks={{ 1.0: '1.00', 1.05: '1.05', 1.10: '1.10', 1.2: '1.20' }}
                  />
                </Col>
              </Row>
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--ant-text-color-secondary, #64748b)' }}>
                * Temperature越低输出越稳定（推荐0.08-0.12）；Repeat Penalty防止重复输出（推荐1.05-1.10）
              </div>
            </div>
          )}
        </Card>

        {/* Input Card */}
        <Card className="demo-card">
          <div className="card-title">
            <span style={{ marginRight: 8 }}>✏️</span>
            文本输入
          </div>
          <TextArea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="请输入要分析的文本内容，或点击上方示例文本快速加载..."
            rows={6}
            maxLength={2000}
            showCount
            className="demo-input"
          />
          <Space size="large" className="demo-actions">
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handleInfer}
              loading={loading}
              size="large"
              className="btn-analyze"
            >
              开始分析
            </Button>
            <Button
              danger
              icon={<StopOutlined />}
              onClick={() => {
                setInputText('')
                setResult(null)
                setError('')
                setStreamingOutput('')
                setStreamingCot({})
              }}
              size="large"
            >
              清空
            </Button>
            <Button
              type={useStreaming ? 'default' : 'dashed'}
              icon={<ThunderboltOutlined />}
              onClick={() => setUseStreaming(!useStreaming)}
              size="large"
              className={useStreaming ? 'btn-streaming-active' : ''}
            >
              流式输出：{useStreaming ? '开' : '关'}
            </Button>
          </Space>
        </Card>

        {error && (
          <Alert
            message="错误"
            description={error}
            type="error"
            showIcon
            className="error-alert"
          />
        )}

        {riskWarning && (
          <Alert
            message="⚠️ 合规性校验警告"
            description={riskWarning}
            type="warning"
            showIcon
            className="warning-alert"
          />
        )}

        {retryInfo && (
          <Alert
            message="🔄 重试中"
            description={`第 ${retryInfo.count} 次重试：${retryInfo.error}`}
            type="info"
            showIcon
            className="retry-alert"
          />
        )}

        {loading && useStreaming && (
          <div className="loading-container">
            <Spin size="large" tip="🤖 流式推理中，请稍候..." />
            {streamingOutput && (
              <Card className="demo-card streaming-output-card">
                <div className="card-title">
                  <span style={{ marginRight: 8 }}>📝</span>
                  实时输出
                </div>
                <div className="streaming-output">
                  <pre>{streamingOutput}</pre>
                </div>
                {streamingLatency > 0 && (
                  <div className="streaming-latency">
                    推理耗时：{formatLatency(streamingLatency)}
                  </div>
                )}
              </Card>
            )}
          </div>
        )}

        {loading && !useStreaming && (
          <div className="loading-container">
            <Spin size="large" tip="🤖 正在分析中，请稍候..." />
          </div>
        )}

        {result && !loading && (
          <>
            {/* Main Result Card */}
            <Card
              className="demo-card result-card"
              style={{
                border: `2px solid ${EMOTION_COLORS[result.primary_emotion]}`,
                boxShadow: `0 8px 32px ${EMOTION_COLORS[result.primary_emotion]}30`,
              }}
            >
              <div
                className="card-title result-title"
                style={{ color: EMOTION_COLORS[result.primary_emotion] }}
              >
                📊 分析结果
              </div>

              <Row gutter={24} className="result-metrics">
                <Col xs={12} sm={6}>
                  <div className="result-metric">
                    <div className="metric-label">主要情绪</div>
                    <Tag
                      className="result-tag"
                      style={{
                        background: EMOTION_COLORS[result.primary_emotion],
                        borderColor: EMOTION_COLORS[result.primary_emotion],
                      }}
                    >
                      {EMOTION_LABELS[result.primary_emotion] || result.primary_emotion}
                    </Tag>
                  </div>
                </Col>
                <Col xs={12} sm={6}>
                  <div className="result-metric">
                    <div className="metric-label">置信度</div>
                    <div className="metric-value" style={{ color: EMOTION_COLORS[result.primary_emotion] }}>
                      {(result.confidence * 100).toFixed(2)}%
                    </div>
                    <Progress
                      percent={Number((result.confidence * 100).toFixed(2))}
                      strokeColor={EMOTION_COLORS[result.primary_emotion]}
                      showInfo={false}
                      size="small"
                    />
                  </div>
                </Col>
                <Col xs={12} sm={6}>
                  <div className="result-metric">
                    <div className="metric-label">不确定性</div>
                    <Tag
                      className="result-tag"
                      style={{
                        background: (result.uncertainty_level ? UNCERTAINTY_LABELS[result.uncertainty_level]?.color : null) || '#faad14',
                        borderColor: (result.uncertainty_level ? UNCERTAINTY_LABELS[result.uncertainty_level]?.color : null) || '#faad14',
                        color: '#fff',
                      }}
                    >
                      {(result.uncertainty_level ? UNCERTAINTY_LABELS[result.uncertainty_level]?.icon : null) || '⚪'} {(result.uncertainty_level ? UNCERTAINTY_LABELS[result.uncertainty_level]?.label : null) || '待定'}
                    </Tag>
                  </div>
                </Col>
                <Col xs={12} sm={6}>
                  <div className="result-metric">
                    <div className="metric-label">推理耗时</div>
                    <div className="metric-value metric-mono">
                      {formatLatency(result.latency_ms)}
                    </div>
                  </div>
                </Col>
              </Row>

              {/* VAD Dimensions Display */}
              {result.vad_dimensions && (() => {
                const step2Data = result.cot?.step2_dimensional_analysis
                let emotionMapping = ''
                if (step2Data) {
                  try {
                    const parsed = typeof step2Data === 'string' ? JSON.parse(step2Data) : step2Data
                    emotionMapping = parsed.emotion_mapping || ''
                  } catch { emotionMapping = '' }
                }
                
                const getVadLabelFromMapping = (mapping: string): Record<string, string> => {
                  const labels: Record<string, string> = { valence: '中', arousal: '中', dominance: '中' }
                  if (!mapping) return labels
                  
                  if (/高效价|高正效价|高效/.test(mapping) && !/低效价/.test(mapping)) labels.valence = '高'
                  else if (/低效价|低负效价/.test(mapping)) labels.valence = '低'
                  
                  if (/高唤醒/.test(mapping) && !/低唤醒/.test(mapping)) labels.arousal = '高'
                  else if (/低唤醒/.test(mapping)) labels.arousal = '低'
                  
                  if (/高支配/.test(mapping) && !/低支配/.test(mapping)) labels.dominance = '高'
                  else if (/低支配/.test(mapping)) labels.dominance = '低'
                  
                  return labels
                }

                const vadLabels = getVadLabelFromMapping(emotionMapping)
                
                return (
                  <>
                    <Divider style={{ borderColor: 'rgba(148, 163, 184, 0.1)' }} />
                    <div className="vad-section">
                      <Title level={5} style={{ marginBottom: 12, textAlign: 'left' }}>
                        📊 VAD 维度分析
                      </Title>
                      <Row gutter={16}>
                        {(Object.entries(result.vad_dimensions) as [keyof typeof result.vad_dimensions, number][]).map(([key, value]) => {
                          const absValue = Math.abs(value)
                          const label = vadLabels[key]
                          const color = label === '高' ? '#52c41a' : label === '低' ? '#ff4d4f' : '#faad14'
                          return (
                            <Col span={8} key={key}>
                              <div className="vad-item">
                                <div className="vad-label">{VAD_LABELS[key]}</div>
                                <Progress
                                  percent={Math.round(absValue * 100)}
                                  strokeColor={color}
                                  trailColor="rgba(148, 163, 184, 0.2)"
                                  size="small"
                                />
                                <div className="vad-value">
                                  {value.toFixed(2)} ({VAD_VALUE_LABELS[key](value)})
                                </div>
                              </div>
                            </Col>
                          )
                        })}
                      </Row>
                    </div>
                  </>
                )
              })()}

              {/* Emotion Cause Display */}
              {result.emotion_cause && (
                <>
                  <Divider style={{ borderColor: 'rgba(148, 163, 184, 0.1)' }} />
                  <div className="emotion-cause-section">
                    <Title level={5} style={{ marginBottom: 8, textAlign: 'left' }}>
                      💡 情感归因
                    </Title>
                    <Paragraph
                      style={{
                        background: 'rgba(148, 163, 184, 0.1)',
                        padding: '12px 16px',
                        borderRadius: 8,
                        margin: 0,
                        fontStyle: 'italic',
                      }}
                    >
                      "{result.emotion_cause}"
                    </Paragraph>
                  </div>
                </>
              )}

              <Divider style={{ borderColor: 'rgba(148, 163, 184, 0.1)' }} />

              <Row gutter={24}>
                <Col span={12}>
                  <Title level={5} className="chart-title">📈 原始强度分布 (CoT 推理前)</Title>
                  <ReactECharts option={getBarChartOption(getRawScores(result))} style={{ height: 320 }} />
                </Col>
                <Col span={12}>
                  <Title level={5} className="chart-title">📈 归一化概率分布 (CoT 推理后)</Title>
                  <ReactECharts option={getBarChartOption(getDisplayScores(result))} style={{ height: 320 }} />
                </Col>
              </Row>
            </Card>

            {/* Streaming CoT Display - Show progressive reasoning */}
            {streamingCot && Object.keys(streamingCot).length > 0 && !result && (
              <Card className="demo-card cot-card streaming-cot-card">
                <div className="card-title">
                  <span style={{ marginRight: 8 }}>🧠</span>
                  思维链推理 (CoT) - 流式生成中
                </div>
                <div className="cot-steps">
                  {Object.entries(streamingCot).map(([stepKey, content], index) => {
                    const parsed = parseCotContent(stepKey, content, 'neutral')
                    return (
                      <div key={stepKey} className="cot-step" style={{ borderLeftColor: EMOTION_COLORS.neutral }}>
                        <div className="cot-step-header">
                          <span className="cot-step-number">{index + 1}</span>
                          <span className="cot-step-title">{STEP_TITLES[stepKey] || stepKey}</span>
                        </div>
                        <Paragraph className="cot-step-content">
                          {parsed.content}
                        </Paragraph>
                      </div>
                    )
                  })}
                </div>
              </Card>
            )}

            {/* CoT Inference Chain */}
            <Card className="demo-card cot-card">
              <div className="card-title">
                <span style={{ marginRight: 8 }}>🧠</span>
                思维链推理 (CoT)
              </div>
              <Collapse
                activeKey={activeKey}
                onChange={(keys) => setActiveKey(keys as string[])}
                items={[{
                  key: 'cot',
                  label: '点击展开/收起 7 步推理过程',
                  children: (
                    <div className="cot-steps">
                      {Object.entries(result.cot).map(([stepKey, content], index) => {
                        const parsed = parseCotContent(stepKey, content, result.primary_emotion)
                        return (
                          <div
                            key={stepKey}
                            className="cot-step"
                            style={{
                              borderLeftColor: EMOTION_COLORS[result.primary_emotion],
                            }}
                          >
                            <div className="cot-step-header">
                              <span className="cot-step-number">{index + 1}</span>
                              <span className="cot-step-title">
                                {STEP_TITLES[stepKey] || stepKey}
                              </span>
                            </div>
                            <Paragraph className="cot-step-content">
                              {parsed.content}
                            </Paragraph>
                          </div>
                        )
                      })}
                    </div>
                  ),
                }]}
                className="cot-collapse"
              />
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

export default Demo
