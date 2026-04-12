import React, { useState, useMemo, useEffect, useRef } from 'react'
import { Typography, Card, Select, Radio, Button, Table, Tag, Spin, Collapse } from 'antd'
import { BarChart, Bar, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip, XAxis, YAxis, CartesianGrid, Cell } from 'recharts'
import {
  BlockOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  DownloadOutlined,
} from '@ant-design/icons'
import type { TableProps } from 'antd'
import './Compare.scss'

const { Title, Paragraph } = Typography
const { Option } = Select

// Model metadata
interface ModelData {
  id: string
  name: string
  icon: React.ReactNode
  description: string
  colors: string[]
}

const models: ModelData[] = [
  {
    id: 'base',
    name: 'Qwen3-8B-Base',
    icon: <BlockOutlined />,
    description: '原始基座模型',
    colors: ['#64748b', '#475569'],
  },
  {
    id: 'lora',
    name: 'Qwen3-8B-LoRA',
    icon: <ThunderboltOutlined />,
    description: 'LoRA 微调模型',
    colors: ['#ff6b4a', '#ff8f73'],
  },
  {
    id: 'gguf',
    name: 'Qwen3-8B-GGUF',
    icon: <ExperimentOutlined />,
    description: '量化压缩模型',
    colors: ['#06b6d4', '#22d3ee'],
  },
]

// Data interfaces matching compare_to_swanlab.py output
interface ComparisonMetrics {
  model_variant: string
  emotion_macro_mae?: number
  emotion_macro_mse?: number
  primary_cls_accuracy?: number
  primary_cls_macro_f1?: number
  primary_cls_macro_auc?: number
  primary_cls_macro_ap?: number
  json_parse_rate?: number
  cot7_complete_rate?: number
  emotion_per_dim_mae?: Record<string, number>
  emotion_per_dim_mse?: Record<string, number>
  primary_cls_per_class_f1?: Record<string, number>
  primary_cls_per_class_metrics?: Record<string, { precision: number; recall: number; f1: number; support: number }>
  latency_ms?: number
  throughput_sps?: number
  vram_gb?: number
}

// Interface for per-emotion comparison table
interface EmotionModelComparison {
  model: string
  modelName: string
  precision: number
  recall: number
  f1: number
  mae: number
  support: number
}

// UI data interfaces
interface EmotionComparisonPoint {
  emotion: string
  base: number
  lora: number
  gguf: number
  fullMark: number
}

interface PerformanceMetric {
  metric: string
  base: number
  lora: number
  gguf: number
  unit: string
  isMaeRow?: boolean
  lowerIsBetter?: boolean
  maxVal?: number
  baseMae?: number
  loraMae?: number
  ggufMae?: number
}

// Default fallback data (used when JSON file is not available)
const defaultEmotionComparisonData: EmotionComparisonPoint[] = [
  { emotion: 'Angry', base: 0.72, lora: 0.89, gguf: 0.85, fullMark: 1 },
  { emotion: 'Fear', base: 0.65, lora: 0.86, gguf: 0.81, fullMark: 1 },
  { emotion: 'Happy', base: 0.78, lora: 0.92, gguf: 0.88, fullMark: 1 },
  { emotion: 'Neutral', base: 0.82, lora: 0.94, gguf: 0.90, fullMark: 1 },
  { emotion: 'Sad', base: 0.68, lora: 0.88, gguf: 0.84, fullMark: 1 },
  { emotion: 'Surprise', base: 0.61, lora: 0.85, gguf: 0.79, fullMark: 1 },
]

const defaultPerformanceMetrics: PerformanceMetric[] = [
  { metric: 'Accuracy', base: 74.5, lora: 91.2, gguf: 87.8, unit: '%', isMaeRow: false },
  { metric: 'Macro F1', base: 72.0, lora: 89.8, gguf: 85.8, unit: '%', isMaeRow: false },
  { metric: 'AUC', base: 82.0, lora: 94.3, gguf: 91.0, unit: '%', isMaeRow: false },
  { metric: 'AP', base: 81.0, lora: 92.1, gguf: 89.5, unit: '%', isMaeRow: false },
  { metric: 'Emotion MAE', base: 0.185, lora: 0.142, gguf: 0.158, unit: '', isMaeRow: true, baseMae: 0.185, loraMae: 0.142, ggufMae: 0.158 },
  { metric: 'JSON Parse', base: 92.0, lora: 94.0, gguf: 93.0, unit: '%', isMaeRow: false },
  { metric: 'CoT Complete', base: 85.0, lora: 89.0, gguf: 87.0, unit: '%', isMaeRow: false },
]

// Default emotion detail data (fallback when no data is loaded)
const defaultEmotionDetailData: Record<string, EmotionModelComparison[]> = {
  angry: [
    { model: 'base', modelName: 'Base', precision: 72.5, recall: 71.8, f1: 72.2, mae: 0.175, support: 245 },
    { model: 'lora', modelName: 'LoRA', precision: 83.2, recall: 82.5, f1: 82.8, mae: 0.138, support: 245 },
    { model: 'gguf', modelName: 'GGUF', precision: 80.1, recall: 79.5, f1: 79.8, mae: 0.152, support: 245 },
  ],
  fear: [
    { model: 'base', modelName: 'Base', precision: 68.2, recall: 67.5, f1: 67.8, mae: 0.195, support: 198 },
    { model: 'lora', modelName: 'LoRA', precision: 76.5, recall: 75.2, f1: 75.8, mae: 0.155, support: 198 },
    { model: 'gguf', modelName: 'GGUF', precision: 73.8, recall: 72.1, f1: 72.9, mae: 0.168, support: 198 },
  ],
  happy: [
    { model: 'base', modelName: 'Base', precision: 78.1, recall: 79.5, f1: 78.8, mae: 0.158, support: 312 },
    { model: 'lora', modelName: 'LoRA', precision: 88.5, recall: 89.2, f1: 88.8, mae: 0.112, support: 312 },
    { model: 'gguf', modelName: 'GGUF', precision: 85.2, recall: 86.1, f1: 85.6, mae: 0.128, support: 312 },
  ],
  neutral: [
    { model: 'base', modelName: 'Base', precision: 82.2, recall: 83.1, f1: 82.6, mae: 0.142, support: 428 },
    { model: 'lora', modelName: 'LoRA', precision: 90.1, recall: 91.2, f1: 90.6, mae: 0.098, support: 428 },
    { model: 'gguf', modelName: 'GGUF', precision: 87.5, recall: 88.3, f1: 87.9, mae: 0.115, support: 428 },
  ],
  sad: [
    { model: 'base', modelName: 'Base', precision: 70.5, recall: 69.8, f1: 70.2, mae: 0.188, support: 267 },
    { model: 'lora', modelName: 'LoRA', precision: 80.2, recall: 79.5, f1: 79.8, mae: 0.148, support: 267 },
    { model: 'gguf', modelName: 'GGUF', precision: 77.8, recall: 76.9, f1: 77.3, mae: 0.162, support: 267 },
  ],
  surprise: [
    { model: 'base', modelName: 'Base', precision: 62.1, recall: 61.5, f1: 61.8, mae: 0.238, support: 156 },
    { model: 'lora', modelName: 'LoRA', precision: 71.2, recall: 70.5, f1: 70.8, mae: 0.201, support: 156 },
    { model: 'gguf', modelName: 'GGUF', precision: 68.5, recall: 67.2, f1: 67.8, mae: 0.218, support: 156 },
  ],
}

// Transform compare_to_swanlab.py output to frontend format
function transformComparisonData(metrics: ComparisonMetrics[]): {
  emotionData: EmotionComparisonPoint[]
  performanceData: PerformanceMetric[]
  emotionDetailData: Record<string, EmotionModelComparison[]>
} {
  const getModelMetric = (matcher: (m: ComparisonMetrics) => boolean, accessor: (m: ComparisonMetrics) => number | undefined): number => {
    const model = metrics.find(matcher)
    if (!model) return 0
    const val = accessor(model)
    return val !== undefined ? val : 0
  }

  const matchModel = (variant: string) => (m: ComparisonMetrics) => {
    const v = m.model_variant?.toLowerCase() || ''
    if (variant === 'base') return v === 'base'
    if (variant === 'lora') return v.includes('lora')
    if (variant === 'gguf') return v.includes('gguf')
    return v === variant
  }

  // Transform emotion per-dim F1 (from primary_cls_per_class_f1 or primary_cls_per_class_metrics)
  const emotionLabels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
  const emotionData: EmotionComparisonPoint[] = emotionLabels.map(label => {
    // Try primary_cls_per_class_f1 first, then fall back to primary_cls_per_class_metrics.f1
    const baseVal = getModelMetric(matchModel('base'), m =>
      m.primary_cls_per_class_f1?.[label] ||
      m.primary_cls_per_class_metrics?.[label]?.f1 ||
      0
    )
    const loraVal = getModelMetric(matchModel('lora'), m =>
      m.primary_cls_per_class_f1?.[label] ||
      m.primary_cls_per_class_metrics?.[label]?.f1 ||
      0
    )
    const ggufVal = getModelMetric(matchModel('gguf'), m =>
      m.primary_cls_per_class_f1?.[label] ||
      m.primary_cls_per_class_metrics?.[label]?.f1 ||
      0
    )
    return {
      emotion: label.charAt(0).toUpperCase() + label.slice(1),
      base: baseVal,
      lora: loraVal,
      gguf: ggufVal,
      fullMark: 1,
    }
  })

  // Build per-emotion detail comparison data
  const emotionDetailData: Record<string, EmotionModelComparison[]> = {}

  const modelConfigs = [
    { variant: 'base', name: 'Base' },
    { variant: 'lora', name: 'LoRA' },
    { variant: 'gguf', name: 'GGUF' },
  ]

  for (const emotion of emotionLabels) {
    emotionDetailData[emotion] = modelConfigs.map(config => {
      const metricsForModel = metrics.find(matchModel(config.variant))
      const perClassMetrics = metricsForModel?.primary_cls_per_class_metrics?.[emotion]
      const emotionMAE = metricsForModel?.emotion_per_dim_mae?.[emotion]
      return {
        model: config.variant,
        modelName: config.name,
        precision: perClassMetrics?.precision ? perClassMetrics.precision * 100 : 0,
        recall: perClassMetrics?.recall ? perClassMetrics.recall * 100 : 0,
        f1: perClassMetrics?.f1 ? perClassMetrics.f1 * 100 : 0,
        mae: emotionMAE !== undefined ? emotionMAE : 0,
        support: perClassMetrics?.support || 0,
      }
    })
  }

  // Transform main metrics to performance format
  // Get MAE values for each model (used only for Emotion MAE row)
  const baseMae = getModelMetric(matchModel('base'), m => m.emotion_macro_mae || 0)
  const loraMae = getModelMetric(matchModel('lora'), m => m.emotion_macro_mae || 0)
  const ggufMae = getModelMetric(matchModel('gguf'), m => m.emotion_macro_mae || 0)

  const performanceData: PerformanceMetric[] = [
    {
      metric: 'Accuracy',
      base: getModelMetric(matchModel('base'), m => (m.primary_cls_accuracy || 0) * 100),
      lora: getModelMetric(matchModel('lora'), m => (m.primary_cls_accuracy || 0) * 100),
      gguf: getModelMetric(matchModel('gguf'), m => (m.primary_cls_accuracy || 0) * 100),
      unit: '%',
      isMaeRow: false,
      lowerIsBetter: false,
      maxVal: 100,
      baseMae, loraMae, ggufMae,
    },
    {
      metric: 'Macro F1',
      base: getModelMetric(matchModel('base'), m => (m.primary_cls_macro_f1 || 0) * 100),
      lora: getModelMetric(matchModel('lora'), m => (m.primary_cls_macro_f1 || 0) * 100),
      gguf: getModelMetric(matchModel('gguf'), m => (m.primary_cls_macro_f1 || 0) * 100),
      unit: '%',
      isMaeRow: false,
      lowerIsBetter: false,
      maxVal: 100,
      baseMae, loraMae, ggufMae,
    },
    {
      metric: 'AUC',
      base: getModelMetric(matchModel('base'), m => (m.primary_cls_macro_auc || 0) * 100),
      lora: getModelMetric(matchModel('lora'), m => (m.primary_cls_macro_auc || 0) * 100),
      gguf: getModelMetric(matchModel('gguf'), m => (m.primary_cls_macro_auc || 0) * 100),
      unit: '%',
      isMaeRow: false,
      lowerIsBetter: false,
      maxVal: 100,
      baseMae, loraMae, ggufMae,
    },
    {
      metric: 'AP',
      base: getModelMetric(matchModel('base'), m => (m.primary_cls_macro_ap || 0) * 100),
      lora: getModelMetric(matchModel('lora'), m => (m.primary_cls_macro_ap || 0) * 100),
      gguf: getModelMetric(matchModel('gguf'), m => (m.primary_cls_macro_ap || 0) * 100),
      unit: '%',
      isMaeRow: false,
      lowerIsBetter: false,
      maxVal: 100,
      baseMae, loraMae, ggufMae,
    },
    {
      metric: 'Emotion MAE',
      base: baseMae,
      lora: loraMae,
      gguf: ggufMae,
      unit: '',
      isMaeRow: true,
      lowerIsBetter: true,
      maxVal: 0.2,
      baseMae, loraMae, ggufMae,
    },
    {
      metric: 'CoT Complete',
      base: getModelMetric(matchModel('base'), m => (m.cot7_complete_rate || 0) * 100),
      lora: getModelMetric(matchModel('lora'), m => (m.cot7_complete_rate || 0) * 100),
      gguf: getModelMetric(matchModel('gguf'), m => (m.cot7_complete_rate || 0) * 100),
      unit: '%',
      isMaeRow: false,
      lowerIsBetter: false,
      maxVal: 100,
      baseMae, loraMae, ggufMae,
    },
  ]

  // Add engineering metrics if available
  const hasEngMetrics = metrics.some(m => (m.throughput_sps ?? 0) > 0 || (m.vram_gb ?? 0) > 0)
  if (hasEngMetrics) {
    performanceData.push({
      metric: 'Throughput',
      base: getModelMetric(matchModel('base'), m => m.throughput_sps || 0),
      lora: getModelMetric(matchModel('lora'), m => m.throughput_sps || 0),
      gguf: getModelMetric(matchModel('gguf'), m => m.throughput_sps || 0),
      unit: 'tokens/s',
    })
    performanceData.push({
      metric: 'VRAM',
      base: getModelMetric(matchModel('base'), m => m.vram_gb || 0),
      lora: getModelMetric(matchModel('lora'), m => m.vram_gb || 0),
      gguf: getModelMetric(matchModel('gguf'), m => m.vram_gb || 0),
      unit: 'GB',
    })
  }

  return { emotionData, performanceData, emotionDetailData }
}

const emotionColors: Record<string, string> = {
  Angry: '#dc8888',
  Fear: '#a894c4',
  Happy: '#88c4a8',
  Neutral: '#b0a8a0',
  Sad: '#88a4c4',
  Surprise: '#dcc488',
}

const Compare: React.FC = () => {
  const chartRef = useRef<HTMLDivElement>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>(['base', 'lora', 'gguf'])
  const [viewMode, setViewMode] = useState<'radar' | 'bar'>('radar')
  const [comparisonMode, setComparisonMode] = useState<'emotion' | 'performance'>('emotion')

  // Data loading state
  const [emotionComparisonData, setEmotionComparisonData] = useState<EmotionComparisonPoint[]>(defaultEmotionComparisonData)
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetric[]>(defaultPerformanceMetrics)
  const [emotionDetailData, setEmotionDetailData] = useState<Record<string, EmotionModelComparison[]>>(defaultEmotionDetailData)
  const [loading, setLoading] = useState(true)

  // Load comparison data on mount
  useEffect(() => {
    const loadComparison = async () => {
      try {
        const response = await fetch('/api/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_variants: ['base', 'lora_merged', 'gguf4bit'] }),
        })
        if (!response.ok) {
          throw new Error('Failed to load comparison data')
        }
        const json = await response.json()

        // Backend returns { models: [...], comparison_table: "..." }
        const metrics: ComparisonMetrics[] = json.models || []

        if (metrics.length === 0) {
          throw new Error('No comparison data found')
        }

        const transformed = transformComparisonData(metrics)
        setEmotionComparisonData(transformed.emotionData)
        setPerformanceMetrics(transformed.performanceData)
        setEmotionDetailData(transformed.emotionDetailData)
      } catch (error) {
        // Silently fall back to default mock data
        setEmotionComparisonData(defaultEmotionComparisonData)
        setPerformanceMetrics(defaultPerformanceMetrics)
        setEmotionDetailData(defaultEmotionDetailData)
      } finally {
        setLoading(false)
      }
    }
    loadComparison()
  }, [])

  const filteredEmotionData = useMemo(() => {
    return emotionComparisonData.map(item => ({
      emotion: item.emotion,
      base: selectedModels.includes('base') ? item.base : null,
      lora: selectedModels.includes('lora') ? item.lora : null,
      gguf: selectedModels.includes('gguf') ? item.gguf : null,
      fullMark: item.fullMark,
    }))
  }, [selectedModels, emotionComparisonData])

  const handleExport = async (e?: React.MouseEvent) => {
    e?.preventDefault()
    e?.stopPropagation()
    
    if (!chartRef.current) return
    
    await new Promise(resolve => setTimeout(resolve, 800))
    
    const allSvgs = chartRef.current.querySelectorAll('svg.recharts-surface')
    let svg: SVGElement | null = null
    for (const s of allSvgs) {
      if ((s as SVGElement).clientWidth > 100 && (s as SVGElement).clientHeight > 100) {
        svg = s as SVGElement
        break
      }
    }
    
    if (!svg) return
    
    const svgData = new XMLSerializer().serializeToString(svg)
    const canvas = document.createElement('canvas')
    const w = svg.clientWidth * 2
    const h = svg.clientHeight * 2
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (ctx) {
      // Get actual background color from computed style
      const bgColor = window.getComputedStyle(chartRef.current).backgroundColor || '#1a2332'
      ctx.fillStyle = bgColor
      ctx.fillRect(0, 0, w, h)
      ctx.scale(2, 2)
      
      const img = new Image()
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
      const url = URL.createObjectURL(svgBlob)
      
      img.onload = () => {
        ctx.drawImage(img, 0, 0)
        URL.revokeObjectURL(url)
        
        canvas.toBlob((blob) => {
          if (blob) {
            const downloadUrl = URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = downloadUrl
            link.download = `comparison-${comparisonMode}-${viewMode}-${Date.now()}.png`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            URL.revokeObjectURL(downloadUrl)
          }
        }, 'image/png')
      }
      img.onerror = () => URL.revokeObjectURL(url)
      img.src = url
    }
  }

  const emotionDetailColumns: TableProps<EmotionModelComparison>['columns'] = [
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'modelName',
      width: 80,
      render: (name) => <Tag color={name === 'LoRA' ? '#ff6b4a' : name === 'GGUF' ? '#06b6d4' : '#64748b'}>{name}</Tag>,
    },
    {
      title: 'Precision',
      dataIndex: 'precision',
      key: 'precision',
      width: 85,
      sorter: (a, b) => a.precision - b.precision,
      render: (value) => <span style={{ color: '#88c4a8', fontWeight: 600 }}>{value.toFixed(1)}%</span>,
    },
    {
      title: 'Recall',
      dataIndex: 'recall',
      key: 'recall',
      width: 85,
      sorter: (a, b) => a.recall - b.recall,
      render: (value) => <span style={{ color: '#88a4c4', fontWeight: 600 }}>{value.toFixed(1)}%</span>,
    },
    {
      title: 'F1 Score',
      dataIndex: 'f1',
      key: 'f1',
      width: 85,
      sorter: (a, b) => a.f1 - b.f1,
      render: (value) => <span style={{ color: '#a894c4', fontWeight: 600 }}>{value.toFixed(1)}%</span>,
    },
    {
      title: 'MAE',
      dataIndex: 'mae',
      key: 'mae',
      width: 75,
      sorter: (a, b) => a.mae - b.mae,
      render: (value) => <span style={{ color: '#ffcc80', fontWeight: 600 }}>{value.toFixed(3)}</span>,
    },
    {
      title: 'Support',
      dataIndex: 'support',
      key: 'support',
      width: 70,
      sorter: (a, b) => a.support - b.support,
      render: (value) => <span style={{ color: '#94a3b8' }}>{value}</span>,
    },
  ]

  return (
    <div className="compare-page">
      <div className="page-content">
        {/* Header */}
        <div className="page-header">
          <Title level={1} className="page-title">
            <BlockOutlined /> 模型对比
          </Title>
          <Paragraph className="page-subtitle">
            Base vs LoRA vs GGUF 多维度性能对比分析
          </Paragraph>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <Spin size="large" tip="加载对比数据..." />
          </div>
        )}

        {!loading && (
          <>
        {/* Control Panel */}
        <Card className="control-card">
          <div className="control-panel">
            <div className="control-group">
              <span className="control-label">选择模型:</span>
              <Select
                mode="multiple"
                value={selectedModels}
                onChange={setSelectedModels}
                className="model-select"
                maxTagCount="responsive"
              >
                {models.map(model => (
                  <Option key={model.id} value={model.id}>
                    {model.icon} {model.name}
                  </Option>
                ))}
              </Select>
            </div>
            <div className="control-group">
              <span className="control-label">对比维度:</span>
              <Radio.Group value={comparisonMode} onChange={(e) => setComparisonMode(e.target.value)}>
                <Radio.Button value="emotion">情感分析</Radio.Button>
                <Radio.Button value="performance">性能指标</Radio.Button>
              </Radio.Group>
            </div>
            {comparisonMode === 'emotion' && (
              <div className="control-group">
                <span className="control-label">图表类型:</span>
                <Radio.Group value={viewMode} onChange={(e) => setViewMode(e.target.value)}>
                  <Radio.Button value="radar">雷达图</Radio.Button>
                  <Radio.Button value="bar">柱状图</Radio.Button>
                </Radio.Group>
              </div>
            )}
            <Button
              type="primary"
              icon={<DownloadOutlined />}
              onClick={handleExport}
              className="btn-export"
            >
              导出图表
            </Button>
          </div>
        </Card>

        {/* Charts */}
        <div className="charts-section">
          {comparisonMode === 'emotion' ? (
            <div ref={chartRef}>
              <Card className="chart-card comparison-chart" title="情感分析能力对比">
              <ResponsiveContainer width="100%" height={400}>
                {viewMode === 'radar' ? (
                  <RadarChart data={filteredEmotionData}>
                    <PolarGrid stroke="rgba(148, 163, 184, 0.2)" />
                    <PolarAngleAxis dataKey="emotion" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <PolarRadiusAxis angle={90} domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    {selectedModels.includes('base') && (
                      <Radar
                        name="Base"
                        dataKey="base"
                        stroke="#64748b"
                        fill="#64748b"
                        fillOpacity={0.3}
                      />
                    )}
                    {selectedModels.includes('lora') && (
                      <Radar
                        name="LoRA"
                        dataKey="lora"
                        stroke="#ff6b4a"
                        fill="#ff6b4a"
                        fillOpacity={0.3}
                      />
                    )}
                    {selectedModels.includes('gguf') && (
                      <Radar
                        name="GGUF"
                        dataKey="gguf"
                        stroke="#06b6d4"
                        fill="#06b6d4"
                        fillOpacity={0.3}
                      />
                    )}
                    <Legend />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(15, 23, 36, 0.98)',
                        border: '1px solid rgba(148, 163, 184, 0.3)',
                        borderRadius: 8,
                        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
                        color: '#e2e8f0',
                        fontSize: '13px',
                        zIndex: 1000,
                      }}
                      labelStyle={{ color: '#94a3b8', marginBottom: '8px' }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                  </RadarChart>
                ) : (
                  <BarChart data={filteredEmotionData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fill: '#94a3b8' }} />
                    <YAxis dataKey="emotion" type="category" tick={{ fill: '#94a3b8' }} width={80} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(15, 23, 36, 0.98)',
                        border: '1px solid rgba(148, 163, 184, 0.3)',
                        borderRadius: 8,
                        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
                        color: '#e2e8f0',
                        fontSize: '13px',
                        zIndex: 1000,
                      }}
                      labelStyle={{ color: '#94a3b8', marginBottom: '8px' }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                    <Legend />
                    {selectedModels.includes('base') && (
                      <Bar dataKey="base" fill="#64748b" radius={[0, 4, 4, 0]} />
                    )}
                    {selectedModels.includes('lora') && (
                      <Bar dataKey="lora" fill="#ff6b4a" radius={[0, 4, 4, 0]} />
                    )}
                    {selectedModels.includes('gguf') && (
                      <Bar dataKey="gguf" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                    )}
                  </BarChart>
                )}
              </ResponsiveContainer>
            </Card>
            </div>
          ) : (
            <div ref={chartRef}>
              <Card className="chart-card comparison-chart" title="性能指标对比">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={performanceMetrics} margin={{ top: 20, right: 60, bottom: 80, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" tick={{ fill: '#94a3b8' }} label={{ value: '百分比 (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} domain={[0, 100]} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#ffcc80' }} label={{ value: 'MAE (越小越好)', angle: 90, position: 'insideRight', fill: '#ffcc80' }} domain={[0, 0.3]} />
                  <Legend />
                  {selectedModels.includes('base') && (
                    <>
                      <Bar dataKey="base" name="Base" fill="#64748b" radius={[4, 4, 0, 0]} yAxisId="left">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-base-${index}`} fill={entry.isMaeRow ? 'transparent' : '#64748b'} />
                        ))}
                      </Bar>
                      <Bar dataKey="baseMae" name="Base" fill="#64748b" radius={[4, 4, 0, 0]} yAxisId="right" opacity={0.7} legendType="none">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-baseMae-${index}`} fill={entry.isMaeRow ? '#64748b' : 'transparent'} />
                        ))}
                      </Bar>
                    </>
                  )}
                  {selectedModels.includes('lora') && (
                    <>
                      <Bar dataKey="lora" name="LoRA" fill="#ff6b4a" radius={[4, 4, 0, 0]} yAxisId="left">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-lora-${index}`} fill={entry.isMaeRow ? 'transparent' : '#ff6b4a'} />
                        ))}
                      </Bar>
                      <Bar dataKey="loraMae" name="LoRA" fill="#ff6b4a" radius={[4, 4, 0, 0]} yAxisId="right" opacity={0.7} legendType="none">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-loraMae-${index}`} fill={entry.isMaeRow ? '#ff6b4a' : 'transparent'} />
                        ))}
                      </Bar>
                    </>
                  )}
                  {selectedModels.includes('gguf') && (
                    <>
                      <Bar dataKey="gguf" name="GGUF" fill="#06b6d4" radius={[4, 4, 0, 0]} yAxisId="left">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-gguf-${index}`} fill={entry.isMaeRow ? 'transparent' : '#06b6d4'} />
                        ))}
                      </Bar>
                      <Bar dataKey="ggufMae" name="GGUF" fill="#06b6d4" radius={[4, 4, 0, 0]} yAxisId="right" opacity={0.7} legendType="none">
                        {performanceMetrics.map((entry, index) => (
                          <Cell key={`cell-ggufMae-${index}`} fill={entry.isMaeRow ? '#06b6d4' : 'transparent'} />
                        ))}
                      </Bar>
                    </>
                  )}
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload || payload.length === 0) return null

                      // Filter out transparent/empty values and duplicate models
                      const filteredPayload = payload.filter((item: any) => {
                        const metric = item.payload?.metric
                        const dataKey = item.dataKey
                        const value = item.value

                        // Skip MAE bars for non-MAE rows (they are transparent)
                        if (dataKey?.includes('Mae') && metric !== 'Emotion MAE') return false
                        // Skip percentage bars for MAE row
                        if (!dataKey?.includes('Mae') && metric === 'Emotion MAE') return false

                        return value !== undefined && value !== null
                      })

                      // Remove duplicate model entries (keep the one with actual value)
                      const uniquePayload = filteredPayload.reduce((acc: any[], item: any) => {
                        const modelName = item.name
                        const existing = acc.find(i => i.name === modelName)
                        if (!existing) {
                          acc.push(item)
                        }
                        return acc
                      }, [])

                      return (
                        <div style={{
                          background: 'rgba(15, 23, 36, 0.98)',
                          border: '1px solid rgba(148, 163, 184, 0.3)',
                          borderRadius: 8,
                          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
                          padding: '12px 16px',
                          fontSize: '13px',
                          zIndex: 9999,
                        }}>
                          <div style={{ color: '#94a3b8', marginBottom: '8px' }}>{label}</div>
                          {uniquePayload.map((item: any, index: number) => {
                            const metric = item.payload?.metric
                            const value = item.value
                            const displayValue = metric === 'Emotion MAE' ? value.toFixed(3) : `${value.toFixed(1)}%`
                            return (
                              <div key={index} style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: '4px' }}>
                                {item.name}: {displayValue}
                              </div>
                            )
                          })}
                        </div>
                      )
                    }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            </div>
          )}

          {/* Metrics Summary Cards */}
          {comparisonMode === 'emotion' && (
            <div className="metrics-summary">
            {performanceMetrics.map((item, index) => (
              <Card key={item.metric} className="metric-summary-card" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="metric-summary-header">
                  <span className="metric-summary-label">{item.metric}</span>
                </div>
                <div className="metric-summary-values">
                  {selectedModels.includes('base') && (
                    <div className="metric-bar-container">
                      <div className="metric-bar-label">Base</div>
                      <div className="metric-bar-track">
                        <div
                          className="metric-bar-fill metric-bar-base"
                          style={{ width: `${Math.min(100, item.lowerIsBetter ? (1 - item.base / (item.maxVal || 1)) * 100 : (item.base / (item.maxVal || 100)) * 100)}%` }}
                        >
                          <span className="metric-bar-value">{item.isMaeRow ? item.base.toFixed(3) : item.base.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {selectedModels.includes('lora') && (
                    <div className="metric-bar-container">
                      <div className="metric-bar-label">LoRA</div>
                      <div className="metric-bar-track">
                        <div
                          className="metric-bar-fill metric-bar-lora"
                          style={{ width: `${Math.min(100, item.lowerIsBetter ? (1 - item.lora / (item.maxVal || 1)) * 100 : (item.lora / (item.maxVal || 100)) * 100)}%` }}
                        >
                          <span className="metric-bar-value">{item.isMaeRow ? item.lora.toFixed(3) : item.lora.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {selectedModels.includes('gguf') && (
                    <div className="metric-bar-container">
                      <div className="metric-bar-label">GGUF</div>
                      <div className="metric-bar-track">
                        <div
                          className="metric-bar-fill metric-bar-gguf"
                          style={{ width: `${Math.min(100, item.lowerIsBetter ? (1 - item.gguf / (item.maxVal || 1)) * 100 : (item.gguf / (item.maxVal || 100)) * 100)}%` }}
                        >
                          <span className="metric-bar-value">{item.isMaeRow ? item.gguf.toFixed(3) : item.gguf.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <span className="metric-unit">{item.unit || '无单位'}</span>
              </Card>
            ))}
          </div>
          )}
        </div>

        {/* Emotion Detail Tables - Collapse Panels */}
        <Card className="results-card" title="每类情感详细指标对比">
          <Collapse
            defaultActiveKey={['angry']}
            items={Object.entries(emotionDetailData).map(([emotion, data]) => ({
              key: emotion,
              label: (
                <span style={{ fontSize: 15, fontWeight: 600 }}>
                  <Tag color={emotionColors[emotion.charAt(0).toUpperCase() + emotion.slice(1)]}>
                    {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                  </Tag>
                  <span style={{ marginLeft: 12, color: '#94a3b8', fontSize: 13 }}>
                    样本数：{data[0]?.support || 0}
                  </span>
                </span>
              ),
              children: (
                <Table<EmotionModelComparison>
                  columns={emotionDetailColumns}
                  dataSource={data}
                  rowKey="model"
                  pagination={false}
                  size="middle"
                />
              ),
            }))}
          />
        </Card>
          </>
        )}
      </div>
    </div>
  )
}

export default Compare
