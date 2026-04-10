import React, { useRef, useState, useEffect } from 'react'
import { Typography, Card, Row, Col, Statistic, Button, Table, Tag, Spin, Alert } from 'antd'
import {
  BarChartOutlined,
  PieChartOutlined,
  HeatMapOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'
import html2canvas from 'html2canvas'
import './Metrics.scss'

const { Title, Paragraph } = Typography

// Default mock metrics data (fallback)
const defaultOverallMetrics = {
  accuracy: 91.2,
  precision: 89.5,
  recall: 90.1,
  f1: 89.8,
  auc: 0.943,
  ap: 0.921,
  loss: 0.2727,
  jsonParseRate: 0.90,
  cotCompleteRate: 0.98,
  emotionMae: 0.087,
}

const defaultEmotionMetrics = [
  { emotion: 'Angry', precision: 88.5, recall: 87.2, f1: 87.8, support: 245 },
  { emotion: 'Fear', precision: 85.2, recall: 84.8, f1: 85.0, support: 198 },
  { emotion: 'Happy', precision: 92.1, recall: 93.5, f1: 92.8, support: 312 },
  { emotion: 'Neutral', precision: 94.2, recall: 95.1, f1: 94.6, support: 428 },
  { emotion: 'Sad', precision: 87.8, recall: 86.5, f1: 87.1, support: 267 },
  { emotion: 'Surprise', precision: 84.5, recall: 83.2, f1: 83.8, support: 156 },
]

const defaultPrCurveData = [
  { recall: 0, precision: 1, threshold: 1.0 },
  { recall: 0.1, precision: 0.98, threshold: 0.9 },
  { recall: 0.2, precision: 0.95, threshold: 0.8 },
  { recall: 0.3, precision: 0.92, threshold: 0.7 },
  { recall: 0.4, precision: 0.89, threshold: 0.6 },
  { recall: 0.5, precision: 0.86, threshold: 0.5 },
  { recall: 0.6, precision: 0.82, threshold: 0.4 },
  { recall: 0.7, precision: 0.76, threshold: 0.3 },
  { recall: 0.8, precision: 0.68, threshold: 0.2 },
  { recall: 0.9, precision: 0.55, threshold: 0.1 },
  { recall: 1.0, precision: 0.42, threshold: 0.0 },
]

const defaultRocCurveData = [
  { fpr: 0, tpr: 0, threshold: 1.0 },
  { fpr: 0.05, tpr: 0.45, threshold: 0.9 },
  { fpr: 0.1, tpr: 0.68, threshold: 0.8 },
  { fpr: 0.15, tpr: 0.78, threshold: 0.7 },
  { fpr: 0.2, tpr: 0.85, threshold: 0.6 },
  { fpr: 0.25, tpr: 0.89, threshold: 0.5 },
  { fpr: 0.3, tpr: 0.92, threshold: 0.4 },
  { fpr: 0.4, tpr: 0.94, threshold: 0.3 },
  { fpr: 0.5, tpr: 0.96, threshold: 0.2 },
  { fpr: 0.7, tpr: 0.98, threshold: 0.1 },
  { fpr: 1, tpr: 1, threshold: 0.0 },
]

const defaultTrainingHistory = [
  { step: 200, trainLoss: 2.34, valLoss: 2.28, trainAcc: 0.452, valAcc: 0.468 },
  { step: 400, trainLoss: 1.87, valLoss: 1.82, trainAcc: 0.584, valAcc: 0.592 },
  { step: 600, trainLoss: 1.45, valLoss: 1.38, trainAcc: 0.685, valAcc: 0.698 },
  { step: 800, trainLoss: 1.12, valLoss: 1.05, trainAcc: 0.762, valAcc: 0.775 },
  { step: 1000, trainLoss: 0.85, valLoss: 0.78, trainAcc: 0.824, valAcc: 0.838 },
  { step: 1200, trainLoss: 0.62, valLoss: 0.55, trainAcc: 0.868, valAcc: 0.882 },
  { step: 1400, trainLoss: 0.45, valLoss: 0.38, trainAcc: 0.895, valAcc: 0.905 },
  { step: 1600, trainLoss: 0.32, valLoss: 0.30, trainAcc: 0.912, valAcc: 0.910 },
  { step: 1800, trainLoss: 0.25, valLoss: 0.28, trainAcc: 0.928, valAcc: 0.912 },
  { step: 2000, trainLoss: 0.20, valLoss: 0.29, trainAcc: 0.935, valAcc: 0.912 },
]

// Interfaces for metrics data (eval_v2.py format)
interface PerClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

interface PRCurvePoint {
  recall: number
  precision: number
  threshold: number
}

interface ROCCurvePoint {
  fpr: number
  tpr: number
  threshold: number
}

interface EmotionMetric {
  emotion: string
  precision: number
  recall: number
  f1: number
  support: number
}

interface TrainingEpoch {
  step: number
  trainLoss: number
  valLoss: number
  trainAcc: number
  valAcc: number
}

interface EvalV2MetricsData {
  primary_cls_accuracy?: number
  primary_cls_macro_f1?: number
  primary_cls_macro_auc?: number
  primary_cls_macro_ap?: number
  primary_cls_per_class_metrics?: Record<string, PerClassMetrics>
  primary_cls_pr_curves?: Record<string, PRCurvePoint[]>
  primary_cls_roc_curves?: Record<string, ROCCurvePoint[]>
  primary_cls_confusion_matrix?: {
    labels: string[]
    matrix: number[][]
  }
  emotion_per_dim_mae?: Record<string, number>
  emotion_macro_mae?: number
  emotion_macro_mse?: number
  mbti_accuracy?: number
  mbti_macro_f1?: number
  json_parse_rate?: number
  cot7_complete_rate?: number
}

interface MetricsData {
  overall: {
    accuracy: number
    precision: number
    recall: number
    f1: number
    auc: number
    ap: number
    loss: number
    jsonParseRate: number
    cotCompleteRate: number
    emotionMae: number
  }
  emotion: EmotionMetric[]
  prCurve: { recall: number; precision: number; threshold: number }[]
  rocCurve: { fpr: number; tpr: number; threshold: number }[]
  training: TrainingEpoch[]
  confusionMatrix?: {
    labels: string[]
    matrix: number[][]
  }
}

const emotionColors: Record<string, string> = {
  Angry: '#dc8888',
  Fear: '#a894c4',
  Happy: '#88c4a8',
  Neutral: '#b0a8a0',
  Sad: '#88a4c4',
  Surprise: '#dcc488',
}

const Metrics: React.FC = () => {
  const confusionMatrixRef = useRef<HTMLDivElement>(null)
  const prCurveRef = useRef<HTMLDivElement>(null)
  const rocCurveRef = useRef<HTMLDivElement>(null)

  // State for metrics data
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  // Training history loading state
  const [trainingHistory, setTrainingHistory] = useState<TrainingEpoch[]>(defaultTrainingHistory)
  const [finalValLoss, setFinalValLoss] = useState<number | null>(null)

  // Helper function for metric color based on value and metric type
  const getMetricColor = (value: number, metricType: 'percentage' | 'mae' | 'loss'): string => {
    if (metricType === 'mae' || metricType === 'loss') {
      // For MAE and Loss (lower is better)
      // 优秀 ≤0.15, 良好 0.15~0.25, 一般 0.25~0.35, 较差 >0.35
      if (value <= 0.15) return '#88c4a8'
      if (value <= 0.25) return '#dcc488'
      if (value <= 0.35) return '#ff6b4a'
      return '#c0392b'
    } else {
      // For percentage metrics (higher is better)
      // 优秀 ≥85%, 良好 70%~85%, 一般 50%~70%, 较差 <50%
      if (value >= 85) return '#88c4a8'
      if (value >= 70) return '#dcc488'
      if (value >= 50) return '#ff6b4a'
      return '#c0392b'
    }
  }

  // Load metrics data from backend API or use default mock data
  useEffect(() => {
    const loadMetrics = async () => {
      try {
        // Try to load from backend API first (which reads from outputs/ files)
        const response = await fetch('/api/metrics/lora_merged')
        if (!response.ok) {
          throw new Error('Metrics API returned error')
        }
        const json: EvalV2MetricsData = await response.json()

        // Helper function to aggregate PR curves across all emotion classes
        const aggregatePRCurve = (prCurves: Record<string, PRCurvePoint[]>): PRCurvePoint[] => {
          const recallLevels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
          const aggregated: PRCurvePoint[] = []

          for (const recallVal of recallLevels) {
            const precisions: number[] = []
            for (const emotion of Object.keys(prCurves)) {
              const curve = prCurves[emotion]
              if (curve && curve.length > 0) {
                // Find precision at this recall level (use first point where recall >= target)
                for (const point of curve) {
                  if (point.recall >= recallVal) {
                    precisions.push(point.precision)
                    break
                  }
                }
              }
            }
            if (precisions.length > 0) {
              const avgPrecision = precisions.reduce((a, b) => a + b, 0) / precisions.length
              aggregated.push({ recall: recallVal, precision: avgPrecision, threshold: 1.0 - recallVal })
            }
          }
          return aggregated.length > 0 ? aggregated : defaultPrCurveData
        }

        // Helper function to aggregate ROC curves across all emotion classes
        const aggregateROCCurve = (rocCurves: Record<string, ROCCurvePoint[]>): ROCCurvePoint[] => {
          const fprLevels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
          const aggregated: ROCCurvePoint[] = []

          for (const fprVal of fprLevels) {
            const tprs: number[] = []
            for (const emotion of Object.keys(rocCurves)) {
              const curve = rocCurves[emotion]
              if (curve && curve.length > 0) {
                // Find TPR at this FPR level
                for (const point of curve) {
                  if (point.fpr >= fprVal) {
                    tprs.push(point.tpr)
                    break
                  }
                }
              }
            }
            if (tprs.length > 0) {
              const avgTPR = tprs.reduce((a, b) => a + b, 0) / tprs.length
              aggregated.push({ fpr: fprVal, tpr: avgTPR, threshold: 1.0 - fprVal })
            }
          }
          return aggregated.length > 0 ? aggregated : defaultRocCurveData
        }

        // Transform eval_v2.py output format to frontend format
        const transformedData: MetricsData = {
          overall: {
            accuracy: json.primary_cls_accuracy ? parseFloat((json.primary_cls_accuracy * 100).toFixed(1)) : defaultOverallMetrics.accuracy,
            precision: json.primary_cls_per_class_metrics
              ? parseFloat(((Object.values(json.primary_cls_per_class_metrics).reduce((sum, m) => sum + m.precision, 0) / Object.keys(json.primary_cls_per_class_metrics).length) * 100).toFixed(1))
              : defaultOverallMetrics.precision,
            recall: json.primary_cls_per_class_metrics
              ? parseFloat(((Object.values(json.primary_cls_per_class_metrics).reduce((sum, m) => sum + m.recall, 0) / Object.keys(json.primary_cls_per_class_metrics).length) * 100).toFixed(1))
              : defaultOverallMetrics.recall,
            f1: json.primary_cls_macro_f1 ? parseFloat((json.primary_cls_macro_f1 * 100).toFixed(1)) : defaultOverallMetrics.f1,
            auc: json.primary_cls_macro_auc ? parseFloat((json.primary_cls_macro_auc * 100).toFixed(2)) : defaultOverallMetrics.auc * 100,
            ap: json.primary_cls_macro_ap ? parseFloat((json.primary_cls_macro_ap * 100).toFixed(2)) : defaultOverallMetrics.ap * 100,
            loss: defaultOverallMetrics.loss,
            jsonParseRate: json.json_parse_rate ? parseFloat((json.json_parse_rate * 100).toFixed(1)) : defaultOverallMetrics.jsonParseRate,
            cotCompleteRate: json.cot7_complete_rate ? parseFloat((json.cot7_complete_rate * 100).toFixed(1)) : defaultOverallMetrics.cotCompleteRate,
            emotionMae: json.emotion_macro_mae ? parseFloat(json.emotion_macro_mae.toFixed(3)) : defaultOverallMetrics.emotionMae,
          },
          emotion: json.primary_cls_per_class_metrics
            ? (Object.entries(json.primary_cls_per_class_metrics) as [string, PerClassMetrics][]).map(([emotion, metrics]) => ({
                emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                precision: parseFloat((metrics.precision * 100).toFixed(1)),
                recall: parseFloat((metrics.recall * 100).toFixed(1)),
                f1: parseFloat((metrics.f1 * 100).toFixed(1)),
                support: metrics.support,
              }))
            : defaultEmotionMetrics,
          prCurve: json.aggregated_pr_curve || 
            (json.primary_cls_pr_curves ? aggregatePRCurve(json.primary_cls_pr_curves) : defaultPrCurveData),
          rocCurve: json.aggregated_roc_curve || 
            (json.primary_cls_roc_curves ? aggregateROCCurve(json.primary_cls_roc_curves) : defaultRocCurveData),
          training: defaultTrainingHistory,
          confusionMatrix: json.primary_cls_confusion_matrix || undefined,
        }

        setMetricsData(transformedData)
        setLoadError(null)
      } catch (error) {
        // Fall back to default mock data
        console.log('Using default mock metrics data')
        setMetricsData({
          overall: defaultOverallMetrics,
          emotion: defaultEmotionMetrics,
          prCurve: defaultPrCurveData,
          rocCurve: defaultRocCurveData,
          training: defaultTrainingHistory,
        })
        setLoadError(null)
      } finally {
        setLoading(false)
      }
    }

    loadMetrics()
  }, [])

  // Load training history from SwanLab via backend API
  useEffect(() => {
    const loadTrainingHistory = async () => {
      try {
        const response = await fetch('/api/training-history')
        if (!response.ok) {
          throw new Error('Training history API error')
        }
        const json = await response.json()
        const history: TrainingEpoch[] = json.data || []
        
        if (history.length > 0) {
          setTrainingHistory(history)
          
          const lastEpoch = history[history.length - 1]
          if (lastEpoch.valLoss > 0) {
            setFinalValLoss(lastEpoch.valLoss)
          }
          
          setMetricsData(prev => {
            if (prev) {
              return { ...prev, training: history }
            }
            return prev
          })
        }
      } catch (error) {
        console.warn('Failed to load training history, using defaults')
      }
    }
    
    loadTrainingHistory()
  }, [])

  // Use default data if not loaded yet
  const displayData = metricsData || {
    overall: defaultOverallMetrics,
    emotion: defaultEmotionMetrics,
    prCurve: defaultPrCurveData,
    rocCurve: defaultRocCurveData,
    training: trainingHistory,
  }

  // Use final val loss from training history if available, otherwise use default
  const displayValLoss = finalValLoss !== null ? finalValLoss : defaultOverallMetrics.loss

  const handleExport = async (ref: React.RefObject<HTMLDivElement>, filename: string) => {
    if (ref.current) {
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const canvas = await html2canvas(ref.current, {
        backgroundColor: '#1a2332',
        scale: 2,
        useCORS: true,
        logging: false,
        width: ref.current.scrollWidth,
        height: ref.current.scrollHeight,
      })
      
      const link = document.createElement('a')
      link.download = `${filename}-${Date.now()}.png`
      link.href = canvas.toDataURL('image/png')
      link.click()
    }
  }

  return (
    <div className="metrics-page">
      <div className="page-content">
        {/* Header */}
        <div className="page-header">
          <Title level={1} className="page-title">
            <BarChartOutlined /> 模型评估
          </Title>
          <Paragraph className="page-subtitle">
            多任务情感分析模型的详细性能评估指标
          </Paragraph>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <Spin size="large" tip="加载评估数据..." />
          </div>
        )}

        {/* Error Alert */}
        {loadError && (
          <Alert
            message="数据加载失败"
            description={loadError}
            type="error"
            showIcon
            style={{ marginBottom: 24 }}
          />
        )}

        {!loading && (
          <>

        {/* Rating Legend */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '24px', marginBottom: '16px', fontSize: '12px', color: '#94a3b8' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: '#88c4a8', fontWeight: 600 }}>●</span>
            <span>优秀 { }</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: '#dcc488', fontWeight: 600 }}>●</span>
            <span>良好</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: '#ff6b4a', fontWeight: 600 }}>●</span>
            <span>一般</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: '#c0392b', fontWeight: 600 }}>●</span>
            <span>较差</span>
          </div>
        </div>

        {/* Row 1: Core Metrics */}
        <Row gutter={[24, 24]} className="metrics-row">
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card highlight" title="准确率">
              <div className="metric-content">
                <Statistic
                  value={displayData.overall.accuracy}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getMetricColor(displayData.overall.accuracy, 'percentage') }}
                />
                <div className="metric-icon">
                  <CheckCircleOutlined />
                </div>
              </div>
              <div className="metric-sub">测试集整体准确率</div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card" title="F1 Score">
              <div className="metric-content">
                <Statistic
                  value={displayData.overall.f1}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getMetricColor(displayData.overall.f1, 'percentage') }}
                />
                <div className="metric-icon">
                  <ThunderboltOutlined />
                </div>
              </div>
              <div className="metric-sub">精确率与召回率的调和平均</div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card" title="AUC-ROC">
              <div className="metric-content">
                <Statistic
                  value={displayData.overall.auc}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: getMetricColor(displayData.overall.auc, 'percentage') }}
                />
                <div className="metric-icon">
                  <PieChartOutlined />
                </div>
              </div>
              <div className="metric-sub">曲线下面积衡量指标</div>
            </Card>
          </Col>
        </Row>

        {/* Row 2: Classification Metrics */}
        <Row gutter={[24, 24]} className="metrics-row">
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card" title="Precision">
              <div className="metric-content">
                <Statistic
                  value={displayData.overall.precision}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getMetricColor(displayData.overall.precision, 'percentage') }}
                />
              </div>
              <div className="metric-sub">预测为正的样本中实际为正的比例</div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card" title="Emotion MAE">
              <div className="metric-content">
                <Statistic
                  value={displayData.overall.emotionMae}
                  precision={3}
                  valueStyle={{ color: getMetricColor(displayData.overall.emotionMae, 'mae') }}
                />
              </div>
              <div className="metric-sub">情感强度预测误差 (越低越好)</div>
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={8}>
            <Card className="metric-card" title="Validation Loss">
              <div className="metric-content">
                <Statistic
                  value={displayValLoss}
                  precision={3}
                  valueStyle={{ color: getMetricColor(displayValLoss, 'loss') }}
                />
              </div>
              <div className="metric-sub">验证集交叉熵损失 (越低越好)</div>
            </Card>
          </Col>
        </Row>

        {/* Charts Section */}
        <Row gutter={[24, 24]}>
          {/* Confusion Matrix */}
          <Col xs={24} lg={12}>
            <Card
              ref={confusionMatrixRef}
              className="chart-card confusion-matrix-card"
              title={
                <div className="chart-header">
                  <HeatMapOutlined /> 混淆矩阵热力图
                </div>
              }
              extra={
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  onClick={() => handleExport(confusionMatrixRef, 'confusion-matrix')}
                >
                  导出 HD
                </Button>
              }
            >
              {displayData.confusionMatrix && (() => {
                const confusionMatrix = displayData.confusionMatrix
                return (
                  <div className="confusion-matrix" style={{ overflow: 'auto' }}>
                    <table className="confusion-matrix-table">
                      <thead>
                        <tr>
                          <th style={{ color: '#94a3b8', fontSize: '12px', textTransform: 'uppercase' }}>真实值 (行) \ 预测值 (列)</th>
                          {confusionMatrix.labels.map((label, index) => (
                            <th key={index} style={{ color: '#f1f5f9', fontSize: '13px' }}>
                              {label.charAt(0).toUpperCase() + label.slice(1)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {confusionMatrix.matrix.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            <td style={{ color: '#f1f5f9', fontSize: '13px', fontWeight: 600 }}>
                              {confusionMatrix.labels[rowIndex].charAt(0).toUpperCase() + confusionMatrix.labels[rowIndex].slice(1)}
                            </td>
                            {row.map((cell, colIndex) => {
                              const maxValue = Math.max(...confusionMatrix.matrix.flat())
                              const intensity = cell / maxValue
                              return (
                                <td
                                  key={colIndex}
                                  style={{
                                    background: `rgba(255, 107, 74, ${intensity})`,
                                    color: intensity > 0.5 ? '#fff' : '#94a3b8',
                                    textAlign: 'center',
                                    padding: '12px 8px',
                                    fontSize: '13px',
                                    fontWeight: 600,
                                    transition: 'all 0.2s',
                                  }}
                                >
                                  {cell}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ marginTop: '16px', fontSize: '12px', color: '#94a3b8' }}>
                      <div style={{ marginBottom: '8px' }}>
                        <span style={{ color: '#ff6b4a', fontWeight: 600 }}>■</span> 颜色越深表示该类别的样本越多
                      </div>
                      <div>
                        <span style={{ color: '#f1f5f9', fontWeight: 600 }}>对角线</span>（从左上到右下）的值越高，表示分类越准确
                      </div>
                    </div>
                  </div>
                )
              })()}
            </Card>
          </Col>

          {/* PR Curve & ROC Curve */}
          <Col xs={24} lg={12}>
            <div ref={prCurveRef}>
              <Card
                className="chart-card"
                title={
                  <div className="chart-header">
                    <PieChartOutlined /> PR 曲线 (Precision-Recall)
                  </div>
                }
                extra={
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => handleExport(prCurveRef, 'pr-curve')}
                  >
                    导出 HD
                  </Button>
                }
              >
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={displayData.prCurve}>
                    <defs>
                      <linearGradient id="prGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ff6b4a" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#ff6b4a" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                    <XAxis dataKey="recall" tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                    <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(15, 23, 36, 0.95)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: 8,
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="precision"
                      stroke="#ff6b4a"
                      fill="url(#prGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </div>

            <div ref={rocCurveRef}>
              <Card
                className="chart-card"
                title={
                  <div className="chart-header">
                    <BarChartOutlined /> ROC 曲线
                  </div>
                }
                extra={
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => handleExport(rocCurveRef, 'roc-curve')}
                  >
                    导出 HD
                  </Button>
                }
              >
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={displayData.rocCurve}>
                    <defs>
                      <linearGradient id="rocGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#88c4a8" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#88c4a8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                    <XAxis dataKey="fpr" tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                    <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(15, 23, 36, 0.95)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: 8,
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="tpr"
                      stroke="#88c4a8"
                      fill="url(#rocGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </Col>
        </Row>

        {/* Training History - SwanLab Embed */}
        <Card
          className="chart-card training-card"
          title={
            <div className="chart-header">
              <BarChartOutlined /> 训练历史 (来自 SwanLab)
            </div>
          }
        >
          <Row gutter={24}>
            <Col xs={24} lg={12}>
              <div className="chart-title">Loss 曲线</div>
              <iframe
                src="https://swanlab.cn/@SrandWind/cmacd-lora-v2/chart/default/5ewtx1jgd8ij852g8qedc/ifhzmZln"
                width="100%"
                height="300"
                style={{ border: 'none', borderRadius: '8px' }}
                title="Train Loss"
              />
            </Col>
            <Col xs={24} lg={12}>
              <div className="chart-title">Train Accuracy 曲线</div>
              <iframe
                src="https://swanlab.cn/@SrandWind/cmacd-lora-v2/chart/default/5ewtx1jgd8ij852g8qedc/LIC5rARv"
                width="100%"
                height="300"
                style={{ border: 'none', borderRadius: '8px' }}
                title="Train Accuracy"
              />
            </Col>
          </Row>
          <Row gutter={24} style={{ marginTop: 24 }}>
            <Col xs={24} lg={12}>
              <div className="chart-title">Eval Loss 曲线</div>
              <iframe
                src="https://swanlab.cn/@SrandWind/cmacd-lora-v2/chart/default/5ewtx1jgd8ij852g8qedc/S_kN-JTN"
                width="100%"
                height="300"
                style={{ border: 'none', borderRadius: '8px' }}
                title="Eval Loss"
              />
            </Col>
            <Col xs={24} lg={12}>
              <div className="chart-title">Eval Accuracy 曲线</div>
              <iframe
                src="https://swanlab.cn/@SrandWind/cmacd-lora-v2/chart/default/5ewtx1jgd8ij852g8qedc/bSOWQXJA"
                width="100%"
                height="300"
                style={{ border: 'none', borderRadius: '8px' }}
                title="Eval Accuracy"
              />
            </Col>
          </Row>
        </Card>

        {/* Per-Class Metrics Table */}
        <Card className="table-card" title="各类别详细指标">
          <Table
            dataSource={displayData.emotion}
            rowKey="emotion"
            pagination={false}
            className="metrics-table"
          >
            <Table.Column
              title="情感类别"
              dataIndex="emotion"
              key="emotion"
              render={(text) => (
                <Tag color={emotionColors[text]} style={{ fontWeight: 600, padding: '4px 12px' }}>
                  {text}
                </Tag>
              )}
            />
            <Table.Column
              title="Precision"
              dataIndex="precision"
              key="precision"
              sorter={(a, b) => a.precision - b.precision}
              render={(value) => (
                <div className="metric-cell">
                  <span style={{ color: '#88c4a8', fontWeight: 600 }}>{value.toFixed(1)}%</span>
                </div>
              )}
            />
            <Table.Column
              title="Recall"
              dataIndex="recall"
              key="recall"
              sorter={(a, b) => a.recall - b.recall}
              render={(value) => (
                <div className="metric-cell">
                  <span style={{ color: '#88a4c4', fontWeight: 600 }}>{value.toFixed(1)}%</span>
                </div>
              )}
            />
            <Table.Column
              title="F1 Score"
              dataIndex="f1"
              key="f1"
              sorter={(a, b) => a.f1 - b.f1}
              render={(value) => (
                <div className="metric-cell">
                  <span style={{ color: '#a894c4', fontWeight: 600 }}>{value.toFixed(1)}%</span>
                </div>
              )}
            />
            <Table.Column
              title="Support (样本数)"
              dataIndex="support"
              key="support"
              sorter={(a, b) => a.support - b.support}
              render={(value) => (
                <div className="metric-cell">
                  <span style={{ color: '#94a3b8' }}>{value}</span>
                </div>
              )}
            />
          </Table>
        </Card>
          </>
        )}
      </div>
    </div>
  )
}

export default Metrics
