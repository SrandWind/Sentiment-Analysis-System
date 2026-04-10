import React, { useState, useEffect } from 'react'
import { Typography, Card, Table, Tag, Space, Button, Modal, Spin, Alert, Empty, Pagination, Select, Descriptions } from 'antd'
import {
  HistoryOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  EyeOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import {
  EMOTION_COLORS,
  formatEmotionLabel,
  formatVadPreview,
  formatUncertaintyBadge,
  parseCotContent,
  STEP_TITLES,
  formatLatency,
} from '@/utils/cotParser'

const emotionColors = EMOTION_COLORS
const vadLabels: Record<string, string> = {
  valence: '效价 (Valence)',
  arousal: '唤醒度 (Arousal)',
  dominance: '支配度 (Dominance)',
}
import './History.scss'

const { Title, Paragraph } = Typography

interface HistoryItem {
  id: number
  text: string
  primary_emotion: string
  mbti_type: string
  scores?: Record<string, number>
  raw_intensity_scores?: Record<string, number>
  target_scores?: Record<string, number>
  confidence: number
  latency_ms: number
  json_parse_ok: boolean
  created_at: string
  vad_dimensions?: { valence: number; arousal: number; dominance: number }
  uncertainty_level?: 'low' | 'medium' | 'high'
}

interface HistoryResponse {
  total: number
  items: HistoryItem[]
}

interface DetailItem {
  id: number
  text: string
  output: string
  scores: Record<string, number>
  raw_intensity_scores?: Record<string, number>
  parsed_result?: {
    raw_intensity_scores?: Record<string, number>
    target_scores?: Record<string, number>
    emotion_analysis?: Record<string, number>
    vad_dimensions?: { valence: number; arousal: number; dominance: number }
    emotion_cause?: string
    uncertainty_level?: 'low' | 'medium' | 'high'
    [key: string]: any
  }
  primary_emotion: string
  mbti_type: string
  cot_reasoning: Record<string, string>
  latency_ms: number
  json_parse_ok: boolean
  cot_complete: boolean
  created_at: string
  vad_dimensions?: { valence: number; arousal: number; dominance: number }
  emotion_cause?: string
  uncertainty_level?: 'low' | 'medium' | 'high'
  confidence?: number
}

const History: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [historyData, setHistoryData] = useState<HistoryItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedItem, setSelectedItem] = useState<DetailItem | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const fetchHistory = async (pageNum: number, size: number) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/history?limit=${size}&offset=${(pageNum - 1) * size}`)
      if (!response.ok) {
        throw new Error('获取历史记录失败')
      }
      const data: HistoryResponse = await response.json()
      setHistoryData(data.items)
      setTotal(data.total)
    } catch (err: any) {
      setError(err.message || '加载历史记录时出错')
    } finally {
      setLoading(false)
    }
  }

  const fetchDetail = async (id: number) => {
    setDetailLoading(true)
    setSelectedItem(null)
    setDetailModalVisible(true)
    try {
      const response = await fetch(`/api/history/${id}`)
      if (!response.ok) {
        throw new Error('获取详情失败')
      }
      const data: DetailItem = await response.json()
      setSelectedItem(data)
    } catch (err: any) {
      setError(err.message || '加载详情时出错')
    } finally {
      setDetailLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory(page, pageSize)
  }, [page, pageSize])

  const handlePageChange = (newPage: number, newPageSize: number) => {
    if (newPageSize !== pageSize) {
      setPageSize(newPageSize)
      setPage(1)
    } else {
      setPage(newPage)
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  const columns: ColumnsType<HistoryItem> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 70,
      render: (id) => <span style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>#{id}</span>,
    },
    {
      title: '文本',
      dataIndex: 'text',
      key: 'text',
      width: '30%',
      render: (text) => (
        <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 0, maxWidth: 350 }}>
          {text}
        </Paragraph>
      ),
    },
    {
      title: '情感',
      key: 'emotion',
      width: 90,
      render: (_, record) => (
        <Tag color={EMOTION_COLORS[record.primary_emotion] || '#b0a8a0'} style={{ fontWeight: 600 }}>
          {formatEmotionLabel(record.primary_emotion)}
        </Tag>
      ),
    },
    {
      title: 'VAD',
      key: 'vad',
      width: 80,
      render: (_, record) => (
        <span style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'pre-line' }}>
          {formatVadPreview(record.vad_dimensions)}
        </span>
      ),
    },
    {
      title: '置信度',
      key: 'confidence',
      width: 90,
      render: (_, record) => {
        const confidence = record.confidence || 0
        return (
          <span style={{ color: confidence > 0.7 ? '#10b981' : confidence > 0.5 ? '#f59e0b' : '#ef4444' }}>
            {confidence > 0 ? (confidence * 100).toFixed(2) + '%' : '-'}
          </span>
        )
      },
    },
    {
      title: '不确定性',
      key: 'uncertainty',
      width: 80,
      render: (_, record) => formatUncertaintyBadge(record.uncertainty_level),
    },
    {
      title: 'JSON',
      key: 'json_parse_ok',
      width: 60,
      render: (_, record) =>
        record.json_parse_ok ? (
          <CheckCircleOutlined style={{ color: '#10b981' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ef4444' }} />
        ),
    },
    {
      title: '耗时',
      dataIndex: 'latency_ms',
      key: 'latency_ms',
      width: 80,
      render: (ms) => (
        <span style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
          {formatLatency(ms)}
        </span>
      ),
    },
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date) => (
        <span style={{ color: '#64748b', fontSize: 11 }}>
          {formatDate(date)}
        </span>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => fetchDetail(record.id)}
        >
          详情
        </Button>
      ),
    },
  ]

  return (
    <div className="history-page">
      <div className="page-content">
        <div className="page-header">
          <Title level={1} className="page-title">
            <HistoryOutlined /> 历史记录
          </Title>
          <Paragraph className="page-subtitle">
            查看所有情感分析的历史记录，包括输入文本、分析结果和处理详情
          </Paragraph>
        </div>

        <Card className="history-card">
          <div className="history-toolbar">
            <Space>
              <span style={{ color: '#94a3b8' }}>共 {total} 条记录</span>
            </Space>
            <Space>
              <Select
                value={pageSize}
                onChange={(value) => {
                  setPageSize(value)
                  setPage(1)
                }}
                options={[
                  { value: 10, label: '10条/页' },
                  { value: 20, label: '20条/页' },
                  { value: 50, label: '50条/页' },
                ]}
                style={{ width: 100 }}
              />
              <Button
                icon={<ReloadOutlined />}
                onClick={() => fetchHistory(page, pageSize)}
              >
                刷新
              </Button>
            </Space>
          </div>

          {loading ? (
            <div className="loading-container">
              <Spin size="large" />
              <span style={{ marginTop: 16, color: '#94a3b8' }}>加载历史记录...</span>
            </div>
          ) : error ? (
            <Alert
              message="加载失败"
              description={error}
              type="error"
              showIcon
              action={
                <Button size="small" onClick={() => fetchHistory(page, pageSize)}>
                  重试
                </Button>
              }
            />
          ) : historyData.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="暂无历史记录"
            >
              <Paragraph style={{ color: '#64748b' }}>
                在"在线演示"或"批量推理"页面进行分析后，记录将显示在这里
              </Paragraph>
            </Empty>
          ) : (
            <>
              <Table
                dataSource={historyData}
                columns={columns}
                rowKey="id"
                pagination={false}
                className="history-table"
                onRow={(record) => ({
                  onClick: () => fetchDetail(record.id),
                  style: { cursor: 'pointer' },
                })}
              />
              <div className="pagination-container">
                <Pagination
                  current={page}
                  pageSize={pageSize}
                  total={total}
                  onChange={handlePageChange}
                  showSizeChanger={false}
                  showQuickJumper
                  showTotal={(total) => `共 ${total} 条`}
                />
              </div>
            </>
          )}
        </Card>
      </div>

      <Modal
        title="推理详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
        ]}
        width={700}
      >
        {detailLoading ? (
          <div className="modal-loading">
            <Spin />
          </div>
        ) : selectedItem ? (
          <div className="detail-content">
            <Descriptions column={2} bordered size="small">
              <Descriptions.Item label="ID">{selectedItem.id}</Descriptions.Item>
              <Descriptions.Item label="主要情感">
                <Tag color={emotionColors[selectedItem.primary_emotion]}>
                  {formatEmotionLabel(selectedItem.primary_emotion)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="置信度">
                {selectedItem.confidence !== undefined ? (
                  <span style={{ 
                    color: selectedItem.confidence > 0.7 ? '#10b981' : selectedItem.confidence > 0.5 ? '#f59e0b' : '#ef4444',
                    fontWeight: 600 
                  }}>
                    {(selectedItem.confidence * 100).toFixed(2)}%
                  </span>
                ) : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="不确定性">
                {formatUncertaintyBadge(selectedItem.uncertainty_level)}
              </Descriptions.Item>
              <Descriptions.Item label="JSON解析">
                {selectedItem.json_parse_ok ? (
                  <CheckCircleOutlined style={{ color: '#10b981' }} />
                ) : (
                  <CloseCircleOutlined style={{ color: '#ef4444' }} />
                )}
              </Descriptions.Item>
              <Descriptions.Item label="CoT完成">
                {selectedItem.cot_complete ? (
                  <CheckCircleOutlined style={{ color: '#10b981' }} />
                ) : (
                  <CloseCircleOutlined style={{ color: '#ef4444' }} />
                )}
              </Descriptions.Item>
              <Descriptions.Item label="处理耗时" span={2}>{formatLatency(selectedItem.latency_ms)}</Descriptions.Item>
            </Descriptions>

            {(selectedItem.vad_dimensions || selectedItem.parsed_result?.vad_dimensions) && (
              <div className="detail-section">
                <h4>VAD 维度分析</h4>
                <div className="vad-display">
                  {Object.entries(selectedItem.vad_dimensions || selectedItem.parsed_result?.vad_dimensions || {}).map(([key, value]) => (
                    <div key={key} className="vad-item">
                      <span className="vad-label">{vadLabels[key] || key}</span>
                      <div className="vad-bar">
                        <div
                          className="vad-fill"
                          style={{ width: `${(value as number) * 100}%` }}
                        />
                      </div>
                      <span className="vad-value">{(value as number).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {(selectedItem.emotion_cause || selectedItem.parsed_result?.emotion_cause) && (
              <div className="detail-section">
                <h4>情感归因</h4>
                <div className="emotion-cause">
                  "{selectedItem.emotion_cause || selectedItem.parsed_result?.emotion_cause}"
                </div>
              </div>
            )}

            <div className="detail-section">
              <h4>输入文本</h4>
              <div className="detail-text">{selectedItem.text}</div>
            </div>

            {(selectedItem.scores || selectedItem.raw_intensity_scores || selectedItem.parsed_result?.raw_intensity_scores) && (
              <div className="detail-section">
                <h4>情感评分对比</h4>
                {(selectedItem.raw_intensity_scores || selectedItem.parsed_result?.raw_intensity_scores) ? (
                  <>
                    <div style={{ marginBottom: 16 }}>
                      <strong style={{ color: '#94a8b8' }}>原始强度分 (CoT 推理前):</strong>
                      <div className="emotion-scores">
                        {Object.entries(selectedItem.raw_intensity_scores || selectedItem.parsed_result?.raw_intensity_scores || {}).map(([emotion, score]) => (
                          <div key={`raw-${emotion}`} className="emotion-score-item">
                            <span className="emotion-label">{formatEmotionLabel(emotion)}</span>
                            <div className="score-bar">
                              <div
                                className="score-fill"
                                style={{
                                  width: `${(score as number) * 100}%`,
                                  backgroundColor: EMOTION_COLORS[emotion] || '#b0a8a0',
                                }}
                              />
                            </div>
                            <span className="score-value">{(score as number * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <strong style={{ color: '#94a8b8' }}>归一化概率分 (CoT 推理后):</strong>
                      <div className="emotion-scores">
                        {Object.entries(selectedItem.scores || selectedItem.parsed_result?.target_scores || {}).map(([emotion, score]) => (
                          <div key={`norm-${emotion}`} className="emotion-score-item">
                            <span className="emotion-label">{formatEmotionLabel(emotion)}</span>
                            <div className="score-bar">
                              <div
                                className="score-fill"
                                style={{
                                  width: `${(score as number) * 100}%`,
                                  backgroundColor: EMOTION_COLORS[emotion] || '#b0a8a0',
                                }}
                              />
                            </div>
                            <span className="score-value">{(score as number * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="emotion-scores">
                    {Object.entries(selectedItem.scores || {}).map(([emotion, score]) => (
                      <div key={emotion} className="emotion-score-item">
                        <span className="emotion-label">{formatEmotionLabel(emotion)}</span>
                        <div className="score-bar">
                          <div
                            className="score-fill"
                            style={{
                              width: `${(score as number) * 100}%`,
                              backgroundColor: EMOTION_COLORS[emotion] || '#b0a8a0',
                            }}
                          />
                        </div>
                        <span className="score-value">{(score as number * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {selectedItem.cot_reasoning && Object.keys(selectedItem.cot_reasoning).length > 0 && (
              <div className="detail-section">
                <h4>思维链推理 (CoT)</h4>
                <div className="cot-reasoning">
                  {Object.entries(selectedItem.cot_reasoning).map(([step, content]) => {
                    const parsed = parseCotContent(step, content, selectedItem.primary_emotion)
                    return (
                      <div key={step} className="cot-step">
                        <div className="cot-step-label">{STEP_TITLES[step] || step}</div>
                        <div className="cot-step-content">
                          {parsed.content}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {selectedItem.output && (
              <div className="detail-section">
                <h4>模型原始输出</h4>
                <pre className="raw-output">{selectedItem.output}</pre>
              </div>
            )}
          </div>
        ) : (
          <Empty description="无法加载详情" />
        )}
      </Modal>
    </div>
  )
}

export default History
