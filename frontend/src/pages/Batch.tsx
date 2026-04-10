import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Typography, Card, Button, Upload, Table, Space, Tag, Progress, Input, Tabs, message, Select, Switch, Tooltip } from 'antd'
import {
  TableOutlined,
  UploadOutlined,
  FileTextOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  StopOutlined,
} from '@ant-design/icons'
import type { UploadFile, UploadProps } from 'antd'
import type { TableProps } from 'antd'
import { sentimentApi, BatchItem, BatchResultsResponse } from '@/services/api'
import { formatLatency } from '@/utils/cotParser'
import './Batch.scss'

const { Title, Paragraph } = Typography
const { TextArea } = Input

interface BatchResult extends BatchItem {
  processingTime: number
}

const Batch: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'file' | 'terminal'>('file')
  const [files, setFiles] = useState<UploadFile[]>([])
  const [terminalInput, setTerminalInput] = useState('')
  const [terminalResults, setTerminalResults] = useState<BatchResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<BatchResult[]>([])
  const [modelVariant, setModelVariant] = useState<'base' | 'lora_merged' | 'gguf4bit'>('gguf4bit')
  const [processedCount, setProcessedCount] = useState(0)
  const [totalCount, setTotalCount] = useState(0)
  const [currentText, setCurrentText] = useState('')
  const terminalRef = useRef<HTMLDivElement>(null)
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const [useQuickPreset, setUseQuickPreset] = useState(true)
  const [_currentBatchId, setCurrentBatchId] = useState<string | null>(null)
  const [isCancelled, setIsCancelled] = useState(false)

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearTimeout(progressIntervalRef.current)
      }
    }
  }, [])

  const stopPolling = useCallback(() => {
    if (progressIntervalRef.current) {
      clearTimeout(progressIntervalRef.current)
      progressIntervalRef.current = null
    }
  }, [])

  const handleCancel = useCallback(() => {
    stopPolling()
    setIsCancelled(true)
    setIsProcessing(false)
    setCurrentBatchId(null)
    message.info('已取消批量处理')
  }, [stopPolling])

  const emotionColors: Record<string, string> = {
    angry: '#dc8888',
    fear: '#a894c4',
    happy: '#88c4a8',
    neutral: '#b0a8a0',
    sad: '#88a4c4',
    surprise: '#dcc488',
  }

  const handleFileChange: UploadProps['onChange'] = ({ fileList }) => {
    setFiles(fileList)
  }

  const uploadProps: UploadProps = {
    multiple: true,
    accept: '.txt,.csv',
    fileList: files,
    onChange: handleFileChange,
    beforeUpload: (file) => {
      const isValidType = ['text/plain', 'text/csv'].includes(file.type) ||
        file.name.endsWith('.txt') || file.name.endsWith('.csv')
      const isValidSize = file.size < 10 * 1024 * 1024
      if (!isValidType) {
        message.error('只能上传 TXT/CSV 文件')
        return false
      }
      if (!isValidSize) {
        message.error('文件大小不能超过 10MB')
        return false
      }
      return false // Prevent auto upload
    },
  }

  const parseFileContent = async (file: File): Promise<string[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string
          // Split by newlines and filter empty lines
          const lines = text.split(/\r?\n/).filter(line => line.trim())
          resolve(lines)
        } catch (err) {
          reject(new Error(`解析文件失败：${file.name}`))
        }
      }
      reader.onerror = () => reject(new Error(`读取文件失败：${file.name}`))
      reader.readAsText(file, 'utf-8')
    })
  }

  const handleBatchProcess = async () => {
    if (files.length === 0) {
      message.warning('请先上传文件')
      return
    }

    stopPolling()
    setIsCancelled(false)
    setIsProcessing(true)
    setProgress(0)
    setResults([])
    setProcessedCount(0)
    setCurrentText('')
    setCurrentBatchId(null)

    try {
      const allTexts: string[] = []
      for (const file of files) {
        const originFileObj = file.originFileObj
        if (originFileObj) {
          const lines = await parseFileContent(originFileObj)
          allTexts.push(...lines)
        }
      }

      if (allTexts.length === 0) {
        message.warning('文件中没有找到任何文本')
        setIsProcessing(false)
        return
      }

      setTotalCount(allTexts.length)
      message.info(`开始处理 ${allTexts.length} 条文本（${useQuickPreset ? '快速' : '标准'}模式）...`)

      const startResponse = await sentimentApi.startBatchInfer({
        texts: allTexts,
        model_variant: modelVariant,
        use_quick_preset: useQuickPreset,
      })

      const batchId = startResponse.batch_id
      setCurrentBatchId(batchId)

      const pollProgress = async () => {
        if (isCancelled) {
          stopPolling()
          return
        }

        try {
          const progressData = await sentimentApi.getBatchProgress(batchId)
          
          setProcessedCount(progressData.processed)
          setProgress(progressData.progress_percent)
          setCurrentText(progressData.current_text || '')

          if (progressData.status === 'completed') {
            const resultsResponse = await sentimentApi.getBatchResults(batchId) as BatchResultsResponse
            const batchResults: BatchResult[] = resultsResponse.results.map((item: BatchItem) => ({
              ...item,
              processingTime: item.result?.latency_ms || 0,
            }))
            setResults(batchResults)
            setProgress(100)
            setCurrentText('')
            setCurrentBatchId(null)
            setIsProcessing(false)
            message.success(`处理完成！成功：${resultsResponse.success}, 失败：${resultsResponse.failed}`)
          } else if (progressData.status === 'not_found') {
            if (!isCancelled) {
              setIsProcessing(false)
              message.error('批量处理任务未找到或已过期')
            }
          } else {
            progressIntervalRef.current = setTimeout(pollProgress, 500)
          }
        } catch (err: any) {
          if (!isCancelled) {
            setIsProcessing(false)
            message.error(`获取进度失败：${err.message}`)
          }
        }
      }

      pollProgress()

    } catch (err: any) {
      message.error(`批量处理失败：${err.message || '未知错误'}`)
      setProgress(0)
      setIsProcessing(false)
    }
  }

  const handleTerminalAnalyze = async () => {
    if (!terminalInput.trim()) {
      message.warning('请输入要分析的文本')
      return
    }

    const lines = terminalInput.trim().split('\n').filter(line => line.trim())
    if (lines.length === 0) {
      message.warning('请输入至少一条文本')
      return
    }

    stopPolling()
    setIsCancelled(false)
    setIsProcessing(true)
    setTerminalResults([])
    setProgress(0)
    setProcessedCount(0)
    setTotalCount(lines.length)
    setCurrentText('')
    setCurrentBatchId(null)

    try {
      const startResponse = await sentimentApi.startBatchInfer({
        texts: lines,
        model_variant: modelVariant,
        use_quick_preset: useQuickPreset,
      })

      const batchId = startResponse.batch_id
      setCurrentBatchId(batchId)

      const pollProgress = async () => {
        if (isCancelled) {
          stopPolling()
          return
        }

        try {
          const progressData = await sentimentApi.getBatchProgress(batchId)
          
          setProcessedCount(progressData.processed)
          setProgress(progressData.progress_percent)
          setCurrentText(progressData.current_text || '')

          if (progressData.status === 'completed') {
            const resultsResponse = await sentimentApi.getBatchResults(batchId) as BatchResultsResponse
            const transformedResults: BatchResult[] = resultsResponse.results.map((item: BatchItem) => ({
              ...item,
              processingTime: item.result?.latency_ms || 0,
            }))
            setTerminalResults(transformedResults)
            setProgress(100)
            setCurrentText('')
            setCurrentBatchId(null)
            setIsProcessing(false)
            message.success(`分析完成！成功：${resultsResponse.success}, 失败：${resultsResponse.failed}`)
          } else if (progressData.status === 'not_found') {
            if (!isCancelled) {
              setIsProcessing(false)
              message.error('批量处理任务未找到或已过期')
            }
          } else {
            progressIntervalRef.current = setTimeout(pollProgress, 500)
          }
        } catch (err: any) {
          if (!isCancelled) {
            setIsProcessing(false)
            message.error(`获取进度失败：${err.message}`)
          }
        }
      }

      pollProgress()

    } catch (err: any) {
      message.error(`分析失败：${err.message || '未知错误'}`)
      setIsProcessing(false)
    }
  }

  const handleExport = () => {
    const dataToExport = activeTab === 'file' ? results : terminalResults
    const csv = [
      ['ID', '文本', '预测情感', '置信度', '不确定性', '第二情感', '第二置信度', '效价(V)', '唤醒度(A)', '支配度(D)', '情感归因', '处理时间 (ms)', '状态'],
      ...dataToExport.map(r => {
        const second = r.result?.scores ? getSecondEmotion(r.result.scores, r.result.primary_emotion) : null
        const vad = r.result?.vad_dimensions
        return [
          r.id,
          `"${r.text.replace(/"/g, '""')}"`,
          r.result?.primary_emotion || '-',
          r.result ? (r.result.confidence * 100).toFixed(2) + '%' : '-',
          r.result?.uncertainty_level || '-',
          second ? (second.emotion.charAt(0).toUpperCase() + second.emotion.slice(1)) : '',
          second ? (second.confidence * 100).toFixed(2) + '%' : '-',
          vad ? vad.valence.toFixed(2) : '-',
          vad ? vad.arousal.toFixed(2) : '-',
          vad ? vad.dominance.toFixed(2) : '-',
          r.result?.emotion_cause ? `"${r.result.emotion_cause.replace(/"/g, '""')}"` : '-',
          r.processingTime,
          r.success ? 'success' : 'error',
        ]
      }),
    ].map(row => row.join(',')).join('\n')

    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `batch-results-${new Date().getTime()}.csv`
    link.click()
    message.success('结果已导出')
  }

  const getSecondEmotion = (scores: Record<string, number>, primary: string) => {
    const entries = Object.entries(scores)
      .filter(([key]) => key !== primary)
      .sort((a, b) => b[1] - a[1])
    if (entries.length === 0) return null
    return { emotion: entries[0][0], confidence: entries[0][1] }
  }

  const batchColumns: TableProps<BatchResult>['columns'] = [
    {
      title: '#',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '文本',
      dataIndex: 'text',
      key: 'text',
      width: '35%',
      render: (text) => (
        <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 0 }}>
          {text}
        </Paragraph>
      ),
    },
    {
      title: '预测情感',
      key: 'prediction',
      width: 140,
      render: (_, record) => {
        if (!record.success || !record.result) {
          return <Tag color="gray">失败</Tag>
        }
        const primary = record.result.primary_emotion
        const confidence = record.result.confidence
        return (
          <div>
            <Tag color={emotionColors[primary]} style={{ fontWeight: 600, marginBottom: 4 }}>
              {primary.charAt(0).toUpperCase() + primary.slice(1)}
            </Tag>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>
              {(confidence * 100).toFixed(2)}%
            </div>
          </div>
        )
      },
    },
    {
      title: '置信度',
      key: 'confidence',
      width: 140,
      render: (_, record) => {
        if (!record.success || !record.result) {
          return <span style={{ color: '#94a3b8' }}>-</span>
        }
        const value = record.result.confidence
        return (
          <div className="confidence-bar">
            <Progress
              percent={value * 100}
              strokeColor={value > 0.8 ? '#10b981' : value > 0.6 ? '#f59e0b' : '#ef4444'}
              trailColor="rgba(148, 163, 184, 0.2)"
              format={(val) => `${val?.toFixed(1)}%`}
              size="small"
            />
          </div>
        )
      },
    },
    {
      title: '第二情感',
      key: 'top2',
      width: 140,
      render: (_, record) => {
        if (!record.success || !record.result?.scores) {
          return <span style={{ color: '#94a3b8' }}>-</span>
        }
        const second = getSecondEmotion(record.result.scores, record.result.primary_emotion)
        if (!second || second.confidence < 0.05) {
          return <span style={{ color: '#94a3b8', fontSize: 12 }}>-</span>
        }
        return (
          <Space size={4}>
            <Tag color={emotionColors[second.emotion]} style={{ fontSize: 12 }}>
              {second.emotion.charAt(0).toUpperCase() + second.emotion.slice(1)}
            </Tag>
            <span style={{ color: '#64748b', fontSize: 12 }}>
              {(second.confidence * 100).toFixed(2)}%
            </span>
          </Space>
        )
      },
    },
    {
      title: '处理时间',
      dataIndex: 'processingTime',
      key: 'processingTime',
      width: 100,
      render: (value) => (
        <span style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
          {value}ms
        </span>
      ),
    },
    {
      title: '状态',
      key: 'status',
      width: 80,
      render: (_, record) =>
        record.success ? (
          <CheckCircleOutlined style={{ color: '#10b981' }} />
        ) : (
          <CloseCircleOutlined style={{ color: '#ef4444' }} />
        ),
    },
  ]

  return (
    <div className="batch-page">
      <div className="page-content">
        {/* Header */}
        <div className="page-header">
          <Title level={1} className="page-title">
            <TableOutlined /> 批量推理
          </Title>
          <Paragraph className="page-subtitle">
            批量处理多条文本的情感分析，支持文件上传和终端输入两种模式
          </Paragraph>
        </div>

        {/* Mode Tabs */}
        <Card className="mode-card">
          <Tabs
            activeKey={activeTab}
            onChange={(key: string) => setActiveTab(key as 'file' | 'terminal')}
            items={[
              {
                key: 'file',
                label: (
                  <span>
                    <UploadOutlined /> 文件上传
                  </span>
                ),
                children: (
                  <div className="tab-content">
                    <div className="upload-section">
                      <Upload.Dragger {...uploadProps}>
                        <p className="ant-upload-drag-icon">
                          <UploadOutlined style={{ color: '#ff6b4a' }} />
                        </p>
                        <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
                        <p className="ant-upload-hint">
                          支持 TXT、CSV 文件格式，单个文件不超过 10MB
                        </p>
                      </Upload.Dragger>
                    </div>

                    {files.length > 0 && (
                      <div className="file-list">
                        <div className="file-list-header">
                          <span>已选择 {files.length} 个文件</span>
                          <Button
                            type="link"
                            danger
                            size="small"
                            onClick={() => setFiles([])}
                          >
                            清空
                          </Button>
                        </div>
                        {files.map((file) => (
                          <div key={file.uid} className="file-item">
                            <FileTextOutlined className="file-icon" />
                            <span className="file-name">{file.name}</span>
                            <span className="file-size">
                              {(file.size || 0 / 1024).toFixed(1)} KB
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="model-select-section">
                      <div className="select-row">
                        <span className="model-select-label">选择模型：</span>
                        <Select
                          value={modelVariant}
                          onChange={(value: 'base' | 'lora_merged' | 'gguf4bit') => setModelVariant(value)}
                          options={[
                            { value: 'base', label: 'Base (基础版)' },
                            { value: 'lora_merged', label: 'LoRA (微调版)' },
                            { value: 'gguf4bit', label: 'GGUF 4bit (量化版)' },
                          ]}
                          className="model-select"
                          disabled={isProcessing}
                        />
                      </div>
                      <div className="select-row">
                        <span className="model-select-label">推理模式：</span>
                        <Space>
                          <Switch
                            checked={useQuickPreset}
                            onChange={setUseQuickPreset}
                            disabled={isProcessing}
                            checkedChildren="快速"
                            unCheckedChildren="标准"
                          />
                          <Tooltip title={useQuickPreset ? '快速模式：推理速度更快，适合大批量处理' : '标准模式：更详细的推理分析'}>
                            <span style={{ fontSize: 12, color: '#94a3b8' }}>
                              {useQuickPreset ? '⚡ 快速' : '🎯 标准'}
                            </span>
                          </Tooltip>
                        </Space>
                      </div>
                    </div>

                    {isProcessing && (
                      <div className="processing-section">
                        <div className="processing-header">
                          <SyncOutlined spin /> 正在处理...
                          <span style={{ marginLeft: 8, fontSize: 12, color: '#94a3b8' }}>
                            {useQuickPreset ? '⚡ 快速模式' : '🎯 标准模式'}
                          </span>
                        </div>
                        <Progress
                          percent={progress}
                          strokeColor="#ff6b4a"
                          status={progress === 100 ? 'success' : 'active'}
                          format={(percent) => `${percent}%`}
                        />
                        <div className="processing-stats">
                          <span>已处理：{processedCount} / {totalCount} 条</span>
                          {currentText && (
                            <span style={{ marginLeft: 16, color: '#64748b', fontSize: 12 }}>
                              当前：{currentText.substring(0, 30)}{currentText.length > 30 ? '...' : ''}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="batch-actions">
                      <Button
                        type="primary"
                        size="large"
                        icon={isProcessing ? <SyncOutlined spin /> : <PlayCircleOutlined />}
                        onClick={handleBatchProcess}
                        disabled={files.length === 0 || isProcessing}
                        className="btn-process"
                      >
                        {isProcessing ? '处理中...' : '开始批量处理'}
                      </Button>
                      {isProcessing && (
                        <Button
                          danger
                          size="large"
                          icon={<StopOutlined />}
                          onClick={handleCancel}
                        >
                          取消
                        </Button>
                      )}
                      <Button
                        size="large"
                        icon={<DownloadOutlined />}
                        onClick={handleExport}
                        disabled={results.length === 0}
                      >
                        导出结果
                      </Button>
                    </div>
                  </div>
                ),
              },
              {
                key: 'terminal',
                label: (
                  <span>
                    <FileTextOutlined /> 终端输入
                  </span>
                ),
                children: (
                  <div className="tab-content terminal-content">
                    <div className="terminal-section">
                      <div className="terminal-header">
                        <span className="terminal-title">多行文本输入</span>
                        <span className="terminal-hint">每行一条文本，点击分析批量处理</span>
                      </div>

                      <div className="model-select-section">
                        <div className="select-row">
                          <span className="model-select-label">选择模型：</span>
                          <Select
                            value={modelVariant}
                            onChange={(value: 'base' | 'lora_merged' | 'gguf4bit') => setModelVariant(value)}
                            options={[
                              { value: 'base', label: 'Base (基础版)' },
                              { value: 'lora_merged', label: 'LoRA (微调版)' },
                              { value: 'gguf4bit', label: 'GGUF 4bit (量化版)' },
                            ]}
                            className="model-select"
                            disabled={isProcessing}
                          />
                        </div>
                        <div className="select-row">
                          <span className="model-select-label">推理模式：</span>
                          <Space>
                            <Switch
                              checked={useQuickPreset}
                              onChange={setUseQuickPreset}
                              disabled={isProcessing}
                              checkedChildren="快速"
                              unCheckedChildren="标准"
                            />
                            <span style={{ fontSize: 12, color: '#94a3b8' }}>
                              {useQuickPreset ? '⚡ 快速' : '🎯 标准'}
                            </span>
                          </Space>
                        </div>
                      </div>

                      {isProcessing && (
                        <div className="processing-section">
                          <div className="processing-header">
                            <SyncOutlined spin /> 正在处理...
                            <span style={{ marginLeft: 8, fontSize: 12, color: '#94a3b8' }}>
                              {useQuickPreset ? '⚡ 快速模式' : '🎯 标准模式'}
                            </span>
                          </div>
                          <Progress
                            percent={progress}
                            strokeColor="#ff6b4a"
                            status={progress === 100 ? 'success' : 'active'}
                            format={(percent) => `${percent}%`}
                          />
                          <div className="processing-stats">
                            <span>已处理：{processedCount} / {totalCount} 条</span>
                            {currentText && (
                              <span style={{ marginLeft: 16, color: '#64748b', fontSize: 12 }}>
                                当前：{currentText.substring(0, 30)}{currentText.length > 30 ? '...' : ''}
                              </span>
                            )}
                          </div>
                        </div>
                      )}

                      <TextArea
                        ref={terminalRef}
                        value={terminalInput}
                        onChange={(e) => setTerminalInput(e.target.value)}
                        placeholder={`请输入要分析的文本，每行一条：

今天天气真好，心情特别愉快！
这个电影太难看了，浪费了我的时间。
没什么特别的感觉，普普通通吧。
太令人惊讶了，完全没想到会这样！`}
                        rows={10}
                        className="terminal-input"
                      />
                      <div className="terminal-actions">
                        <Button
                          type="primary"
                          icon={isProcessing ? <SyncOutlined spin /> : <PlayCircleOutlined />}
                          onClick={handleTerminalAnalyze}
                          disabled={!terminalInput.trim() || isProcessing}
                          loading={isProcessing && activeTab === 'terminal'}
                        >
                          {isProcessing && activeTab === 'terminal' ? '分析中...' : '分析'}
                        </Button>
                        {isProcessing && activeTab === 'terminal' && (
                          <Button
                            danger
                            icon={<StopOutlined />}
                            onClick={handleCancel}
                          >
                            取消
                          </Button>
                        )}
                        <Button
                          icon={<DeleteOutlined />}
                          onClick={() => {
                            setTerminalInput('')
                            setTerminalResults([])
                          }}
                        >
                          清空
                        </Button>
                        <Button
                          icon={<DownloadOutlined />}
                          onClick={handleExport}
                          disabled={terminalResults.length === 0}
                        >
                          导出结果
                        </Button>
                      </div>
                    </div>

                    {terminalResults.length > 0 && (
                      <div className="terminal-results">
                        <div className="results-header">
                          <Title level={5}>分析结果 ({terminalResults.length} 条)</Title>
                        </div>
                        <Table
                          dataSource={terminalResults}
                          columns={batchColumns}
                          rowKey="id"
                          pagination={false}
                          size="small"
                          scroll={{ y: 400 }}
                        />
                      </div>
                    )}
                  </div>
                ),
              },
            ]}
          />
        </Card>

        {/* Results Section */}
        {results.length > 0 && (
          <Card className="results-card" title={`批量处理结果 (${results.length} 条)`}>
            <div className="results-header">
              <Space>
                <Tag color="#10b981">
                  {processedCount > 0 ? <CheckCircleOutlined /> : null} 成功：{processedCount}
                </Tag>
                <Tag color="#ef4444">
                  {results.filter(r => !r.success).length > 0 ? <CloseCircleOutlined /> : null} 失败：{results.filter(r => !r.success).length}
                </Tag>
                <Tag color="#ff6b4a">平均耗时：{formatLatency(results.reduce((sum, r) => sum + (r.processingTime || 0), 0) / results.length)}</Tag>
                <Tag color="#88c4a8">
                  平均置信度：{results.filter(r => r.success && r.result).length > 0
                    ? ((results.reduce((sum, r) => sum + (r.result?.confidence || 0), 0) / results.filter(r => r.success && r.result).length) * 100).toFixed(2)
                    : 0}%
                </Tag>
              </Space>
            </div>
            <Table
              dataSource={results}
              columns={batchColumns}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              className="batch-table"
            />
          </Card>
        )}
      </div>
    </div>
  )
}

export default Batch
