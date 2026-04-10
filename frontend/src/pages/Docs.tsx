import React, { useState } from 'react'
import { Typography, Card, Menu, Button, Space, Tag, Timeline, Table, Descriptions, Collapse } from 'antd'
import {
  FileTextOutlined,
  BookOutlined,
  CodeOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  DownloadOutlined,
} from '@ant-design/icons'
import type { MenuProps } from 'antd'
import './Docs.scss'

const { Title, Paragraph } = Typography
const { Panel } = Collapse

const Docs: React.FC = () => {
  const [activeSection, setActiveSection] = useState('overview')

  const menuItems: MenuProps['items'] = [
    { key: 'overview', label: '系统概述', icon: <BookOutlined /> },
    { key: 'architecture', label: '系统架构', icon: <ApiOutlined /> },
    { key: 'cot', label: 'CoT 推理过程', icon: <ThunderboltOutlined /> },
    { key: 'params', label: '技术参数', icon: <CodeOutlined /> },
    { key: 'api', label: 'API 接口', icon: <FileTextOutlined /> },
  ]

  const handleMenuClick: MenuProps['onClick'] = (e) => {
    setActiveSection(e.key)
  }

  const cotSteps = [
    {
      step: 1,
      title: '文本线索识别',
      description: '识别文本中的情感关键词、标点符号和表达方式',
      icon: '🔍',
      details: '提取情感触发词、程度副词、否定词、标点符号等语言线索',
    },
    {
      step: 2,
      title: '候选情绪列表',
      description: '基于文本线索生成可能的情绪候选列表',
      icon: '📋',
      details: '根据线索匹配六种基本情绪：Angry、Fear、Happy、Neutral、Sad、Surprise',
    },
    {
      step: 3,
      title: '强度评估依据',
      description: '评估每种情绪候选的强度等级',
      icon: '📊',
      details: '结合程度副词、感叹号、重复表达等因素量化情绪强度 (0.0-1.0)',
    },
    {
      step: 4,
      title: '复合情绪关系',
      description: '分析多种情绪共存的复合情绪状态',
      icon: '🔗',
      details: '识别人类情感的多维度特性，如"喜极而泣"(Happy+Sad) 等复合情绪',
    },
    {
      step: 5,
      title: 'MBTI 调节分析',
      description: '基于文本推断作者的人格特质倾向',
      icon: '🧩',
      details: '分析 E-I(外向 - 内向)、S-N(感觉 - 直觉)、T-F(思维 - 情感)、J-P(判断 - 知觉) 四个维度',
    },
    {
      step: 6,
      title: '反事实检验',
      description: '通过反事实推理验证情绪判断的合理性',
      icon: '🤔',
      details: '假设"如果不是这种情绪会怎样"，检验当前判断的稳健性',
    },
    {
      step: 7,
      title: '不确定性说明和最终结论',
      description: '说明判断的不确定性并给出最终情绪结论',
      icon: '✅',
      details: '输出主要情绪、置信度、MBTI 类型及完整的 JSON 格式分析结果',
    },
  ]

  const technicalParams = [
    { param: '基座模型', value: 'Qwen2.5-7B-Instruct' },
    { param: '参数量', value: '7.2B' },
    { param: '注意力机制', value: 'Grouped Query Attention (GQA)' },
    { param: '词表大小', value: '151,936 (vocab size)' },
    { param: '最大序列长度', value: '512 tokens' },
    { param: 'LoRA Rank', value: 'r=16' },
    { param: 'LoRA Alpha', value: 'α=32' },
    { param: 'LoRA Dropout', value: '0.05' },
    { param: '微调方法', value: 'LoRA (QLoRA 4-bit quantization)' },
    { param: '批次大小', value: 'batch_size=32' },
    { param: '学习率', value: '2e-4 (cosine decay)' },
    { param: '训练轮次', value: '10 epochs' },
    { param: '优化器', value: 'AdamW (β1=0.9, β2=0.999)' },
    { param: '推理服务', value: 'LMStudio API' },
    { param: '模型变体', value: 'base / lora_merged / gguf4bit' },
    { param: '后端框架', value: 'FastAPI + SQLite' },
    { param: '前端框架', value: 'React 18 + TypeScript' },
  ]

  const apiEndpoints = [
    {
      method: 'POST',
      endpoint: '/api/infer',
      description: '单条文本情感分析推理',
      input: '{ text: string, model_variant?: "base" | "lora_merged" | "gguf4bit" }',
      output: '{ output: string, scores: Record<string, number>, primary_emotion: string, confidence: number, mbti_type: string, cot: Record<string, string>, latency_ms: number }',
    },
    {
      method: 'POST',
      endpoint: '/api/infer/stream',
      description: '流式推理接口 (SSE)，实时返回 CoT 推理过程',
      input: '{ text: string, model_variant?: "base" | "lora_merged" | "gguf4bit" }',
      output: 'SSE Stream: { delta: string, output: string, done: boolean, latency_ms?: number }',
    },
    {
      method: 'POST',
      endpoint: '/api/batch',
      description: '批量文本情感分析',
      input: '{ texts: string[], model_variant?: "base" | "lora_merged" | "gguf4bit" }',
      output: '{ total: number, success: number, failed: number, results: BatchItem[] }',
    },
    {
      method: 'GET',
      endpoint: '/api/metrics/:model_variant',
      description: '获取模型评估指标',
      input: '-',
      output: '{ emotion_macro_mae: number, primary_cls_accuracy: number, primary_cls_macro_f1: number, mbti_accuracy: number, json_parse_rate: number, cot7_complete_rate: number, ... }',
    },
    {
      method: 'POST',
      endpoint: '/api/compare',
      description: '模型对比分析',
      input: '{ model_variants?: string[] }',
      output: '{ models: MetricsResponse[], comparison_table: string }',
    },
    {
      method: 'GET',
      endpoint: '/api/health',
      description: '健康检查',
      input: '-',
      output: '{ status: string, lmstudio_connected: boolean, database_connected: boolean, version: string }',
    },
  ]

  const renderContent = () => {
    switch (activeSection) {
      case 'overview':
        return (
          <div className="section-content">
            <Title level={2}>系统概述</Title>
            <Paragraph className="section-intro">
              本系统是一个基于大语言模型的情感分析平台，专为社交平台场景设计。
              采用 Qwen2.5-7B-Instruct 作为基座模型，通过 LoRA 技术进行多任务指令微调，
              实现高效、准确的情感分析和人格预测。
            </Paragraph>

            <Card className="feature-card" title="核心功能">
              <div className="feature-grid">
                <div className="feature-item">
                  <div className="feature-icon">🎭</div>
                  <h4>六类情绪识别</h4>
                  <p>Angry、Fear、Happy、Neutral、Sad、Surprise</p>
                </div>
                <div className="feature-item">
                  <div className="feature-icon">🧩</div>
                  <h4>MBTI 人格预测</h4>
                  <p>基于文本内容预测 16 种人格类型</p>
                </div>
                <div className="feature-item">
                  <div className="feature-icon">🔬</div>
                  <h4>CoT 可解释推理</h4>
                  <p>7 步思维链展示推理过程</p>
                </div>
                <div className="feature-item">
                  <div className="feature-icon">⚡</div>
                  <h4>批量处理</h4>
                  <p>支持文件上传和多行文本输入</p>
                </div>
              </div>
            </Card>

            <Card className="tech-stack-card" title="技术栈">
              <Descriptions column={2} bordered>
                <Descriptions.Item label="后端框架">FastAPI + Uvicorn</Descriptions.Item>
                <Descriptions.Item label="深度学习">PyTorch + Transformers</Descriptions.Item>
                <Descriptions.Item label="前端框架">React 18 + TypeScript</Descriptions.Item>
                <Descriptions.Item label="UI 组件库">Ant Design 5</Descriptions.Item>
                <Descriptions.Item label="可视化">ECharts 5 + Recharts</Descriptions.Item>
                <Descriptions.Item label="部署方式">Docker + AutoDL</Descriptions.Item>
              </Descriptions>
            </Card>
          </div>
        )

      case 'architecture':
        return (
          <div className="section-content">
            <Title level={2}>系统架构</Title>
            <Paragraph className="section-intro">
              系统采用前后端分离架构，支持高并发推理请求和实时结果可视化。
            </Paragraph>

            <Card className="architecture-card" title="整体架构图">
              <div className="architecture-diagram">
                <div className="arch-layer">
                  <div className="arch-title">前端层 (Frontend)</div>
                  <div className="arch-components">
                    <Tag color="#ff6b4a">React 18</Tag>
                    <Tag color="#ff6b4a">TypeScript</Tag>
                    <Tag color="#ff6b4a">Ant Design 5</Tag>
                    <Tag color="#ff6b4a">ECharts 5</Tag>
                  </div>
                </div>
                <div className="arch-arrow">⬇️</div>
                <div className="arch-layer">
                  <div className="arch-title">API 网关层 (API Gateway)</div>
                  <div className="arch-components">
                    <Tag color="#88c4a8">FastAPI</Tag>
                    <Tag color="#88c4a8">CORS</Tag>
                    <Tag color="#88c4a8">请求验证</Tag>
                    <Tag color="#88c4a8">错误处理</Tag>
                  </div>
                </div>
                <div className="arch-arrow">⬇️</div>
                <div className="arch-layer">
                  <div className="arch-title">推理服务层 (Inference Service)</div>
                  <div className="arch-components">
                    <Tag color="#a894c4">Qwen2.5-7B</Tag>
                    <Tag color="#a894c4">LoRA Adapters</Tag>
                    <Tag color="#a894c4">vLLM 加速</Tag>
                    <Tag color="#a894c4">Batch Processing</Tag>
                  </div>
                </div>
                <div className="arch-arrow">⬇️</div>
                <div className="arch-layer">
                  <div className="arch-title">数据存储层 (Storage)</div>
                  <div className="arch-components">
                    <Tag color="#88a4c4">SQLite</Tag>
                    <Tag color="#88a4c4">文件系统</Tag>
                    <Tag color="#88a4c4">缓存机制</Tag>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="flow-card" title="推理流程">
              <Timeline
                items={[
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>用户输入文本</strong>
                        <p>通过前端界面提交待分析的文本内容</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>文本预处理</strong>
                        <p>分词、标准化、序列化为模型输入格式</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>LoRA 模型推理</strong>
                        <p>Qwen2.5-7B + LoRA 适配器进行前向传播计算</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>多任务输出解析</strong>
                        <p>解析情感分类、MBTI 预测、CoT 推理步骤</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>结果返回与可视化</strong>
                        <p>前端接收并展示分析结果，支持图表导出</p>
                      </div>
                    ),
                  },
                ]}
              />
            </Card>
          </div>
        )

      case 'cot':
        return (
          <div className="section-content">
            <Title level={2}>CoT 推理过程</Title>
            <Paragraph className="section-intro">
              Chain-of-Thought (思维链) 技术使模型能够展示其推理过程，提高决策透明度和可解释性。
            </Paragraph>

            <Card className="cot-card" title="7 步 CoT 推理流程">
              <Timeline
                items={cotSteps.map((step) => ({
                  color: '#ff6b4a',
                  dot: (
                    <div className="cot-timeline-dot">
                      <span className="dot-icon">{step.icon}</span>
                    </div>
                  ),
                  children: (
                    <div className="cot-step">
                      <div className="cot-step-header">
                        <span className="cot-step-number">{step.step}</span>
                        <h4 className="cot-step-title">{step.title}</h4>
                      </div>
                      <p className="cot-step-desc">{step.description}</p>
                      <div className="cot-step-details">
                        <CodeOutlined /> 技术细节：{step.details}
                      </div>
                    </div>
                  ),
                }))}
              />
            </Card>

            <Card className="example-card" title="CoT 推理示例">
              <div className="example-input">
                <strong>输入文本：</strong>
                <p>「今天收到了期待的礼物，真的太开心了！感谢朋友们的用心安排～」</p>
              </div>
              <div className="example-output">
                <strong>推理输出：</strong>
                <Collapse ghost>
                  <Panel header="步骤 1: 文本线索识别" key="1">
                    <p>• 情感关键词：「开心」「感谢」「期待」</p>
                    <p>• 程度副词：「真的」「太」→ 强度增强</p>
                    <p>• 感叹号 → 情绪强烈</p>
                    <p>• 波浪号「～」→ 轻松愉快的语气</p>
                  </Panel>
                  <Panel header="步骤 2: 候选情绪列表" key="2">
                    <p>• Happy (高兴) - 主要候选</p>
                    <p>• Surprise (惊讶) - 次要候选</p>
                    <p>• Neutral (中性) - 可能性低</p>
                    <p>• 其他情绪可能性极低</p>
                  </Panel>
                  <Panel header="步骤 3: 强度评估依据" key="3">
                    <p>• 「太开心了」→ 高强度正向情绪</p>
                    <p>• 「真的」→ 强调真实性，增强可信度</p>
                    <p>• 「！」→ 情绪表达强烈</p>
                    <p>• 综合评估：Happy 强度 0.92</p>
                  </Panel>
                  <Panel header="步骤 4: 复合情绪关系" key="4">
                    <p>• 主要情绪：Happy (0.92)</p>
                    <p>• 次要情绪：Surprise (0.05) -「期待」暗示事先未知</p>
                    <p>• 感谢表达：强化社交联结的积极情绪</p>
                    <p>• 整体以单一正向情绪为主</p>
                  </Panel>
                  <Panel header="步骤 5: MBTI 调节分析" key="5">
                    <p>• E-I: E (外向) 0.72 - 表达感谢、提及「朋友们」</p>
                    <p>• S-N: S (感觉) 0.58 - 描述具体事件 (礼物)</p>
                    <p>• T-F: F (情感) 0.85 - 情感表达丰富、重视他人用心</p>
                    <p>• J-P: P (知觉) 0.54 - 开放性、随意性表达</p>
                    <p><strong>→ ESFJ</strong> (执政官型 - 热情、负责任、受欢迎)</p>
                  </Panel>
                  <Panel header="步骤 6: 反事实检验" key="6">
                    <p>• 如果不是 Happy，可能是 Surprise？</p>
                    <p>• 但「期待」一词表明是预期内的惊喜</p>
                    <p>• 「开心」直接表达正向情绪，排除负面可能</p>
                    <p>• 检验结果：Happy 判断稳健</p>
                  </Panel>
                  <Panel header="步骤 7: 不确定性说明和最终结论" key="7">
                    <p><strong>最终结论：</strong></p>
                    <p>• 主要情绪：Happy (置信度 0.92)</p>
                    <p>• MBTI 类型：ESFJ</p>
                    <p>• 不确定性：低 (高置信度)</p>
                    <p>• JSON 输出完整，CoT 七步完整</p>
                  </Panel>
                </Collapse>
              </div>
            </Card>
          </div>
        )

      case 'params':
        return (
          <div className="section-content">
            <Title level={2}>技术参数</Title>
            <Paragraph className="section-intro">
              详细的模型配置、训练超参数和硬件规格说明。
            </Paragraph>

            <Card className="params-card" title="模型与训练参数">
              <Table
                dataSource={technicalParams}
                columns={[
                  { title: '参数名称', dataIndex: 'param', key: 'param', width: '40%' },
                  { title: '参数值', dataIndex: 'value', key: 'value' },
                ]}
                rowKey="param"
                pagination={false}
                size="small"
                bordered
              />
            </Card>

            <Card className="hardware-card" title="硬件要求" style={{ marginTop: 24 }}>
              <Descriptions column={1} bordered>
                <Descriptions.Item label="最低配置">
                  <ul>
                    <li>GPU: NVIDIA RTX 3090 (24GB)</li>
                    <li>内存：32GB RAM</li>
                    <li>存储：50GB 可用空间</li>
                  </ul>
                </Descriptions.Item>
                <Descriptions.Item label="推荐配置">
                  <ul>
                    <li>GPU: NVIDIA A100 (40GB+) 或 RTX 4090 (24GB)</li>
                    <li>内存：64GB RAM</li>
                    <li>存储：100GB NVMe SSD</li>
                  </ul>
                </Descriptions.Item>
                <Descriptions.Item label="部署环境">
                  <ul>
                    <li>Docker 20.10+</li>
                    <li>CUDA 11.8+</li>
                    <li>Python 3.10+</li>
                  </ul>
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </div>
        )

      case 'api':
        return (
          <div className="section-content">
            <Title level={2}>API 接口文档</Title>
            <Paragraph className="section-intro">
              RESTful API 接口定义和使用说明。
            </Paragraph>

            <Card className="api-card" title="接口列表">
              <div className="api-list">
                {apiEndpoints.map((api, index) => (
                  <Card key={index} className="api-item" size="small">
                    <div className="api-header">
                      <Tag
                        color={
                          api.method === 'POST' ? '#ff6b4a' : '#88c4a8'
                        }
                        className="api-method"
                      >
                        {api.method}
                      </Tag>
                      <code className="api-endpoint">{api.endpoint}</code>
                    </div>
                    <p className="api-description">{api.description}</p>
                    <div className="api-details">
                      <div className="api-detail-row">
                        <strong>输入:</strong>
                        <pre>{api.input}</pre>
                      </div>
                      <div className="api-detail-row">
                        <strong>输出:</strong>
                        <pre>{api.output}</pre>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Card>

            <Card className="download-card" title="下载文档" style={{ marginTop: 24 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button block icon={<DownloadOutlined />} href="/api-docs.pdf">
                  下载完整 API 文档 (PDF)
                </Button>
                <Button block icon={<DownloadOutlined />} href="/openapi.json">
                  下载 OpenAPI Specification (JSON)
                </Button>
              </Space>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="docs-page">
      <div className="docs-layout">
        {/* Left Navigation */}
        <div className="docs-nav">
          <Card className="nav-card">
            <Menu
              mode="inline"
              selectedKeys={[activeSection]}
              items={menuItems}
              onClick={handleMenuClick}
            />
          </Card>
        </div>

        {/* Main Content */}
        <div className="docs-content">
          {renderContent()}
        </div>
      </div>
    </div>
  )
}

export default Docs
