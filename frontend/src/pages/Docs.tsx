import React, { useState } from 'react'
import { Typography, Card, Menu, Button, Space, Tag, Timeline, Table, Descriptions, Collapse } from 'antd'
import {
  FileTextOutlined,
  BookOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  SafetyOutlined,
  SettingOutlined,
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
    { key: 'vad', label: 'VAD 情感模型', icon: <ExperimentOutlined /> },
    { key: 'cot', label: 'CoT 推理过程', icon: <ThunderboltOutlined /> },
    { key: 'presets', label: '推理预设', icon: <SettingOutlined /> },
    { key: 'rules', label: '兜底机制', icon: <SafetyOutlined /> },
    { key: 'params', label: '技术参数', icon: <FileTextOutlined /> },
    { key: 'api', label: 'API 接口', icon: <ApiOutlined /> },
  ]

  const handleMenuClick: MenuProps['onClick'] = (e) => {
    setActiveSection(e.key)
  }

  const cotSteps = [
    {
      step: 1,
      title: 'step1_lexical_grounding',
      subtitle: '词汇证据锚定',
      description: '提取情感线索并与原文证据一一对应',
      icon: '🔍',
      details: '从文本中提取强情绪线索、反讽线索、弱情绪线索、中性线索，与6种情绪的evidence一一匹配',
    },
    {
      step: 2,
      title: 'step2_dimensional_analysis',
      subtitle: 'VAD 维度分析',
      description: '基于证据完成细粒度情感量化',
      icon: '📊',
      details: 'Valence效价(愉悦度)、Arousal唤醒度(激活强度)、Dominance支配度(控制感)三维分析',
    },
    {
      step: 3,
      title: 'step3_negation_detection',
      subtitle: '否定与反讽检测',
      description: '识别否定词和反讽表达',
      icon: '🔄',
      details: '检测否定词(如不、没、非)和反讽表达，调整对应情绪分数',
    },
    {
      step: 4,
      title: 'step4_cause_extraction',
      subtitle: '情绪诱因提取',
      description: '锁定情绪触发原因',
      icon: '💡',
      details: '提取primary_cause和secondary_causes，原因必须100%来自原文',
    },
    {
      step: 5,
      title: 'step5_consistency_check',
      subtitle: '一致性校验',
      description: '逐条核对VAD与情绪分数的一致性',
      icon: '✅',
      details: '校验VAD与情绪分数的映射关系，确保逻辑自洽',
    },
    {
      step: 6,
      title: 'step6_uncertainty_calibration',
      subtitle: '不确定性校准',
      description: '校准推理置信度',
      icon: '📈',
      details: '评估推理可靠性(high/medium/low)，标记不确定区域',
    },
    {
      step: 7,
      title: 'step7_faithful_synthesis',
      subtitle: '可信性合成',
      description: '汇总推理结果，标记风险',
      icon: '🎯',
      details: '记录全部分数调整，标记幻觉风险环节，输出最终结果',
    },
  ]

  const technicalParams = [
    { param: '基座模型', value: 'Qwen3-8B-Instruct' },
    { param: '参数量', value: '8B' },
    { param: '注意力机制', value: 'Grouped Query Attention (GQA)' },
    { param: '词表大小', value: '151,936' },
    { param: '最大序列长度', value: '5120 tokens' },
    { param: 'LoRA Rank', value: 'r=16' },
    { param: 'LoRA Alpha', value: 'α=32' },
    { param: 'LoRA Dropout', value: '0.05' },
    { param: '微调方法', value: 'LoRA (QLoRA 4-bit quantization)' },
    { param: '批次大小', value: 'batch_size=32' },
    { param: '学习率', value: '2e-4 (cosine decay)' },
    { param: '训练轮次', value: '3 epochs' },
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
      input: '{ text: string, model_variant?: "base" | "lora_merged" | "gguf4bit", preset?: "quick" | "standard" | "deep" }',
      output: '{ scores, primary_emotion, confidence, cot, vad_dimensions, emotion_cause, uncertainty_level, ... }',
    },
    {
      method: 'POST',
      endpoint: '/api/infer/stream',
      description: '流式推理接口 (SSE)，实时返回 CoT 推理过程',
      input: '{ text: string, model_variant?: string, preset?: string }',
      output: 'SSE Stream: { delta: string, done: boolean, latency_ms?: number }',
    },
    {
      method: 'POST',
      endpoint: '/api/batch',
      description: '批量文本情感分析',
      input: '{ texts: string[], model_variant?: string, use_quick_preset?: boolean }',
      output: '{ total, success, failed, results: BatchItem[] }',
    },
    {
      method: 'GET',
      endpoint: '/api/metrics/:model_variant',
      description: '获取模型评估指标',
      input: '-',
      output: '{ emotion_macro_mae, primary_cls_accuracy, primary_cls_macro_f1, json_parse_rate, cot7_complete_rate, ... }',
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
              采用 Qwen3-8B-Instruct 作为基座模型，通过 LoRA 技术进行多任务指令微调，
              实现高效、准确的情感分析和 VAD 三维情感量化。
            </Paragraph>

            <Card className="feature-card" title="核心功能">
              <div className="feature-grid">
                <div className="feature-item">
                  <div className="feature-icon">🎭</div>
                  <h4>六类情绪识别</h4>
                  <p>Angry、Fear、Happy、Neutral、Sad、Surprise</p>
                </div>
                <div className="feature-item">
                  <div className="feature-icon">📊</div>
                  <h4>VAD 情感量化</h4>
                  <p>效价、唤醒度、支配度三维情感分析</p>
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
                    <Tag color="#a894c4">Qwen3-8B</Tag>
                    <Tag color="#a894c4">LoRA Adapters</Tag>
                    <Tag color="#a894c4">LMStudio / vLLM</Tag>
                    <Tag color="#a894c4">SwanLab</Tag>
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
                        <p>Qwen3-8B + LoRA 适配器进行前向传播计算</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>多任务输出解析</strong>
                        <p>解析情感分类、VAD 维度、CoT 推理步骤</p>
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

      case 'vad':
        return (
          <div className="section-content">
            <Title level={2}>VAD 情感模型</Title>
            <Paragraph className="section-intro">
              VAD (Valence-Arousal-Dominance) 是心理学中经典的三维情感模型，
              能够更细粒度地描述情感的多个维度。
            </Paragraph>

            <Card className="vad-card" title="三维情感定义">
              <div className="vad-grid">
                <div className="vad-item">
                  <div className="vad-header">
                    <span className="vad-icon">💢</span>
                    <h3>Valence (效价)</h3>
                  </div>
                  <p className="vad-desc">情绪愉悦程度</p>
                  <div className="vad-scale">
                    <div className="scale-bar">
                      <div className="scale-gradient" />
                    </div>
                    <div className="scale-labels">
                      <span>极度负面 (0.00)</span>
                      <span>中性 (0.50)</span>
                      <span>极度正面 (1.00)</span>
                    </div>
                  </div>
                </div>

                <div className="vad-item">
                  <div className="vad-header">
                    <span className="vad-icon">⚡</span>
                    <h3>Arousal (唤醒度)</h3>
                  </div>
                  <p className="vad-desc">情绪激活强度</p>
                  <div className="vad-scale">
                    <div className="scale-bar">
                      <div className="scale-gradient arousal" />
                    </div>
                    <div className="scale-labels">
                      <span>极度平静 (0.00)</span>
                      <span>中等激活 (0.50)</span>
                      <span>极度激动 (1.00)</span>
                    </div>
                  </div>
                </div>

                <div className="vad-item">
                  <div className="vad-header">
                    <span className="vad-icon">🎮</span>
                    <h3>Dominance (支配度)</h3>
                  </div>
                  <p className="vad-desc">对情境的控制感</p>
                  <div className="vad-scale">
                    <div className="scale-bar">
                      <div className="scale-gradient dominance" />
                    </div>
                    <div className="scale-labels">
                      <span>极度被动 (0.00)</span>
                      <span>中等控制 (0.50)</span>
                      <span>极度主动 (1.00)</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="vad-mapping-card" title="VAD 与情绪映射">
              <Table
                dataSource={[
                  { emotion: 'Happy (喜悦)', valence: '≥0.70', arousal: '≥0.30', dominance: '≥0.40', example: '考试通过、收到礼物' },
                  { emotion: 'Sad (悲伤)', valence: '≤0.39', arousal: '≤0.69', dominance: '≤0.50', example: '失业、失恋、分离' },
                  { emotion: 'Angry (愤怒)', valence: '≤0.39', arousal: '≥0.70', dominance: '≥0.50', example: '售后推诿、被歧视' },
                  { emotion: 'Fear (恐惧)', valence: '≤0.39', arousal: '≥0.70', dominance: '≤0.49', example: '未知风险、被胁迫' },
                  { emotion: 'Surprise (惊讶)', valence: '0.40-0.69', arousal: '≥0.70', dominance: '0.30-0.70', example: '突发新闻、意外相遇' },
                  { emotion: 'Neutral (中性)', valence: '0.40-0.69', arousal: '≤0.39', dominance: '0.40-0.69', example: '客观陈述事实' },
                ]}
                columns={[
                  { title: '情绪', dataIndex: 'emotion', key: 'emotion' },
                  { title: '效价区间', dataIndex: 'valence', key: 'valence' },
                  { title: '唤醒度区间', dataIndex: 'arousal', key: 'arousal' },
                  { title: '支配度区间', dataIndex: 'dominance', key: 'dominance' },
                  { title: '典型场景', dataIndex: 'example', key: 'example' },
                ]}
                rowKey="emotion"
                pagination={false}
                size="small"
                bordered
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
              系统采用基于证据的 7 步 CoT 推理方法，每一步都锚定原文证据。
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
                        <div>
                          <h4 className="cot-step-title">{step.subtitle}</h4>
                          <code className="cot-step-code">{step.title}</code>
                        </div>
                      </div>
                      <p className="cot-step-desc">{step.description}</p>
                      <div className="cot-step-details">
                        <code>{step.details}</code>
                      </div>
                    </div>
                  ),
                }))}
              />
            </Card>

            <Card className="example-card" title="CoT 推理示例">
              <div className="example-input">
                <strong>输入文本：</strong>
                <p>「吐槽，用了一年电池就不耐用了，充电还发烫，售后服务态度也超差，踢皮球踢得比球队还专业，再也不买他们家东西了」</p>
              </div>
              <div className="example-output">
                <strong>推理输出：</strong>
                <Collapse ghost>
                  <Panel header="步骤 1: 词汇证据锚定" key="1">
                    <p><strong>cues 线索提取：</strong></p>
                    <p>• 强情绪线索：「不耐用」「发烫」「超差」「后悔」</p>
                    <p>• 反讽线索：「踢皮球」</p>
                    <p>• 中性线索：「用了一年」「某国产手机」</p>
                    <br />
                    <p><strong>evidence 证据匹配：</strong></p>
                    <p>• angry: 「用了一年电池就不耐用了」「充电还发烫」「售后服务态度也超差」「踢皮球踢得比球队还专业」「再也不买他们家东西了」</p>
                    <p>• 其他情绪: 均为空数组</p>
                  </Panel>
                  <Panel header="步骤 2: VAD 维度分析" key="2">
                    <p>• valence: 0.12 (低效价 - 负面)</p>
                    <p>• arousal: 0.88 (高唤醒度 - 激动)</p>
                    <p>• dominance: 0.72 (高支配度 - 主动表达)</p>
                    <p>• 情绪映射：低效价+高唤醒度+高支配度 → 指向 Angry (愤怒)</p>
                  </Panel>
                  <Panel header="步骤 3: 否定与反讽检测" key="3">
                    <p>• 检测到否定词：「不」</p>
                    <p>• 检测到反讽：「踢皮球踢得比球队还专业」</p>
                    <p>• 分数调整：happy → 0.00，angry 强化</p>
                  </Panel>
                  <Panel header="步骤 4: 情绪诱因提取" key="4">
                    <p>• 主要原因：电池不耐用、充电发烫、售后服务差</p>
                    <p>• 次要原因：性价比极低，表示不会再购买</p>
                  </Panel>
                  <Panel header="步骤 5-7: 校验与合成" key="5">
                    <p>• VAD 与情绪分数一致性校验：通过</p>
                    <p>• 置信度评估：高置信度</p>
                    <p>• 最终输出：Angry (置信度 0.92)</p>
                  </Panel>
                </Collapse>
              </div>
            </Card>
          </div>
        )

      case 'presets':
        return (
          <div className="section-content">
            <Title level={2}>推理预设</Title>
            <Paragraph className="section-intro">
              系统提供三种推理预设，适用于不同的分析场景。用户可根据文本长度和复杂度选择合适的预设。
            </Paragraph>

            <Card className="presets-card" title="推理预设配置">
              <Table
                dataSource={[
                  {
                    preset: 'quick',
                    name: '快速模式',
                    max_tokens: '1536',
                    temperature: '0.03',
                    top_p: '0.75',
                    scenario: '短句/单情绪简单文本 (≤50字)',
                  },
                  {
                    preset: 'standard',
                    name: '标准模式',
                    max_tokens: '2560',
                    temperature: '0.05',
                    top_p: '0.80',
                    scenario: '常规长度文本 (50-300字)',
                  },
                  {
                    preset: 'deep',
                    name: '深度模式',
                    max_tokens: '4096',
                    temperature: '0.07',
                    top_p: '0.85',
                    scenario: '长文本/多段落 (≥300字)',
                  },
                ]}
                columns={[
                  { title: '预设', dataIndex: 'preset', key: 'preset' },
                  { title: '名称', dataIndex: 'name', key: 'name' },
                  { title: 'Max Tokens', dataIndex: 'max_tokens', key: 'max_tokens' },
                  { title: 'Temperature', dataIndex: 'temperature', key: 'temperature' },
                  { title: 'Top P', dataIndex: 'top_p', key: 'top_p' },
                  { title: '适用场景', dataIndex: 'scenario', key: 'scenario' },
                ]}
                rowKey="preset"
                pagination={false}
                bordered
              />
            </Card>

            <Card className="preset-tips-card" title="使用建议" style={{ marginTop: 24 }}>
              <ul className="preset-tips">
                <li>
                  <strong>quick（快速模式）</strong>：适用于社交媒体的短评论、简单句子，
                  追求快速响应，适合批量处理场景。
                </li>
                <li>
                  <strong>standard（标准模式）【推荐】</strong>：适用于大部分常规文本分析，
                  平衡推理质量和速度，是默认推荐选项。
                </li>
                <li>
                  <strong>deep（深度模式）</strong>：适用于长文本、复杂情感、混合情绪场景，
                  如产品评论、论坛帖子等需要深度分析的内容。
                </li>
              </ul>
            </Card>
          </div>
        )

      case 'rules':
        return (
          <div className="section-content">
            <Title level={2}>兜底机制</Title>
            <Paragraph className="section-intro">
              系统内置多层兜底机制，确保模型输出的情绪分数与原文证据严格对应，
              从机制上消除幻觉和无依据打分。
            </Paragraph>

            <Card className="rules-card" title="分数兜底规则">
              <div className="rule-item">
                <div className="rule-header">
                  <Tag color="red">规则 1</Tag>
                  <h4>证据缺失下的强制收敛机制</h4>
                </div>
                <p className="rule-desc">
                  鉴于大模型在无有效信息输入时存在“幻觉填充”倾向，为防止凭空捏造情感判断，系统设立<strong>最低信任基线</strong>。当上下文窗口内未能提取到任何可采信的情感证据时，无论模型初始倾向如何，分数将被强制压缩至<strong>[0.00, 0.03]</strong>的极小值区间内，从而彻底切断无据打分的干扰路径。
                </p>
                <div className="rule-example">
                  <strong>示例：</strong>若文本中未出现悲伤相关词汇，则 sad 分数不会超过 0.03。
                </div>
              </div>

              <div className="rule-item">
                <div className="rule-header">
                  <Tag color="red">规则 2</Tag>
                  <h4>缺省归零与防误判保护</h4>
                </div>
                <p className="rule-desc">
                  鉴于 Happy 维度的易感性与高误判风险，设置系统最高优先级的硬性缺省值。凡未捕获明确愉悦信号，一律强制赋分 <strong>0.00</strong>，杜绝虚假愉悦输出。
                </p>
                <div className="rule-example">
                  <strong>示例：</strong>若文本仅为客观陈述（如「今天气温25度」），则 happy 必须为 0.00。
                </div>
              </div>

              <div className="rule-item">
                <div className="rule-header">
                  <Tag color="orange">规则 3</Tag>
                  <h4>VAD 三维空间的逻辑自洽约束</h4>
                </div>
                <p className="rule-desc">
                  鉴于 Valence（效价）、Arousal（唤醒）、Dominance（支配）在情感心理学模型中存在固有的内在映射关系，系统建立<strong>三维联动校验门控</strong>。凡模型输出的 VAD 分布违背该理论空间的基本拓扑结构（如高唤醒伴随低效价时的逻辑矛盾），系统将自动触发纠偏流程，依据最近邻理论分布对分数进行重映射，确保输出结果始终落于合理的心理学象限内。
                </p>
                <div className="rule-example">
                  <strong>示例：</strong>若 valence=0.12 (低) 而 happy=0.90 (高)，则校验不通过。
                </div>
              </div>

              <div className="rule-item">
                <div className="rule-header">
                  <Tag color="orange">规则 4</Tag>
                  <h4>语义反转信号的动态感知修正</h4>
                </div>
                <p className="rule-desc">
                  鉴于自然语言中否定前缀（“不”、“无”）与反讽修辞（预期违背）会从根本上反转情感极性，系统部署了<strong>高阶语义探测器</strong>。一旦识别到上述逆向逻辑算子，系统将不再信任初始情感分类结果，而是对原始分数向量执行<strong>反向镜像操作</strong>，以消除字面浅层理解与深层语用意图之间的鸿沟。
                </p>
                <div className="rule-example">
                  <strong>示例：</strong>「服务真不错」(反讽) → happy → 0.00, angry 强化
                </div>
              </div>
            </Card>

            <Card className="rules-flow-card" title="兜底机制执行流程" style={{ marginTop: 24 }}>
              <Timeline
                items={[
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 1: 线索提取</strong>
                        <p>从原文提取情感线索并分类到6种情绪</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 2: 初步打分</strong>
                        <p>基于线索给各情绪分配原始强度分</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 3: 否定/反讽调整</strong>
                        <p>检测否定词和反讽，自动调整分数</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 4: 兜底校验</strong>
                        <p>检查无证据情绪分数是否超标，违规则强制修正</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 5: VAD 一致性校验</strong>
                        <p>核对 VAD 维度与情绪分数的逻辑一致性</p>
                      </div>
                    ),
                  },
                  {
                    children: (
                      <div className="timeline-item">
                        <strong>Step 6: 输出最终结果</strong>
                        <p>通过所有校验后输出结构化 JSON</p>
                      </div>
                    ),
                  },
                ]}
              />
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
                    <li>CUDA 12.0+</li>
                    <li>Python 3.11+</li>
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

        <div className="docs-content">
          {renderContent()}
        </div>
      </div>
    </div>
  )
}

export default Docs
