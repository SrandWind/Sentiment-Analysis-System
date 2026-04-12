import React from 'react'
import { Button, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  ThunderboltOutlined,
  CheckCircleOutlined,
  SafetyCertificateOutlined,
  PlayCircleOutlined,
  BarChartOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons'
import NeuralBackground from '@/components/NeuralBackground'
import '../assets/styles/design-system.scss'
import './Home.scss'

const { Title, Paragraph } = Typography

const Home: React.FC = () => {
  const navigate = useNavigate()

  const features = [
    {
      icon: <ThunderboltOutlined />,
      title: '多任务联合预测',
      description: '同时预测情绪强度、主要情绪和 7 步思维链推理',
      color: '#ff6b4a',
    },
    {
      icon: <CheckCircleOutlined />,
      title: 'LoRA 微调优化',
      description: '基于 Qwen3-8B-Instruct 的 LoRA 微调，性能提升显著',
      color: '#10b981',
    },
    {
      icon: <SafetyCertificateOutlined />,
      title: '4bit 量化部署',
      description: 'GGUF Q4_K_M 量化，大幅降低推理资源需求',
      color: '#f59e0b',
    },
  ]

  const metrics = [
    { label: '情绪分类准确率', value: '82', suffix: '%', sub: 'LoRA 微调后', trend: '+24%' },
    { label: '情绪回归 MAE', value: '0.072', suffix: '', sub: '越低越好', trend: '+56%' },
    { label: 'Macro F1 SCORE', value: '0.74', suffix: '', sub: '分类均衡性', trend: '+53%' },
    { label: 'AUC-ROC', value: '0.94', suffix: '', sub: '模型区分度', trend: '+25%' },
  ]

  return (
    <div className="home-page">
      {/* 神经网络背景 */}
      <NeuralBackground density={0.4} speed={0.6} opacity={0.5} />

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-dot" />
            毕业设计作品
          </div>

          <Title level={1} className="hero-title">
            情感分析系统
          </Title>

          <Paragraph className="hero-description">
            基于多任务 LoRA 微调大模型的社交平台情感分析系统
            <br />
            支持情绪强度预测、主要情绪分类和 7 步思维链推理生成
          </Paragraph>

          <div className="hero-actions">
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              onClick={() => navigate('/demo')}
              className="btn-primary"
            >
              开始体验
              <ArrowRightOutlined style={{ marginLeft: 8 }} />
            </Button>
            <Button
              size="large"
              icon={<BarChartOutlined />}
              onClick={() => navigate('/compare')}
              className="btn-secondary"
            >
              模型对比
            </Button>
          </div>
        </div>

        {/* 滚动提示 */}
        <div className="scroll-indicator">
          <div className="scroll-mouse">
            <div className="scroll-wheel" />
          </div>
          <span>向下滚动</span>
        </div>
      </section>

      {/* Metrics Section */}
      <section className="metrics-section">
        <div className="section-container">
          <div className="section-header">
            <Title level={2} className="section-title">核心性能指标</Title>
            <Paragraph className="section-subtitle">
              在测试集上的评估结果，展示 LoRA 微调带来的显著性能提升
            </Paragraph>
          </div>

          <div className="metrics-grid">
            {metrics.map((metric, index) => (
              <div
                key={index}
                className="metric-card"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="metric-header">
                  <span className="metric-label">{metric.label}</span>
                  {metric.trend && (
                    <span className={`metric-trend ${metric.trend.startsWith('+') ? 'positive' : 'negative'}`}>
                      {metric.trend}
                    </span>
                  )}
                </div>
                <div className="metric-value">
                  {metric.value}<span className="metric-suffix">{metric.suffix}</span>
                </div>
                <div className="metric-sub">{metric.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-container">
          <div className="section-header">
            <Title level={2} className="section-title">系统特点</Title>
            <Paragraph className="section-subtitle">
              采用前沿的深度学习技术，实现多任务联合情感分析
            </Paragraph>
          </div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <div
                key={index}
                className="feature-card"
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div
                  className="feature-icon"
                  style={{ color: feature.color, background: `${feature.color}15` }}
                >
                  {feature.icon}
                </div>
                <Title level={4} className="feature-title">{feature.title}</Title>
                <Paragraph className="feature-description">
                  {feature.description}
                </Paragraph>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <Title level={2} className="cta-title">准备好开始体验了吗？</Title>
          <Paragraph className="cta-description">
            立即试用情感分析系统，感受多任务 LoRA 微调大模型的强大能力
          </Paragraph>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate('/demo')}
            className="btn-primary btn-large"
          >
            立即体验
          </Button>
        </div>
      </section>
    </div>
  )
}

export default Home
