import React, { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  HomeOutlined,
  ExperimentOutlined,
  TableOutlined,
  BarChartOutlined,
  AppstoreOutlined,
  FileTextOutlined,
  HistoryOutlined,
  LineChartOutlined,
} from '@ant-design/icons'

import Home from './pages/Home'
import Demo from './pages/Demo'
import Batch from './pages/Batch'
import Metrics from './pages/Metrics'
import Compare from './pages/Compare'
import Docs from './pages/Docs'
import WeiboAnalysis from './pages/WeiboAnalysis'
import History from './pages/History'
import ThemeToggle from './components/ThemeToggle'

import './assets/styles/design-system.scss'
import './App.scss'

const { Header, Content, Footer } = Layout

const menuItems = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  { key: '/weibo-analysis', icon: <LineChartOutlined />, label: '微博用户分析' },
  { key: '/demo', icon: <ExperimentOutlined />, label: '在线演示' },
  { key: '/batch', icon: <TableOutlined />, label: '批量推理' },
  {
    key: 'model-center',
    icon: <AppstoreOutlined />,
    label: '模型中心',
    children: [
      { key: '/metrics', icon: <BarChartOutlined />, label: '模型评估' },
      { key: '/compare', icon: <TableOutlined />, label: '模型对比' },
    ]
  },
  { key: '/history', icon: <HistoryOutlined />, label: '历史记录' },
  { key: '/docs', icon: <FileTextOutlined />, label: '技术文档' },
]

const AppContent: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const [current, setCurrent] = React.useState('/')

  useEffect(() => {
    setCurrent(location.pathname)
  }, [location.pathname])

  const handleClick = (e: any) => {
    setCurrent(e.key)
    navigate(e.key)
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header className="app-header">
        <div className="header-inner">
          <div className="logo-container" onClick={() => navigate('/')}>
            <span className="logo-emoji">🧠</span>
            <div className="logo-text-wrapper">
              <div className="logo-title">情感分析系统</div>
              <div className="logo-subtitle">Sentiment Analysis</div>
            </div>
          </div>

          <Menu
            theme="dark"
            mode="horizontal"
            selectedKeys={[current]}
            items={menuItems}
            onClick={handleClick}
            className="app-menu"
          />

          <div className="theme-toggle-wrapper">
            <ThemeToggle />
          </div>
        </div>
      </Header>

      <Content className="app-content">
        <div className="content-wrapper">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/demo" element={<Demo />} />
            <Route path="/weibo-analysis" element={<WeiboAnalysis />} />
            <Route path="/batch" element={<Batch />} />
            <Route path="/history" element={<History />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Content>

      <Footer className="app-footer">
        <div className="footer-content">
          <div className="footer-main">
            面向社交平台的指令微调大模型情感分析系统设计与实现
          </div>
          <div className="footer-sub">
            毕业设计 © 2026 | 基于 Qwen3-8B-Instruct + LoRA 多任务学习
          </div>
        </div>
      </Footer>
    </Layout>
  )
}

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}

export default App
