import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Select, Spin, message } from 'antd'
import ReactECharts from 'echarts-for-react'
import ReactWordcloud from 'react-wordcloud'
import NeuralBackground from '@/components/NeuralBackground'
import './WeiboAnalysis.scss'
import weekData from "@/data/weibo/weekly_2024-01-01.json";  // 新文件名

const { Title, Paragraph } = Typography

interface User {
  id: string
  name: string
  avatar: string
  period: string
}

interface Summary {
  avg_valence: number
  avg_confidence: number
  volatility_index: string
  volatility_std: number
  lowest_day: string
  lowest_valence: number
  highest_day: string
  highest_valence: number
  radar_summary: string
}

interface Radar {
  user: number[]
  platform_avg: number[]
  labels: string[]
}

interface VadDaily {
  day: string
  valence: number
  v_q1: number
  v_q3: number
  v_std: number
  arousal: number
  a_q1: number
  a_q3: number
  a_std: number
  dominance: number
  d_q1: number
  d_q3: number
  d_std: number
  is_anchor: boolean
}

interface Heatmap {
  rows: string[]
  cols: string[]
  values: number[][]
}

interface EmotionWheelItem {
  name: string
  color: string
  count: number
  avg_intensity: number
}

interface WordCloudItem {
  word: string
  size: number
  bg: string
  color: string
  sentiment?: string
}

interface WordCloudWord {
  text: string
  value: number
  sentiment: 'pos' | 'neg' | 'neu'
  postIds?: string[]
}

interface ScatterPost {
  x: number
  y: number
  d: number
  label: string
  day_idx: number
  is_anchor: boolean
  is_outlier: boolean
  anchor_type?: string
  text: string
}

interface ValenceBoxItem {
  day: string
  min: number
  q1: number
  median: number
  q3: number
  max: number
  outliers: number[]
}

interface SocialStats {
  positive: { avg_likes: number; avg_comments: number }
  neutral: { avg_likes: number; avg_comments: number }
  negative: { avg_likes: number; avg_comments: number }
}

interface WeiboPost {
  id: string
  content: string
  day: string
  time: string
  v: number
  a: number
  d: number
  emotion: string
  confidence: number
  cause: string
  rule?: string | null
  step: number
  likes: number
  comments: number
}

interface WeiboWeeklyData {
  user: User
  summary: Summary
  radar: Radar
  vad_daily: VadDaily[]
  heatmap: Heatmap
  emotion_wheel: EmotionWheelItem[]
  wordcloud: WordCloudItem[]
  wordcloud_by_day: Record<string, WordCloudItem[]>
  scatter_posts: ScatterPost[]
  valence_box: ValenceBoxItem[]
  social_stats: SocialStats
  posts: WeiboPost[]
}

const EMOTION_COLORS: Record<string, string> = {
  happy: '#639922',
  neutral: '#B4B2A9',
  sad: '#E24B4A',
  angry: '#BA7517',
  fear: '#7F77DD',
  surprise: '#378ADD',
}

const DAY_ORDER = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

const DEFAULT_DATA: WeiboWeeklyData = weekData as WeiboWeeklyData

// 发散色计算：低(蓝) -> 中(白) -> 高(红)，柔和化处理
const divergingColor = (v: number): string => {
  if (v >= 0.5) {
    const intensity = (v - 0.5) * 2
    const r = Math.round(255)
    const g = Math.round(245 * (1 - intensity * 0.6))
    const b = Math.round(235 * (1 - intensity * 0.8))
    return `rgb(${r}, ${g}, ${b})`
  } else {
    const intensity = (0.5 - v) * 2
    const r = Math.round(235 * (1 + intensity * 0.5))
    const g = Math.round(245 * (1 + intensity * 0.4))
    const b = 255
    return `rgb(${r}, ${g}, ${b})`
  }
}

// 计算情绪堆叠数据
const calcEmotionStack = (posts: WeiboPost[]): Record<string, { positive: number; neutral: number; negative: number }> => {
  const stack: Record<string, { positive: number; neutral: number; negative: number }> = {}
  DAY_ORDER.forEach(day => {
    const dayPosts = posts.filter(p => p.day === day)
    const total = dayPosts.length || 1
    const pos = dayPosts.filter(p => p.emotion === 'happy').length
    const neg = dayPosts.filter(p => ['sad', 'angry', 'fear'].includes(p.emotion)).length
    const neu = total - pos - neg
    stack[day] = {
      positive: Math.round((pos / total) * 100),
      neutral: Math.round((neu / total) * 100),
      negative: Math.round((neg / total) * 100),
    }
  })
return stack
}

// 词云配色映射
const WORDCLOUD_COLORS = {
  pos: '#3B6D11',
  neg: '#A32D2D',
  neu: '#5F5E5A',
}

const WeiboAnalysis: React.FC = () => {
  const [selectedWeek, setSelectedWeek] = useState('2024-01-01')
  const [data, setData] = useState<WeiboWeeklyData | null>(null)
  const [loading, setLoading] = useState(false)
  const [anchorTooltip, setAnchorTooltip] = useState<{ visible: boolean; x: number; y: number; content: string }>({ visible: false, x: 0, y: 0, content: '' })
  const [selectedWord, setSelectedWord] = useState<string | null>(null)

  const loadData = async (weekId: string) => {
    setLoading(true)
    
    try {
      const dataModule = await import(`@/data/weibo/weekly_${weekId}.json`)
      setData(dataModule.default as WeiboWeeklyData)
    } catch (err) {
      console.error('Failed to load data:', err)
      setData(DEFAULT_DATA)
      message.warning('加载数据失败，使用默认数据')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData(selectedWeek)
  }, [selectedWeek])

  useEffect(() => {
    const handleClickOutside = () => setAnchorTooltip(prev => ({ ...prev, visible: false }))
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [])

  const currentData = data || DEFAULT_DATA
  const emotionStack = calcEmotionStack(currentData.posts)
  
  // 词云状态：null=本周总览，0-6=周一到周日
  const [selectedDay, setSelectedDay] = useState<number | null>(null)
  const [hoveredWord, setHoveredWord] = useState<string | null>(null)
  // Tab 标签状态：0=总览, 1=趋势, 2=细节, 3=洞察
  const [activeTab, setActiveTab] = useState(0)
  // 博文列表折叠状态
  const [postsExpanded, setPostsExpanded] = useState(false)
  
  // 根据选中词获取对应的 postIds（不依赖 selectedDay，保持稳定）
  const getWordPostIds = (word: string): string[] => {
    // 从本周总览词云中查找
    const weekWord = currentData.wordcloud.find(w => w.word === word)
    if ((weekWord as any)?.postIds?.length > 0) {
      return (weekWord as any).postIds
    }
    // 从每日词云中查找
    for (let day = 0; day <= 6; day++) {
      if (currentData.wordcloud_by_day && currentData.wordcloud_by_day[day.toString()]) {
        const dayWord = currentData.wordcloud_by_day[day.toString()].find(w => w.word === word)
        if ((dayWord as any)?.postIds?.length > 0) {
          return (dayWord as any).postIds
        }
      }
    }
    return []
  }
  
  const matchedPostIds = selectedWord ? getWordPostIds(selectedWord) : []

  const radarOption = {
    tooltip: {},
    legend: {
      data: ['本周', '平台均值'],
      bottom: 0,
      textStyle: { fontSize: 11 }
    },
    radar: {
      indicator: currentData.radar.labels.map(l => ({ name: l, max: 100 })),
      radius: '65%',
      axisName: { fontSize: 10 },
      splitArea: { areaStyle: { color: ['rgba(0,0,0,0.02)', 'rgba(0,0,0,0.04)'] } },
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: currentData.radar.user,
            name: '本周',
            itemStyle: { color: '#7F77DD' },
            areaStyle: { color: 'rgba(127,119,221,0.15)' },
            lineStyle: { width: 2 },
          },
          {
            value: currentData.radar.platform_avg,
            name: '平台均值',
            itemStyle: { color: '#1D9E75' },
            areaStyle: { color: 'rgba(29,158,117,0.06)' },
            lineStyle: { type: 'dashed', width: 1.5, color: '#1D9E75' },
          },
        ],
      },
    ],
  }

  const trendOption = {
    tooltip: { trigger: 'axis' },
    legend: { show: false },
    grid: { top: 25, right: 20, bottom: 25, left: 35 },
    xAxis: {
      type: 'category',
      data: DAY_ORDER,
      axisLabel: { fontSize: 10 },
      axisTick: { alignWithLabel: true },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => v.toFixed(1), fontSize: 10 },
    },
    series: [
      // vQ3: 上界，fill: '+1' 与 vQ1 形成置信区间阴影
      {
        name: 'vQ3',
        type: 'line',
        data: currentData.vad_daily.map(d => d.v_q3),
        smooth: true,
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(29,158,117,0.12)' },
        stack: 'confidence',
        tooltip: { show: false },
        symbol: 'none',
        z: 1,
      },
      // vQ1: 下界，fill: false 关闭填充，形成区间
      {
        name: 'vQ1',
        type: 'line',
        data: currentData.vad_daily.map(d => d.v_q1),
        smooth: true,
        lineStyle: { opacity: 0 },
        stack: 'confidence',
        symbol: 'none',
        tooltip: { show: false },
        z: 1,
      },
      // Valence 主线，锚点特殊样式
      {
        name: 'Valence',
        type: 'line',
        data: currentData.vad_daily.map(d => ({
          value: d.valence,
          isAnchor: d.is_anchor,
        })),
        smooth: true,
        itemStyle: { color: '#1D9E75' },
        lineStyle: { width: 2 },
        z: 2,
        // 锚点: 6px 白底红边框，非锚点: 3.5px
        symbolSize: (val: any) => val.isAnchor ? 6 : 3.5,
        symbol: 'circle',
        symbolKeepAspect: true,
        onClick: (params: any) => {
          const idx = params.dataIndex
          const day = currentData.vad_daily[idx]
          if (day.is_anchor) {
            // 查找该天的锚点博文
            const anchorPost = currentData.scatter_posts.find(p => p.day_idx === idx && p.is_anchor)
            if (anchorPost) {
              const badge = anchorPost.anchor_type === 'highest' 
                ? '<span class="badge b-pos">最高</span>' 
                : '<span class="badge b-neg">最低</span>'
              const rule = anchorPost.label === 'sad' ? '<span class="badge b-warn" style="font-size:9px;padding:1px 5px;">规则03</span>' : ''
              const text = anchorPost.text.length > 40 ? anchorPost.text.slice(0, 40) + '…' : anchorPost.text
              const content = `<div style="font-size:10px;color:var(--text3);margin-bottom:4px;">${day.day} · V=${day.valence.toFixed(2)} · 置信${(day.valence > 0.5 ? 0.85 : 0.80).toFixed(2)} ${badge} ${rule}</div><div style="font-size:11px;line-height:1.5;">${text}</div>`
              setAnchorTooltip({ 
                visible: true, 
                x: params.event.offsetX + 10, 
                y: params.event.offsetY - 8, 
                content 
              })
            }
          }
        },
      },
      {
        name: 'Arousal',
        type: 'line',
        data: currentData.vad_daily.map(d => d.arousal),
        smooth: true,
        lineStyle: { type: 'dashed', width: 1.5 },
        itemStyle: { color: '#BA7517' },
        symbol: 'emptyTriangle',
        symbolSize: 4,
      },
      {
        name: 'Dominance',
        type: 'line',
        data: currentData.vad_daily.map(d => d.dominance),
        smooth: true,
        lineStyle: { type: 'dotted', width: 1.5 },
        itemStyle: { color: '#378ADD' },
        symbol: 'rect',
        symbolSize: 3,
      },
    ],
  }

  const stackBarOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { show: false },
    grid: { top: 10, right: 10, bottom: 25, left: 35 },
    xAxis: {
      type: 'category',
      data: DAY_ORDER,
      axisLabel: { fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLabel: { formatter: '{value}%', fontSize: 10 },
    },
    series: [
      {
        name: '积极',
        type: 'bar',
        stack: 'total',
        data: DAY_ORDER.map(d => emotionStack[d]?.positive || 0),
        itemStyle: { color: '#639922', borderRadius: 0 },
      },
      {
        name: '中性',
        type: 'bar',
        stack: 'total',
        data: DAY_ORDER.map(d => emotionStack[d]?.neutral || 0),
        itemStyle: { color: '#B4B2A9', borderRadius: 0 },
      },
      {
        name: '消极',
        type: 'bar',
        stack: 'total',
        data: DAY_ORDER.map(d => emotionStack[d]?.negative || 0),
        itemStyle: { color: '#E24B4A', borderRadius: 0 },
      },
    ],
  }

  const scatterOption = {
    tooltip: { formatter: ({ data }: any) => `V: ${data.value[0].toFixed(2)} A: ${data.value[1].toFixed(2)}` },
    grid: { top: 15, right: 15, bottom: 30, left: 35 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1.05,
      name: 'Valence',
      nameLocation: 'middle',
      nameGap: 20,
      nameTextStyle: { fontSize: 10 },
      axisLabel: { fontSize: 9 },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1.05,
      name: 'Arousal',
      nameLocation: 'middle',
      nameGap: 25,
      nameTextStyle: { fontSize: 10 },
      axisLabel: { fontSize: 9 },
    },
    series: [
      // 时间轨迹线：按 day_idx 排序后连线，虚线 + 箭头
      {
        name: '轨迹',
        type: 'line',
        data: [...currentData.scatter_posts]
          .sort((a, b) => a.day_idx - b.day_idx)
          .map(p => [p.x, p.y]),
        lineStyle: { type: 'dashed', width: 1.8, color: 'rgba(100,100,100,0.35)' },
        symbol: 'none',
        tooltip: { show: false },
        z: 1,
      },
      {
        name: '散点',
        type: 'scatter',
        symbolSize: (val: any) => Math.max(4, val[2] * 12),
        data: currentData.scatter_posts.map(p => ({
          value: [p.x, p.y, p.d],
          itemStyle: { 
            color: EMOTION_COLORS[p.label] || '#888',
            opacity: 0.72 
          },
        })),
        tooltip: {
          formatter: (params: any) => {
            const post = currentData.scatter_posts[params.dataIndex]
            return `${post.text.slice(0, 30)}…<br/>V:${post.x.toFixed(2)} A:${post.y.toFixed(2)} D:${post.d.toFixed(2)}`
          }
        },
        z: 2,
      },
    ],
  }

  const socialBarOption = {
    tooltip: { trigger: 'axis' },
    legend: { show: false },
    grid: { top: 10, right: 10, bottom: 25, left: 45 },
    xAxis: {
      type: 'category',
      data: ['积极博文', '中性博文', '消极博文'],
      axisLabel: { fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 10 },
    },
    series: [
      {
        name: '平均点赞',
        type: 'bar',
        data: [
          currentData.social_stats.positive.avg_likes,
          currentData.social_stats.neutral.avg_likes,
          currentData.social_stats.negative.avg_likes
        ],
        itemStyle: { color: '#378ADD', borderRadius: 3 },
      },
      {
        name: '平均评论',
        type: 'bar',
        data: [
          currentData.social_stats.positive.avg_comments,
          currentData.social_stats.neutral.avg_comments,
          currentData.social_stats.negative.avg_comments
        ],
        itemStyle: { color: 'rgba(127,119,221,0.75)', borderRadius: 3 },
      },
    ],
  }

  const boxBarOption = {
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(255,255,255,0.95)',
      borderColor: 'rgba(0,0,0,0.08)',
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { fontSize: 11, color: '#5f5e5a' },
      formatter: (params: any) => {
        const box = currentData.valence_box[params.dataIndex]
        if (!box) return ''
        return `
          <div style="font-weight:500;margin-bottom:6px;color:#1a1a18">${DAY_ORDER[params.dataIndex]}</div>
          <div style="line-height:1.6">
            <div>最小值: <b style="color:#1D9E75">${box.min.toFixed(2)}</b></div>
            <div>下四分位数(Q1): <b style="color:#1D9E75">${box.q1.toFixed(2)}</b></div>
            <div>中位数: <b style="color:#1D9E75">${box.median.toFixed(2)}</b></div>
            <div>上四分位数(Q3): <b style="color:#1D9E75">${box.q3.toFixed(2)}</b></div>
            <div>最大值: <b style="color:#1D9E75">${box.max.toFixed(2)}</b></div>
            ${box.outliers.length > 0 ? `<div style="margin-top:4px">异常值: <b style="color:#E24B4A">${box.outliers.length}个 (${box.outliers.map((o: number) => o.toFixed(2)).join(', ')})</b></div>` : ''}
          </div>
        `
      },
    },
    legend: { show: false },
    grid: { top: 10, right: 10, bottom: 25, left: 35 },
    xAxis: {
      type: 'category',
      data: DAY_ORDER,
      axisLabel: { fontSize: 10 },
      axisTick: { alignWithLabel: true },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      interval: 0.2,
      axisLabel: { formatter: (v: number) => v.toFixed(1), fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.05)' } },
    },
    series: [
      {
        name: '箱线图',
        type: 'custom',
        renderItem: (params: any, api: any) => {
          const box = currentData.valence_box[params.dataIndex]
          if (!box) return null
          
          const x = api.coord([params.dataIndex, 0])[0]
          const boxWidth = 36  // 统一箱体宽度
          const capWidth = 8   // 须线帽线宽度
          const tealFill = 'rgba(29, 158, 117, 0.12)'
          const tealStroke = '#1D9E75'
          const outlierRed = '#E24B4A'
          
          const yMin = api.coord([0, box.min])[1]
          const yQ1 = api.coord([0, box.q1])[1]
          const yMedian = api.coord([0, box.median])[1]
          const yQ3 = api.coord([0, box.q3])[1]
          const yMax = api.coord([0, box.max])[1]
          
          const group: any = { type: 'group', children: [] }
          
          // 1. 下须线: min -> q1 (虚线)
          group.children.push({
            type: 'line',
            shape: { x1: x, y1: yMin, x2: x, y2: yQ1 },
            style: { stroke: tealStroke, lineWidth: 1, lineDash: [3, 2] },
          })
          
          // 1.1 下须线帽线
          group.children.push({
            type: 'line',
            shape: { x1: x - capWidth, y1: yMin, x2: x + capWidth, y2: yMin },
            style: { stroke: tealStroke, lineWidth: 1 },
          })
          
          // 2. 上须线: q3 -> max (虚线)
          group.children.push({
            type: 'line',
            shape: { x1: x, y1: yQ3, x2: x, y2: yMax },
            style: { stroke: tealStroke, lineWidth: 1, lineDash: [3, 2] },
          })
          
          // 2.1 上须线帽线
          group.children.push({
            type: 'line',
            shape: { x1: x - capWidth, y1: yMax, x2: x + capWidth, y2: yMax },
            style: { stroke: tealStroke, lineWidth: 1 },
          })
          
          // 3. 箱体: q1 -> q3 (浅青色填充 + teal 边框)
          group.children.push({
            type: 'rect',
            shape: { x: x - boxWidth / 2, y: yQ3, width: boxWidth, height: yQ1 - yQ3 },
            style: { fill: tealFill, stroke: tealStroke, lineWidth: 1 },
          })
          
          // 4. 中位线: 实线 2px
          group.children.push({
            type: 'line',
            shape: { x1: x - boxWidth / 2, y1: yMedian, x2: x + boxWidth / 2, y2: yMedian },
            style: { stroke: tealStroke, lineWidth: 2 },
          })
          
          // 5. 异常值: 红色实心圆 4px
          box.outliers.forEach((ov: number) => {
            const yOutlier = api.coord([0, ov])[1]
            group.children.push({
              type: 'circle',
              shape: { cx: x, cy: yOutlier, r: 4 },
              style: { fill: outlierRed },
            })
          })
          
          return group
        },
        data: currentData.valence_box.map((_, i) => i),
        z: 10,
      },
    ],
  }

  if (loading) {
    return (
      <div className="weibo-analysis-page">
        <NeuralBackground density={0.3} speed={0.5} opacity={0.4} />
        <div className="page-content" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <Spin size="large" tip="加载数据中..." />
        </div>
      </div>
    )
  }

  return (
    <div className="weibo-analysis-page">
      <NeuralBackground density={0.3} speed={0.5} opacity={0.4} />

      <div className="page-content">
        <div className="page-header">
          <Title level={1} className="page-title">
            📊 微博用户分析
          </Title>
          <Paragraph className="page-subtitle">
            分析微博用户 {currentData.user.name} 一周内的情绪变化趋势与情感特征
          </Paragraph>
        </div>

        <div className="week-selector">
          <span style={{ color: '#94a3b8', marginRight: 8 }}>选择周：</span>
          <Select
            value={selectedWeek}
            onChange={setSelectedWeek}
            style={{ width: 160 }}
            options={[
              { value: '2024-01-01', label: '2024年第1周' },
              { value: '2024-01-08', label: '2024年第2周' },
              { value: '2024-01-15', label: '2024年第3周' },
            ]}
          />
          <span style={{ marginLeft: 16, fontSize: 12, color: '#64748b' }}>
            {currentData.user.period}
          </span>
        </div>

        {/* 侧边导航 + 内容区域 */}
        <div style={{ display: 'flex', gap: 0, minHeight: 'calc(100vh - 200px)' }}>
          {/* 侧边导航 */}
          <div className="tab-sidebar">
            <div 
              className={`tab-nav-item ${activeTab === 0 ? 'active' : ''}`}
              onClick={() => setActiveTab(0)}
            >
              <span className="tab-nav-icon">📊</span>
              <span className="tab-nav-text">总览</span>
            </div>
            <div 
              className={`tab-nav-item ${activeTab === 1 ? 'active' : ''}`}
              onClick={() => setActiveTab(1)}
            >
              <span className="tab-nav-icon">📈</span>
              <span className="tab-nav-text">趋势</span>
            </div>
            <div 
              className={`tab-nav-item ${activeTab === 2 ? 'active' : ''}`}
              onClick={() => setActiveTab(2)}
            >
              <span className="tab-nav-icon">🔍</span>
              <span className="tab-nav-text">细节</span>
            </div>
            <div 
              className={`tab-nav-item ${activeTab === 3 ? 'active' : ''}`}
              onClick={() => setActiveTab(3)}
            >
              <span className="tab-nav-icon">💡</span>
              <span className="tab-nav-text">洞察</span>
            </div>
          </div>

          {/* 内容区域 */}
          <div className="tab-content">
            
            {/* 第一层 — 总览 */}
            {activeTab === 0 && (
              <div className="summary-section">
                <Row gutter={[12, 12]} className="metrics-row">
                  <Col xs={12} sm={8}>
                    <div className="metric-card">
                      <div className="metric-label">平均情感得分</div>
                      <div className="metric-value" style={{ color: '#3B6D11' }}>
                        +{currentData.summary.avg_valence.toFixed(2)}
                      </div>
                      <div className="metric-sub">整体偏积极</div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8}>
                    <div className="metric-card">
                      <div className="metric-label">置信度均值</div>
                      <div className="metric-value" style={{ color: '#3B6D11' }}>
                        {currentData.summary.avg_confidence.toFixed(2)}
                      </div>
                      <div className="metric-sub">模型整体可靠性</div>
                    </div>
                  </Col>
                  <Col xs={12} sm={8}>
                    <div className="metric-card">
                      <div className="metric-label">情绪最低点</div>
                      <div className="metric-value" style={{ color: '#A32D2D' }}>
                        {currentData.summary.lowest_day}
                      </div>
                      <div className="metric-sub">V={currentData.summary.lowest_valence.toFixed(2)}</div>
                    </div>
                  </Col>
                </Row>
                <Row gutter={[12, 12]} className="metrics-row">
                  <Col xs={12} sm={12}>
                    <div className="metric-card">
                      <div className="metric-label">情绪最高点</div>
                      <div className="metric-value" style={{ color: '#3B6D11' }}>
                        {currentData.summary.highest_day}
                      </div>
                      <div className="metric-sub">V={currentData.summary.highest_valence.toFixed(2)}</div>
                    </div>
                  </Col>
                  <Col xs={12} sm={12}>
                    <div className="metric-card">
                      <div className="metric-label">博文总数</div>
                      <div className="metric-value">{currentData.posts.length}</div>
                      <div className="metric-sub">活跃度良好</div>
                    </div>
                  </Col>
                </Row>
                <Row gutter={[12, 12]} style={{ marginTop: 16 }}>
                  <Col xs={24}>
                    <Card className="card radar-card">
                      <div className="card-title">
                        <span className="card-num">综</span> 情感综合雷达
                      </div>
                      <div style={{ width: '100%', height: 220 }}>
                        <ReactECharts option={radarOption} style={{ height: '100%' }} />
                      </div>
                      <p style={{ fontSize: 11, color: '#888780', margin: '8px 0 0' }}>
                        {currentData.summary.radar_summary}
                      </p>
                    </Card>
                  </Col>
                </Row>
              </div>
            )}

            {/* 第二层 — 趋势 */}
            {activeTab === 1 && (
              <div className="trend-section">
                <Card className="card card-detail">
                  <div className="card-title">
                    <span className="card-num">1</span> VAD 三维情感趋势
                  </div>
                  <div style={{ display: 'flex', gap: 14, marginBottom: 8, fontSize: 11, color: '#64748b' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 14, height: 2, background: '#1D9E75', display: 'inline-block' }}></span>Valence 效价
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 14, height: 0, borderTop: '2px dashed #BA7517', display: 'inline-block' }}></span>Arousal 唤醒
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 14, height: 0, borderTop: '2px dotted #378ADD', display: 'inline-block' }}></span>Dominance 支配
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 9, height: 9, borderRadius: '50%', border: '1.5px solid #E24B4A', background: 'white', display: 'inline-block' }}></span>关键锚点
                    </span>
                  </div>
                  <div style={{ position: 'relative' }}>
                    <div style={{ width: '100%', height: 140 }}>
                      <ReactECharts option={trendOption} style={{ height: '100%' }} />
                    </div>
                    {anchorTooltip.visible && (
                      <div 
                        className="anchor-tooltip"
                        dangerouslySetInnerHTML={{ __html: anchorTooltip.content }}
                        style={{ 
                          left: anchorTooltip.x, 
                          top: anchorTooltip.y,
                          position: 'absolute'
                        }}
                      />
                    )}
                  </div>
                  <p style={{ fontSize: 10, color: '#888780', margin: '8px 0 0' }}>
                    阴影区域为该日 VAD 分布的 Q1–Q3 置信区间，体现单日情绪的波动范围
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, padding: '0 2px' }}>
                    <span style={{ fontSize: 10, color: '#888780' }}>周一</span>
                    <span style={{ fontSize: 10, color: '#888780' }}>周日</span>
                  </div>
                </Card>

                <Row gutter={[12, 12]} className="trend-row">
                  <Col xs={24} md={12}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">2</span> 每日情绪结构堆叠图
                      </div>
                      <div style={{ width: '100%', height: 130 }}>
                        <ReactECharts option={stackBarOption} style={{ height: '100%' }} />
                      </div>
                      <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 11, flexWrap: 'wrap' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#639922', display: 'inline-block' }}></span>积极
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#B4B2A9', display: 'inline-block' }}></span>中性
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#E24B4A', display: 'inline-block' }}></span>消极
                        </span>
                      </div>
                      <p style={{ fontSize: 10, color: '#888780', margin: '6px 0 0' }}>
                        与趋势图互补：趋势图看走向，此图看单日情绪结构
                      </p>
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">3</span> 发帖时段 × 情绪热力图
                      </div>
                      <div className="heat-grid">
                        <div></div>
                        {currentData.heatmap.cols.map(d => (
                          <div key={d} className="heat-day">{d}</div>
                        ))}
                        {currentData.heatmap.rows.map((time, rowIdx) => (
                          <React.Fragment key={time}>
                            <div className="heat-lbl">{time}</div>
                            {(currentData.heatmap.values[rowIdx] || []).map((v, i) => (
                              <div 
                                key={i} 
                                className="heat-cell" 
                                style={{ background: divergingColor(v) }}
                                title={`Valence: ${v.toFixed(2)}`}
                              ></div>
                            ))}
                          </React.Fragment>
                        ))}
                      </div>
                      <div style={{ display: 'flex', gap: 5, marginTop: 7, alignItems: 'center', fontSize: 10, color: '#888780' }}>
                        <span>消极</span>
                        <div style={{ display: 'flex', gap: 2 }}>
                          {[0.1, 0.25, 0.5, 0.75, 0.9].map((v, i) => (
                            <div key={i} style={{ width: 11, height: 6, borderRadius: 1, background: divergingColor(v) }}></div>
                          ))}
                        </div>
                        <span>积极（Valence均值）</span>
                      </div>
                    </Card>
                  </Col>
                </Row>
              </div>
            )}

            {/* 第三层 — 细节 */}
            {activeTab === 2 && (
              <div className="detail-section">
                <Row gutter={[12, 12]} className="detail-row">
                  <Col xs={24} md={8}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">4</span> 情绪轮细分
                      </div>
                      {currentData.emotion_wheel.map(item => (
                        <div key={item.name} className="wheel-row">
                          <div className="wdot" style={{ background: item.color }}></div>
                          <span className="wname">{item.name}</span>
                          <div className="wtrack">
                            <div className="wfill" style={{ width: `${item.avg_intensity * 100}%`, background: item.color }}></div>
                          </div>
                          <span className="wcnt">{item.count}次</span>
                          <span className="winterp">强度 {item.avg_intensity.toFixed(2)}</span>
                        </div>
                      ))}
                      <p style={{ fontSize: 10, color: '#888780', margin: '7px 0 0' }}>
                        由 CoT 第2步 VAD 量化映射至 Plutchik 模型
                      </p>
                    </Card>
                  </Col>
                  <Col xs={24} md={8}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">5</span> 高频情感词云
                      </div>
                      {/* 时间轴滑块 */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, padding: '0 4px' }}>
                        <span style={{ fontSize: 10, color: '#8B8A85', whiteSpace: 'nowrap' }}>本周总览</span>
                <input
                  type="range"
                  min="0"
                  max="7"
                  value={selectedDay ?? 7}
                  onChange={(e) => {
                    const val = parseInt(e.target.value)
                    setSelectedDay(val === 7 ? null : val)
                  }}
                  style={{ flex: 1, cursor: 'pointer', accentColor: '#534AB7' }}
                />
                <span style={{ fontSize: 10, color: '#8B8A85', whiteSpace: 'nowrap' }}>
                  {selectedDay === null ? '周日' : DAY_ORDER[selectedDay]}
                </span>
              </div>
              {/* react-wordcloud 词云 */}
              <div style={{ height: 160, width: '100%' }}>
                <ReactWordcloud
                  words={selectedDay === null 
                    ? currentData.wordcloud
                        .filter(w => (w as any).postIds && (w as any).postIds.length > 0)
                        .map(w => ({ text: w.word, value: w.size, sentiment: (w.sentiment as 'pos' | 'neg' | 'neu') || 'neu', postIds: (w as any).postIds || [] }))
                    : ((currentData.wordcloud_by_day && currentData.wordcloud_by_day[selectedDay.toString()]) 
                        ? currentData.wordcloud_by_day[selectedDay.toString()]
                            .filter(w => (w as any).postIds && (w as any).postIds.length > 0)
                            .map(w => ({ text: w.word, value: w.size, sentiment: (w.sentiment as 'pos' | 'neg' | 'neu') || 'neu', postIds: (w as any).postIds || [] }))
                        : [])
                  }
                  options={{
                    fontSizes: [11, 24] as [number, number],
                    rotations: 0,
                    rotationAngles: [0, 0] as [number, number],
                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
                    deterministic: true,
                  }}
                  callbacks={{
                    onWordClick: (word: any) => {
                      setSelectedWord(selectedWord === word.text ? null : word.text)
                    },
                    onWordMouseOver: (word: any) => setHoveredWord(word.text),
                    onWordMouseOut: () => setHoveredWord(null),
                  }}
                />
              </div>
              {/* 自定义悬浮提示 */}
              {hoveredWord && (() => {
                const allWords: WordCloudWord[] = selectedDay === null 
                  ? currentData.wordcloud
                      .filter(w => (w as any).postIds && (w as any).postIds.length > 0)
                      .map(w => ({ text: w.word, value: w.size, sentiment: (w.sentiment as 'pos' | 'neg' | 'neu') || 'neu', postIds: (w as any).postIds || [] }))
                  : ((currentData.wordcloud_by_day && currentData.wordcloud_by_day[selectedDay.toString()])
                      ? currentData.wordcloud_by_day[selectedDay.toString()]
                          .filter(w => (w as any).postIds && (w as any).postIds.length > 0)
                          .map(w => ({ text: w.word, value: w.size, sentiment: (w.sentiment as 'pos' | 'neg' | 'neu') || 'neu', postIds: (w as any).postIds || [] }))
                      : [])
                const wordData = allWords.find(w => w.text === hoveredWord)
                const postIds = wordData?.postIds || []
                const sentiment = (wordData?.sentiment || 'neu') as 'pos' | 'neg' | 'neu'
                return (
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    padding: '8px 12px',
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    border: '1px solid #E0DFD9',
                    borderRadius: 8,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                    fontSize: 12,
                    zIndex: 100,
                    pointerEvents: 'none',
                  }}>
                    <div style={{ fontWeight: 600, marginBottom: 4, color: '#1A1A18' }}>{hoveredWord}</div>
                    <div style={{ color: '#8B8A85', marginBottom: 2 }}>
                      关联 <strong>{postIds.length}</strong> 篇博文
                    </div>
                    <div style={{ color: WORDCLOUD_COLORS[sentiment] }}>
                      {sentiment === 'pos' ? '积极' : sentiment === 'neg' ? '消极' : '中性'}
                    </div>
                  </div>
                )
              })()}
              <p style={{ fontSize: 10, color: '#888780', margin: '8px 0 0' }}>
                CoT 第1步词汇锚定 · 字号=频率 · 颜色=极性{selectedDay !== null && ` · 当前：${DAY_ORDER[selectedDay]}`}
                {selectedWord && <span style={{ color: '#534AB7' }}> · 已筛选：{selectedWord}</span>}
              </p>
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card className="card card-detail">
              <div className="card-title">
                <span className="card-num">6</span> VAD 情感空间散点
                <span style={{ marginLeft: 'auto', fontSize: 10, color: '#888780' }}>含时间轨迹</span>
              </div>
              <div style={{ width: '100%', height: 160 }}>
                <ReactECharts option={scatterOption} style={{ height: '100%' }} />
              </div>
              <div style={{ display: 'flex', gap: 10, marginTop: 6, fontSize: 10, color: '#64748b', flexWrap: 'wrap' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'rgba(99,153,34,0.75)', display: 'inline-block' }}></span>积极
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'rgba(186,117,23,0.75)', display: 'inline-block' }}></span>中性
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'rgba(226,75,74,0.75)', display: 'inline-block' }}></span>消极
                </span>
                <span style={{ color: '#888780' }}>点大小=Dominance</span>
                <span style={{ color: '#888780' }}>· 箭头=时间顺序轨迹</span>
              </div>
            </Card>
          </Col>
        </Row>

        {/* 博文列表（默认折叠，点击展开） */}
        <div 
          className="posts-collapsible"
          onClick={() => setPostsExpanded(!postsExpanded)}
          style={{ 
            padding: '12px 16px', 
            background: postsExpanded ? 'rgba(83, 74, 183, 0.08)' : 'rgba(83, 74, 183, 0.04)',
            borderRadius: 8, 
            cursor: 'pointer',
            marginTop: 12,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            transition: 'all 0.2s ease'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="card-num">7</span>
            <span style={{ fontWeight: 500, color: 'var(--theme-text-primary)' }}>博文列表 — 情感精细标注</span>
            {selectedWord && (
              <span style={{ fontSize: 11, color: '#534AB7' }}>筛选词："{selectedWord}"</span>
            )}
          </div>
          <span style={{ color: '#64748b', fontSize: 12 }}>
            {postsExpanded ? '收起 ▲' : '展开 ▼'}
          </span>
        </div>

        {postsExpanded && (
          <Card className="card card-detail" style={{ marginTop: 12 }}>
            {currentData.posts
              .filter(post => !selectedWord || matchedPostIds.includes(post.id))
              .slice(0, selectedWord ? undefined : 5)
              .map(post => (
              <div key={post.id} className="post-row">
                <div style={{ textAlign: 'center', minWidth: 42 }}>
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 500,
                      color: post.v >= 0.5 ? '#3B6D11' : post.v >= 0.3 ? '#854F0B' : '#A32D2D',
                    }}
                  >
                    {post.v >= 0 ? '+' : ''}{post.v.toFixed(2)}
                  </div>
                  <div className={`badge ${post.v >= 0.5 ? 'badge-pos' : post.v >= 0.3 ? 'badge-warn' : 'badge-neg'}`} style={{ marginTop: 3 }}>
                    {post.v >= 0.5 ? '积极' : post.v >= 0.3 ? '低置信' : '消极'}
                  </div>
                </div>
                <div className="post-main">
                  <p className="post-text">{post.content}</p>
                  <div className="vmi">
                    <div className="vmi-item">
                      <div className="vmi-lbl">V {post.v.toFixed(2)}</div>
                      <div className="vmi-track">
                        <div className="vmi-fill" style={{ width: `${post.v * 100}%`, background: EMOTION_COLORS[post.emotion] }}></div>
                      </div>
                    </div>
                    <div className="vmi-item">
                      <div className="vmi-lbl">A {post.a.toFixed(2)}</div>
                      <div className="vmi-track">
                        <div className="vmi-fill" style={{ width: `${post.a * 100}%`, background: '#BA7517' }}></div>
                      </div>
                    </div>
                    <div className="vmi-item">
                      <div className="vmi-lbl">D {post.d.toFixed(2)}</div>
                      <div className="vmi-track">
                        <div className="vmi-fill" style={{ width: `${post.d * 100}%`, background: '#378ADD' }}></div>
                      </div>
                    </div>
                    <span className="confidence-label">
                      置信 <b style={{ color: post.confidence >= 0.8 ? '#3B6D11' : '#854F0B' }}>{post.confidence.toFixed(2)}</b>
                    </span>
                  </div>
                  <div className="post-meta">
                    {post.day} {post.time} · 诱因：{post.cause}
                    {post.rule && <span className="rule-tag">{post.rule}</span>}
                    · 点赞{post.likes}
                  </div>
                </div>
              </div>
            ))}
          </Card>
        )}
              </div>
            )}

            {/* 第四层 — 洞察 */}
            {activeTab === 3 && (
              <div className="insight-section">
                <Row gutter={[12, 12]} className="insight-row">
                  <Col xs={24} md={12}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">8</span> 情绪 × 社交互动关联
                      </div>
                      <div style={{ width: '100%', height: 130 }}>
                        <ReactECharts option={socialBarOption} style={{ height: '100%' }} />
                      </div>
                      <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 11, flexWrap: 'wrap' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#378ADD', display: 'inline-block' }}></span>平均点赞
                        </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 2, border: '1px dashed #7F77DD', background: 'rgba(127,119,221,0.75)', display: 'inline-block' }}></span>平均评论
                        </span>
                      </div>
                      <p style={{ fontSize: 10, color: '#888780', margin: '6px 0 0' }}>
                        洞察：消极博文平均评论量是积极博文的 2.1 倍，负面情绪更易引发社交共鸣
                      </p>
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card className="card card-detail">
                      <div className="card-title">
                        <span className="card-num">9</span> Valence 日内分布
                        <span style={{ marginLeft: 'auto', fontSize: 10, color: '#888780' }}>箱线图 · 含异常值标记</span>
                      </div>
                      <div style={{ width: '100%', height: 140 }}>
                        <ReactECharts option={boxBarOption} style={{ height: '100%' }} />
                      </div>
                      <p style={{ fontSize: 10, color: '#888780', margin: '10px 0 0' }}>
                        圆点 = 当日极端情绪博文（异常值）· <strong>周三离散度最高（σ={currentData.summary.volatility_std}）</strong>，箱体最长，单日内情绪起伏最剧烈
                      </p>
                    </Card>
                  </Col>
                </Row>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default WeiboAnalysis