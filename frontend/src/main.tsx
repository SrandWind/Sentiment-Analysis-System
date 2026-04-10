import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './assets/styles/design-system.scss'
import './index.scss'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: undefined, // 让 design-system.scss 的 CSS 变量控制
        token: {
          colorPrimary: '#ff6b4a',
          borderRadius: 8,
          fontFamily: '"Source Han Sans SC", "IBM Plex Sans", sans-serif',
          fontSize: 15,
        },
        components: {
          Layout: {
            headerBg: 'rgba(15, 23, 36, 0.9)',
            headerHeight: 64,
          },
          Menu: {
            darkItemSelectedBg: 'rgba(255, 107, 74, 0.15)',
            darkItemColor: '#94a3b8',
          },
          Card: {
            borderRadiusLG: 16,
          },
          Button: {
            borderRadiusLG: 12,
          },
        },
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>
)
