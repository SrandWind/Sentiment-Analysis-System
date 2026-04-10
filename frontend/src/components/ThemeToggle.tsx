import React from 'react'
import { Switch } from 'antd'
import { SunOutlined, MoonOutlined } from '@ant-design/icons'
import './ThemeToggle.scss'

interface ThemeToggleProps {
  onThemeChange?: (isDark: boolean) => void
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ onThemeChange }) => {
  const [isDark, setIsDark] = React.useState(true)

  React.useEffect(() => {
    // 初始化主题 - 默认浅色模式
    const savedTheme = localStorage.getItem('theme')

    if (savedTheme) {
      const shouldUseDark = savedTheme === 'dark'
      setIsDark(shouldUseDark)
      document.documentElement.setAttribute('data-theme', shouldUseDark ? 'dark' : 'light')
    } else {
      // 默认浅色模式
      setIsDark(false)
      document.documentElement.setAttribute('data-theme', 'light')
    }
  }, [])

  const toggleTheme = (checked: boolean) => {
    const newIsDark = checked
    setIsDark(newIsDark)
    document.documentElement.setAttribute('data-theme', newIsDark ? 'dark' : 'light')
    localStorage.setItem('theme', newIsDark ? 'dark' : 'light')
    onThemeChange?.(newIsDark)
  }

  return (
    <div className="theme-toggle">
      <span className="theme-label">
        <SunOutlined className="icon sun-icon" />
      </span>
      <Switch
        checked={isDark}
        onChange={toggleTheme}
        checkedChildren={<MoonOutlined className="switch-icon moon-icon" />}
        unCheckedChildren={<SunOutlined className="switch-icon sun-icon" />}
        className="theme-switch"
        size="default"
      />
      <span className="theme-label">
        <MoonOutlined className="icon moon-icon" />
      </span>
    </div>
  )
}

export default ThemeToggle
