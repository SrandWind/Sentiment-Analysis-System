import React, { useEffect, useRef } from 'react'
import './NeuralBackground.scss'

interface NeuralBackgroundProps {
  density?: number // 粒子密度 0.3-1.0
  speed?: number   // 动画速度 0.5-2.0
  opacity?: number // 透明度 0.3-1.0
}

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  alpha: number
}

const NeuralBackground: React.FC<NeuralBackgroundProps> = ({
  density = 0.5,
  speed = 0.8,
  opacity = 0.6,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const animationRef = useRef<number>()
  const mouseRef = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 设置画布尺寸
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // 初始化粒子
    const initParticles = () => {
      const particleCount = Math.floor((canvas.width * canvas.height) / 15000 * density)
      particlesRef.current = []

      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.3 * speed,
          vy: (Math.random() - 0.5) * 0.3 * speed,
          radius: Math.random() * 2 + 1,
          alpha: Math.random() * 0.5 + 0.3,
        })
      }
    }
    initParticles()

    // 鼠标交互
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', handleMouseMove)

    // 绘制函数（低帧率 30fps）
    let lastTime = 0
    const fpsInterval = 1000 / 30 // 30 FPS

    const draw = (currentTime: number) => {
      animationRef.current = requestAnimationFrame(draw)

      const elapsed = currentTime - lastTime
      if (elapsed < fpsInterval) return

      lastTime = currentTime - (elapsed % fpsInterval)

      // 清空画布
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const particles = particlesRef.current
      const connectionDistance = 150
      const mouseDistance = 200

      // 更新和绘制粒子
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i]

        // 更新位置
        p.x += p.vx
        p.y += p.vy

        // 边界检测
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1

        // 鼠标交互（轻微排斥）
        const dx = mouseRef.current.x - p.x
        const dy = mouseRef.current.y - p.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance < mouseDistance) {
          const force = (mouseDistance - distance) / mouseDistance
          p.x -= (dx / distance) * force * 0.5
          p.y -= (dy / distance) * force * 0.5
        }

        // 绘制粒子
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(255, 107, 74, ${p.alpha * opacity})`
        ctx.fill()

        // 绘制连接线
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j]
          const dx2 = p.x - p2.x
          const dy2 = p.y - p2.y
          const dist = Math.sqrt(dx2 * dx2 + dy2 * dy2)

          if (dist < connectionDistance) {
            const lineAlpha = (1 - dist / connectionDistance) * 0.3 * opacity
            ctx.beginPath()
            ctx.moveTo(p.x, p.y)
            ctx.lineTo(p2.x, p2.y)
            ctx.strokeStyle = `rgba(255, 107, 74, ${lineAlpha})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }
    }

    animationRef.current = requestAnimationFrame(draw)

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('mousemove', handleMouseMove)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [density, speed, opacity])

  return (
    <canvas
      ref={canvasRef}
      className="neural-background"
      style={{ pointerEvents: 'none' }}
    />
  )
}

export default NeuralBackground
