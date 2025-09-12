import { useState, useEffect, useCallback, useRef } from 'react'
import { message } from 'antd'

// 无障碍配置接口
interface AccessibilityConfig {
  announcePageChanges: boolean
  enableKeyboardNavigation: boolean
  enableHighContrast: boolean
  enableReducedMotion: boolean
  fontSize: 'small' | 'medium' | 'large' | 'extra-large'
  enableScreenReader: boolean
}

// 无障碍状态接口
interface AccessibilityState {
  isScreenReaderActive: boolean
  isKeyboardUser: boolean
  isHighContrastMode: boolean
  isReducedMotionMode: boolean
  currentFocusElement: Element | null
  announcements: string[]
}

// 键盘快捷键映射
interface KeyboardShortcuts {
  [key: string]: () => void
}

/**
 * 无障碍支持Hook
 * 提供屏幕阅读器支持、键盘导航、高对比度模式等无障碍功能
 */
export const useAccessibility = (config: Partial<AccessibilityConfig> = {}) => {
  const defaultConfig: AccessibilityConfig = {
    announcePageChanges: true,
    enableKeyboardNavigation: true,
    enableHighContrast: false,
    enableReducedMotion: false,
    fontSize: 'medium',
    enableScreenReader: true,
    ...config,
  }

  const [accessibilityState, setAccessibilityState] = useState<AccessibilityState>({
    isScreenReaderActive: false,
    isKeyboardUser: false,
    isHighContrastMode: false,
    isReducedMotionMode: false,
    currentFocusElement: null,
    announcements: [],
  })

  const [currentConfig, setCurrentConfig] = useState<AccessibilityConfig>(defaultConfig)
  const announcementRef = useRef<HTMLDivElement>(null)
  const keyboardShortcutsRef = useRef<KeyboardShortcuts>({})

  // 检测屏幕阅读器
  const detectScreenReader = useCallback(() => {
    // 检测常见的屏幕阅读器
    const userAgent = navigator.userAgent.toLowerCase()
    
    // 检测媒体查询支持
    let supportsReducedMotion = false
    try {
      if (typeof window !== 'undefined' && 'matchMedia' in window) {
        supportsReducedMotion = (window as any).matchMedia('(prefers-reduced-motion: reduce)').matches
      }
    } catch (e) {
      // 忽略错误
    }
    
    const hasScreenReader = 
      userAgent.includes('nvda') ||
      userAgent.includes('jaws') ||
      userAgent.includes('voiceover') ||
      userAgent.includes('talkback') ||
      // 检测是否有辅助技术API
      'speechSynthesis' in window ||
      // 检测媒体查询
      supportsReducedMotion

    setAccessibilityState(prev => ({
      ...prev,
      isScreenReaderActive: hasScreenReader,
    }))

    return hasScreenReader
  }, [])

  // 检测键盘用户
  const detectKeyboardUser = useCallback(() => {
    let isKeyboardUser = false

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        isKeyboardUser = true
        setAccessibilityState(prev => ({
          ...prev,
          isKeyboardUser: true,
        }))
        document.body.classList.add('keyboard-user')
      }
    }

    const handleMouseDown = () => {
      isKeyboardUser = false
      setAccessibilityState(prev => ({
        ...prev,
        isKeyboardUser: false,
      }))
      document.body.classList.remove('keyboard-user')
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('mousedown', handleMouseDown)

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('mousedown', handleMouseDown)
    }
  }, [])

  // 检测系统偏好设置
  const detectSystemPreferences = useCallback(() => {
    // 检测高对比度模式
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)')
    const handleHighContrastChange = (e: MediaQueryListEvent) => {
      setAccessibilityState(prev => ({
        ...prev,
        isHighContrastMode: e.matches,
      }))
      
      if (e.matches) {
        document.body.classList.add('high-contrast')
      } else {
        document.body.classList.remove('high-contrast')
      }
    }
    
    highContrastQuery.addEventListener('change', handleHighContrastChange)
    handleHighContrastChange({ matches: highContrastQuery.matches } as MediaQueryListEvent)

    // 检测减少动画偏好
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handleReducedMotionChange = (e: MediaQueryListEvent) => {
      setAccessibilityState(prev => ({
        ...prev,
        isReducedMotionMode: e.matches,
      }))
      
      if (e.matches) {
        document.body.classList.add('reduced-motion')
      } else {
        document.body.classList.remove('reduced-motion')
      }
    }
    
    reducedMotionQuery.addEventListener('change', handleReducedMotionChange)
    handleReducedMotionChange({ matches: reducedMotionQuery.matches } as MediaQueryListEvent)

    return () => {
      highContrastQuery.removeEventListener('change', handleHighContrastChange)
      reducedMotionQuery.removeEventListener('change', handleReducedMotionChange)
    }
  }, [])

  // 焦点管理
  const manageFocus = useCallback(() => {
    const handleFocusChange = (e: FocusEvent) => {
      setAccessibilityState(prev => ({
        ...prev,
        currentFocusElement: e.target as Element,
      }))
    }

    document.addEventListener('focusin', handleFocusChange)
    document.addEventListener('focusout', handleFocusChange)

    return () => {
      document.removeEventListener('focusin', handleFocusChange)
      document.removeEventListener('focusout', handleFocusChange)
    }
  }, [])

  // 语音播报
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!currentConfig.enableScreenReader) return

    // 添加到播报队列
    setAccessibilityState(prev => ({
      ...prev,
      announcements: [...prev.announcements, message],
    }))

    // 创建临时的aria-live区域进行播报
    const announcement = document.createElement('div')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.style.position = 'absolute'
    announcement.style.left = '-10000px'
    announcement.style.width = '1px'
    announcement.style.height = '1px'
    announcement.style.overflow = 'hidden'
    announcement.textContent = message

    document.body.appendChild(announcement)

    // 清理
    setTimeout(() => {
      if (announcement.parentNode) {
        announcement.parentNode.removeChild(announcement)
      }
    }, 1000)

    // 使用Web Speech API（如果可用）
    if ('speechSynthesis' in window && window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(message)
      utterance.rate = 0.8
      utterance.pitch = 1
      utterance.volume = 0.8
      window.speechSynthesis.speak(utterance)
    }
  }, [currentConfig.enableScreenReader])

  // 设置字体大小
  const setFontSize = useCallback((size: AccessibilityConfig['fontSize']) => {
    setCurrentConfig(prev => ({ ...prev, fontSize: size }))
    
    const sizeMap = {
      'small': '14px',
      'medium': '16px',
      'large': '18px',
      'extra-large': '20px',
    }
    
    document.documentElement.style.fontSize = sizeMap[size]
    document.body.classList.remove('font-small', 'font-medium', 'font-large', 'font-extra-large')
    document.body.classList.add(`font-${size}`)
    
    announce(`字体大小已调整为${size === 'extra-large' ? '特大' : size === 'large' ? '大' : size === 'medium' ? '中等' : '小'}`)
  }, [announce])

  // 切换高对比度模式
  const toggleHighContrast = useCallback(() => {
    const newValue = !currentConfig.enableHighContrast
    setCurrentConfig(prev => ({ ...prev, enableHighContrast: newValue }))
    
    if (newValue) {
      document.body.classList.add('high-contrast-forced')
      announce('高对比度模式已开启')
    } else {
      document.body.classList.remove('high-contrast-forced')
      announce('高对比度模式已关闭')
    }
  }, [currentConfig.enableHighContrast, announce])

  // 切换减少动画模式
  const toggleReducedMotion = useCallback(() => {
    const newValue = !currentConfig.enableReducedMotion
    setCurrentConfig(prev => ({ ...prev, enableReducedMotion: newValue }))
    
    if (newValue) {
      document.body.classList.add('reduced-motion-forced')
      announce('减少动画模式已开启')
    } else {
      document.body.classList.remove('reduced-motion-forced')
      announce('减少动画模式已关闭')
    }
  }, [currentConfig.enableReducedMotion, announce])

  // 注册键盘快捷键
  const registerShortcut = useCallback((key: string, callback: () => void) => {
    keyboardShortcutsRef.current[key] = callback
  }, [])

  // 注销键盘快捷键
  const unregisterShortcut = useCallback((key: string) => {
    delete keyboardShortcutsRef.current[key]
  }, [])

  // 处理键盘快捷键
  const handleKeyboardShortcuts = useCallback((e: KeyboardEvent) => {
    if (!currentConfig.enableKeyboardNavigation) return

    const key = `${e.ctrlKey ? 'ctrl+' : ''}${e.altKey ? 'alt+' : ''}${e.shiftKey ? 'shift+' : ''}${e.key.toLowerCase()}`
    const callback = keyboardShortcutsRef.current[key]
    
    if (callback) {
      e.preventDefault()
      callback()
    }
  }, [currentConfig.enableKeyboardNavigation])

  // 跳转到主要内容
  const skipToMainContent = useCallback(() => {
    const mainContent = document.querySelector('main, [role="main"], #main-content')
    if (mainContent) {
      (mainContent as HTMLElement).focus()
      announce('已跳转到主要内容')
    }
  }, [announce])

  // 显示无障碍帮助
  const showAccessibilityHelp = useCallback(() => {
    const helpText = `
      无障碍快捷键：
      Alt+1: 跳转到主要内容
      Alt+2: 切换高对比度模式
      Alt+3: 切换减少动画模式
      Alt+Plus: 增大字体
      Alt+Minus: 减小字体
      Tab: 在可交互元素间导航
      Enter/Space: 激活按钮或链接
      Escape: 关闭对话框或菜单
    `
    
    message.info({
      content: helpText,
      duration: 10,
      style: { whiteSpace: 'pre-line' },
    })
    
    announce('无障碍帮助信息已显示')
  }, [announce])

  // 初始化
  useEffect(() => {
    const cleanupFunctions: (() => void)[] = []

    // 检测各种无障碍特性
    detectScreenReader()
    cleanupFunctions.push(detectKeyboardUser())
    cleanupFunctions.push(detectSystemPreferences())
    cleanupFunctions.push(manageFocus())

    // 注册默认快捷键
    registerShortcut('alt+1', skipToMainContent)
    registerShortcut('alt+2', toggleHighContrast)
    registerShortcut('alt+3', toggleReducedMotion)
    registerShortcut('alt+h', showAccessibilityHelp)
    registerShortcut('alt+=', () => {
      const sizes: AccessibilityConfig['fontSize'][] = ['small', 'medium', 'large', 'extra-large']
      const currentIndex = sizes.indexOf(currentConfig.fontSize)
      if (currentIndex < sizes.length - 1) {
        setFontSize(sizes[currentIndex + 1])
      }
    })
    registerShortcut('alt+-', () => {
      const sizes: AccessibilityConfig['fontSize'][] = ['small', 'medium', 'large', 'extra-large']
      const currentIndex = sizes.indexOf(currentConfig.fontSize)
      if (currentIndex > 0) {
        setFontSize(sizes[currentIndex - 1])
      }
    })

    // 监听键盘事件
    document.addEventListener('keydown', handleKeyboardShortcuts)
    cleanupFunctions.push(() => {
      document.removeEventListener('keydown', handleKeyboardShortcuts)
    })

    // 页面加载完成后播报
    if (currentConfig.announcePageChanges) {
      setTimeout(() => {
        announce('页面已加载完成')
      }, 1000)
    }

    return () => {
      cleanupFunctions.forEach(cleanup => cleanup())
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // 状态
    accessibilityState,
    config: currentConfig,
    
    // 方法
    announce,
    setFontSize,
    toggleHighContrast,
    toggleReducedMotion,
    registerShortcut,
    unregisterShortcut,
    skipToMainContent,
    showAccessibilityHelp,
    
    // 配置更新
    updateConfig: setCurrentConfig,
  }
}

export type { AccessibilityConfig, AccessibilityState }