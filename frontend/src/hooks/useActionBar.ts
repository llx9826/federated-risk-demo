import { useMemo } from 'react'
import { useBreakpoint } from './useBreakpoint'

// 操作栏配置接口
interface ActionBarConfig {
  // 布局配置
  direction: 'horizontal' | 'vertical'
  align: 'start' | 'center' | 'end'
  wrap: boolean
  
  // 样式配置
  buttonWidth: string | number
  buttonHeight: string | number
  spacing: number
  
  // 安全区域
  safeAreaBottom: string
  
  // 响应式标识
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
}

// 操作栏样式接口
interface ActionBarStyles {
  container: React.CSSProperties
  button: React.CSSProperties
  dangerSection: React.CSSProperties
  mainSection: React.CSSProperties
}

// 操作栏行为接口
interface ActionBarBehavior {
  // 按钮排序规则
  sortActions: (actions: any[]) => any[]
  
  // 危险操作处理
  shouldSeparateDangerActions: boolean
  
  // 更多操作菜单
  shouldUseMoreMenu: boolean
  maxVisibleActions: number
}

// useActionBar返回值接口
interface UseActionBarReturn {
  config: ActionBarConfig
  styles: ActionBarStyles
  behavior: ActionBarBehavior
  
  // 工具方法
  getButtonProps: (index: number, total: number) => React.CSSProperties
  getContainerProps: (sticky?: boolean) => React.CSSProperties
}

/**
 * useActionBar Hook
 * 根据断点自动切换纵/横布局与安全区域支持
 */
export const useActionBar = (): UseActionBarReturn => {
  const { 
    isMobile, 
    isTablet, 
    isDesktop,
    screen 
  } = useBreakpoint()
  
  // 基础配置
  const config = useMemo((): ActionBarConfig => {
    return {
      // 移动端强制纵向布局
      direction: isMobile ? 'vertical' : 'horizontal',
      align: isMobile ? 'start' : 'end',
      wrap: !isDesktop,
      
      // 按钮尺寸
      buttonWidth: isMobile ? '100%' : isTablet ? '100px' : '112px',
      buttonHeight: isMobile ? '44px' : '32px',
      spacing: isMobile ? 12 : 8,
      
      // iOS安全区域支持
      safeAreaBottom: isMobile ? 'env(safe-area-inset-bottom, 0px)' : '0px',
      
      // 设备标识
      isMobile,
      isTablet,
      isDesktop,
    }
  }, [isMobile, isTablet, isDesktop])
  
  // 样式配置
  const styles = useMemo((): ActionBarStyles => {
    const baseSpacing = config.spacing
    
    return {
      container: {
        display: 'flex',
        flexDirection: config.direction === 'vertical' ? 'column' : 'row',
        justifyContent: config.align === 'start' ? 'flex-start' : 
                       config.align === 'center' ? 'center' : 'flex-end',
        alignItems: isMobile ? 'stretch' : 'center',
        gap: baseSpacing,
        padding: isMobile ? '12px 16px' : '16px',
        paddingBottom: `calc(16px + ${config.safeAreaBottom})`,
        flexWrap: config.wrap ? 'wrap' : 'nowrap',
      },
      
      button: {
        minWidth: config.buttonWidth,
        height: config.buttonHeight,
        width: isMobile ? '100%' : 'auto',
        transition: 'all 0.2s ease',
      },
      
      dangerSection: {
        display: 'flex',
        flexDirection: config.direction === 'vertical' ? 'column' : 'row',
        gap: baseSpacing,
        order: isMobile ? 2 : 0,
        paddingTop: isMobile ? baseSpacing : 0,
        borderTop: isMobile ? '1px solid var(--ant-color-border-secondary)' : 'none',
        marginTop: isMobile ? baseSpacing : 0,
      },
      
      mainSection: {
        display: 'flex',
        flexDirection: config.direction === 'vertical' ? 'column' : 'row',
        gap: baseSpacing,
        order: isMobile ? 1 : 0,
        marginLeft: !isMobile ? 'auto' : 0,
      },
    }
  }, [config, isMobile])
  
  // 行为配置
  const behavior = useMemo((): ActionBarBehavior => {
    return {
      // 按钮优先级排序：primary > default > ghost > link
      sortActions: (actions: Array<{type?: string}>) => {
        const priority: Record<string, number> = { 
          primary: 4, 
          default: 3, 
          dashed: 2, 
          ghost: 1, 
          link: 0, 
          text: 0 
        }
        
        return [...actions].sort((a, b) => {
          const aPriority = priority[a.type || 'default'] || 0
          const bPriority = priority[b.type || 'default'] || 0
          return bPriority - aPriority
        })
      },
      
      // 危险操作分离
      shouldSeparateDangerActions: true,
      
      // 更多操作菜单（小屏幕时启用）
      shouldUseMoreMenu: isMobile && screen.width < 375,
      maxVisibleActions: isMobile ? (screen.width < 375 ? 2 : 3) : 6,
    }
  }, [isMobile, screen.width])
  
  // 获取按钮属性
  const getButtonProps = (index: number, total: number): React.CSSProperties => {
    const baseProps = { ...styles.button }
    
    // 移动端按钮间距
    if (isMobile && index < total - 1) {
      baseProps.marginBottom = config.spacing
    }
    
    // 主按钮强调（通常是第一个primary按钮）
    if (index === 0) {
      baseProps.fontWeight = 500
    }
    
    return baseProps
  }
  
  // 获取容器属性
  const getContainerProps = (sticky = false): React.CSSProperties => {
    const baseProps = { ...styles.container }
    
    if (sticky) {
      baseProps.position = 'sticky'
      baseProps.bottom = 0
      baseProps.zIndex = 100
      baseProps.backgroundColor = 'var(--ant-color-bg-container)'
      baseProps.borderTop = '1px solid var(--ant-color-split)'
      baseProps.boxShadow = '0 -2px 8px rgba(0, 0, 0, 0.1)'
      
      // 暗色主题阴影
      if (document.documentElement.getAttribute('data-theme') === 'dark') {
        baseProps.boxShadow = '0 -2px 8px rgba(0, 0, 0, 0.3)'
      }
    }
    
    return baseProps
  }
  
  return {
    config,
    styles,
    behavior,
    getButtonProps,
    getContainerProps,
  }
}

export type {
  ActionBarConfig,
  ActionBarStyles,
  ActionBarBehavior,
  UseActionBarReturn,
}