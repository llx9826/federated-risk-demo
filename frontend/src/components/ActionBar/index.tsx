import React from 'react'
import { Space, Button } from 'antd'
import { useBreakpoint } from '@/hooks/useBreakpoint'
import './index.less'

// 操作按钮配置接口
interface ActionConfig {
  key: string
  type?: 'primary' | 'default' | 'dashed' | 'link' | 'text'
  danger?: boolean
  ghost?: boolean
  icon?: React.ReactNode
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  loading?: boolean
  href?: string
  target?: string
  htmlType?: 'button' | 'submit' | 'reset'
  size?: 'large' | 'middle' | 'small'
  className?: string
  'aria-label'?: string
}

// ActionBar属性接口
interface ActionBarProps {
  children?: React.ReactNode
  actions?: ActionConfig[]
  align?: 'start' | 'center' | 'end'
  sticky?: boolean
  className?: string
  style?: React.CSSProperties
  size?: 'small' | 'middle' | 'large'
  wrap?: boolean
  split?: React.ReactNode
  direction?: 'horizontal' | 'vertical'
  // 危险操作区域（左侧单独显示）
  dangerActions?: ActionConfig[]
  // 更多操作菜单
  moreActions?: ActionConfig[]
}

/**
 * ActionBar 操作栏组件
 * 统一页面与卡片底部操作区的对齐规范
 */
const ActionBar: React.FC<ActionBarProps> = ({
  children,
  actions = [],
  align = 'end',
  sticky = false,
  className = '',
  style,
  size = 'middle',
  wrap = true,
  split,
  direction,
  dangerActions = [],
  moreActions = [],
}) => {
  const { isMobile, isTablet } = useBreakpoint()
  
  // 移动端强制纵向布局
  const finalDirection = direction || (isMobile ? 'vertical' : 'horizontal')
  const finalAlign = isMobile ? 'stretch' : align
  
  // 渲染按钮
  const renderButton = (action: ActionConfig) => {
    const {
      key,
      type = 'default',
      danger = false,
      ghost = false,
      icon,
      children,
      onClick,
      disabled = false,
      loading = false,
      href,
      target,
      htmlType = 'button',
      size: buttonSize = size,
      'aria-label': ariaLabel,
    } = action
    
    const buttonProps = {
      key,
      type,
      danger,
      ghost,
      icon,
      onClick,
      disabled,
      loading,
      href,
      target,
      htmlType,
      size: buttonSize,
      'aria-label': ariaLabel || (typeof children === 'string' ? children : undefined),
      className: 'action-bar-button',
      style: {
        minWidth: isMobile ? 'auto' : '112px', // 按钮最小宽度
        width: isMobile ? '100%' : 'auto',     // 移动端全宽
      },
    }
    
    return (
      <Button {...buttonProps}>
        {children}
      </Button>
    )
  }
  
  // 渲染操作按钮组
  const renderActions = () => {
    if (children) {
      return children
    }
    
    if (actions.length === 0 && dangerActions.length === 0) {
      return null
    }
    
    // 按钮优先级排序：primary > default > ghost > link
    const sortedActions = [...actions].sort((a, b) => {
      const priority = { primary: 4, default: 3, dashed: 2, ghost: 1, link: 0, text: 0 }
      return (priority[b.type || 'default'] || 0) - (priority[a.type || 'default'] || 0)
    })
    
    // 移动端布局
    if (isMobile) {
      return (
        <div className="action-bar-mobile">
          {/* 主要操作按钮（置顶） */}
          {sortedActions.map(renderButton)}
          
          {/* 危险操作按钮（底部） */}
          {dangerActions.length > 0 && (
            <div className="action-bar-danger-section">
              {dangerActions.map(renderButton)}
            </div>
          )}
        </div>
      )
    }
    
    // 桌面端布局
    return (
      <div className="action-bar-desktop">
        {/* 危险操作区域（左侧） */}
        {dangerActions.length > 0 && (
          <div className="action-bar-danger-section">
            <Space size={8} wrap={wrap} split={split}>
              {dangerActions.map(renderButton)}
            </Space>
          </div>
        )}
        
        {/* 主要操作区域（右侧） */}
        <div className="action-bar-main-section">
          <Space 
            size={8} 
            wrap={wrap} 
            split={split}
            direction={finalDirection as any}
          >
            {sortedActions.map(renderButton)}
          </Space>
        </div>
      </div>
    )
  }
  
  const classNames = [
    'action-bar',
    `action-bar-${finalAlign}`,
    `action-bar-${finalDirection}`,
    sticky ? 'action-bar-sticky' : '',
    isMobile ? 'action-bar-mobile-layout' : '',
    isTablet ? 'action-bar-tablet-layout' : '',
    className,
  ].filter(Boolean).join(' ')
  
  return (
    <div 
      className={classNames}
      style={style}
    >
      {renderActions()}
    </div>
  )
}

export default ActionBar
export type { ActionBarProps, ActionConfig }