import React, { Component, ErrorInfo, ReactNode } from 'react'
import { Result, Button, Typography, Card, Collapse, Space } from 'antd'
import {
  ExclamationCircleOutlined,
  ReloadOutlined,
  BugOutlined,
  HomeOutlined,
} from '@ant-design/icons'
import './index.less'

const { Text, Paragraph } = Typography
const { Panel } = Collapse

// 错误边界状态接口
interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
  errorId: string
}

// 错误边界属性接口
interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
  showErrorDetails?: boolean
  level?: 'page' | 'component' | 'critical'
  title?: string
  description?: string
}

/**
 * 错误边界组件
 * 用于捕获和处理React组件树中的JavaScript错误
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    }
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // 更新状态以显示错误UI
    return {
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    }
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 记录错误信息
    this.setState({
      error,
      errorInfo,
    })
    
    // 调用外部错误处理函数
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }
    
    // 发送错误报告到监控系统
    this.reportError(error, errorInfo)
  }
  
  // 错误报告方法
  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    const errorReport = {
      errorId: this.state.errorId,
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      level: this.props.level || 'component',
    }
    
    // 这里可以集成错误监控服务，如Sentry、LogRocket等
    console.error('Error Boundary caught an error:', errorReport)
    
    // 可以发送到后端API
    // fetch('/api/errors', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(errorReport)
    // })
  }
  
  // 重置错误状态
  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    })
  }
  
  // 刷新页面
  private handleRefresh = () => {
    window.location.reload()
  }
  
  // 返回首页
  private handleGoHome = () => {
    window.location.href = '/'
  }
  
  // 获取错误级别对应的样式
  private getErrorLevelConfig = () => {
    const { level = 'component' } = this.props
    
    switch (level) {
      case 'critical':
        return {
          status: 'error' as const,
          icon: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
          title: '系统发生严重错误',
          subTitle: '应用程序遇到了无法恢复的错误，请联系技术支持',
        }
      case 'page':
        return {
          status: 'error' as const,
          icon: <BugOutlined style={{ color: '#ff4d4f' }} />,
          title: '页面加载失败',
          subTitle: '当前页面遇到了错误，请尝试刷新页面或返回首页',
        }
      default:
        return {
          status: 'error' as const,
          icon: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
          title: '组件渲染失败',
          subTitle: '部分内容无法正常显示，请尝试重新加载',
        }
    }
  }
  
  // 渲染错误详情
  private renderErrorDetails = () => {
    const { error, errorInfo, errorId } = this.state
    const { showErrorDetails = process.env.NODE_ENV === 'development' } = this.props
    
    if (!showErrorDetails || !error) {
      return null
    }
    
    return (
      <Card 
        size="small" 
        style={{ marginTop: 16, textAlign: 'left' }}
        title="错误详情"
      >
        <Space direction="vertical" style={{ width: '100%' }} size={12}>
          <div>
            <Text strong>错误ID: </Text>
            <Text code>{errorId}</Text>
          </div>
          
          <Collapse size="small">
            <Panel header="错误信息" key="error">
              <Paragraph>
                <Text strong>消息: </Text>
                <Text code>{error.message}</Text>
              </Paragraph>
              {error.stack && (
                <Paragraph>
                  <Text strong>堆栈跟踪:</Text>
                  <pre style={{ 
                    background: '#f5f5f5', 
                    padding: 8, 
                    borderRadius: 4,
                    fontSize: 12,
                    overflow: 'auto',
                    maxHeight: 200
                  }}>
                    {error.stack}
                  </pre>
                </Paragraph>
              )}
            </Panel>
            
            {errorInfo?.componentStack && (
              <Panel header="组件堆栈" key="component">
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: 8, 
                  borderRadius: 4,
                  fontSize: 12,
                  overflow: 'auto',
                  maxHeight: 200
                }}>
                  {errorInfo.componentStack}
                </pre>
              </Panel>
            )}
          </Collapse>
        </Space>
      </Card>
    )
  }
  
  // 渲染操作按钮
  private renderActions = () => {
    const { level = 'component' } = this.props
    
    const actions = []
    
    // 重试按钮
    actions.push(
      <Button 
        key="retry"
        type="primary" 
        icon={<ReloadOutlined />} 
        onClick={this.handleReset}
      >
        重试
      </Button>
    )
    
    // 刷新页面按钮
    if (level === 'page' || level === 'critical') {
      actions.push(
        <Button 
          key="refresh"
          icon={<ReloadOutlined />} 
          onClick={this.handleRefresh}
        >
          刷新页面
        </Button>
      )
    }
    
    // 返回首页按钮
    if (level === 'critical') {
      actions.push(
        <Button 
          key="home"
          icon={<HomeOutlined />} 
          onClick={this.handleGoHome}
        >
          返回首页
        </Button>
      )
    }
    
    return actions
  }
  
  render() {
    const { hasError } = this.state
    const { children, fallback, title, description } = this.props
    
    if (hasError) {
      // 如果提供了自定义fallback，使用它
      if (fallback) {
        return fallback
      }
      
      const config = this.getErrorLevelConfig()
      
      return (
        <div className="error-boundary">
          <Result
            status={config.status}
            icon={config.icon}
            title={title || config.title}
            subTitle={description || config.subTitle}
            extra={this.renderActions()}
          />
          {this.renderErrorDetails()}
        </div>
      )
    }
    
    return children
  }
}

export default ErrorBoundary
export type { ErrorBoundaryProps }