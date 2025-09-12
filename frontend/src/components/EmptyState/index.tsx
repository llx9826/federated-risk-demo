import React from 'react'
import { Empty, Button, Space, Typography } from 'antd'
import {
  InboxOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  PlusOutlined,
  SearchOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
} from '@ant-design/icons'
import './index.less'

const { Text } = Typography

// 空态类型枚举
enum EmptyStateType {
  NO_DATA = 'no-data',
  NO_SEARCH_RESULTS = 'no-search-results',
  ERROR = 'error',
  NETWORK_ERROR = 'network-error',
  PERMISSION_DENIED = 'permission-denied',
  LOADING_FAILED = 'loading-failed',
  NO_CONTENT = 'no-content',
  MAINTENANCE = 'maintenance',
}

// 空态配置接口
interface EmptyStateConfig {
  icon: React.ReactNode
  title: string
  description?: string
  image?: string
}

// 空态组件属性
interface EmptyStateProps {
  type?: EmptyStateType
  title?: string
  description?: string
  icon?: React.ReactNode
  image?: string
  actions?: React.ReactNode[]
  onRetry?: () => void
  onCreate?: () => void
  onRefresh?: () => void
  loading?: boolean
  className?: string
  style?: React.CSSProperties
  size?: 'small' | 'default' | 'large'
}

// 预定义空态配置
const EMPTY_STATE_CONFIGS: Record<EmptyStateType, EmptyStateConfig> = {
  [EmptyStateType.NO_DATA]: {
    icon: <InboxOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />,
    title: '暂无数据',
    description: '当前没有可显示的数据',
  },
  [EmptyStateType.NO_SEARCH_RESULTS]: {
    icon: <SearchOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />,
    title: '无搜索结果',
    description: '未找到符合条件的数据，请尝试调整搜索条件',
  },
  [EmptyStateType.ERROR]: {
    icon: <ExclamationCircleOutlined style={{ fontSize: 64, color: '#ff4d4f' }} />,
    title: '加载失败',
    description: '数据加载时发生错误，请稍后重试',
  },
  [EmptyStateType.NETWORK_ERROR]: {
    icon: <CloudServerOutlined style={{ fontSize: 64, color: '#ff4d4f' }} />,
    title: '网络连接失败',
    description: '请检查网络连接后重试',
  },
  [EmptyStateType.PERMISSION_DENIED]: {
    icon: <ExclamationCircleOutlined style={{ fontSize: 64, color: '#faad14' }} />,
    title: '权限不足',
    description: '您没有访问此内容的权限',
  },
  [EmptyStateType.LOADING_FAILED]: {
    icon: <DatabaseOutlined style={{ fontSize: 64, color: '#ff4d4f' }} />,
    title: '数据加载失败',
    description: '服务器响应异常，请稍后重试',
  },
  [EmptyStateType.NO_CONTENT]: {
    icon: <FileTextOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />,
    title: '暂无内容',
    description: '还没有任何内容，快来创建第一个吧',
  },
  [EmptyStateType.MAINTENANCE]: {
    icon: <CloudServerOutlined style={{ fontSize: 64, color: '#faad14' }} />,
    title: '系统维护中',
    description: '系统正在维护升级，请稍后访问',
  },
}

/**
 * 空态组件
 * 用于显示各种空状态、错误状态和加载失败状态
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  type = EmptyStateType.NO_DATA,
  title,
  description,
  icon,
  image,
  actions,
  onRetry,
  onCreate,
  onRefresh,
  loading = false,
  className,
  style,
  size = 'default',
}) => {
  const config = EMPTY_STATE_CONFIGS[type]
  
  // 获取最终显示的配置
  const finalIcon = icon || config.icon
  const finalTitle = title || config.title
  const finalDescription = description || config.description
  const finalImage = image || config.image
  
  // 生成默认操作按钮
  const getDefaultActions = () => {
    const defaultActions: React.ReactNode[] = []
    
    // 重试按钮
    if (onRetry && [EmptyStateType.ERROR, EmptyStateType.NETWORK_ERROR, EmptyStateType.LOADING_FAILED].includes(type)) {
      defaultActions.push(
        <Button
          key="retry"
          type="primary"
          icon={<ReloadOutlined />}
          onClick={onRetry}
          loading={loading}
        >
          重试
        </Button>
      )
    }
    
    // 刷新按钮
    if (onRefresh && type !== EmptyStateType.PERMISSION_DENIED) {
      defaultActions.push(
        <Button
          key="refresh"
          icon={<ReloadOutlined />}
          onClick={onRefresh}
          loading={loading}
        >
          刷新
        </Button>
      )
    }
    
    // 创建按钮
    if (onCreate && [EmptyStateType.NO_DATA, EmptyStateType.NO_CONTENT].includes(type)) {
      defaultActions.push(
        <Button
          key="create"
          type="primary"
          icon={<PlusOutlined />}
          onClick={onCreate}
        >
          立即创建
        </Button>
      )
    }
    
    return defaultActions
  }
  
  const finalActions = actions || getDefaultActions()
  
  // 根据尺寸调整样式
  const getSizeStyle = () => {
    switch (size) {
      case 'small':
        return { padding: '24px 16px' }
      case 'large':
        return { padding: '80px 24px' }
      default:
        return { padding: '48px 24px' }
    }
  }
  
  return (
    <div 
      className={`empty-state empty-state-${size} ${className || ''}`}
      style={{ ...getSizeStyle(), ...style }}
    >
      <Empty
        image={finalImage || finalIcon}
        description={
          <Space direction="vertical" size={8}>
            <Text strong style={{ fontSize: size === 'large' ? 18 : 16 }}>
              {finalTitle}
            </Text>
            {finalDescription && (
              <Text type="secondary" style={{ fontSize: size === 'small' ? 12 : 14 }}>
                {finalDescription}
              </Text>
            )}
          </Space>
        }
      >
        {finalActions.length > 0 && (
          <Space size={12} wrap>
            {finalActions}
          </Space>
        )}
      </Empty>
    </div>
  )
}

export default EmptyState
export { EmptyStateType }
export type { EmptyStateProps }