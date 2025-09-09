import React from 'react'
import { Space, Tag, Popover, Button, Divider } from 'antd'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import { useAppStore, selectSystemStatus } from '@store/app'
import { checkSystemHealth } from '@services/api'
import dayjs from 'dayjs'

const SystemStatus: React.FC = () => {
  const systemStatus = useAppStore(selectSystemStatus)
  const { updateSystemStatus } = useAppStore()

  // 服务状态配置
  const serviceConfig = {
    psiService: { name: 'PSI服务', port: '7001' },
    consentService: { name: '同意管理', port: '7002' },
    trainService: { name: '训练服务', port: '7003' },
    servingService: { name: '推理服务', port: '7004' },
  }

  // 获取状态图标和颜色
  const getStatusConfig = (status: 'online' | 'offline' | 'error') => {
    switch (status) {
      case 'online':
        return {
          icon: <CheckCircleOutlined />,
          color: 'success',
          text: '在线',
        }
      case 'offline':
        return {
          icon: <ExclamationCircleOutlined />,
          color: 'warning',
          text: '离线',
        }
      case 'error':
        return {
          icon: <CloseCircleOutlined />,
          color: 'error',
          text: '错误',
        }
      default:
        return {
          icon: <InfoCircleOutlined />,
          color: 'default',
          text: '未知',
        }
    }
  }

  // 计算整体状态
  const getOverallStatus = () => {
    const statuses = Object.values(systemStatus).filter(status => 
      typeof status === 'string'
    ) as string[]
    
    if (statuses.every(status => status === 'online')) {
      return 'online'
    } else if (statuses.some(status => status === 'error')) {
      return 'error'
    } else {
      return 'offline'
    }
  }

  // 刷新系统状态
  const handleRefresh = async () => {
    try {
      const status = await checkSystemHealth()
      updateSystemStatus(status)
    } catch (error) {
      console.error('Failed to refresh system status:', error)
    }
  }

  const overallStatus = getOverallStatus()
  const overallConfig = getStatusConfig(overallStatus)

  // 详细状态内容
  const statusContent = (
    <div style={{ width: 300 }}>
      <div style={{ marginBottom: 16 }}>
        <Space>
          <strong>系统状态概览</strong>
          <Button
            type="text"
            size="small"
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
          >
            刷新
          </Button>
        </Space>
      </div>
      
      <Divider style={{ margin: '12px 0' }} />
      
      {Object.entries(serviceConfig).map(([key, config]) => {
        const status = systemStatus[key as keyof typeof systemStatus] as string
        const statusConfig = getStatusConfig(status as any)
        
        return (
          <div
            key={key}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 8,
            }}
          >
            <Space>
              <span style={{ fontSize: 12, color: '#666' }}>:{config.port}</span>
              <span>{config.name}</span>
            </Space>
            <Tag
              icon={statusConfig.icon}
              color={statusConfig.color as any}
              style={{ margin: 0 }}
            >
              {statusConfig.text}
            </Tag>
          </div>
        )
      })}
      
      <Divider style={{ margin: '12px 0' }} />
      
      <div style={{ fontSize: 12, color: '#999' }}>
        最后检查: {systemStatus.lastCheck ? 
          dayjs(systemStatus.lastCheck).format('HH:mm:ss') : 
          '未检查'
        }
      </div>
    </div>
  )

  return (
    <Popover
      content={statusContent}
      title={null}
      trigger="hover"
      placement="bottomLeft"
    >
      <div style={{ cursor: 'pointer' }}>
        <Space size="small">
          <span style={{ fontSize: 12, color: '#666' }}>系统状态:</span>
          <Tag
            icon={overallConfig.icon}
            color={overallConfig.color as any}
            style={{ margin: 0, cursor: 'pointer' }}
          >
            {overallConfig.text}
          </Tag>
        </Space>
      </div>
    </Popover>
  )
}

export default SystemStatus