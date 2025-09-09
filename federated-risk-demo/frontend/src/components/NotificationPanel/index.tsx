import React, { useState } from 'react'
import {
  Drawer,
  List,
  Badge,
  Button,
  Space,
  Typography,
  Empty,
  Tag,
  Tooltip,
} from 'antd'
import {
  BellOutlined,
  CheckOutlined,
  DeleteOutlined,
  InfoCircleOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons'
import { useAppStore, selectNotifications, selectUnreadNotifications } from '@store/app'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import 'dayjs/locale/zh-cn'

dayjs.extend(relativeTime)
dayjs.locale('zh-cn')

const { Text } = Typography

interface NotificationPanelProps {
  children: React.ReactNode
}

const NotificationPanel: React.FC<NotificationPanelProps> = ({ children }) => {
  const [visible, setVisible] = useState(false)
  
  const notifications = useAppStore(selectNotifications)
  const unreadNotifications = useAppStore(selectUnreadNotifications)
  const { markNotificationRead, clearNotifications } = useAppStore()

  // 通知类型配置
  const getNotificationConfig = (type: string) => {
    switch (type) {
      case 'info':
        return {
          icon: <InfoCircleOutlined style={{ color: '#1890ff' }} />,
          color: '#1890ff',
        }
      case 'success':
        return {
          icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
          color: '#52c41a',
        }
      case 'warning':
        return {
          icon: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
          color: '#faad14',
        }
      case 'error':
        return {
          icon: <CloseCircleOutlined style={{ color: '#ff4d4f' }} />,
          color: '#ff4d4f',
        }
      default:
        return {
          icon: <InfoCircleOutlined style={{ color: '#1890ff' }} />,
          color: '#1890ff',
        }
    }
  }

  // 处理通知点击
  const handleNotificationClick = (id: string) => {
    markNotificationRead(id)
  }

  // 删除通知
  const handleDeleteNotification = (_id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    clearNotifications()
  }

  // 标记为已读
  const handleMarkAsRead = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    markNotificationRead(id)
  }

  return (
    <>
      <div onClick={() => setVisible(true)}>
        {children}
      </div>
      
      <Drawer
        title={
          <Space>
            <BellOutlined />
            <span>通知中心</span>
            <Badge count={unreadNotifications.length} size="small" />
          </Space>
        }
        placement="right"
        width={400}
        open={visible}
        onClose={() => setVisible(false)}
        extra={
          <Space>
            {unreadNotifications.length > 0 && (
              <Button
                type="text"
                size="small"
                onClick={() => clearNotifications()}
              >
                全部已读
              </Button>
            )}
            {notifications.length > 0 && (
              <Button
                type="text"
                size="small"
                danger
                onClick={() => clearNotifications()}
              >
                清空全部
              </Button>
            )}
          </Space>
        }
      >
        {notifications.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="暂无通知"
            style={{ marginTop: 100 }}
          />
        ) : (
          <List
            dataSource={notifications}
            renderItem={(notification) => {
              const config = getNotificationConfig(notification.type)
              const isUnread = !notification.read
              
              return (
                <List.Item
                  style={{
                    padding: '16px 0',
                    cursor: 'pointer',
                    backgroundColor: isUnread ? '#f6ffed' : 'transparent',
                    borderRadius: 6,
                    marginBottom: 8,
                    border: isUnread ? '1px solid #b7eb8f' : '1px solid transparent',
                  }}
                  onClick={() => handleNotificationClick(notification.id)}
                >
                  <List.Item.Meta
                    avatar={config.icon}
                    title={
                      <Space>
                        <Text strong={isUnread}>{notification.title}</Text>
                        {isUnread && (
                          <Badge status="processing" />
                        )}
                      </Space>
                    }
                    description={
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {notification.message}
                        </Text>
                        <br />
                        <Space style={{ marginTop: 8 }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {dayjs(notification.timestamp).fromNow()}
                          </Text>
                          {notification.type && (
                            <Tag color={config.color}>
                              {notification.type}
                            </Tag>
                          )}
                        </Space>
                      </div>
                    }
                  />
                  <Space>
                    {isUnread && (
                      <Tooltip title="标记为已读">
                        <Button
                          type="text"
                          size="small"
                          icon={<CheckOutlined />}
                          onClick={(e) => handleMarkAsRead(notification.id, e)}
                        />
                      </Tooltip>
                    )}
                    <Tooltip title="删除">
                      <Button
                        type="text"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={(e) => handleDeleteNotification(notification.id, e)}
                      />
                    </Tooltip>
                  </Space>
                </List.Item>
              )
            }}
          />
        )}
      </Drawer>
    </>
  )
}

export default NotificationPanel