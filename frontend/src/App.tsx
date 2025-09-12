import React, { useState, useEffect } from 'react'
import { RouterProvider } from 'react-router-dom'
import { Spin, App as AntdApp } from 'antd'
import { useAppStore } from '@store/app'
import { messageService } from '@/utils/messageService'
import { router } from '@/router'
import AppThemeProvider from '@/components/AppThemeProvider'
import '@/styles/global.less'

function AppContent() {
  const [loading, setLoading] = useState(true)
  const { theme, notifications } = useAppStore()
  const { message } = AntdApp.useApp()

  // 初始化message服务
  useEffect(() => {
    messageService.setMessageApi(message)
  }, [message])

  // 初始化应用
  useEffect(() => {
    const initApp = async () => {
      try {
        // 应用初始化逻辑
        console.log('应用初始化完成')
      } catch (error) {
        console.error('应用初始化失败:', error)
        message.error('应用初始化失败，请刷新页面重试')
      } finally {
        setLoading(false)
      }
    }

    initApp()
  }, [])

  // 显示通知
  useEffect(() => {
    notifications.forEach(notification => {
      const { type, title, content, duration = 4.5 } = notification
      
      switch (type) {
        case 'success':
          message.success({ content: title, duration })
          break
        case 'error':
          message.error({ content: title, duration })
          break
        case 'warning':
          message.warning({ content: title, duration })
          break
        case 'info':
        default:
          message.info({ content: title, duration })
          break
      }
    })
  }, [notifications])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spin size="large">
          <div className="mt-4">正在加载...</div>
        </Spin>
      </div>
    )
  }

  return (
    <div className={`app ${theme}`}>
      <RouterProvider router={router} />
    </div>
  )
}

function App() {
  return (
    <AppThemeProvider>
      <AntdApp>
        <AppContent />
      </AntdApp>
    </AppThemeProvider>
  )
}

export default App