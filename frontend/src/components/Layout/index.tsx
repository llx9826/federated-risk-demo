import React from 'react'
import { Layout as AntLayout, Menu, Badge } from 'antd'
import { BellOutlined } from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAppStore } from '@/store/app'

const { Header, Sider, Content } = AntLayout

interface AppLayoutProps {
  children: React.ReactNode
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { notifications } = useAppStore()

  const menuItems = [
    {
      key: '/dashboard',
      label: '仪表板',
      onClick: () => navigate('/dashboard')
    },
    {
      key: '/consent',
      label: '同意管理',
      onClick: () => navigate('/consent')
    },
    {
      key: '/psi',
      label: 'PSI对齐',
      onClick: () => navigate('/psi')
    }
  ]

  const unreadCount = notifications.filter(n => !n.read).length

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="light">
        <div className="p-4">
          <h2 className="text-lg font-bold">联邦风控</h2>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
        />
      </Sider>
      <AntLayout>
        <Header className="bg-white px-6 flex justify-between items-center shadow-sm">
          <div className="text-lg font-medium">
            联邦风控管理平台
          </div>
          <div className="flex items-center space-x-4">
            <Badge count={unreadCount}>
              <BellOutlined className="text-lg cursor-pointer" />
            </Badge>
          </div>
        </Header>
        <Content className="p-6">
          {children}
        </Content>
      </AntLayout>
    </AntLayout>
  )
}

export default AppLayout