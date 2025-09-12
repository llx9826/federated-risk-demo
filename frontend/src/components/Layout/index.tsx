import React from 'react'
import { Badge, Avatar, Dropdown, Space, Input } from 'antd'
import './index.less'
import { 
  BellOutlined, 
  DashboardOutlined,
  SafetyCertificateOutlined,
  ShareAltOutlined,
  ExperimentOutlined,
  CloudServerOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  SearchOutlined,
  EnvironmentOutlined
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import { ProLayout, PageContainer } from '@ant-design/pro-components'
import { useAppStore } from '@/store/app'

interface AppLayoutProps {
  children: React.ReactNode
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { notifications, pageTitle, breadcrumbs } = useAppStore()

  // ProLayout路由配置
  const route = {
    path: '/',
    routes: [
      {
        path: '/dashboard',
        name: '仪表板',
        icon: <DashboardOutlined />,
      },
      {
        path: '/consent',
        name: '同意管理',
        icon: <SafetyCertificateOutlined />,
      },
      {
        path: '/psi',
        name: 'PSI对齐',
        icon: <ShareAltOutlined />,
      },
      {
        path: '/federated',
        name: '模型训练',
        icon: <ExperimentOutlined />,
      },
      {
        path: '/models',
        name: '推理服务',
        icon: <CloudServerOutlined />,
      },
      {
        path: '/settings',
        name: '系统设置',
        icon: <SettingOutlined />,
      },
    ],
  }

  const unreadCount = notifications.filter(n => !n.read).length

  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '账户设置',
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
    },
  ]

  return (
    <ProLayout
      title="联邦风控"
      logo={(
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <span className="text-white font-bold text-sm">联</span>
        </div>
      )}
      route={route}
      location={{
        pathname: location.pathname,
      }}
      menuItemRender={(item, dom) => (
        <div
          onClick={() => {
            navigate(item.path || '/')
          }}
          style={{ cursor: 'pointer' }}
        >
          {dom}
        </div>
      )}
      headerContentRender={() => (
        <div className="flex items-center space-x-4">
          <Input.Search
            placeholder="全局搜索..."
            prefix={<SearchOutlined />}
            style={{ width: 300 }}
            allowClear
          />
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-green-100 text-green-700">
            <EnvironmentOutlined />
            <span className="text-sm font-medium">生产环境</span>
          </div>
        </div>
      )}
      actionsRender={() => [
        <Badge key="notification" count={unreadCount}>
          <BellOutlined className="text-lg cursor-pointer hover:text-blue-500 transition-colors" />
        </Badge>,
        <Dropdown key="user" menu={{ items: userMenuItems }} placement="bottomRight">
          <Avatar 
            size="default" 
            icon={<UserOutlined />} 
            className="cursor-pointer"
            style={{ backgroundColor: 'var(--ant-color-primary)' }}
          />
        </Dropdown>,
      ]}
      layout="side"
      fixSiderbar
      fixedHeader
      contentWidth="Fluid"
      siderWidth={220}
      style={{
        minHeight: '100vh',
      }}
    >
      <PageContainer
        title={pageTitle}
        breadcrumb={{
          items: breadcrumbs.map(item => ({
            title: item.title,
            path: item.path,
          })),
        }}
        content={null}
        extra={null}
      >
        <div className="animate-fadeInUp">
          {children}
        </div>
      </PageContainer>
    </ProLayout>
  )
}

export default AppLayout