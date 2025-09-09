import React, { useEffect } from 'react'
import { Layout, Menu, Avatar, Dropdown, Badge, Button, Space, Tooltip } from 'antd'
import {
  DashboardOutlined,
  ShareAltOutlined,
  SafetyCertificateOutlined,
  ExperimentOutlined,
  CloudServerOutlined,
  AuditOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  GithubOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons'
import { useLocation, useNavigate } from 'react-router-dom'
import { useAppStore, selectUnreadNotifications } from '@store/app'
import { checkSystemHealth } from '@services/api'
import SystemStatus from '@components/SystemStatus'
import NotificationPanel from '@components/NotificationPanel'

const { Header, Sider, Content } = Layout

interface AppLayoutProps {
  children: React.ReactNode
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  
  const {
    sidebarCollapsed,
    setSidebarCollapsed,
    user,
    setUser,
    updateSystemStatus,
    settings,
  } = useAppStore()
  
  const unreadNotifications = useAppStore(selectUnreadNotifications)

  // 菜单配置
  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '仪表板',
    },
    {
      key: '/psi',
      icon: <ShareAltOutlined />,
      label: 'PSI对齐',
    },
    {
      key: '/consent',
      icon: <SafetyCertificateOutlined />,
      label: '同意管理',
    },
    {
      key: '/training',
      icon: <ExperimentOutlined />,
      label: '联邦训练',
    },
    {
      key: '/inference',
      icon: <CloudServerOutlined />,
      label: '模型推理',
    },
    {
      key: '/audit',
      icon: <AuditOutlined />,
      label: '审计日志',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
  ]

  // 用户菜单
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '个人资料',
      onClick: () => {
        // TODO: 打开个人资料页面
      },
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '设置',
      onClick: () => navigate('/settings'),
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: () => {
        setUser(null)
        navigate('/dashboard')
      },
    },
  ]

  // 检查系统健康状态
  const checkHealth = async () => {
    try {
      const status = await checkSystemHealth()
      updateSystemStatus(status)
    } catch (error) {
      console.error('Health check failed:', error)
    }
  }

  // 定期检查系统状态
  useEffect(() => {
    checkHealth()
    
    if (settings.autoRefresh) {
      const interval = setInterval(checkHealth, settings.refreshInterval * 1000)
      return () => clearInterval(interval)
    }
  }, [settings.autoRefresh, settings.refreshInterval])

  // 处理菜单点击
  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key)
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 侧边栏 */}
      <Sider
        trigger={null}
        collapsible
        collapsed={sidebarCollapsed}
        width={240}
        style={{
          background: '#001529',
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          zIndex: 100,
        }}
      >
        {/* Logo */}
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: sidebarCollapsed ? 'center' : 'flex-start',
            padding: sidebarCollapsed ? 0 : '0 24px',
            color: '#fff',
            fontSize: 18,
            fontWeight: 600,
            borderBottom: '1px solid #1f1f1f',
          }}
        >
          {sidebarCollapsed ? (
            <ShareAltOutlined style={{ fontSize: 24 }} />
          ) : (
            <>
              <ShareAltOutlined style={{ marginRight: 12, fontSize: 24 }} />
              联邦风控
            </>
          )}
        </div>

        {/* 菜单 */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ borderRight: 0 }}
        />
      </Sider>

      {/* 主布局 */}
      <Layout
        style={{
          marginLeft: sidebarCollapsed ? 80 : 240,
          transition: 'margin-left 0.2s',
        }}
      >
        {/* 顶部导航 */}
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid #f0f0f0',
            position: 'sticky',
            top: 0,
            zIndex: 99,
          }}
        >
          {/* 左侧 */}
          <Space>
            <Button
              type="text"
              icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              style={{ fontSize: 16 }}
            />
            
            <SystemStatus />
          </Space>

          {/* 右侧 */}
          <Space size="middle">
            {/* 帮助 */}
            <Tooltip title="帮助文档">
              <Button
                type="text"
                icon={<QuestionCircleOutlined />}
                onClick={() => window.open('https://github.com/your-repo', '_blank')}
              />
            </Tooltip>

            {/* GitHub */}
            <Tooltip title="GitHub">
              <Button
                type="text"
                icon={<GithubOutlined />}
                onClick={() => window.open('https://github.com/your-repo', '_blank')}
              />
            </Tooltip>

            {/* 通知 */}
            <NotificationPanel>
              <Badge count={unreadNotifications.length} size="small">
                <Button
                  type="text"
                  icon={<BellOutlined />}
                  style={{ fontSize: 16 }}
                />
              </Badge>
            </NotificationPanel>

            {/* 用户菜单 */}
            <Dropdown
              menu={{ items: userMenuItems }}
              placement="bottomRight"
              trigger={['click']}
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar
                  size="small"
                  icon={<UserOutlined />}
                  src={user?.avatar}
                />
                {!sidebarCollapsed && (
                  <span style={{ color: '#262626' }}>
                    {user?.name || '演示用户'}
                  </span>
                )}
              </Space>
            </Dropdown>
          </Space>
        </Header>

        {/* 内容区域 */}
        <Content>
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default AppLayout