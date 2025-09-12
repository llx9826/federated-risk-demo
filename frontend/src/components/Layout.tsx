import React, { useState } from 'react'
import { Layout as AntLayout, Menu, Avatar, Dropdown, Button, Space, Breadcrumb } from 'antd'
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  DashboardOutlined,
  SafetyOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  DeploymentUnitOutlined,
  AuditOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import { useAppStore } from '@/store/app'

const { Header, Sider, Content } = AntLayout

const Layout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false)
  const [openKeys, setOpenKeys] = useState<string[]>([])
  const navigate = useNavigate()
  const location = useLocation()
  
  const { user, logout } = useAuthStore()
  const { notifications } = useAppStore()

  // 菜单配置
  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '仪表盘',
    },
    {
      key: '/consent',
      icon: <SafetyOutlined />,
      label: '同意管理',
    },
    {
      key: '/psi',
      icon: <DatabaseOutlined />,
      label: '数据对齐',
    },
    {
      key: '/federated',
      icon: <CloudServerOutlined />,
      label: '联邦训练',
    },
    {
      key: '/models',
      icon: <DeploymentUnitOutlined />,
      label: '模型服务',
    },
    {
      key: '/audit',
      icon: <AuditOutlined />,
      label: '审计监控',
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
      onClick: () => {
        logout()
        navigate('/login')
      },
    },
  ]

  // 处理菜单点击
  const handleMenuClick = ({ key }: { key: string }) => {
    // 只对实际路由路径进行导航，忽略父级菜单项
    if (key.startsWith('/')) {
      navigate(key)
    }
  }

  // 处理菜单展开/收起
  const handleOpenChange = (keys: string[]) => {
    setOpenKeys(keys)
  }

  // 获取当前选中的菜单项
  const getSelectedKeys = () => {
    return [location.pathname]
  }

  // 获取展开的菜单项
  const getOpenKeys = () => {
    // 由于菜单已简化为单级，不需要展开逻辑
    return []
  }

  // 生成面包屑
  const getBreadcrumbs = () => {
    const path = location.pathname
    const breadcrumbs = [{ title: '首页' }]
    
    if (path === '/dashboard') {
      breadcrumbs.push({ title: '仪表盘' })
    } else if (path === '/consent') {
      breadcrumbs.push({ title: '同意管理' })
    } else if (path === '/psi') {
      breadcrumbs.push({ title: '数据对齐' })
    } else if (path === '/federated') {
      breadcrumbs.push({ title: '联邦训练' })
    } else if (path === '/models') {
      breadcrumbs.push({ title: '模型服务' })
    } else if (path === '/audit') {
      breadcrumbs.push({ title: '审计监控' })
    } else if (path === '/settings') {
      breadcrumbs.push({ title: '系统设置' })
    }
    
    return breadcrumbs
  }

  return (
    <AntLayout className="min-h-screen">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={256}
        className="bg-white shadow-md border-r border-gray-100"
        style={{
          position: 'fixed',
          height: '100vh',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 100,
        }}
      >
        <div className="h-16 flex items-center justify-center border-b border-gray-100" style={{ backgroundColor: '#1677ff' }}>
          <div className="text-lg font-bold text-white">
            {collapsed ? 'FRP' : '联邦风控平台'}
          </div>
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={getSelectedKeys()}
          openKeys={openKeys.length > 0 ? openKeys : getOpenKeys()}
          items={menuItems}
          onClick={handleMenuClick}
          onOpenChange={handleOpenChange}
          className="border-r-0 h-full"
          style={{
            borderRight: 'none',
            fontSize: '14px',
            fontWeight: 500,
          }}
        />
      </Sider>
      
      <AntLayout style={{ marginLeft: collapsed ? 80 : 256, transition: 'margin-left 0.2s' }}>
        <Header className="bg-white px-6 shadow-sm flex items-center justify-between border-b border-gray-100" style={{ position: 'fixed', top: 0, right: 0, left: collapsed ? 80 : 256, zIndex: 99, transition: 'left 0.2s' }}>
          <div className="flex items-center">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              className="text-lg hover:bg-gray-100"
              style={{ color: '#666' }}
            />
          </div>
          
          <div className="flex items-center space-x-4">
            <Button
              type="text"
              icon={<BellOutlined />}
              className="relative"
            >
              {notifications.length > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {notifications.length}
                </span>
              )}
            </Button>
            
            <Dropdown
              menu={{ items: userMenuItems }}
              placement="bottomRight"
              arrow
            >
              <Space className="cursor-pointer hover:bg-gray-50 px-2 py-1 rounded">
                <Avatar
                  size="small"
                  icon={<UserOutlined />}
                  src={user?.avatar}
                />
                <span className="text-sm font-medium">
                  {user?.username || '用户'}
                </span>
              </Space>
            </Dropdown>
          </div>
        </Header>
        
        <Content className="bg-gray-50" style={{ marginTop: 64, minHeight: 'calc(100vh - 64px)' }}>
          <div className="px-6 py-4 bg-white border-b border-gray-100">
            <Breadcrumb items={getBreadcrumbs()} className="text-sm" />
          </div>
          
          <div className="p-6">
            <Outlet />
          </div>
        </Content>
      </AntLayout>
    </AntLayout>
  )
}

export default Layout