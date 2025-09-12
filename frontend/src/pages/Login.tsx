import React, { useState } from 'react'
import { Input, Button, Card, Typography, Space, Alert } from 'antd'
import { ProForm } from '@ant-design/pro-components'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import { useAppStore } from '@/store/app'
import { errorUtils } from '@/utils'

const { Title, Text } = Typography

interface LoginForm {
  username: string
  password: string
}

const Login: React.FC = () => {
  const [form] = ProForm.useForm<LoginForm>()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  
  const navigate = useNavigate()
  const { login } = useAuthStore()
  const { showNotification } = useAppStore()

  const handleSubmit = async (values: LoginForm) => {
    setLoading(true)
    setError('')
    
    try {
      await login(values)
      
      showNotification({
        type: 'success',
        title: '登录成功',
        message: '欢迎回来！',
      })
      
      navigate('/dashboard')
      
    } catch (err: any) {
      const errorMessage = err?.response?.data?.message || err?.message || '登录失败，请检查用户名和密码'
      setError(errorMessage)
      errorUtils.showError(err, '登录失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#f5f5f5' }}>
      <div className="w-full max-w-md">
        <Card className="shadow-lg border-0">
          <div className="text-center mb-8">
            <Title level={2} className="mb-2">
              联邦风控平台
            </Title>
            <Text type="secondary">
              Federated Risk Management Platform
            </Text>
          </div>
          
          {error && (
            <Alert
              message={error}
              type="error"
              showIcon
              className="mb-6"
              closable
              onClose={() => setError('')}
            />
          )}
          
          <ProForm
            form={form}
            name="login"
            onFinish={handleSubmit}
            layout="vertical"
            size="large"
            submitter={{
              render: () => (
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  className="w-full h-12 text-lg"
                >
                  {loading ? '登录中...' : '登录'}
                </Button>
              ),
            }}
          >
            <ProForm.Item
              name="username"
              label="用户名"
              rules={[
                { required: true, message: '请输入用户名' },
                { min: 3, message: '用户名至少3个字符' },
              ]}
            >
              <Input
                prefix={<UserOutlined />}
                placeholder="请输入用户名"
                autoComplete="username"
              />
            </ProForm.Item>
            
            <ProForm.Item
              name="password"
              label="密码"
              rules={[
                { required: true, message: '请输入密码' },
                { min: 6, message: '密码至少6个字符' },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="请输入密码"
                autoComplete="current-password"
              />
            </ProForm.Item>
          </ProForm>
          
          <div className="text-center">
            <Space direction="vertical" size="small">
              <Text type="secondary" className="text-sm">
                演示账号：admin / password
              </Text>
              <Text type="secondary" className="text-sm">
                如有问题，请联系系统管理员
              </Text>
            </Space>
          </div>
        </Card>
        
        <div className="text-center mt-8">
          <Text type="secondary" className="text-sm">
            © 2024 联邦风控平台. All rights reserved.
          </Text>
        </div>
      </div>
    </div>
  )
}

export default Login