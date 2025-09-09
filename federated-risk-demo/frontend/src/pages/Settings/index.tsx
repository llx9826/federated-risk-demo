import React, { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Switch,
  Button,
  Select,
  InputNumber,
  Divider,
  Space,
  Typography,
  Row,
  Col,
  Alert,
  Modal,
  Tabs,
  Upload,
  message,
  Tag,
  Tooltip,
  Slider,
  Radio,
  Collapse,
  Table,
  Popconfirm,
} from 'antd'
import {
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
  ExportOutlined,
  ImportOutlined,
  SecurityScanOutlined,
  DatabaseOutlined,
  CloudOutlined,
  BellOutlined,
  UserOutlined,
  LockOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  QuestionCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  UploadOutlined,
  DeleteOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import { useAppStore } from '@store/app'
import type { UploadProps } from 'antd'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
// const { TabPane } = Tabs // 已弃用，使用items属性替代
const { Panel } = Collapse
const { TextArea } = Input

interface SystemSettings {
  // 基础设置
  systemName: string
  systemDescription: string
  timezone: string
  language: string
  theme: 'light' | 'dark' | 'auto'
  
  // 安全设置
  sessionTimeout: number
  maxLoginAttempts: number
  passwordPolicy: {
    minLength: number
    requireUppercase: boolean
    requireLowercase: boolean
    requireNumbers: boolean
    requireSpecialChars: boolean
    expirationDays: number
  }
  twoFactorAuth: boolean
  ipWhitelist: string[]
  
  // 隐私设置
  privacyBudget: number
  noiseMultiplier: number
  maxGradNorm: number
  enableAuditLog: boolean
  dataRetentionDays: number
  anonymizationLevel: 'low' | 'medium' | 'high'
  
  // 联邦学习设置
  maxParticipants: number
  minParticipants: number
  roundTimeout: number
  aggregationMethod: 'fedavg' | 'fedprox' | 'scaffold'
  clientSamplingRate: number
  
  // 通知设置
  emailNotifications: boolean
  smsNotifications: boolean
  webhookUrl: string
  notificationTypes: string[]
  
  // 存储设置
  storageType: 'local' | 's3' | 'hdfs'
  storageConfig: Record<string, any>
  backupEnabled: boolean
  backupSchedule: string
  
  // 监控设置
  metricsEnabled: boolean
  loggingLevel: 'debug' | 'info' | 'warning' | 'error'
  healthCheckInterval: number
  alertThresholds: {
    cpuUsage: number
    memoryUsage: number
    diskUsage: number
    errorRate: number
  }
}

interface ApiKey {
  id: string
  name: string
  key: string
  permissions: string[]
  createdAt: string
  lastUsed?: string
  expiresAt?: string
  status: 'active' | 'inactive'
}

const SettingsPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState<SystemSettings | null>(null)
  const [activeTab, setActiveTab] = useState('basic')
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({})
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [createApiKeyVisible, setCreateApiKeyVisible] = useState(false)
  const [newApiKeyForm] = Form.useForm()
  const { addNotification, theme, setTheme } = useAppStore()

  // 默认设置
  const defaultSettings: SystemSettings = {
    systemName: '联邦风控演示系统',
    systemDescription: '基于联邦学习的金融风险控制演示平台',
    timezone: 'Asia/Shanghai',
    language: 'zh-CN',
    theme: 'light',
    
    sessionTimeout: 30,
    maxLoginAttempts: 5,
    passwordPolicy: {
      minLength: 8,
      requireUppercase: true,
      requireLowercase: true,
      requireNumbers: true,
      requireSpecialChars: true,
      expirationDays: 90,
    },
    twoFactorAuth: false,
    ipWhitelist: [],
    
    privacyBudget: 1.0,
    noiseMultiplier: 1.1,
    maxGradNorm: 1.0,
    enableAuditLog: true,
    dataRetentionDays: 365,
    anonymizationLevel: 'medium',
    
    maxParticipants: 10,
    minParticipants: 2,
    roundTimeout: 300,
    aggregationMethod: 'fedavg',
    clientSamplingRate: 1.0,
    
    emailNotifications: true,
    smsNotifications: false,
    webhookUrl: '',
    notificationTypes: ['training_complete', 'error_alert', 'security_event'],
    
    storageType: 'local',
    storageConfig: {},
    backupEnabled: true,
    backupSchedule: '0 2 * * *',
    
    metricsEnabled: true,
    loggingLevel: 'info',
    healthCheckInterval: 60,
    alertThresholds: {
      cpuUsage: 80,
      memoryUsage: 85,
      diskUsage: 90,
      errorRate: 5,
    },
  }

  // 模拟API密钥数据
  const mockApiKeys: ApiKey[] = [
    {
      id: '1',
      name: '生产环境API',
      key: 'sk-prod-abc123...xyz789',
      permissions: ['read', 'write', 'admin'],
      createdAt: '2024-01-15T10:30:00Z',
      lastUsed: '2024-01-20T14:25:00Z',
      expiresAt: '2024-12-31T23:59:59Z',
      status: 'active',
    },
    {
      id: '2',
      name: '测试环境API',
      key: 'sk-test-def456...uvw012',
      permissions: ['read', 'write'],
      createdAt: '2024-01-10T09:15:00Z',
      lastUsed: '2024-01-19T16:45:00Z',
      status: 'active',
    },
    {
      id: '3',
      name: '只读API',
      key: 'sk-readonly-ghi789...rst345',
      permissions: ['read'],
      createdAt: '2024-01-05T11:20:00Z',
      status: 'inactive',
    },
  ]

  useEffect(() => {
    loadSettings()
    loadApiKeys()
  }, [])

  // 加载设置
  const loadSettings = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setSettings(defaultSettings)
      form.setFieldsValue(defaultSettings)
    } catch (error) {
      addNotification({
        type: 'error',
        title: '加载失败',
        message: '系统设置加载失败，请重试',
      })
    } finally {
      setLoading(false)
    }
  }

  // 加载API密钥
  const loadApiKeys = async () => {
    try {
      await new Promise(resolve => setTimeout(resolve, 500))
      setApiKeys(mockApiKeys)
    } catch (error) {
      console.error('Failed to load API keys:', error)
    }
  }

  // 保存设置
  const saveSettings = async (values: SystemSettings) => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500))
      setSettings(values)
      
      // 如果主题发生变化，更新全局主题
      if (values.theme !== settings?.theme) {
        setTheme(values.theme === 'dark' ? 'dark' : 'light')
      }
      
      addNotification({
        type: 'success',
        title: '保存成功',
        message: '系统设置已保存',
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '保存失败',
        message: '系统设置保存失败，请重试',
      })
    } finally {
      setLoading(false)
    }
  }

  // 重置设置
  const resetSettings = () => {
    Modal.confirm({
      title: '确认重置',
      content: '确定要重置所有设置到默认值吗？此操作不可撤销。',
      onOk: () => {
        form.setFieldsValue(defaultSettings)
        setSettings(defaultSettings)
        addNotification({
          type: 'success',
          title: '重置成功',
          message: '所有设置已重置到默认值',
        })
      },
    })
  }

  // 导出设置
  const exportSettings = () => {
    if (!settings) return
    
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `system_settings_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    addNotification({
      type: 'success',
      title: '导出成功',
      message: '系统设置已导出到本地',
    })
  }

  // 导入设置
  const importSettings: UploadProps['customRequest'] = ({ file, onSuccess, onError }) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const importedSettings = JSON.parse(e.target?.result as string)
        form.setFieldsValue(importedSettings)
        setSettings(importedSettings)
        onSuccess?.({})
        addNotification({
          type: 'success',
          title: '导入成功',
          message: '系统设置已导入，请检查并保存',
        })
      } catch (error) {
        onError?.(error as Error)
        addNotification({
          type: 'error',
          title: '导入失败',
          message: '设置文件格式错误',
        })
      }
    }
    reader.readAsText(file as File)
  }

  // 创建API密钥
  const createApiKey = async (values: { name: string; permissions: string[]; expiresAt?: string }) => {
    try {
      const newKey: ApiKey = {
        id: Date.now().toString(),
        name: values.name,
        key: `sk-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        permissions: values.permissions,
        createdAt: new Date().toISOString(),
        expiresAt: values.expiresAt,
        status: 'active',
      }
      
      setApiKeys(prev => [...prev, newKey])
      setCreateApiKeyVisible(false)
      newApiKeyForm.resetFields()
      
      addNotification({
        type: 'success',
        title: 'API密钥创建成功',
        message: '请妥善保存密钥，创建后将无法再次查看完整密钥',
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '创建失败',
        message: 'API密钥创建失败，请重试',
      })
    }
  }

  // 删除API密钥
  const deleteApiKey = (id: string) => {
    setApiKeys(prev => prev.filter(key => key.id !== id))
    addNotification({
      type: 'success',
      title: '删除成功',
      message: 'API密钥已删除',
    })
  }

  // 切换API密钥状态
  const toggleApiKeyStatus = (id: string) => {
    setApiKeys(prev => prev.map(key => 
      key.id === id 
        ? { ...key, status: key.status === 'active' ? 'inactive' : 'active' }
        : key
    ))
  }

  // API密钥表格列
  const apiKeyColumns = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'API密钥',
      dataIndex: 'key',
      key: 'key',
      render: (key: string, record: ApiKey) => (
        <Space>
          <Text code>
            {showApiKey[record.id] ? key : key.replace(/(?<=sk-[^-]+-).+(?=.{6})/g, '***')}
          </Text>
          <Button
            type="text"
            size="small"
            icon={showApiKey[record.id] ? <EyeInvisibleOutlined /> : <EyeOutlined />}
            onClick={() => setShowApiKey(prev => ({ ...prev, [record.id]: !prev[record.id] }))}
          />
        </Space>
      ),
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <Space>
          {permissions.map(permission => (
            <Tag key={permission} color="blue">{permission}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: ApiKey) => (
        <Switch
          checked={status === 'active'}
          onChange={() => toggleApiKeyStatus(record.id)}
          checkedChildren="启用"
          unCheckedChildren="禁用"
        />
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ApiKey) => (
        <Popconfirm
          title="确定要删除这个API密钥吗？"
          onConfirm={() => deleteApiKey(record.id)}
        >
          <Button type="text" danger icon={<DeleteOutlined />} size="small">
            删除
          </Button>
        </Popconfirm>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          <SettingOutlined style={{ marginRight: 8 }} />
          系统设置
        </Title>
        <Text type="secondary">配置系统参数和安全策略</Text>
      </div>

      {/* 操作按钮 */}
      <Card style={{ marginBottom: 16 }}>
        <Space>
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={() => form.submit()}
            loading={loading}
          >
            保存设置
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={resetSettings}
          >
            重置默认
          </Button>
          <Button
            icon={<ExportOutlined />}
            onClick={exportSettings}
          >
            导出设置
          </Button>
          <Upload
            accept=".json"
            showUploadList={false}
            customRequest={importSettings}
          >
            <Button icon={<ImportOutlined />}>
              导入设置
            </Button>
          </Upload>
        </Space>
      </Card>

      {/* 设置表单 */}
      <Form
        form={form}
        layout="vertical"
        onFinish={saveSettings}
        initialValues={defaultSettings}
      >
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={[
            {
              key: 'basic',
              label: '基础设置',
              children: (
            <Card>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="系统名称"
                    name="systemName"
                    rules={[{ required: true, message: '请输入系统名称' }]}
                  >
                    <Input placeholder="请输入系统名称" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="时区"
                    name="timezone"
                    rules={[{ required: true, message: '请选择时区' }]}
                  >
                    <Select placeholder="请选择时区">
                      <Option value="Asia/Shanghai">Asia/Shanghai (UTC+8)</Option>
                      <Option value="UTC">UTC (UTC+0)</Option>
                      <Option value="America/New_York">America/New_York (UTC-5)</Option>
                      <Option value="Europe/London">Europe/London (UTC+0)</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="语言"
                    name="language"
                    rules={[{ required: true, message: '请选择语言' }]}
                  >
                    <Select placeholder="请选择语言">
                      <Option value="zh-CN">简体中文</Option>
                      <Option value="en-US">English</Option>
                      <Option value="ja-JP">日本語</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="主题"
                    name="theme"
                    rules={[{ required: true, message: '请选择主题' }]}
                  >
                    <Radio.Group>
                      <Radio value="light">浅色</Radio>
                      <Radio value="dark">深色</Radio>
                      <Radio value="auto">自动</Radio>
                    </Radio.Group>
                  </Form.Item>
                </Col>
                <Col span={24}>
                  <Form.Item
                    label="系统描述"
                    name="systemDescription"
                  >
                    <TextArea rows={3} placeholder="请输入系统描述" />
                  </Form.Item>
                </Col>
              </Row>
            </Card>
              )
            },
            {
              key: 'security',
              label: '安全设置',
              children: (
            <Card>
              <Collapse defaultActiveKey={['session', 'password']}>
                <Panel header="会话管理" key="session">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="会话超时时间（分钟）"
                        name="sessionTimeout"
                        rules={[{ required: true, message: '请输入会话超时时间' }]}
                      >
                        <InputNumber min={5} max={480} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="最大登录尝试次数"
                        name="maxLoginAttempts"
                        rules={[{ required: true, message: '请输入最大登录尝试次数' }]}
                      >
                        <InputNumber min={3} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={24}>
                      <Form.Item
                        label="启用双因子认证"
                        name="twoFactorAuth"
                        valuePropName="checked"
                      >
                        <Switch checkedChildren="启用" unCheckedChildren="禁用" />
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>
                
                <Panel header="密码策略" key="password">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="最小长度"
                        name={['passwordPolicy', 'minLength']}
                        rules={[{ required: true, message: '请输入最小长度' }]}
                      >
                        <InputNumber min={6} max={32} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="密码有效期（天）"
                        name={['passwordPolicy', 'expirationDays']}
                        rules={[{ required: true, message: '请输入密码有效期' }]}
                      >
                        <InputNumber min={30} max={365} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="要求大写字母"
                        name={['passwordPolicy', 'requireUppercase']}
                        valuePropName="checked"
                      >
                        <Switch checkedChildren="是" unCheckedChildren="否" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="要求小写字母"
                        name={['passwordPolicy', 'requireLowercase']}
                        valuePropName="checked"
                      >
                        <Switch checkedChildren="是" unCheckedChildren="否" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="要求数字"
                        name={['passwordPolicy', 'requireNumbers']}
                        valuePropName="checked"
                      >
                        <Switch checkedChildren="是" unCheckedChildren="否" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="要求特殊字符"
                        name={['passwordPolicy', 'requireSpecialChars']}
                        valuePropName="checked"
                      >
                        <Switch checkedChildren="是" unCheckedChildren="否" />
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>
              </Collapse>
            </Card>
              )
            },
            {
              key: 'privacy',
              label: '隐私设置',
              children: (
            <Card>
              <Alert
                message="隐私保护配置"
                description="这些设置影响差分隐私和数据保护的强度，请谨慎调整。"
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label={
                      <Space>
                        隐私预算 (ε)
                        <Tooltip title="较小的值提供更强的隐私保护，但可能影响模型性能">
                          <QuestionCircleOutlined />
                        </Tooltip>
                      </Space>
                    }
                    name="privacyBudget"
                    rules={[{ required: true, message: '请输入隐私预算' }]}
                  >
                    <Slider
                      min={0.1}
                      max={10.0}
                      step={0.1}
                      marks={{
                        0.1: '0.1',
                        1.0: '1.0',
                        5.0: '5.0',
                        10.0: '10.0',
                      }}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label={
                      <Space>
                        噪声倍数
                        <Tooltip title="控制添加到梯度的噪声量">
                          <QuestionCircleOutlined />
                        </Tooltip>
                      </Space>
                    }
                    name="noiseMultiplier"
                    rules={[{ required: true, message: '请输入噪声倍数' }]}
                  >
                    <InputNumber min={0.1} max={5.0} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="最大梯度范数"
                    name="maxGradNorm"
                    rules={[{ required: true, message: '请输入最大梯度范数' }]}
                  >
                    <InputNumber min={0.1} max={10.0} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="匿名化级别"
                    name="anonymizationLevel"
                    rules={[{ required: true, message: '请选择匿名化级别' }]}
                  >
                    <Select placeholder="请选择匿名化级别">
                      <Option value="low">低</Option>
                      <Option value="medium">中</Option>
                      <Option value="high">高</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="数据保留天数"
                    name="dataRetentionDays"
                    rules={[{ required: true, message: '请输入数据保留天数' }]}
                  >
                    <InputNumber min={30} max={3650} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="启用审计日志"
                    name="enableAuditLog"
                    valuePropName="checked"
                  >
                    <Switch checkedChildren="启用" unCheckedChildren="禁用" />
                  </Form.Item>
                </Col>
              </Row>
            </Card>
              )
            },
            {
              key: 'federated',
              label: '联邦学习',
              children: (
            <Card>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="最大参与方数量"
                    name="maxParticipants"
                    rules={[{ required: true, message: '请输入最大参与方数量' }]}
                  >
                    <InputNumber min={2} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="最小参与方数量"
                    name="minParticipants"
                    rules={[{ required: true, message: '请输入最小参与方数量' }]}
                  >
                    <InputNumber min={2} max={10} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="轮次超时时间（秒）"
                    name="roundTimeout"
                    rules={[{ required: true, message: '请输入轮次超时时间' }]}
                  >
                    <InputNumber min={60} max={3600} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="聚合方法"
                    name="aggregationMethod"
                    rules={[{ required: true, message: '请选择聚合方法' }]}
                  >
                    <Select placeholder="请选择聚合方法">
                      <Option value="fedavg">FedAvg</Option>
                      <Option value="fedprox">FedProx</Option>
                      <Option value="scaffold">SCAFFOLD</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="客户端采样率"
                    name="clientSamplingRate"
                    rules={[{ required: true, message: '请输入客户端采样率' }]}
                  >
                    <Slider
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      marks={{
                        0.1: '10%',
                        0.5: '50%',
                        1.0: '100%',
                      }}
                    />
                  </Form.Item>
                </Col>
              </Row>
            </Card>
              )
            },
            {
              key: 'apikeys',
              label: 'API密钥',
              children: (
            <Card
              title="API密钥管理"
              extra={
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setCreateApiKeyVisible(true)}
                >
                  创建密钥
                </Button>
              }
            >
              <Table
                dataSource={apiKeys}
                columns={apiKeyColumns}
                rowKey="id"
                pagination={false}
                size="small"
              />
            </Card>
              )
            }
          ]}
        />
      </Form>

      {/* 创建API密钥弹窗 */}
      <Modal
        title="创建API密钥"
        open={createApiKeyVisible}
        onCancel={() => setCreateApiKeyVisible(false)}
        onOk={() => newApiKeyForm.submit()}
        width={600}
      >
        <Form
          form={newApiKeyForm}
          layout="vertical"
          onFinish={createApiKey}
        >
          <Form.Item
            label="密钥名称"
            name="name"
            rules={[{ required: true, message: '请输入密钥名称' }]}
          >
            <Input placeholder="请输入密钥名称" />
          </Form.Item>
          
          <Form.Item
            label="权限"
            name="permissions"
            rules={[{ required: true, message: '请选择权限' }]}
          >
            <Select mode="multiple" placeholder="请选择权限">
              <Option value="read">读取</Option>
              <Option value="write">写入</Option>
              <Option value="admin">管理</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            label="过期时间（可选）"
            name="expiresAt"
          >
            <Input type="datetime-local" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default SettingsPage