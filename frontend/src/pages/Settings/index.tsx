import React, { useState, useEffect } from 'react'
import {
  Card,
  Switch,
  Button,
  Space,
  Typography,
  Row,
  Col,
  Alert,
  Modal,
  Tabs,
  Tag,
  Tooltip,
  Popconfirm,
  message,
} from 'antd'
import {
  PageContainer,
  ProForm,
  ProFormText,
  ProFormTextArea,
  ProFormSelect,
  ProFormDigit,
  ProFormSwitch,
  ProFormSlider,
  ProFormRadio,
  ProFormList,
  ProTable,
} from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import {
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
  ExportOutlined,
  ImportOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
  QuestionCircleOutlined,
  DeleteOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import { formatDate } from '@/utils'
import dayjs from 'dayjs'

const { Title, Text } = Typography

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
  const [form] = ProForm.useForm()
  const [apiKeyForm] = ProForm.useForm()
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState<SystemSettings | null>(null)
  const [activeTab, setActiveTab] = useState('basic')
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({})
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [createApiKeyVisible, setCreateApiKeyVisible] = useState(false)

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
      createdAt: dayjs().subtract(5, 'day').toISOString(),
      lastUsed: dayjs().subtract(1, 'hour').toISOString(),
      expiresAt: dayjs().add(11, 'month').toISOString(),
      status: 'active',
    },
    {
      id: '2',
      name: '测试环境API',
      key: 'sk-test-def456...uvw012',
      permissions: ['read', 'write'],
      createdAt: dayjs().subtract(10, 'day').toISOString(),
      lastUsed: dayjs().subtract(2, 'day').toISOString(),
      status: 'active',
    },
    {
      id: '3',
      name: '只读API',
      key: 'sk-readonly-ghi789...rst345',
      permissions: ['read'],
      createdAt: dayjs().subtract(15, 'day').toISOString(),
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
      await new Promise(resolve => setTimeout(resolve, 1000))
      setSettings(defaultSettings)
      form.setFieldsValue(defaultSettings)
    } catch (error) {
      message.error('系统设置加载失败，请重试')
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
      await new Promise(resolve => setTimeout(resolve, 1500))
      setSettings(values)
      message.success('系统设置已保存')
    } catch (error) {
      message.error('系统设置保存失败，请重试')
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
        message.success('所有设置已重置到默认值')
      },
    })
  }

  // 导出设置
  const exportSettings = () => {
    if (!settings) return
    
    const dataStr = JSON.stringify(settings, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `settings_${dayjs().format('YYYY-MM-DD_HH-mm-ss')}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    message.success('设置已导出到本地文件')
  }

  // 创建API密钥
  const createApiKey = async (values: any) => {
    try {
      const newKey: ApiKey = {
        id: Date.now().toString(),
        name: values.name,
        key: `sk-${values.name.toLowerCase().replace(/\s+/g, '-')}-${Math.random().toString(36).substr(2, 9)}...${Math.random().toString(36).substr(2, 6)}`,
        permissions: values.permissions,
        createdAt: dayjs().toISOString(),
        status: 'active',
      }
      
      setApiKeys(prev => [newKey, ...prev])
      setCreateApiKeyVisible(false)
      apiKeyForm.resetFields()
      
      message.success('API密钥创建成功')
    } catch (error) {
      message.error('API密钥创建失败')
    }
  }

  // 删除API密钥
  const deleteApiKey = (keyId: string) => {
    setApiKeys(prev => prev.filter(key => key.id !== keyId))
    message.success('API密钥已删除')
  }

  // 切换API密钥状态
  const toggleApiKeyStatus = (keyId: string) => {
    setApiKeys(prev => prev.map(key => {
      if (key.id === keyId) {
        return { ...key, status: key.status === 'active' ? 'inactive' : 'active' }
      }
      return key
    }))
  }

  // API密钥表格列
  const apiKeyColumns: ProColumns<ApiKey>[] = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'API密钥',
      dataIndex: 'key',
      key: 'key',
      render: (_, record) => (
        <Space>
          <Text code style={{ fontFamily: 'monospace' }}>
            {showApiKey[record.id] ? record.key : record.key.replace(/(?<=.{8}).+(?=.{6}$)/, '***')}
          </Text>
          <Button
            type="link"
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
      render: (_, record) => (
        <Space>
          {record.permissions.map(permission => (
            <Tag key={permission} color={permission === 'admin' ? 'red' : permission === 'write' ? 'orange' : 'blue'}>
              {permission}
            </Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (_, record) => (
        <Tag color={record.status === 'active' ? 'green' : 'default'}>
          {record.status === 'active' ? '活跃' : '禁用'}
        </Tag>
      ),
    },
    {
      title: '最后使用',
      dataIndex: 'lastUsed',
      key: 'lastUsed',
      render: (_, record) => record.lastUsed ? formatDate(record.lastUsed) : '-',
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (_, record) => formatDate(record.createdAt),
    },
    {
      title: '操作',
      key: 'actions',
      valueType: 'option',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            size="small"
            onClick={() => toggleApiKeyStatus(record.id)}
          >
            {record.status === 'active' ? '禁用' : '启用'}
          </Button>
          <Popconfirm
            title="确定要删除这个API密钥吗？"
            onConfirm={() => deleteApiKey(record.id)}
          >
            <Button type="link" size="small" danger>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  const tabItems = [
    {
      key: 'basic',
      label: '基础设置',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormText
                name="systemName"
                label="系统名称"
                placeholder="请输入系统名称"
                rules={[{ required: true, message: '请输入系统名称' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="language"
                label="系统语言"
                options={[
                  { label: '中文', value: 'zh-CN' },
                  { label: 'English', value: 'en-US' },
                ]}
                rules={[{ required: true, message: '请选择系统语言' }]}
              />
            </Col>
          </Row>
          
          <ProFormTextArea
            name="systemDescription"
            label="系统描述"
            placeholder="请输入系统描述"
            rules={[{ required: true, message: '请输入系统描述' }]}
          />
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSelect
                name="timezone"
                label="时区"
                options={[
                  { label: '北京时间 (UTC+8)', value: 'Asia/Shanghai' },
                  { label: '东京时间 (UTC+9)', value: 'Asia/Tokyo' },
                  { label: 'UTC时间 (UTC+0)', value: 'UTC' },
                ]}
                rules={[{ required: true, message: '请选择时区' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormRadio.Group
                name="theme"
                label="主题模式"
                options={[
                  { label: '浅色', value: 'light' },
                  { label: '深色', value: 'dark' },
                  { label: '自动', value: 'auto' },
                ]}
                rules={[{ required: true, message: '请选择主题模式' }]}
              />
            </Col>
          </Row>
        </ProForm>
      ),
    },
    {
      key: 'security',
      label: '安全设置',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="sessionTimeout"
                label="会话超时时间（分钟）"
                min={5}
                max={480}
                rules={[{ required: true, message: '请设置会话超时时间' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormDigit
                name="maxLoginAttempts"
                label="最大登录尝试次数"
                min={3}
                max={10}
                rules={[{ required: true, message: '请设置最大登录尝试次数' }]}
              />
            </Col>
          </Row>
          
          <ProFormSwitch
            name="twoFactorAuth"
            label="双因素认证"
            tooltip="启用后用户登录需要额外的验证步骤"
          />
          
          <ProFormList
            name="ipWhitelist"
            label="IP白名单"
            tooltip="只允许白名单中的IP地址访问系统"
            creatorButtonProps={{
              creatorButtonText: '添加IP地址',
            }}
          >
            <ProFormText
              name="ip"
              placeholder="请输入IP地址，如：192.168.1.1"
              rules={[
                { required: true, message: '请输入IP地址' },
                { pattern: /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/, message: '请输入有效的IP地址' },
              ]}
            />
          </ProFormList>
          
          <Card title="密码策略" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <ProFormDigit
                  name={['passwordPolicy', 'minLength']}
                  label="最小长度"
                  min={6}
                  max={32}
                  rules={[{ required: true, message: '请设置密码最小长度' }]}
                />
              </Col>
              <Col span={12}>
                <ProFormDigit
                  name={['passwordPolicy', 'expirationDays']}
                  label="密码有效期（天）"
                  min={30}
                  max={365}
                  rules={[{ required: true, message: '请设置密码有效期' }]}
                />
              </Col>
            </Row>
            
            <Row gutter={16}>
              <Col span={6}>
                <ProFormSwitch
                  name={['passwordPolicy', 'requireUppercase']}
                  label="需要大写字母"
                />
              </Col>
              <Col span={6}>
                <ProFormSwitch
                  name={['passwordPolicy', 'requireLowercase']}
                  label="需要小写字母"
                />
              </Col>
              <Col span={6}>
                <ProFormSwitch
                  name={['passwordPolicy', 'requireNumbers']}
                  label="需要数字"
                />
              </Col>
              <Col span={6}>
                <ProFormSwitch
                  name={['passwordPolicy', 'requireSpecialChars']}
                  label="需要特殊字符"
                />
              </Col>
            </Row>
          </Card>
        </ProForm>
      ),
    },
    {
      key: 'privacy',
      label: '隐私设置',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Alert
            message="隐私保护设置"
            description="这些设置影响联邦学习中的隐私保护强度，请根据实际需求谨慎调整。"
            type="info"
            showIcon
            style={{ marginBottom: 24 }}
          />
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSlider
                name="privacyBudget"
                label="隐私预算 (ε)"
                min={0.1}
                max={10}
                step={0.1}
                marks={{
                  0.1: '0.1',
                  1: '1.0',
                  5: '5.0',
                  10: '10.0',
                }}
                tooltip="较小的值提供更强的隐私保护，但可能影响模型性能"
              />
            </Col>
            <Col span={12}>
              <ProFormSlider
                name="noiseMultiplier"
                label="噪声倍数"
                min={0.1}
                max={5}
                step={0.1}
                marks={{
                  0.1: '0.1',
                  1: '1.0',
                  3: '3.0',
                  5: '5.0',
                }}
                tooltip="控制添加到梯度中的噪声量"
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSlider
                name="maxGradNorm"
                label="最大梯度范数"
                min={0.1}
                max={10}
                step={0.1}
                marks={{
                  0.1: '0.1',
                  1: '1.0',
                  5: '5.0',
                  10: '10.0',
                }}
                tooltip="梯度裁剪的阈值"
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="anonymizationLevel"
                label="匿名化级别"
                options={[
                  { label: '低', value: 'low' },
                  { label: '中', value: 'medium' },
                  { label: '高', value: 'high' },
                ]}
                tooltip="数据匿名化的强度级别"
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="dataRetentionDays"
                label="数据保留天数"
                min={30}
                max={3650}
                rules={[{ required: true, message: '请设置数据保留天数' }]}
                tooltip="系统中数据的最长保留时间"
              />
            </Col>
            <Col span={12}>
              <ProFormSwitch
                name="enableAuditLog"
                label="启用审计日志"
                tooltip="记录所有系统操作的详细日志"
              />
            </Col>
          </Row>
        </ProForm>
      ),
    },
    {
      key: 'federation',
      label: '联邦学习',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="maxParticipants"
                label="最大参与方数量"
                min={2}
                max={100}
                rules={[{ required: true, message: '请设置最大参与方数量' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormDigit
                name="minParticipants"
                label="最小参与方数量"
                min={2}
                max={10}
                rules={[{ required: true, message: '请设置最小参与方数量' }]}
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="roundTimeout"
                label="训练轮超时时间（秒）"
                min={60}
                max={3600}
                rules={[{ required: true, message: '请设置训练轮超时时间' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="aggregationMethod"
                label="聚合算法"
                options={[
                  { label: 'FedAvg', value: 'fedavg' },
                  { label: 'FedProx', value: 'fedprox' },
                  { label: 'SCAFFOLD', value: 'scaffold' },
                ]}
                rules={[{ required: true, message: '请选择聚合算法' }]}
              />
            </Col>
          </Row>
          
          <ProFormSlider
            name="clientSamplingRate"
            label="客户端采样率"
            min={0.1}
            max={1}
            step={0.1}
            marks={{
              0.1: '10%',
              0.5: '50%',
              1: '100%',
            }}
            tooltip="每轮训练中参与的客户端比例"
          />
        </ProForm>
      ),
    },
    {
      key: 'notifications',
      label: '通知设置',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSwitch
                name="emailNotifications"
                label="邮件通知"
                tooltip="通过邮件接收系统通知"
              />
            </Col>
            <Col span={12}>
              <ProFormSwitch
                name="smsNotifications"
                label="短信通知"
                tooltip="通过短信接收重要通知"
              />
            </Col>
          </Row>
          
          <ProFormText
            name="webhookUrl"
            label="Webhook URL"
            placeholder="https://your-webhook-url.com/notify"
            tooltip="系统事件将发送到此URL"
          />
          
          <ProFormSelect
            name="notificationTypes"
            label="通知类型"
            mode="multiple"
            options={[
              { label: '训练完成', value: 'training_complete' },
              { label: '错误警报', value: 'error_alert' },
              { label: '安全事件', value: 'security_event' },
              { label: '系统维护', value: 'system_maintenance' },
              { label: '性能警告', value: 'performance_warning' },
            ]}
            tooltip="选择需要接收的通知类型"
          />
        </ProForm>
      ),
    },
    {
      key: 'monitoring',
      label: '监控设置',
      children: (
        <ProForm
          form={form}
          layout="vertical"
          onFinish={saveSettings}
          submitter={{
            searchConfig: {
              submitText: '保存设置',
              resetText: '重置',
            },
            resetButtonProps: {
              onClick: resetSettings,
            },
            submitButtonProps: {
              loading: loading,
              icon: <SaveOutlined />,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSwitch
                name="metricsEnabled"
                label="启用指标收集"
                tooltip="收集系统性能和使用指标"
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="loggingLevel"
                label="日志级别"
                options={[
                  { label: 'Debug', value: 'debug' },
                  { label: 'Info', value: 'info' },
                  { label: 'Warning', value: 'warning' },
                  { label: 'Error', value: 'error' },
                ]}
                rules={[{ required: true, message: '请选择日志级别' }]}
              />
            </Col>
          </Row>
          
          <ProFormDigit
            name="healthCheckInterval"
            label="健康检查间隔（秒）"
            min={30}
            max={300}
            rules={[{ required: true, message: '请设置健康检查间隔' }]}
          />
          
          <Card title="告警阈值" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <ProFormSlider
                  name={['alertThresholds', 'cpuUsage']}
                  label="CPU使用率 (%)"
                  min={50}
                  max={100}
                  marks={{
                    50: '50%',
                    70: '70%',
                    90: '90%',
                    100: '100%',
                  }}
                />
              </Col>
              <Col span={12}>
                <ProFormSlider
                  name={['alertThresholds', 'memoryUsage']}
                  label="内存使用率 (%)"
                  min={50}
                  max={100}
                  marks={{
                    50: '50%',
                    70: '70%',
                    90: '90%',
                    100: '100%',
                  }}
                />
              </Col>
            </Row>
            
            <Row gutter={16}>
              <Col span={12}>
                <ProFormSlider
                  name={['alertThresholds', 'diskUsage']}
                  label="磁盘使用率 (%)"
                  min={50}
                  max={100}
                  marks={{
                    50: '50%',
                    70: '70%',
                    90: '90%',
                    100: '100%',
                  }}
                />
              </Col>
              <Col span={12}>
                <ProFormSlider
                  name={['alertThresholds', 'errorRate']}
                  label="错误率 (%)"
                  min={1}
                  max={20}
                  marks={{
                    1: '1%',
                    5: '5%',
                    10: '10%',
                    20: '20%',
                  }}
                />
              </Col>
            </Row>
          </Card>
        </ProForm>
      ),
    },
    {
      key: 'api',
      label: 'API管理',
      children: (
        <div>
          <ProTable<ApiKey>
            columns={apiKeyColumns}
            dataSource={apiKeys}
            rowKey="id"
            search={false}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
            }}
            toolBarRender={() => [
              <Button
                key="add"
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setCreateApiKeyVisible(true)}
              >
                创建API密钥
              </Button>,
            ]}
          />
          
          <Modal
            title="创建API密钥"
            open={createApiKeyVisible}
            onCancel={() => setCreateApiKeyVisible(false)}
            footer={null}
            width={600}
          >
            <ProForm
              form={apiKeyForm}
              layout="vertical"
              onFinish={createApiKey}
              submitter={{
                searchConfig: {
                  submitText: '创建密钥',
                  resetText: '取消',
                },
                resetButtonProps: {
                  onClick: () => setCreateApiKeyVisible(false),
                },
              }}
            >
              <ProFormText
                name="name"
                label="密钥名称"
                placeholder="请输入API密钥名称"
                rules={[{ required: true, message: '请输入密钥名称' }]}
              />
              
              <ProFormSelect
                name="permissions"
                label="权限"
                mode="multiple"
                options={[
                  { label: '读取', value: 'read' },
                  { label: '写入', value: 'write' },
                  { label: '管理员', value: 'admin' },
                ]}
                rules={[{ required: true, message: '请选择权限' }]}
                tooltip="管理员权限包含所有操作权限"
              />
            </ProForm>
          </Modal>
        </div>
      ),
    },
  ]

  return (
    <PageContainer
      title="系统设置"
      subTitle="配置系统参数和安全策略"
      extra={[
        <Button
          key="export"
          icon={<ExportOutlined />}
          onClick={exportSettings}
        >
          导出设置
        </Button>,
        <Button
          key="reset"
          icon={<ReloadOutlined />}
          onClick={resetSettings}
        >
          重置设置
        </Button>,
      ]}
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        tabPosition="left"
        style={{ minHeight: 600 }}
      />
    </PageContainer>
  )
}

export default SettingsPage