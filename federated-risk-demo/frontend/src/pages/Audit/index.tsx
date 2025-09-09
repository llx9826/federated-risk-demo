import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Tag,
  Space,
  Button,
  Input,
  Select,
  DatePicker,
  Row,
  Col,
  Statistic,
  Typography,
  Modal,
  Descriptions,
  Timeline,
  Alert,
  Tooltip,
  Badge,
  Divider,
  Form,
} from 'antd'
import {
  AuditOutlined,
  SearchOutlined,
  EyeOutlined,
  DownloadOutlined,
  FilterOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  UserOutlined,
  SafetyCertificateOutlined,
  FileTextOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { Line, Column } from '@ant-design/plots'
import { useAppStore } from '@store/app'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { Option } = Select
const { RangePicker } = DatePicker

interface AuditLog {
  id: string
  timestamp: string
  action: string
  category: 'authentication' | 'authorization' | 'data_access' | 'model_training' | 'model_inference' | 'system_config' | 'user_management'
  severity: 'low' | 'medium' | 'high' | 'critical'
  user: string
  userId: string
  userRole: string
  resource: string
  resourceType: 'model' | 'dataset' | 'job' | 'user' | 'system' | 'api'
  details: string
  ipAddress: string
  userAgent: string
  status: 'success' | 'failure' | 'warning'
  duration?: number
  metadata?: Record<string, any>
}

interface AuditStats {
  totalLogs: number
  todayLogs: number
  successRate: number
  criticalEvents: number
  topUsers: Array<{ user: string; count: number }>
  categoryDistribution: Array<{ category: string; count: number }>
}

const AuditPage: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [filteredLogs, setFilteredLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null)
  const [detailVisible, setDetailVisible] = useState(false)
  const [stats, setStats] = useState<AuditStats | null>(null)
  const [filters, setFilters] = useState({
    category: '',
    severity: '',
    status: '',
    user: '',
    dateRange: null as [dayjs.Dayjs, dayjs.Dayjs] | null,
    keyword: '',
  })
  const { addNotification } = useAppStore()

  // 模拟审计日志数据
  const mockLogs: AuditLog[] = [
    {
      id: '1',
      timestamp: dayjs().subtract(10, 'minute').toISOString(),
      action: '用户登录',
      category: 'authentication',
      severity: 'low',
      user: '张三',
      userId: 'user001',
      userRole: '数据科学家',
      resource: '/api/auth/login',
      resourceType: 'api',
      details: '用户成功登录系统',
      ipAddress: '192.168.1.100',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 1200,
      metadata: {
        loginMethod: 'password',
        sessionId: 'sess_abc123',
      },
    },
    {
      id: '2',
      timestamp: dayjs().subtract(15, 'minute').toISOString(),
      action: '创建训练任务',
      category: 'model_training',
      severity: 'medium',
      user: '李四',
      userId: 'user002',
      userRole: '模型工程师',
      resource: 'training_job_001',
      resourceType: 'job',
      details: '创建信用风险评估模型训练任务',
      ipAddress: '192.168.1.101',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 2500,
      metadata: {
        algorithm: 'FedAvg',
        participants: ['银行A', '银行B'],
        datasetSize: 100000,
      },
    },
    {
      id: '3',
      timestamp: dayjs().subtract(30, 'minute').toISOString(),
      action: '访问敏感数据',
      category: 'data_access',
      severity: 'high',
      user: '王五',
      userId: 'user003',
      userRole: '数据分析师',
      resource: 'customer_data_2024',
      resourceType: 'dataset',
      details: '访问客户个人信息数据集',
      ipAddress: '192.168.1.102',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 5000,
      metadata: {
        datasetId: 'ds_001',
        recordCount: 50000,
        accessType: 'read',
      },
    },
    {
      id: '4',
      timestamp: dayjs().subtract(45, 'minute').toISOString(),
      action: '登录失败',
      category: 'authentication',
      severity: 'medium',
      user: '未知用户',
      userId: 'unknown',
      userRole: '未知',
      resource: '/api/auth/login',
      resourceType: 'api',
      details: '用户登录失败 - 密码错误',
      ipAddress: '203.0.113.45',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'failure',
      duration: 800,
      metadata: {
        attemptedUsername: 'admin',
        failureReason: 'invalid_password',
        attemptCount: 3,
      },
    },
    {
      id: '5',
      timestamp: dayjs().subtract(1, 'hour').toISOString(),
      action: '模型推理',
      category: 'model_inference',
      severity: 'low',
      user: '赵六',
      userId: 'user004',
      userRole: '业务分析师',
      resource: 'inference_job_001',
      resourceType: 'job',
      details: '执行信用风险评估模型推理',
      ipAddress: '192.168.1.103',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 15000,
      metadata: {
        modelId: 'model_001',
        sampleCount: 1000,
        accuracy: 0.89,
      },
    },
    {
      id: '6',
      timestamp: dayjs().subtract(2, 'hour').toISOString(),
      action: '系统配置修改',
      category: 'system_config',
      severity: 'critical',
      user: '管理员',
      userId: 'admin001',
      userRole: '系统管理员',
      resource: 'system_settings',
      resourceType: 'system',
      details: '修改隐私预算配置参数',
      ipAddress: '192.168.1.1',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 3000,
      metadata: {
        configKey: 'privacy_budget',
        oldValue: '1.0',
        newValue: '0.8',
      },
    },
    {
      id: '7',
      timestamp: dayjs().subtract(3, 'hour').toISOString(),
      action: '未授权访问尝试',
      category: 'authorization',
      severity: 'critical',
      user: '张三',
      userId: 'user001',
      userRole: '数据科学家',
      resource: '/api/admin/users',
      resourceType: 'api',
      details: '尝试访问管理员接口被拒绝',
      ipAddress: '192.168.1.100',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'failure',
      duration: 500,
      metadata: {
        requiredRole: 'admin',
        userRole: 'data_scientist',
        endpoint: '/api/admin/users',
      },
    },
    {
      id: '8',
      timestamp: dayjs().subtract(4, 'hour').toISOString(),
      action: '用户权限修改',
      category: 'user_management',
      severity: 'high',
      user: '管理员',
      userId: 'admin001',
      userRole: '系统管理员',
      resource: 'user002',
      resourceType: 'user',
      details: '修改用户李四的权限级别',
      ipAddress: '192.168.1.1',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      status: 'success',
      duration: 2000,
      metadata: {
        targetUser: '李四',
        oldRole: 'data_scientist',
        newRole: 'model_engineer',
      },
    },
  ]

  useEffect(() => {
    loadAuditLogs()
  }, [])

  useEffect(() => {
    applyFilters()
  }, [logs, filters])

  // 加载审计日志
  const loadAuditLogs = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setLogs(mockLogs)
      
      // 计算统计数据
      const stats: AuditStats = {
        totalLogs: mockLogs.length,
        todayLogs: mockLogs.filter(log => dayjs(log.timestamp).isAfter(dayjs().startOf('day'))).length,
        successRate: mockLogs.filter(log => log.status === 'success').length / mockLogs.length,
        criticalEvents: mockLogs.filter(log => log.severity === 'critical').length,
        topUsers: Object.entries(
          mockLogs.reduce((acc, log) => {
            acc[log.user] = (acc[log.user] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        )
          .map(([user, count]) => ({ user, count }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 5),
        categoryDistribution: Object.entries(
          mockLogs.reduce((acc, log) => {
            acc[log.category] = (acc[log.category] || 0) + 1
            return acc
          }, {} as Record<string, number>)
        ).map(([category, count]) => ({ category, count })),
      }
      setStats(stats)
    } catch (error) {
      addNotification({
        type: 'error',
        title: '加载失败',
        message: '审计日志加载失败，请重试',
      })
    } finally {
      setLoading(false)
    }
  }

  // 应用过滤器
  const applyFilters = () => {
    let filtered = [...logs]

    if (filters.category) {
      filtered = filtered.filter(log => log.category === filters.category)
    }

    if (filters.severity) {
      filtered = filtered.filter(log => log.severity === filters.severity)
    }

    if (filters.status) {
      filtered = filtered.filter(log => log.status === filters.status)
    }

    if (filters.user) {
      filtered = filtered.filter(log => log.user.includes(filters.user))
    }

    if (filters.dateRange) {
      const [start, end] = filters.dateRange
      filtered = filtered.filter(log => {
        const logTime = dayjs(log.timestamp)
        return logTime.isAfter(start) && logTime.isBefore(end)
      })
    }

    if (filters.keyword) {
      const keyword = filters.keyword.toLowerCase()
      filtered = filtered.filter(log => 
        log.action.toLowerCase().includes(keyword) ||
        log.details.toLowerCase().includes(keyword) ||
        log.resource.toLowerCase().includes(keyword)
      )
    }

    setFilteredLogs(filtered)
  }

  // 重置过滤器
  const resetFilters = () => {
    setFilters({
      category: '',
      severity: '',
      status: '',
      user: '',
      dateRange: null,
      keyword: '',
    })
  }

  // 导出审计日志
  const exportLogs = () => {
    const exportData = filteredLogs.map(log => ({
      时间: dayjs(log.timestamp).format('YYYY-MM-DD HH:mm:ss'),
      操作: log.action,
      类别: getCategoryText(log.category),
      严重程度: getSeverityText(log.severity),
      用户: log.user,
      资源: log.resource,
      状态: getStatusText(log.status),
      详情: log.details,
      IP地址: log.ipAddress,
    }))

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit_logs_${dayjs().format('YYYYMMDD_HHmmss')}.json`
    a.click()
    URL.revokeObjectURL(url)

    addNotification({
      type: 'success',
      title: '导出成功',
      message: '审计日志已导出到本地',
    })
  }

  // 获取类别文本
  const getCategoryText = (category: string) => {
    const categoryMap: Record<string, string> = {
      authentication: '身份认证',
      authorization: '权限控制',
      data_access: '数据访问',
      model_training: '模型训练',
      model_inference: '模型推理',
      system_config: '系统配置',
      user_management: '用户管理',
    }
    return categoryMap[category] || category
  }

  // 获取严重程度文本和颜色
  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case 'low':
        return { text: '低', color: 'green' }
      case 'medium':
        return { text: '中', color: 'orange' }
      case 'high':
        return { text: '高', color: 'red' }
      case 'critical':
        return { text: '严重', color: 'purple' }
      default:
        return { text: '未知', color: 'default' }
    }
  }

  const getSeverityText = (severity: string) => getSeverityConfig(severity).text

  // 获取状态文本和配置
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'success':
        return { text: '成功', color: 'success', icon: <CheckCircleOutlined /> }
      case 'failure':
        return { text: '失败', color: 'error', icon: <ExclamationCircleOutlined /> }
      case 'warning':
        return { text: '警告', color: 'warning', icon: <WarningOutlined /> }
      default:
        return { text: '未知', color: 'default', icon: <ClockCircleOutlined /> }
    }
  }

  const getStatusText = (status: string) => getStatusConfig(status).text

  // 表格列配置
  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: string) => (
        <div>
          <div>{dayjs(timestamp).format('MM-DD HH:mm')}</div>
          <div style={{ fontSize: 11, color: '#999' }}>
            {dayjs(timestamp).format('YYYY')}
          </div>
        </div>
      ),
      sorter: (a: AuditLog, b: AuditLog) => dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix(),
      defaultSortOrder: 'descend' as const,
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      width: 150,
      render: (action: string, record: AuditLog) => (
        <div>
          <div style={{ fontWeight: 500 }}>{action}</div>
          <Tag size="small" color="blue">{getCategoryText(record.category)}</Tag>
        </div>
      ),
    },
    {
      title: '用户',
      key: 'user',
      width: 120,
      render: (record: AuditLog) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.user}</div>
          <div style={{ fontSize: 11, color: '#666' }}>{record.userRole}</div>
        </div>
      ),
    },
    {
      title: '资源',
      key: 'resource',
      width: 150,
      render: (record: AuditLog) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.resource}</div>
          <Tag size="small">{record.resourceType}</Tag>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => {
        const config = getStatusConfig(status)
        return (
          <Tag icon={config.icon} color={config.color}>
            {config.text}
          </Tag>
        )
      },
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => {
        const config = getSeverityConfig(severity)
        return (
          <Tag color={config.color}>{config.text}</Tag>
        )
      },
    },
    {
      title: 'IP地址',
      dataIndex: 'ipAddress',
      key: 'ipAddress',
      width: 120,
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (record: AuditLog) => (
        <Button
          type="text"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedLog(record)
            setDetailVisible(true)
          }}
        >
          详情
        </Button>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          <AuditOutlined style={{ marginRight: 8 }} />
          审计日志
        </Title>
        <Text type="secondary">系统操作审计与安全监控</Text>
      </div>

      {/* 统计概览 */}
      {stats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总日志数"
                value={stats.totalLogs}
                prefix={<FileTextOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="今日日志"
                value={stats.todayLogs}
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="成功率"
                value={stats.successRate * 100}
                precision={1}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="严重事件"
                value={stats.criticalEvents}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 过滤器 */}
      <Card style={{ marginBottom: 16 }}>
        <Form layout="inline">
          <Form.Item label="关键词">
            <Input
              placeholder="搜索操作、详情或资源"
              value={filters.keyword}
              onChange={(e) => setFilters(prev => ({ ...prev, keyword: e.target.value }))}
              style={{ width: 200 }}
              prefix={<SearchOutlined />}
            />
          </Form.Item>
          
          <Form.Item label="类别">
            <Select
              placeholder="选择类别"
              value={filters.category}
              onChange={(value) => setFilters(prev => ({ ...prev, category: value }))}
              style={{ width: 150 }}
              allowClear
            >
              <Option value="authentication">身份认证</Option>
              <Option value="authorization">权限控制</Option>
              <Option value="data_access">数据访问</Option>
              <Option value="model_training">模型训练</Option>
              <Option value="model_inference">模型推理</Option>
              <Option value="system_config">系统配置</Option>
              <Option value="user_management">用户管理</Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="严重程度">
            <Select
              placeholder="选择严重程度"
              value={filters.severity}
              onChange={(value) => setFilters(prev => ({ ...prev, severity: value }))}
              style={{ width: 120 }}
              allowClear
            >
              <Option value="low">低</Option>
              <Option value="medium">中</Option>
              <Option value="high">高</Option>
              <Option value="critical">严重</Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="状态">
            <Select
              placeholder="选择状态"
              value={filters.status}
              onChange={(value) => setFilters(prev => ({ ...prev, status: value }))}
              style={{ width: 100 }}
              allowClear
            >
              <Option value="success">成功</Option>
              <Option value="failure">失败</Option>
              <Option value="warning">警告</Option>
            </Select>
          </Form.Item>
          
          <Form.Item label="时间范围">
            <RangePicker
              value={filters.dateRange}
              onChange={(dates) => setFilters(prev => ({ ...prev, dateRange: dates }))}
              showTime
              format="YYYY-MM-DD HH:mm"
            />
          </Form.Item>
          
          <Form.Item>
            <Space>
              <Button
                icon={<FilterOutlined />}
                onClick={resetFilters}
              >
                重置
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={loadAuditLogs}
                loading={loading}
              >
                刷新
              </Button>
              <Button
                icon={<DownloadOutlined />}
                onClick={exportLogs}
                disabled={filteredLogs.length === 0}
              >
                导出
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* 审计日志表格 */}
      <Card
        title={`审计日志 (${filteredLogs.length}/${logs.length})`}
        extra={
          <Space>
            <Badge count={stats?.criticalEvents || 0} offset={[10, 0]}>
              <Button icon={<SafetyCertificateOutlined />} size="small">
                安全事件
              </Button>
            </Badge>
          </Space>
        }
      >
        <Table
          dataSource={filteredLogs}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
          scroll={{ x: 1200 }}
          size="small"
        />
      </Card>

      {/* 详情弹窗 */}
      <Modal
        title="审计日志详情"
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        {selectedLog && (
          <div>
            <Descriptions column={2} bordered>
              <Descriptions.Item label="操作" span={2}>
                <Space>
                  {selectedLog.action}
                  <Tag color="blue">{getCategoryText(selectedLog.category)}</Tag>
                  {(() => {
                    const config = getSeverityConfig(selectedLog.severity)
                    return <Tag color={config.color}>{config.text}</Tag>
                  })()}
                  {(() => {
                    const config = getStatusConfig(selectedLog.status)
                    return <Tag icon={config.icon} color={config.color}>{config.text}</Tag>
                  })()}
                </Space>
              </Descriptions.Item>
              
              <Descriptions.Item label="时间">
                {dayjs(selectedLog.timestamp).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
              
              <Descriptions.Item label="持续时间">
                {selectedLog.duration ? `${selectedLog.duration}ms` : 'N/A'}
              </Descriptions.Item>
              
              <Descriptions.Item label="用户">
                <Space>
                  <UserOutlined />
                  {selectedLog.user}
                  <Tag size="small">{selectedLog.userRole}</Tag>
                </Space>
              </Descriptions.Item>
              
              <Descriptions.Item label="用户ID">
                {selectedLog.userId}
              </Descriptions.Item>
              
              <Descriptions.Item label="资源">
                <Space>
                  {selectedLog.resource}
                  <Tag size="small">{selectedLog.resourceType}</Tag>
                </Space>
              </Descriptions.Item>
              
              <Descriptions.Item label="IP地址">
                {selectedLog.ipAddress}
              </Descriptions.Item>
              
              <Descriptions.Item label="详情" span={2}>
                {selectedLog.details}
              </Descriptions.Item>
              
              <Descriptions.Item label="用户代理" span={2}>
                <Text code style={{ fontSize: 11 }}>
                  {selectedLog.userAgent}
                </Text>
              </Descriptions.Item>
            </Descriptions>
            
            {selectedLog.metadata && Object.keys(selectedLog.metadata).length > 0 && (
              <>
                <Divider>元数据</Divider>
                <Descriptions column={1} bordered size="small">
                  {Object.entries(selectedLog.metadata).map(([key, value]) => (
                    <Descriptions.Item key={key} label={key}>
                      <Text code>{JSON.stringify(value)}</Text>
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </>
            )}
            
            {selectedLog.severity === 'critical' && (
              <>
                <Divider />
                <Alert
                  message="严重安全事件"
                  description="此事件被标记为严重级别，建议立即关注并采取相应措施。"
                  type="error"
                  showIcon
                />
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}

export default AuditPage