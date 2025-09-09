import React, { useEffect, useState } from 'react'
import {
  Row,
  Col,
  Card,
  Statistic,
  Progress,
  Table,
  Tag,
  Space,
  Button,
  Alert,
  Spin,
  Typography,
  Divider,
} from 'antd'
import {
  ShareAltOutlined,
  SafetyCertificateOutlined,
  ExperimentOutlined,
  CloudServerOutlined,
  TrophyOutlined,
  UserOutlined,
  FileTextOutlined,
  ReloadOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
} from '@ant-design/icons'
import { Line, Column, Pie } from '@ant-design/plots'
import { useAppStore } from '@store/app'
import { ApiService } from '@services/api'
import dayjs from 'dayjs'

const { Title, Text } = Typography

interface DashboardData {
  overview: {
    totalPsiJobs: number
    totalConsentRequests: number
    totalTrainingJobs: number
    totalInferences: number
    systemUptime: number
    activeUsers: number
  }
  recentActivities: Array<{
    id: string
    type: 'psi' | 'consent' | 'training' | 'inference'
    title: string
    status: 'success' | 'running' | 'failed'
    timestamp: string
    user: string
  }>
  performanceMetrics: {
    psiThroughput: Array<{ time: string; value: number }>
    trainingAccuracy: Array<{ time: string; accuracy: number; loss: number }>
    systemLoad: Array<{ service: string; cpu: number; memory: number }>
  }
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState<DashboardData | null>(null)
  const { addNotification } = useAppStore()

  // 模拟数据加载
  const loadDashboardData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const mockData: DashboardData = {
        overview: {
          totalPsiJobs: 156,
          totalConsentRequests: 2340,
          totalTrainingJobs: 45,
          totalInferences: 8920,
          systemUptime: 99.8,
          activeUsers: 23,
        },
        recentActivities: [
          {
            id: '1',
            type: 'training',
            title: '联邦学习模型训练完成',
            status: 'success',
            timestamp: dayjs().subtract(5, 'minute').toISOString(),
            user: '张三',
          },
          {
            id: '2',
            type: 'psi',
            title: 'PSI隐私求交任务',
            status: 'running',
            timestamp: dayjs().subtract(15, 'minute').toISOString(),
            user: '李四',
          },
          {
            id: '3',
            type: 'inference',
            title: '风险评分推理',
            status: 'success',
            timestamp: dayjs().subtract(30, 'minute').toISOString(),
            user: '王五',
          },
          {
            id: '4',
            type: 'consent',
            title: '数据使用同意申请',
            status: 'success',
            timestamp: dayjs().subtract(1, 'hour').toISOString(),
            user: '赵六',
          },
        ],
        performanceMetrics: {
          psiThroughput: Array.from({ length: 24 }, (_, i) => ({
            time: dayjs().subtract(23 - i, 'hour').format('HH:mm'),
            value: Math.floor(Math.random() * 100) + 50,
          })),
          trainingAccuracy: Array.from({ length: 10 }, (_, i) => ({
            time: `Epoch ${i + 1}`,
            accuracy: 0.7 + Math.random() * 0.25,
            loss: 2 - (i * 0.15) + Math.random() * 0.1,
          })),
          systemLoad: [
            { service: 'PSI服务', cpu: 45, memory: 62 },
            { service: '同意管理', cpu: 23, memory: 38 },
            { service: '训练服务', cpu: 78, memory: 85 },
            { service: '推理服务', cpu: 56, memory: 71 },
          ],
        },
      }
      
      setData(mockData)
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      addNotification({
        type: 'error',
        title: '数据加载失败',
        message: '无法加载仪表板数据，请稍后重试',
      })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDashboardData()
  }, [])

  if (loading) {
    return (
      <div style={{ padding: 24, textAlign: 'center' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>加载仪表板数据...</div>
      </div>
    )
  }

  if (!data) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          message="数据加载失败"
          description="无法加载仪表板数据，请检查网络连接或稍后重试。"
          type="error"
          showIcon
          action={
            <Button size="small" onClick={loadDashboardData}>
              重试
            </Button>
          }
        />
      </div>
    )
  }

  // 活动类型配置
  const getActivityConfig = (type: string) => {
    switch (type) {
      case 'psi':
        return { icon: <ShareAltOutlined />, color: '#1890ff', label: 'PSI' }
      case 'consent':
        return { icon: <SafetyCertificateOutlined />, color: '#52c41a', label: '同意' }
      case 'training':
        return { icon: <ExperimentOutlined />, color: '#722ed1', label: '训练' }
      case 'inference':
        return { icon: <CloudServerOutlined />, color: '#fa8c16', label: '推理' }
      default:
        return { icon: <FileTextOutlined />, color: '#666', label: '其他' }
    }
  }

  // 状态配置
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'success':
        return { color: 'success', text: '成功' }
      case 'running':
        return { color: 'processing', text: '运行中' }
      case 'failed':
        return { color: 'error', text: '失败' }
      default:
        return { color: 'default', text: '未知' }
    }
  }

  // 活动表格列配置
  const activityColumns = [
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (type: string) => {
        const config = getActivityConfig(type)
        return (
          <Tag icon={config.icon} color={config.color}>
            {config.label}
          </Tag>
        )
      },
    },
    {
      title: '活动',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const config = getStatusConfig(status)
        return <Tag color={config.color}>{config.text}</Tag>
      },
    },
    {
      title: '用户',
      dataIndex: 'user',
      key: 'user',
      width: 100,
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: string) => dayjs(timestamp).format('HH:mm:ss'),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Space>
          <Title level={2} style={{ margin: 0 }}>仪表板</Title>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadDashboardData}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
        <Text type="secondary">联邦风控系统运行概览</Text>
      </div>

      {/* 概览统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="PSI任务总数"
              value={data.overview.totalPsiJobs}
              prefix={<ShareAltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="同意请求总数"
              value={data.overview.totalConsentRequests}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="训练任务总数"
              value={data.overview.totalTrainingJobs}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="推理次数"
              value={data.overview.totalInferences}
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统状态 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={8}>
          <Card title="系统运行时间">
            <Progress
              type="circle"
              percent={data.overview.systemUptime}
              format={(percent) => `${percent}%`}
              strokeColor="#52c41a"
            />
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <Text type="secondary">系统稳定运行</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <Card>
            <Statistic
              title="活跃用户"
              value={data.overview.activeUsers}
              prefix={<UserOutlined />}
              suffix={<ArrowUpOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#1890ff' }}
            />
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">较昨日 +12%</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <Card title="系统负载">
            <div>
              {data.performanceMetrics.systemLoad.map((item) => (
                <div key={item.service} style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <Text style={{ fontSize: 12 }}>{item.service}</Text>
                    <Text style={{ fontSize: 12 }}>CPU: {item.cpu}% | 内存: {item.memory}%</Text>
                  </div>
                  <Progress
                    percent={item.cpu}
                    size="small"
                    strokeColor={item.cpu > 80 ? '#ff4d4f' : item.cpu > 60 ? '#faad14' : '#52c41a'}
                    showInfo={false}
                  />
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>

      {/* 图表和活动 */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="PSI吞吐量趋势" extra={<Text type="secondary">最近24小时</Text>}>
            <Line
              data={data.performanceMetrics.psiThroughput}
              xField="time"
              yField="value"
              smooth
              color="#1890ff"
              height={300}
              point={{
                size: 3,
                shape: 'circle',
              }}
              tooltip={{
                formatter: (datum) => {
                  return { name: '吞吐量', value: `${datum.value} 次/小时` }
                },
              }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="最近活动" extra={<Text type="secondary">实时更新</Text>}>
            <Table
              dataSource={data.recentActivities}
              columns={activityColumns}
              pagination={false}
              size="small"
              scroll={{ y: 300 }}
              rowKey="id"
            />
          </Card>
        </Col>
      </Row>

      {/* 训练指标 */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card title="模型训练准确率">
            <Line
              data={data.performanceMetrics.trainingAccuracy}
              xField="time"
              yField="accuracy"
              smooth
              color="#52c41a"
              height={250}
              yAxis={{
                min: 0.7,
                max: 1.0,
              }}
              tooltip={{
                formatter: (datum) => {
                  return { name: '准确率', value: `${(datum.accuracy * 100).toFixed(2)}%` }
                },
              }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="训练损失">
            <Line
              data={data.performanceMetrics.trainingAccuracy}
              xField="time"
              yField="loss"
              smooth
              color="#ff4d4f"
              height={250}
              tooltip={{
                formatter: (datum) => {
                  return { name: '损失', value: datum.loss.toFixed(4) }
                },
              }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard