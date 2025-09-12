import React, { useEffect, useState } from 'react'
import {
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Space,
  Button,
  Spin,
  Typography,
  Alert,
  Tooltip,
} from 'antd'
import {
  ShareAltOutlined,
  SafetyCertificateOutlined,
  ExperimentOutlined,
  CloudServerOutlined,
  UserOutlined,
  FileTextOutlined,
  ReloadOutlined,
  ArrowUpOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons'
import { ProCard, PageContainer, ProTable } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import { Line } from '@ant-design/plots'
import { useAppStore } from '@store/app'
import { useChartTheme } from '@hooks/useChartTheme'
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
  const { getLineConfig } = useChartTheme()

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
        content: '无法加载仪表板数据，请稍后重试',
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
  const activityColumns: ProColumns<DashboardData['recentActivities'][0]>[] = [
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (_, record) => {
        const config = getActivityConfig(record.type)
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
      render: (_, record) => {
        const config = getStatusConfig(record.status)
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
      render: (_, record) => dayjs(record.timestamp).format('HH:mm:ss'),
    },
  ]

  return (
    <PageContainer
      title="仪表盘"
      subTitle="联邦风控系统实时监控中心"
      extra={[
        <Button
          key="refresh"
          icon={<ReloadOutlined />}
          onClick={loadDashboardData}
          loading={loading}
          type="primary"
        >
          刷新数据
        </Button>,
      ]}
    >

      {/* 概览统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <ProCard bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                width: 48, 
                height: 48, 
                margin: '0 auto 16px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--ant-color-primary)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                <ShareAltOutlined style={{ fontSize: '24px', color: 'white' }} />
              </div>
              <Statistic
                title="PSI任务总数"
                value={data.overview.totalPsiJobs}
                valueStyle={{ color: 'var(--ant-color-primary)' }}
              />
              <Text type="success" style={{ fontSize: '12px' }}>↗ +15% 本周</Text>
            </div>
          </ProCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <ProCard bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                width: 48, 
                height: 48, 
                margin: '0 auto 16px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--ant-color-success)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                <SafetyCertificateOutlined style={{ fontSize: '24px', color: 'white' }} />
              </div>
              <Statistic
                title="同意请求总数"
                value={data.overview.totalConsentRequests}
                valueStyle={{ color: 'var(--ant-color-success)' }}
              />
              <Text type="success" style={{ fontSize: '12px' }}>↗ +8% 本周</Text>
            </div>
          </ProCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <ProCard bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                width: 48, 
                height: 48, 
                margin: '0 auto 16px', 
                borderRadius: '50%', 
                backgroundColor: '#722ed1', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                <ExperimentOutlined style={{ fontSize: '24px', color: 'white' }} />
              </div>
              <Statistic
                title="训练任务总数"
                value={data.overview.totalTrainingJobs}
                valueStyle={{ color: '#722ed1' }}
              />
              <Text type="success" style={{ fontSize: '12px' }}>↗ +23% 本周</Text>
            </div>
          </ProCard>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <ProCard bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                width: 48, 
                height: 48, 
                margin: '0 auto 16px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--ant-color-warning)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                <CloudServerOutlined style={{ fontSize: '24px', color: 'white' }} />
              </div>
              <Statistic
                title="推理次数"
                value={data.overview.totalInferences}
                valueStyle={{ color: 'var(--ant-color-warning)' }}
              />
              <Text type="success" style={{ fontSize: '12px' }}>↗ +31% 本周</Text>
            </div>
          </ProCard>
        </Col>
      </Row>

      {/* 系统状态 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={8}>
          <ProCard title="系统运行时间" bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <Progress
                type="circle"
                percent={data.overview.systemUptime}
                format={(percent) => `${percent}%`}
                strokeColor="var(--ant-color-success)"
                size={120}
              />
              <div style={{ marginTop: 16 }}>
                <Text type="success">系统稳定运行</Text>
              </div>
            </div>
          </ProCard>
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <ProCard title="活跃用户" bordered hoverable>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                width: 48, 
                height: 48, 
                margin: '0 auto 16px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--ant-color-primary)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}>
                <UserOutlined style={{ fontSize: '24px', color: 'white' }} />
              </div>
              <Statistic
                value={data.overview.activeUsers}
                suffix={<ArrowUpOutlined style={{ color: 'var(--ant-color-success)' }} />}
                valueStyle={{ color: 'var(--ant-color-primary)' }}
              />
              <div style={{ marginTop: 16 }}>
                <Text type="success">较昨日 +12%</Text>
              </div>
            </div>
          </ProCard>
        </Col>
        <Col xs={24} sm={12} lg={8}>
          <ProCard title="系统负载" bordered hoverable>
            <div>
              {data.performanceMetrics.systemLoad.map((item, index) => (
                <div key={item.service} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text strong>{item.service}</Text>
                    <Text type="secondary">CPU: {item.cpu}% | 内存: {item.memory}%</Text>
                  </div>
                  <Progress
                    percent={item.cpu}
                    strokeColor="var(--ant-color-primary)"
                    size="small"
                    showInfo={false}
                  />
                </div>
              ))}
            </div>
          </ProCard>
        </Col>
      </Row>

      {/* 图表和活动 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <ProCard
            title="PSI吞吐量趋势"
            extra={<Text type="secondary">最近24小时</Text>}
            bordered
            hoverable
          >
              <Line
                data={data.performanceMetrics.psiThroughput}
                xField="time"
                yField="value"
                height={300}
                {...getLineConfig()}
                tooltip={{
                  formatter: (datum: any) => {
                    return { name: '吞吐量', value: `${datum.value} 次/小时` }
                  },
                }}
              />
          </ProCard>
        </Col>
        <Col xs={24} lg={8}>
          <ProCard
            title="最近活动"
            extra={<Text type="secondary">实时更新</Text>}
            bordered
            hoverable
          >
            <ProTable
              dataSource={data.recentActivities}
              columns={activityColumns}
              pagination={false}
              size="small"
              scroll={{ y: 300 }}
              rowKey="id"
              className="modern-table"
              search={false}
              toolBarRender={false}
            />
          </ProCard>
        </Col>
      </Row>

      {/* 训练指标 */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <ProCard title="模型训练准确率" bordered hoverable>
            <Line
              data={data.performanceMetrics.trainingAccuracy}
              xField="time"
              yField="accuracy"
              height={250}
              {...getLineConfig()}
              yAxis={{
                min: 0.7,
                max: 1.0,
              }}
              tooltip={{
                formatter: (datum: any) => {
                  return { name: '准确率', value: `${(datum.accuracy * 100).toFixed(2)}%` }
                },
              }}
            />
          </ProCard>
        </Col>
        <Col xs={24} lg={12}>
          <ProCard title="训练损失" bordered hoverable>
            <Line
              data={data.performanceMetrics.trainingAccuracy}
              xField="time"
              yField="loss"
              height={250}
              {...getLineConfig()}
              tooltip={{
                formatter: (datum: any) => {
                  return { name: '损失', value: datum.loss.toFixed(4) }
                },
              }}
            />
          </ProCard>
        </Col>
      </Row>
    </PageContainer>
  )
}

export default Dashboard