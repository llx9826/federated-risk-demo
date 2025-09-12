import React, { useEffect, useState } from 'react'
import { Row, Col, Card, Statistic, Progress, Tag, Space, Button } from 'antd'
import {
  UserOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  SafetyOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { ProTable } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import { useQuery } from '@tanstack/react-query'
import { systemAPI, federatedAPI, consentAPI, auditAPI } from '@/services/api'
import { dateUtils, numberUtils } from '@/utils'

interface DashboardStats {
  totalUsers: number
  activeJobs: number
  completedJobs: number
  systemHealth: number
  dataVolume: number
  successRate: number
}

interface RecentActivity {
  id: string
  type: string
  description: string
  status: 'success' | 'warning' | 'error'
  timestamp: string
  user: string
}

const Dashboard: React.FC = () => {
  const [refreshKey, setRefreshKey] = useState(0)

  // 获取系统统计数据
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats', refreshKey],
    queryFn: async (): Promise<DashboardStats> => {
      const [systemStatus, jobs, consents] = await Promise.all([
        systemAPI.getSystemStatus(),
        federatedAPI.getTrainingJobs({ limit: 100 }),
        consentAPI.getConsents({ limit: 100 }),
      ])
      
      return {
        totalUsers: systemStatus.data?.users || 0,
        activeJobs: jobs.data?.filter((job: any) => job.status === 'running').length || 0,
        completedJobs: jobs.data?.filter((job: any) => job.status === 'completed').length || 0,
        systemHealth: systemStatus.data?.health || 95,
        dataVolume: systemStatus.data?.dataVolume || 0,
        successRate: systemStatus.data?.successRate || 0.98,
      }
    },
    refetchInterval: 30000, // 30秒刷新一次
  })

  // 获取最近活动
  const { data: activities, isLoading: activitiesLoading } = useQuery({
    queryKey: ['dashboard-activities', refreshKey],
    queryFn: async (): Promise<RecentActivity[]> => {
      const response = await auditAPI.getAuditLogs({ limit: 10, sort: 'desc' })
      return response.data?.map((log: any) => ({
        id: log.id,
        type: log.action,
        description: log.description,
        status: log.level === 'error' ? 'error' : log.level === 'warning' ? 'warning' : 'success',
        timestamp: log.timestamp,
        user: log.user?.username || 'System',
      })) || []
    },
    refetchInterval: 60000, // 1分钟刷新一次
  })

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'green'
      case 'warning': return 'orange'
      case 'error': return 'red'
      default: return 'blue'
    }
  }

  const activityColumns: ProColumns<RecentActivity>[] = [
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 120,
      render: (_, record: RecentActivity) => (
        <Tag color="blue">{record.type}</Tag>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (_, record: RecentActivity) => (
        <Tag color={getStatusColor(record.status)}>
          {record.status === 'success' ? '成功' : record.status === 'warning' ? '警告' : '错误'}
        </Tag>
      ),
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
      render: (_, record: RecentActivity) => dateUtils.formatRelativeTime(record.timestamp),
    },
  ]

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">仪表盘</h1>
          <p className="text-gray-600">系统概览和实时监控</p>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={handleRefresh}
          loading={statsLoading}
        >
          刷新数据
        </Button>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={stats?.totalUsers || 0}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#3f8600' }}
              suffix={<ArrowUpOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={stats?.activeJobs || 0}
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="已完成任务"
              value={stats?.completedJobs || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="成功率"
              value={numberUtils.formatPercentage(stats?.successRate || 0)}
              prefix={<SafetyOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 系统健康状态 */}
        <Col xs={24} lg={8}>
          <Card title="系统健康状态" className="h-full">
            <div className="text-center">
              <Progress
                type="circle"
                percent={stats?.systemHealth || 0}
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
                size={120}
              />
              <div className="mt-4">
                <p className="text-lg font-semibold">
                  {stats?.systemHealth || 0}%
                </p>
                <p className="text-gray-500">整体健康度</p>
              </div>
            </div>
          </Card>
        </Col>

        {/* 数据处理量 */}
        <Col xs={24} lg={8}>
          <Card title="数据处理量" className="h-full">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span>今日处理</span>
                  <span className="font-semibold">
                    {numberUtils.formatFileSize(stats?.dataVolume || 0)}
                  </span>
                </div>
                <Progress percent={75} strokeColor="#1890ff" />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span>本周处理</span>
                  <span className="font-semibold">
                    {numberUtils.formatFileSize((stats?.dataVolume || 0) * 7)}
                  </span>
                </div>
                <Progress percent={60} strokeColor="#52c41a" />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span>本月处理</span>
                  <span className="font-semibold">
                    {numberUtils.formatFileSize((stats?.dataVolume || 0) * 30)}
                  </span>
                </div>
                <Progress percent={45} strokeColor="#722ed1" />
              </div>
            </div>
          </Card>
        </Col>

        {/* 快速操作 */}
        <Col xs={24} lg={8}>
          <Card title="快速操作" className="h-full">
            <Space direction="vertical" className="w-full" size="middle">
              <Button type="primary" block size="large">
                创建新任务
              </Button>
              <Button block size="large">
                查看数据对齐
              </Button>
              <Button block size="large">
                模型管理
              </Button>
              <Button block size="large">
                系统设置
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 最近活动 */}
      <Row className="mt-6">
        <Col span={24}>
          <Card title="最近活动" className="h-full">
            <ProTable<RecentActivity>
              columns={activityColumns}
              dataSource={activities}
              loading={activitiesLoading}
              pagination={{
                pageSize: 10,
                showSizeChanger: false,
                showQuickJumper: true,
              }}
              rowKey="id"
              size="small"
              search={false}
              toolBarRender={false}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard