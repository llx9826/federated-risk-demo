import React, { useState, useEffect } from 'react'
import {
  Card,
  Input,
  Button,
  Upload,
  Tag,
  Space,
  Progress,
  Modal,
  Descriptions,
  Typography,
  Row,
  Col,
  Statistic,
  message,
} from 'antd'
import { ProTable, PageContainer, ProForm } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import {
  UploadOutlined,
  PlayCircleOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  ShareAltOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { useAppStore } from '@store/app'
import dayjs from 'dayjs'

const { Title, Text } = Typography

interface PSIJob {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  datasetA: string
  datasetB: string
  intersectionSize: number
  totalRecordsA: number
  totalRecordsB: number
  createdAt: string
  completedAt?: string
  duration?: number
  creator: string
}

const PSIPage: React.FC = () => {
  const [form] = ProForm.useForm()
  const [jobs, setJobs] = useState<PSIJob[]>([])
  const [loading, setLoading] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<PSIJob | null>(null)
  const { addNotification } = useAppStore()

  // 模拟PSI任务数据
  const mockJobs: PSIJob[] = [
    {
      id: '1',
      name: '银行-电商用户匹配',
      description: '银行客户数据与电商用户数据的隐私求交',
      status: 'completed',
      progress: 100,
      datasetA: 'bank_customers.csv',
      datasetB: 'ecommerce_users.csv',
      intersectionSize: 15420,
      totalRecordsA: 50000,
      totalRecordsB: 80000,
      createdAt: dayjs().subtract(2, 'hour').toISOString(),
      completedAt: dayjs().subtract(1, 'hour').toISOString(),
      duration: 3600,
      creator: '张三',
    },
    {
      id: '2',
      name: '风险用户识别',
      description: '多方风险用户数据求交分析',
      status: 'running',
      progress: 65,
      datasetA: 'risk_users_a.csv',
      datasetB: 'risk_users_b.csv',
      intersectionSize: 0,
      totalRecordsA: 30000,
      totalRecordsB: 25000,
      createdAt: dayjs().subtract(30, 'minute').toISOString(),
      creator: '李四',
    },
    {
      id: '3',
      name: '营销目标用户',
      description: '精准营销目标用户群体求交',
      status: 'pending',
      progress: 0,
      datasetA: 'marketing_targets.csv',
      datasetB: 'potential_customers.csv',
      intersectionSize: 0,
      totalRecordsA: 20000,
      totalRecordsB: 35000,
      createdAt: dayjs().subtract(10, 'minute').toISOString(),
      creator: '王五',
    },
  ]

  useEffect(() => {
    setJobs(mockJobs)
  }, [])

  // 提交PSI任务
  const handleSubmit = async (values: any) => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const newJob: PSIJob = {
        id: Date.now().toString(),
        name: values.name,
        description: values.description || '',
        status: 'pending',
        progress: 0,
        datasetA: values.datasetA?.file?.name || 'dataset_a.csv',
        datasetB: values.datasetB?.file?.name || 'dataset_b.csv',
        intersectionSize: 0,
        totalRecordsA: Math.floor(Math.random() * 50000) + 10000,
        totalRecordsB: Math.floor(Math.random() * 50000) + 10000,
        createdAt: dayjs().toISOString(),
        creator: '当前用户',
      }
      
      setJobs(prev => [newJob, ...prev])
      form.resetFields()
      
      addNotification({
        type: 'success',
        title: 'PSI任务创建成功',
        message: `任务 "${values.name}" 已提交，正在处理中...`,
      })
      
      // 模拟任务进度更新
      setTimeout(() => {
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, status: 'running' as const }
            : job
        ))
      }, 2000)
      
    } catch (error) {
      console.error('Failed to create PSI job:', error)
      addNotification({
        type: 'error',
        title: 'PSI任务创建失败',
        message: '请检查输入数据并重试',
      })
    } finally {
      setLoading(false)
    }
  }

  // 查看任务详情
  const handleViewDetail = (job: PSIJob) => {
    setSelectedJob(job)
    setDetailVisible(true)
  }

  // 删除任务
  const handleDeleteJob = (jobId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个PSI任务吗？此操作不可撤销。',
      onOk: () => {
        setJobs(prev => prev.filter(job => job.id !== jobId))
        addNotification({
          type: 'success',
          title: '任务已删除',
          message: 'PSI任务已成功删除',
        })
      },
    })
  }

  // 下载结果
  const handleDownloadResult = (job: PSIJob) => {
    if (job.status !== 'completed') {
      message.warning('任务尚未完成，无法下载结果')
      return
    }
    
    // 模拟下载
    const blob = new Blob([`PSI结果\n交集大小: ${job.intersectionSize}`], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `psi_result_${job.id}.txt`
    a.click()
    URL.revokeObjectURL(url)
    
    addNotification({
      type: 'success',
      title: '下载完成',
      message: 'PSI结果已下载到本地',
    })
  }

  // 状态配置
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'pending':
        return { color: 'default', icon: <ClockCircleOutlined />, text: '等待中' }
      case 'running':
        return { color: 'processing', icon: <PlayCircleOutlined />, text: '运行中' }
      case 'completed':
        return { color: 'success', icon: <CheckCircleOutlined />, text: '已完成' }
      case 'failed':
        return { color: 'error', icon: <ExclamationCircleOutlined />, text: '失败' }
      default:
        return { color: 'default', icon: <ClockCircleOutlined />, text: '未知' }
    }
  }

  // 表格列配置
  const columns: ProColumns<PSIJob>[] = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          {record.description && (
            <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
              {record.description}
            </div>
          )}
        </div>
      ),
    },
    {
      title: '数据集',
      key: 'datasets',
      render: (_, record) => (
        <div>
          <div style={{ fontSize: 12 }}>A: {record.datasetA}</div>
          <div style={{ fontSize: 12 }}>B: {record.datasetB}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (_, record) => {
        const config = getStatusConfig(record.status)
        return (
          <div>
            <Tag icon={config.icon} color={config.color}>
              {config.text}
            </Tag>
            {record.status === 'running' && (
              <Progress
                percent={record.progress}
                size="small"
                style={{ marginTop: 4 }}
              />
            )}
          </div>
        )
      },
    },
    {
      title: '交集大小',
      dataIndex: 'intersectionSize',
      key: 'intersectionSize',
      width: 100,
      align: 'right',
      render: (_, record) => (
        record.status === 'completed' ? (
          <Text strong style={{ color: '#1890ff' }}>
            {record.intersectionSize.toLocaleString()}
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        )
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 120,
      render: (_, record) => dayjs(record.createdAt).format('MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
          >
            详情
          </Button>
          {record.status === 'completed' && (
            <Button
              type="text"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadResult(record)}
            >
              下载
            </Button>
          )}
          <Button
            type="text"
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteJob(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  // 计算统计数据
  const stats = {
    total: jobs.length,
    completed: jobs.filter(job => job.status === 'completed').length,
    running: jobs.filter(job => job.status === 'running').length,
    pending: jobs.filter(job => job.status === 'pending').length,
  }

  return (
    <PageContainer
      title="PSI隐私求交"
      subTitle="安全多方计算隐私集合求交"
      extra={[
        <Button key="refresh" onClick={() => window.location.reload()}>
          刷新
        </Button>,
      ]}
    >

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="总任务数" value={stats.total} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={stats.completed}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中"
              value={stats.running}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="等待中"
              value={stats.pending}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 创建任务表单 */}
      <Card title="创建PSI任务" style={{ marginBottom: 24 }}>
        <ProForm
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          submitter={{
            render: () => [
              <Button
                key="submit"
                type="primary"
                htmlType="submit"
                loading={loading}
                icon={<PlayCircleOutlined />}
              >
                创建任务
              </Button>
            ]
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProForm.Item
                name="name"
                label="任务名称"
                rules={[{ required: true, message: '请输入任务名称' }]}
              >
                <Input placeholder="请输入任务名称" />
              </ProForm.Item>
            </Col>
            <Col span={12}>
              <ProForm.Item name="description" label="任务描述">
                <Input placeholder="请输入任务描述（可选）" />
              </ProForm.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProForm.Item
                name="datasetA"
                label="数据集A"
                rules={[{ required: true, message: '请上传数据集A' }]}
              >
                <Upload
                  accept=".csv,.txt"
                  beforeUpload={() => false}
                  maxCount={1}
                  fileList={[]}
                >
                  <Button icon={<UploadOutlined />}>选择文件</Button>
                </Upload>
              </ProForm.Item>
            </Col>
            <Col span={12}>
              <ProForm.Item
                name="datasetB"
                label="数据集B"
                rules={[{ required: true, message: '请上传数据集B' }]}
              >
                <Upload
                  accept=".csv,.txt"
                  beforeUpload={() => false}
                  maxCount={1}
                  fileList={[]}
                >
                  <Button icon={<UploadOutlined />}>选择文件</Button>
                </Upload>
              </ProForm.Item>
            </Col>
          </Row>
        </ProForm>
      </Card>

      {/* 任务列表 */}
      <Card title="任务列表">
        <ProTable<PSIJob>
          dataSource={jobs}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
          search={false}
          toolBarRender={false}
        />
      </Card>

      {/* 任务详情弹窗 */}
      <Modal
        title="PSI任务详情"
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
          selectedJob?.status === 'completed' && (
            <Button
              key="download"
              type="primary"
              icon={<DownloadOutlined />}
              onClick={() => {
                if (selectedJob) {
                  handleDownloadResult(selectedJob)
                  setDetailVisible(false)
                }
              }}
            >
              下载结果
            </Button>
          ),
        ]}
        width={600}
      >
        {selectedJob && (
          <Descriptions column={2} bordered>
            <Descriptions.Item label="任务名称" span={2}>
              {selectedJob.name}
            </Descriptions.Item>
            <Descriptions.Item label="任务描述" span={2}>
              {selectedJob.description || '无描述'}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              {(() => {
                const config = getStatusConfig(selectedJob.status)
                return (
                  <Tag icon={config.icon} color={config.color}>
                    {config.text}
                  </Tag>
                )
              })()}
            </Descriptions.Item>
            <Descriptions.Item label="进度">
              <Progress percent={selectedJob.progress} size="small" />
            </Descriptions.Item>
            <Descriptions.Item label="数据集A">
              {selectedJob.datasetA}
            </Descriptions.Item>
            <Descriptions.Item label="数据集B">
              {selectedJob.datasetB}
            </Descriptions.Item>
            <Descriptions.Item label="A记录数">
              {selectedJob.totalRecordsA.toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="B记录数">
              {selectedJob.totalRecordsB.toLocaleString()}
            </Descriptions.Item>
            {selectedJob.status === 'completed' && (
              <>
                <Descriptions.Item label="交集大小" span={2}>
                  <Text strong style={{ color: '#1890ff', fontSize: 16 }}>
                    {selectedJob.intersectionSize.toLocaleString()}
                  </Text>
                </Descriptions.Item>
                <Descriptions.Item label="匹配率">
                  {((selectedJob.intersectionSize / Math.min(selectedJob.totalRecordsA, selectedJob.totalRecordsB)) * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label="执行时长">
                  {selectedJob.duration ? `${Math.floor(selectedJob.duration / 60)}分${selectedJob.duration % 60}秒` : '-'}
                </Descriptions.Item>
              </>
            )}
            <Descriptions.Item label="创建者">
              {selectedJob.creator}
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {dayjs(selectedJob.createdAt).format('YYYY-MM-DD HH:mm:ss')}
            </Descriptions.Item>
            {selectedJob.completedAt && (
              <Descriptions.Item label="完成时间" span={2}>
                {dayjs(selectedJob.completedAt).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </PageContainer>
  )
}

export default PSIPage