import React, { useState, useEffect } from 'react'
import { Button, Space, Tag, Progress, Modal, message, Descriptions, Typography, Row, Col, Statistic } from 'antd'
import { ProTable, PageContainer, ProForm, ProFormText, ProFormSelect, ProFormTextArea, ProFormDigit, ProFormSlider } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  PlusOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { Line } from '@ant-design/plots'
import { useChartTheme } from '@/hooks/useChartTheme'
import { formatDate } from '@/utils'
import dayjs from 'dayjs'
import axios from 'axios'

const { Title, Text } = Typography

interface TrainingJob {
  id: string
  name: string
  description: string
  algorithm: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  progress: number
  currentEpoch: number
  totalEpochs: number
  accuracy: number
  loss: number
  participants: string[]
  datasetSize: number
  createdAt: string
  startedAt?: string
  completedAt?: string
  duration?: number
  creator: string
  config: {
    learningRate: number
    batchSize: number
    privacyBudget: number
    aggregationMethod: string
  }
  metrics: Array<{
    epoch: number
    accuracy: number
    loss: number
    timestamp: string
  }>
}

const TrainingPage: React.FC = () => {
  const { getLineConfig } = useChartTheme()
  const [form] = ProForm.useForm()
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(false)

  // 训练任务数据
  const [jobs, setJobs] = useState<TrainingJob[]>([])

  // 从后端API获取训练任务列表
  const fetchTrainingJobs = async () => {
    try {
      const response = await axios.get('http://localhost:8002/tasks')
      const backendJobs = response.data.map((task: any) => ({
        id: task.task_id,
        name: task.task_id,
        description: `联邦学习训练任务 - ${task.task_id}`,
        algorithm: 'XGBoost',
        status: task.status === 'completed' ? 'completed' : task.status === 'running' ? 'running' : 'pending',
        progress: task.status === 'completed' ? 100 : task.status === 'running' ? 50 : 0,
        currentEpoch: task.status === 'completed' ? 100 : 0,
        totalEpochs: 100,
        accuracy: task.metrics?.accuracy || 0,
        loss: task.metrics?.loss || 0,
        participants: task.participants || [],
        datasetSize: 1000,
        createdAt: task.created_at || dayjs().toISOString(),
        startedAt: task.started_at,
        completedAt: task.completed_at,
        creator: '系统',
        config: {
          learningRate: 0.1,
          batchSize: 100,
          privacyBudget: 1.0,
          aggregationMethod: 'SecureAgg',
        },
        metrics: [],
      }))
      setJobs(backendJobs)
    } catch (error) {
      console.error('获取训练任务失败:', error)
      message.error('获取训练任务失败')
    }
  }

  // 组件加载时获取数据
  useEffect(() => {
    fetchTrainingJobs()
    // 每30秒刷新一次数据
    const interval = setInterval(fetchTrainingJobs, 30000)
    return () => clearInterval(interval)
  }, [])

  // 创建训练任务
  const handleCreateJob = async (values: any) => {
    setLoading(true)
    try {
      const newJob: TrainingJob = {
        id: Date.now().toString(),
        name: values.name,
        description: values.description,
        algorithm: values.algorithm,
        status: 'pending',
        progress: 0,
        currentEpoch: 0,
        totalEpochs: values.totalEpochs,
        accuracy: 0,
        loss: 0,
        participants: values.participants || [],
        datasetSize: values.datasetSize || 0,
        createdAt: dayjs().toISOString(),
        creator: '当前用户',
        config: {
          learningRate: values.learningRate,
          batchSize: values.batchSize,
          privacyBudget: values.privacyBudget,
          aggregationMethod: values.aggregationMethod,
        },
        metrics: [],
      }
      
      setJobs(prev => [newJob, ...prev])
      setCreateModalVisible(false)
      form.resetFields()
      
      message.success('训练任务创建成功')
    } catch (error) {
      message.error('创建失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 开始训练
  const handleStartTraining = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return {
          ...job,
          status: 'running' as const,
          startedAt: dayjs().toISOString(),
        }
      }
      return job
    }))
    
    message.success('训练已开始')
  }

  // 暂停训练
  const handlePauseTraining = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return { ...job, status: 'paused' as const }
      }
      return job
    }))
    
    message.warning('训练已暂停')
  }

  // 停止训练
  const handleStopTraining = (jobId: string) => {
    Modal.confirm({
      title: '确认停止训练',
      content: '确定要停止这个训练任务吗？停止后无法恢复。',
      onOk: () => {
        setJobs(prev => prev.map(job => {
          if (job.id === jobId) {
            return {
              ...job,
              status: 'failed' as const,
              completedAt: dayjs().toISOString(),
            }
          }
          return job
        }))
        
        message.error('训练已停止')
      },
    })
  }

  // 删除任务
  const handleDeleteJob = (jobId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个训练任务吗？此操作不可撤销。',
      onOk: () => {
        setJobs(prev => prev.filter(job => job.id !== jobId))
        message.success('任务已删除')
      },
    })
  }

  // 下载模型
  const handleDownloadModel = (job: TrainingJob) => {
    if (job.status !== 'completed') {
      message.warning('只有已完成的训练任务才能下载模型')
      return
    }
    
    // 模拟下载
    const modelInfo = {
      name: job.name,
      algorithm: job.algorithm,
      accuracy: job.accuracy,
      epochs: job.totalEpochs,
      participants: job.participants,
    }
    
    const blob = new Blob([JSON.stringify(modelInfo, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `model_${job.id}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    message.success('模型文件已下载到本地')
  }

  // 状态配置
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'pending':
        return { color: 'default', icon: <ClockCircleOutlined />, text: '等待中' }
      case 'running':
        return { color: 'processing', icon: <PlayCircleOutlined />, text: '训练中' }
      case 'paused':
        return { color: 'warning', icon: <PauseCircleOutlined />, text: '已暂停' }
      case 'completed':
        return { color: 'success', icon: <CheckCircleOutlined />, text: '已完成' }
      case 'failed':
        return { color: 'error', icon: <ExclamationCircleOutlined />, text: '失败' }
      default:
        return { color: 'default', icon: <ClockCircleOutlined />, text: '未知' }
    }
  }

  // 表格列配置
  const columns: ProColumns<TrainingJob>[] = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.name}</div>
          <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      width: 100,
      render: (_, record) => (
        <Tag color="blue">{record.algorithm}</Tag>
      ),
    },
    {
      title: '参与方',
      dataIndex: 'participants',
      key: 'participants',
      width: 150,
      render: (_, record) => (
        <div>
          {record.participants.slice(0, 2).map(p => (
            <Tag key={p} style={{ marginBottom: 2 }}>{p}</Tag>
          ))}
          {record.participants.length > 2 && (
            <Tag>+{record.participants.length - 2}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (_, record) => {
        const config = getStatusConfig(record.status)
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        )
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 150,
      render: (_, record) => (
        <div>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.currentEpoch}/{record.totalEpochs} epochs
          </Text>
        </div>
      ),
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      width: 100,
      align: 'right',
      render: (_, record) => (
        <Text strong>{(record.accuracy * 100).toFixed(1)}%</Text>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 120,
      render: (_, record) => formatDate(record.createdAt),
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      valueType: 'option',
      render: (_, record) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedJob(record)
              setDetailVisible(true)
            }}
          >
            查看
          </Button>
          {record.status === 'pending' && (
            <Button
              type="link"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStartTraining(record.id)}
            >
              开始
            </Button>
          )}
          {record.status === 'running' && (
            <>
              <Button
                type="link"
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handlePauseTraining(record.id)}
              >
                暂停
              </Button>
              <Button
                type="link"
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleStopTraining(record.id)}
              >
                停止
              </Button>
            </>
          )}
          {record.status === 'paused' && (
            <Button
              type="link"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStartTraining(record.id)}
            >
              恢复
            </Button>
          )}
          {record.status === 'completed' && (
            <Button
              type="link"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadModel(record)}
            >
              下载
            </Button>
          )}
          <Button
            type="link"
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

  return (
    <PageContainer
      title="联邦学习训练"
      subTitle="管理联邦学习模型训练任务"
      extra={[
        <Button
          key="add"
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
        >
          创建训练任务
        </Button>,
      ]}
    >
      <ProTable<TrainingJob>
        columns={columns}
        dataSource={jobs}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
        }}
        search={{
          labelWidth: 'auto',
        }}
        toolBarRender={false}
      />

      {/* 创建训练任务模态框 */}
      <Modal
        title="创建训练任务"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
        width={800}
      >
        <ProForm
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
          submitter={{
            searchConfig: {
              submitText: '创建任务',
              resetText: '取消',
            },
            resetButtonProps: {
              onClick: () => setCreateModalVisible(false),
            },
            submitButtonProps: {
              loading: loading,
            },
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <ProFormText
                name="name"
                label="任务名称"
                placeholder="请输入训练任务名称"
                rules={[{ required: true, message: '请输入任务名称' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="algorithm"
                label="训练算法"
                placeholder="请选择训练算法"
                options={[
                  { label: 'FedAvg', value: 'FedAvg' },
                  { label: 'FedProx', value: 'FedProx' },
                  { label: 'FedNova', value: 'FedNova' },
                  { label: 'SCAFFOLD', value: 'SCAFFOLD' },
                ]}
                rules={[{ required: true, message: '请选择训练算法' }]}
              />
            </Col>
          </Row>
          
          <ProFormTextArea
            name="description"
            label="任务描述"
            placeholder="请输入任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
          />
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSelect
                name="participants"
                label="参与方"
                placeholder="请选择参与方"
                mode="multiple"
                options={[
                  { label: '银行A', value: '银行A' },
                  { label: '银行B', value: '银行B' },
                  { label: '银行C', value: '银行C' },
                  { label: '保险公司A', value: '保险公司A' },
                  { label: '证券公司A', value: '证券公司A' },
                ]}
                rules={[{ required: true, message: '请选择参与方' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormDigit
                name="datasetSize"
                label="数据集大小"
                placeholder="请输入数据集大小"
                min={1}
                rules={[{ required: true, message: '请输入数据集大小' }]}
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={8}>
              <ProFormDigit
                name="totalEpochs"
                label="训练轮数"
                placeholder="请输入训练轮数"
                min={1}
                max={1000}
                initialValue={50}
                rules={[{ required: true, message: '请输入训练轮数' }]}
              />
            </Col>
            <Col span={8}>
              <ProFormDigit
                name="learningRate"
                label="学习率"
                placeholder="请输入学习率"
                min={0.0001}
                max={1}
                initialValue={0.001}
                fieldProps={{ step: 0.0001, precision: 4 }}
                rules={[{ required: true, message: '请输入学习率' }]}
              />
            </Col>
            <Col span={8}>
              <ProFormDigit
                name="batchSize"
                label="批次大小"
                placeholder="请输入批次大小"
                min={1}
                max={1024}
                initialValue={32}
                rules={[{ required: true, message: '请输入批次大小' }]}
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormSlider
                name="privacyBudget"
                label="隐私预算"
                min={0.1}
                max={10}
                step={0.1}
                initialValue={1.0}
                marks={{
                  0.1: '0.1',
                  1: '1.0',
                  5: '5.0',
                  10: '10.0',
                }}
                rules={[{ required: true, message: '请设置隐私预算' }]}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="aggregationMethod"
                label="聚合方法"
                placeholder="请选择聚合方法"
                initialValue="FedAvg"
                options={[
                  { label: 'FedAvg', value: 'FedAvg' },
                  { label: 'FedProx', value: 'FedProx' },
                  { label: 'Weighted Average', value: 'weighted_avg' },
                ]}
                rules={[{ required: true, message: '请选择聚合方法' }]}
              />
            </Col>
          </Row>
        </ProForm>
      </Modal>

      {/* 训练详情模态框 */}
      <Modal
        title="训练任务详情"
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
        ]}
        width={900}
      >
        {selectedJob && (
          <div>
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={6}>
                <Statistic
                  title="当前准确率"
                  value={selectedJob.accuracy * 100}
                  precision={2}
                  suffix="%"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="当前损失"
                  value={selectedJob.loss}
                  precision={3}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="训练进度"
                  value={selectedJob.progress}
                  suffix="%"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="数据集大小"
                  value={selectedJob.datasetSize}
                  formatter={(value) => `${value?.toLocaleString()} 条`}
                />
              </Col>
            </Row>
            
            <Descriptions title="基本信息" bordered column={2}>
              <Descriptions.Item label="任务名称">{selectedJob.name}</Descriptions.Item>
              <Descriptions.Item label="算法">{selectedJob.algorithm}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={getStatusConfig(selectedJob.status).color}>
                  {getStatusConfig(selectedJob.status).text}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="创建者">{selectedJob.creator}</Descriptions.Item>
              <Descriptions.Item label="参与方" span={2}>
                {selectedJob.participants.map(p => (
                  <Tag key={p}>{p}</Tag>
                ))}
              </Descriptions.Item>
              <Descriptions.Item label="任务描述" span={2}>
                {selectedJob.description}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {formatDate(selectedJob.createdAt)}
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {selectedJob.startedAt ? formatDate(selectedJob.startedAt) : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedJob.completedAt ? formatDate(selectedJob.completedAt) : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="训练时长">
                {selectedJob.duration ? `${Math.round(selectedJob.duration / 3600)} 小时` : '-'}
              </Descriptions.Item>
            </Descriptions>
            
            <Descriptions title="训练配置" bordered column={2} style={{ marginTop: 16 }}>
              <Descriptions.Item label="学习率">{selectedJob.config.learningRate}</Descriptions.Item>
              <Descriptions.Item label="批次大小">{selectedJob.config.batchSize}</Descriptions.Item>
              <Descriptions.Item label="隐私预算">{selectedJob.config.privacyBudget}</Descriptions.Item>
              <Descriptions.Item label="聚合方法">{selectedJob.config.aggregationMethod}</Descriptions.Item>
              <Descriptions.Item label="总轮数">{selectedJob.totalEpochs}</Descriptions.Item>
              <Descriptions.Item label="当前轮数">{selectedJob.currentEpoch}</Descriptions.Item>
            </Descriptions>
            
            {selectedJob.metrics.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <Title level={5}>训练指标</Title>
                <Row gutter={16}>
                  <Col span={12}>
                    <Line
                      {...getLineConfig()}
                      data={selectedJob.metrics.map(m => ({ epoch: m.epoch, value: m.accuracy, type: '准确率' }))}
                      xField="epoch"
                      yField="value"
                      height={200}
                    />
                  </Col>
                  <Col span={12}>
                    <Line
                      {...getLineConfig()}
                      data={selectedJob.metrics.map(m => ({ epoch: m.epoch, value: m.loss, type: '损失' }))}
                      xField="epoch"
                      yField="value"
                      height={200}
                    />
                  </Col>
                </Row>
              </div>
            )}
          </div>
        )}
      </Modal>
    </PageContainer>
  )
}

export default TrainingPage