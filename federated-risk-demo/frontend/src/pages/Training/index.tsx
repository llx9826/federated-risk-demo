import React, { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Button,
  Select,
  Table,
  Tag,
  Space,
  Modal,
  Descriptions,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Tabs,
  Steps,
  InputNumber,
  Switch,
  Slider,
  Upload,
} from 'antd'
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  UploadOutlined,
  SettingOutlined,
  LineChartOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import { Line, Column } from '@ant-design/plots'
import { useAppStore } from '@store/app'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { Option } = Select
const { TextArea } = Input
// const { TabPane } = Tabs // 已弃用，使用items属性替代
const { Step } = Steps

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
  const [form] = Form.useForm()
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const { addNotification } = useAppStore()

  // 模拟训练任务数据
  const mockJobs: TrainingJob[] = [
    {
      id: '1',
      name: '信用风险评估模型',
      description: '基于多方数据的联邦学习信用风险评估模型训练',
      algorithm: 'FedAvg',
      status: 'completed',
      progress: 100,
      currentEpoch: 50,
      totalEpochs: 50,
      accuracy: 0.892,
      loss: 0.234,
      participants: ['银行A', '银行B', '金融机构C'],
      datasetSize: 150000,
      createdAt: dayjs().subtract(3, 'day').toISOString(),
      startedAt: dayjs().subtract(3, 'day').add(5, 'minute').toISOString(),
      completedAt: dayjs().subtract(2, 'day').toISOString(),
      duration: 7200,
      creator: '张三',
      config: {
        learningRate: 0.001,
        batchSize: 32,
        privacyBudget: 1.0,
        aggregationMethod: 'weighted_average',
      },
      metrics: Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        accuracy: 0.6 + (i * 0.006) + Math.random() * 0.02,
        loss: 2.0 - (i * 0.035) + Math.random() * 0.1,
        timestamp: dayjs().subtract(3, 'day').add(i * 2, 'minute').toISOString(),
      })),
    },
    {
      id: '2',
      name: '反欺诈检测模型',
      description: '多方联邦学习反欺诈检测模型',
      algorithm: 'FedProx',
      status: 'running',
      progress: 65,
      currentEpoch: 32,
      totalEpochs: 50,
      accuracy: 0.834,
      loss: 0.412,
      participants: ['支付平台A', '电商平台B'],
      datasetSize: 80000,
      createdAt: dayjs().subtract(2, 'hour').toISOString(),
      startedAt: dayjs().subtract(1, 'hour').toISOString(),
      creator: '李四',
      config: {
        learningRate: 0.002,
        batchSize: 64,
        privacyBudget: 0.8,
        aggregationMethod: 'fedprox',
      },
      metrics: Array.from({ length: 32 }, (_, i) => ({
        epoch: i + 1,
        accuracy: 0.65 + (i * 0.005) + Math.random() * 0.015,
        loss: 1.8 - (i * 0.04) + Math.random() * 0.08,
        timestamp: dayjs().subtract(2, 'hour').add(i * 1.5, 'minute').toISOString(),
      })),
    },
    {
      id: '3',
      name: '客户流失预测模型',
      description: '基于用户行为的客户流失预测联邦模型',
      algorithm: 'FedAvg',
      status: 'pending',
      progress: 0,
      currentEpoch: 0,
      totalEpochs: 30,
      accuracy: 0,
      loss: 0,
      participants: ['电信运营商A', '互联网公司B'],
      datasetSize: 200000,
      createdAt: dayjs().subtract(30, 'minute').toISOString(),
      creator: '王五',
      config: {
        learningRate: 0.0015,
        batchSize: 128,
        privacyBudget: 1.2,
        aggregationMethod: 'weighted_average',
      },
      metrics: [],
    },
  ]

  useEffect(() => {
    setJobs(mockJobs)
  }, [])

  // 创建训练任务
  const handleCreateJob = async (values: any) => {
    setLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      
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
      setCurrentStep(0)
      
      addNotification({
        type: 'success',
        title: '训练任务创建成功',
        message: `任务 "${values.name}" 已创建，等待开始训练`,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '创建失败',
        message: '训练任务创建失败，请重试',
      })
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
    
    addNotification({
      type: 'success',
      title: '训练已开始',
      message: '联邦学习训练任务已开始执行',
    })
  }

  // 暂停训练
  const handlePauseTraining = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return { ...job, status: 'paused' as const }
      }
      return job
    }))
    
    addNotification({
      type: 'warning',
      title: '训练已暂停',
      message: '训练任务已暂停，可以随时恢复',
    })
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
        
        addNotification({
          type: 'error',
          title: '训练已停止',
          message: '训练任务已被手动停止',
        })
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
        addNotification({
          type: 'success',
          title: '任务已删除',
          message: '训练任务已成功删除',
        })
      },
    })
  }

  // 下载模型
  const handleDownloadModel = (job: TrainingJob) => {
    if (job.status !== 'completed') {
      addNotification({
        type: 'warning',
        title: '无法下载',
        message: '只有已完成的训练任务才能下载模型',
      })
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
    
    addNotification({
      type: 'success',
      title: '下载完成',
      message: '模型文件已下载到本地',
    })
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
  const columns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: TrainingJob) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
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
      render: (algorithm: string) => (
        <Tag color="blue">{algorithm}</Tag>
      ),
    },
    {
      title: '参与方',
      dataIndex: 'participants',
      key: 'participants',
      width: 150,
      render: (participants: string[]) => (
        <div>
          {participants.slice(0, 2).map(p => (
            <Tag key={p} size="small" style={{ marginBottom: 2 }}>{p}</Tag>
          ))}
          {participants.length > 2 && (
            <Tag size="small">+{participants.length - 2}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string, record: TrainingJob) => {
        const config = getStatusConfig(status)
        return (
          <div>
            <Tag icon={config.icon} color={config.color}>
              {config.text}
            </Tag>
            {status === 'running' && (
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
      title: '进度',
      key: 'progress',
      width: 120,
      render: (record: TrainingJob) => (
        <div>
          <Text style={{ fontSize: 12 }}>
            {record.currentEpoch}/{record.totalEpochs} epochs
          </Text>
          {record.status === 'completed' && (
            <div style={{ fontSize: 12, color: '#52c41a', marginTop: 2 }}>
              准确率: {(record.accuracy * 100).toFixed(1)}%
            </div>
          )}
        </div>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 120,
      render: (time: string) => dayjs(time).format('MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (record: TrainingJob) => (
        <Space>
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedJob(record)
              setDetailVisible(true)
            }}
          >
            详情
          </Button>
          
          {record.status === 'pending' && (
            <Button
              type="text"
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
                type="text"
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handlePauseTraining(record.id)}
              >
                暂停
              </Button>
              <Button
                type="text"
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
              type="text"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStartTraining(record.id)}
            >
              继续
            </Button>
          )}
          
          {record.status === 'completed' && (
            <Button
              type="text"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadModel(record)}
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
    running: jobs.filter(job => job.status === 'running').length,
    completed: jobs.filter(job => job.status === 'completed').length,
    pending: jobs.filter(job => job.status === 'pending').length,
  }

  // 创建任务步骤
  const steps = [
    {
      title: '基本信息',
      content: (
        <>
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="请输入训练任务名称" />
          </Form.Item>
          
          <Form.Item
            name="description"
            label="任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
          >
            <TextArea rows={3} placeholder="请描述训练任务的目标和用途" />
          </Form.Item>
          
          <Form.Item
            name="algorithm"
            label="联邦算法"
            rules={[{ required: true, message: '请选择联邦算法' }]}
          >
            <Select placeholder="请选择联邦学习算法">
              <Option value="FedAvg">FedAvg - 联邦平均</Option>
              <Option value="FedProx">FedProx - 联邦近端</Option>
              <Option value="FedNova">FedNova - 联邦新星</Option>
              <Option value="SCAFFOLD">SCAFFOLD - 控制变量</Option>
            </Select>
          </Form.Item>
        </>
      ),
    },
    {
      title: '参与方配置',
      content: (
        <>
          <Form.Item
            name="participants"
            label="参与方"
            rules={[{ required: true, message: '请选择参与方' }]}
          >
            <Select
              mode="multiple"
              placeholder="请选择参与训练的机构"
              options={[
                { label: '银行A', value: '银行A' },
                { label: '银行B', value: '银行B' },
                { label: '金融机构C', value: '金融机构C' },
                { label: '支付平台A', value: '支付平台A' },
                { label: '电商平台B', value: '电商平台B' },
                { label: '电信运营商A', value: '电信运营商A' },
                { label: '互联网公司B', value: '互联网公司B' },
              ]}
            />
          </Form.Item>
          
          <Form.Item
            name="datasetSize"
            label="数据集大小"
            rules={[{ required: true, message: '请输入数据集大小' }]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder="请输入总数据量"
              min={1000}
              max={10000000}
              formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
              parser={value => value!.replace(/\$\s?|(,*)/g, '')}
            />
          </Form.Item>
          
          <Form.Item name="dataUpload" label="数据文件">
            <Upload
              accept=".csv,.json,.parquet"
              beforeUpload={() => false}
              multiple
            >
              <Button icon={<UploadOutlined />}>选择数据文件</Button>
            </Upload>
          </Form.Item>
        </>
      ),
    },
    {
      title: '训练参数',
      content: (
        <>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="totalEpochs"
                label="训练轮数"
                rules={[{ required: true, message: '请输入训练轮数' }]}
                initialValue={50}
              >
                <InputNumber min={1} max={1000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="batchSize"
                label="批次大小"
                rules={[{ required: true, message: '请输入批次大小' }]}
                initialValue={32}
              >
                <Select>
                  <Option value={16}>16</Option>
                  <Option value={32}>32</Option>
                  <Option value={64}>64</Option>
                  <Option value={128}>128</Option>
                  <Option value={256}>256</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="learningRate"
            label="学习率"
            rules={[{ required: true, message: '请设置学习率' }]}
            initialValue={0.001}
          >
            <Slider
              min={0.0001}
              max={0.01}
              step={0.0001}
              marks={{
                0.0001: '0.0001',
                0.001: '0.001',
                0.01: '0.01',
              }}
            />
          </Form.Item>
          
          <Form.Item
            name="privacyBudget"
            label="隐私预算 (ε)"
            rules={[{ required: true, message: '请设置隐私预算' }]}
            initialValue={1.0}
          >
            <Slider
              min={0.1}
              max={5.0}
              step={0.1}
              marks={{
                0.1: '0.1',
                1.0: '1.0',
                5.0: '5.0',
              }}
            />
          </Form.Item>
          
          <Form.Item
            name="aggregationMethod"
            label="聚合方法"
            rules={[{ required: true, message: '请选择聚合方法' }]}
            initialValue="weighted_average"
          >
            <Select>
              <Option value="weighted_average">加权平均</Option>
              <Option value="simple_average">简单平均</Option>
              <Option value="fedprox">FedProx聚合</Option>
              <Option value="scaffold">SCAFFOLD聚合</Option>
            </Select>
          </Form.Item>
        </>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          <ExperimentOutlined style={{ marginRight: 8 }} />
          联邦训练
        </Title>
        <Text type="secondary">联邦学习模型训练与管理</Text>
      </div>

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
              title="训练中"
              value={stats.running}
              valueStyle={{ color: '#1890ff' }}
            />
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
              title="等待中"
              value={stats.pending}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 任务列表 */}
      <Card
        title="训练任务"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            创建任务
          </Button>
        }
      >
        <Table
          dataSource={jobs}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
        />
      </Card>

      {/* 创建任务弹窗 */}
      <Modal
        title="创建联邦训练任务"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          setCurrentStep(0)
          form.resetFields()
        }}
        footer={null}
        width={800}
      >
        <Steps current={currentStep} style={{ marginBottom: 24 }}>
          {steps.map(item => (
            <Step key={item.title} title={item.title} />
          ))}
        </Steps>
        
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
        >
          <div style={{ minHeight: 300 }}>
            {steps[currentStep].content}
          </div>
          
          <div style={{ textAlign: 'right', marginTop: 24 }}>
            <Space>
              {currentStep > 0 && (
                <Button onClick={() => setCurrentStep(currentStep - 1)}>
                  上一步
                </Button>
              )}
              
              <Button onClick={() => {
                setCreateModalVisible(false)
                setCurrentStep(0)
                form.resetFields()
              }}>
                取消
              </Button>
              
              {currentStep < steps.length - 1 ? (
                <Button
                  type="primary"
                  onClick={() => {
                    form.validateFields().then(() => {
                      setCurrentStep(currentStep + 1)
                    })
                  }}
                >
                  下一步
                </Button>
              ) : (
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                >
                  创建任务
                </Button>
              )}
            </Space>
          </div>
        </Form>
      </Modal>

      {/* 任务详情弹窗 */}
      <Modal
        title="训练任务详情"
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
                  handleDownloadModel(selectedJob)
                  setDetailVisible(false)
                }
              }}
            >
              下载模型
            </Button>
          ),
        ]}
        width={900}
      >
        {selectedJob && (
          <Tabs 
            defaultActiveKey="info"
            items={[
              {
                key: 'info',
                label: '基本信息',
                children: (
              <Descriptions column={2} bordered>
                <Descriptions.Item label="任务名称" span={2}>
                  {selectedJob.name}
                </Descriptions.Item>
                <Descriptions.Item label="任务描述" span={2}>
                  {selectedJob.description}
                </Descriptions.Item>
                <Descriptions.Item label="算法">
                  <Tag color="blue">{selectedJob.algorithm}</Tag>
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
                <Descriptions.Item label="训练进度">
                  {selectedJob.currentEpoch}/{selectedJob.totalEpochs} epochs ({selectedJob.progress}%)
                </Descriptions.Item>
                <Descriptions.Item label="数据集大小">
                  {selectedJob.datasetSize.toLocaleString()} 条记录
                </Descriptions.Item>
                <Descriptions.Item label="参与方" span={2}>
                  {selectedJob.participants.map(p => (
                    <Tag key={p} style={{ marginBottom: 4 }}>{p}</Tag>
                  ))}
                </Descriptions.Item>
                {selectedJob.status === 'completed' && (
                  <>
                    <Descriptions.Item label="最终准确率">
                      <Text strong style={{ color: '#52c41a', fontSize: 16 }}>
                        {(selectedJob.accuracy * 100).toFixed(2)}%
                      </Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="最终损失">
                      <Text strong style={{ color: '#1890ff', fontSize: 16 }}>
                        {selectedJob.loss.toFixed(4)}
                      </Text>
                    </Descriptions.Item>
                  </>
                )}
                <Descriptions.Item label="创建者">
                  {selectedJob.creator}
                </Descriptions.Item>
                <Descriptions.Item label="创建时间">
                  {dayjs(selectedJob.createdAt).format('YYYY-MM-DD HH:mm:ss')}
                </Descriptions.Item>
                {selectedJob.startedAt && (
                  <Descriptions.Item label="开始时间">
                    {dayjs(selectedJob.startedAt).format('YYYY-MM-DD HH:mm:ss')}
                  </Descriptions.Item>
                )}
                {selectedJob.completedAt && (
                  <Descriptions.Item label="完成时间">
                    {dayjs(selectedJob.completedAt).format('YYYY-MM-DD HH:mm:ss')}
                  </Descriptions.Item>
                )}
                {selectedJob.duration && (
                  <Descriptions.Item label="训练时长" span={2}>
                    {Math.floor(selectedJob.duration / 3600)}小时{Math.floor((selectedJob.duration % 3600) / 60)}分钟
                  </Descriptions.Item>
                )}
              </Descriptions>
                )
              },
              {
                key: 'config',
                label: '训练配置',
                children: (
              <Descriptions column={2} bordered>
                <Descriptions.Item label="学习率">
                  {selectedJob.config.learningRate}
                </Descriptions.Item>
                <Descriptions.Item label="批次大小">
                  {selectedJob.config.batchSize}
                </Descriptions.Item>
                <Descriptions.Item label="隐私预算">
                  {selectedJob.config.privacyBudget}
                </Descriptions.Item>
                <Descriptions.Item label="聚合方法">
                  {selectedJob.config.aggregationMethod}
                </Descriptions.Item>
                <Descriptions.Item label="总轮数">
                  {selectedJob.totalEpochs}
                </Descriptions.Item>
                <Descriptions.Item label="当前轮数">
                  {selectedJob.currentEpoch}
                </Descriptions.Item>
              </Descriptions>
                )
              },
              ...(selectedJob.metrics.length > 0 ? [{
                key: 'metrics',
                label: '训练指标',
                children: (
                <Row gutter={16}>
                  <Col span={12}>
                    <Card title="准确率变化" size="small">
                      <Line
                        data={selectedJob.metrics}
                        xField="epoch"
                        yField="accuracy"
                        smooth
                        color="#52c41a"
                        height={200}
                        yAxis={{
                          min: Math.min(...selectedJob.metrics.map(m => m.accuracy)) * 0.95,
                        }}
                        tooltip={{
                          formatter: (datum) => {
                            return { name: '准确率', value: `${(datum.accuracy * 100).toFixed(2)}%` }
                          },
                        }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="损失变化" size="small">
                      <Line
                        data={selectedJob.metrics}
                        xField="epoch"
                        yField="loss"
                        smooth
                        color="#ff4d4f"
                        height={200}
                        tooltip={{
                          formatter: (datum) => {
                            return { name: '损失', value: datum.loss.toFixed(4) }
                          },
                        }}
                      />
                    </Card>
                  </Col>
                </Row>
                )
              }] : [])
            ]}
          />
        )}
      </Modal>
    </div>
  )
}

export default TrainingPage