import * as React from 'react'
import { useState, useEffect } from 'react'
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
  Tabs,
  Upload,
  InputNumber,
  Switch,
  Timeline,
} from 'antd'
import {
  RocketOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  UploadOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import { Pie } from '@ant-design/plots'
import { useAppStore } from '@store/app'
import * as dayjs from 'dayjs'

const { Title, Text } = Typography
const { Option } = Select
const { TextArea } = Input
// const { TabPane } = Tabs // 已弃用，使用 items 属性替代

interface InferenceJob {
  id: string
  name: string
  description: string
  modelId: string
  modelName: string
  modelVersion: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  progress: number
  totalSamples: number
  processedSamples: number
  accuracy?: number
  precision?: number
  recall?: number
  f1Score?: number
  participants: string[]
  createdAt: string
  startedAt?: string
  completedAt?: string
  duration?: number
  creator: string
  inputDataset: string
  outputPath: string
  config: {
    batchSize: number
    confidenceThreshold: number
    enableAudit: boolean
    privacyLevel: 'low' | 'medium' | 'high'
  }
  results?: {
    predictions: number
    positiveRate: number
    averageConfidence: number
    riskDistribution: Array<{
      level: string
      count: number
      percentage: number
    }>
  }
  auditLog?: Array<{
    timestamp: string
    action: string
    user: string
    details: string
  }>
}

interface ModelInfo {
  id: string
  name: string
  version: string
  algorithm: string
  accuracy: number
  trainingDate: string
  status: 'active' | 'deprecated' | 'testing'
  description: string
}

const InferencePage: React.FC = () => {
  const [form] = Form.useForm()
  const [jobs, setJobs] = useState<InferenceJob[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<InferenceJob | null>(null)
  const { addNotification } = useAppStore()

  // 模拟模型数据
  const mockModels: ModelInfo[] = [
    {
      id: '1',
      name: '信用风险评估模型',
      version: 'v1.2.0',
      algorithm: 'FedAvg',
      accuracy: 0.892,
      trainingDate: dayjs().subtract(7, 'day').toISOString(),
      status: 'active',
      description: '基于多方数据的联邦学习信用风险评估模型',
    },
    {
      id: '2',
      name: '实时欺诈检测',
      version: 'v2.1.0',
      algorithm: 'FedProx',
      accuracy: 0.934,
      trainingDate: dayjs().subtract(3, 'day').toISOString(),
      status: 'active',
      description: '多方联邦学习反欺诈检测模型',
    },
    {
      id: '3',
      name: '客户流失预测模型',
      version: 'v1.0.0',
      algorithm: 'FedAvg',
      accuracy: 0.876,
      trainingDate: dayjs().subtract(1, 'day').toISOString(),
      status: 'testing',
      description: '基于用户行为的客户流失预测联邦模型',
    },
  ]

  // 模拟推理任务数据
  const mockJobs: InferenceJob[] = [
    {
      id: '1',
      name: '信用风险批量评估',
      description: '对新客户申请进行批量信用风险评估',
      modelId: '1',
      modelName: '信用风险评估模型',
      modelVersion: 'v1.2.0',
      status: 'completed',
      progress: 100,
      totalSamples: 10000,
      processedSamples: 10000,
      accuracy: 0.889,
      precision: 0.912,
      recall: 0.867,
      f1Score: 0.889,
      participants: ['银行A', '银行B', '金融机构C'],
      createdAt: dayjs().subtract(2, 'hour').toISOString(),
      startedAt: dayjs().subtract(2, 'hour').add(5, 'minute').toISOString(),
      completedAt: dayjs().subtract(1, 'hour').toISOString(),
      duration: 3300,
      creator: '张三',
      inputDataset: 'credit_applications_20240115.csv',
      outputPath: '/results/credit_risk_predictions_20240115.json',
      config: {
        batchSize: 100,
        confidenceThreshold: 0.7,
        enableAudit: true,
        privacyLevel: 'high',
      },
      results: {
        predictions: 10000,
        positiveRate: 0.234,
        averageConfidence: 0.856,
        riskDistribution: [
          { level: '低风险', count: 6800, percentage: 68.0 },
        { level: '中风险', count: 2340, percentage: 23.4 },
        { level: '高风险', count: 860, percentage: 8.6 },
        ],
      },
      auditLog: [
        {
          timestamp: dayjs().subtract(2, 'hour').toISOString(),
          action: '创建推理任务',
          user: '张三',
          details: '创建信用风险批量评估任务',
        },
        {
          timestamp: dayjs().subtract(2, 'hour').add(5, 'minute').toISOString(),
          action: '开始推理',
          user: '系统',
          details: '开始执行推理任务，使用模型 v1.2.0',
        },
        {
          timestamp: dayjs().subtract(1, 'hour').toISOString(),
          action: '推理完成',
          user: '系统',
          details: '推理任务完成，处理10000条记录',
        },
      ],
    },
    {
      id: '2',
      name: '实时欺诈检测',
      description: '对交易数据进行实时欺诈检测',
      modelId: '2',
      modelName: '反欺诈检测模型',
      modelVersion: 'v2.1.0',
      status: 'running',
      progress: 45,
      totalSamples: 5000,
      processedSamples: 2250,
      participants: ['支付平台A', '电商平台B'],
      createdAt: dayjs().subtract(30, 'minute').toISOString(),
      startedAt: dayjs().subtract(25, 'minute').toISOString(),
      creator: '李四',
      inputDataset: 'transaction_data_realtime.json',
      outputPath: '/results/fraud_detection_realtime.json',
      config: {
        batchSize: 50,
        confidenceThreshold: 0.8,
        enableAudit: true,
        privacyLevel: 'medium',
      },
      auditLog: [
        {
          timestamp: dayjs().subtract(30, 'minute').toISOString(),
          action: '创建推理任务',
          user: '李四',
          details: '创建实时欺诈检测任务',
        },
        {
          timestamp: dayjs().subtract(25, 'minute').toISOString(),
          action: '开始推理',
          user: '系统',
          details: '开始执行推理任务，使用模型 v2.1.0',
        },
      ],
    },
    {
      id: '3',
      name: '客户流失预测分析',
      description: '对现有客户进行流失风险预测',
      modelId: '3',
      modelName: '客户流失预测模型',
      modelVersion: 'v1.0.0',
      status: 'pending',
      progress: 0,
      totalSamples: 8000,
      processedSamples: 0,
      participants: ['电信运营商A', '互联网公司B'],
      createdAt: dayjs().subtract(10, 'minute').toISOString(),
      creator: '王五',
      inputDataset: 'customer_behavior_data.csv',
      outputPath: '/results/churn_prediction_analysis.json',
      config: {
        batchSize: 200,
        confidenceThreshold: 0.6,
        enableAudit: false,
        privacyLevel: 'low',
      },
      auditLog: [
        {
          timestamp: dayjs().subtract(10, 'minute').toISOString(),
          action: '创建推理任务',
          user: '王五',
          details: '创建客户流失预测分析任务',
        },
      ],
    },
  ]

  useEffect(() => {
    setJobs(mockJobs)
    setModels(mockModels)
  }, [])

  // 创建推理任务
  const handleCreateJob = async (values: any) => {
    setLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const selectedModel = models.find(m => m.id === values.modelId)
      
      const newJob: InferenceJob = {
        id: Date.now().toString(),
        name: values.name,
        description: values.description,
        modelId: values.modelId,
        modelName: selectedModel?.name || '',
        modelVersion: selectedModel?.version || '',
        status: 'pending',
        progress: 0,
        totalSamples: values.totalSamples || 0,
        processedSamples: 0,
        participants: values.participants || [],
        createdAt: dayjs().toISOString(),
        creator: '当前用户',
        inputDataset: values.inputDataset || '',
        outputPath: values.outputPath || '',
        config: {
          batchSize: values.batchSize,
          confidenceThreshold: values.confidenceThreshold,
          enableAudit: values.enableAudit,
          privacyLevel: values.privacyLevel,
        },
        auditLog: [
          {
            timestamp: dayjs().toISOString(),
            action: '创建推理任务',
            user: '当前用户',
            details: `创建推理任务 "${values.name}"`,
          },
        ],
      }
      
      setJobs(prev => [newJob, ...prev])
      setCreateModalVisible(false)
      form.resetFields()
      
      addNotification({
        type: 'success',
        title: '推理任务创建成功',
        message: `任务 "${values.name}" 已创建，等待开始推理`,
      })
    } catch (error) {
      addNotification({
        type: 'error',
        title: '创建失败',
        message: '推理任务创建失败，请重试',
      })
    } finally {
      setLoading(false)
    }
  }

  // 开始推�?
  const handleStartInference = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        const updatedJob = {
          ...job,
          status: 'running' as const,
          startedAt: dayjs().toISOString(),
        }
        
        // 添加审计日志
        if (updatedJob.auditLog) {
          updatedJob.auditLog.push({
            timestamp: dayjs().toISOString(),
            action: '开始推理',
            user: '系统',
            details: `开始执行推理任务，使用模型 ${job.modelVersion}`,
          })
        }
        
        return updatedJob
      }
      return job
    }))
    
    addNotification({
      type: 'success',
      title: '推理已开始',
      message: '联邦推理任务已开始执行',
    })
  }

  // 暂停推理
  const handlePauseInference = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        const updatedJob = { ...job, status: 'paused' as const }
        
        if (updatedJob.auditLog) {
          updatedJob.auditLog.push({
            timestamp: dayjs().toISOString(),
            action: '暂停推理',
            user: '当前用户',
            details: '推理任务已暂停',
          })
        }
        
        return updatedJob
      }
      return job
    }))
    
    addNotification({
      type: 'warning',
      title: '推理已暂停',
      message: '推理任务已暂停，可以随时恢复',
    })
  }

  // 停止推理
  const handleStopInference = (jobId: string) => {
    Modal.confirm({
      title: '确认停止推理',
      content: '确定要停止这个推理任务吗？停止后无法恢复。',
      onOk: () => {
        setJobs(prev => prev.map(job => {
          if (job.id === jobId) {
            const updatedJob = {
              ...job,
              status: 'failed' as const,
              completedAt: dayjs().toISOString(),
            }
            
            if (updatedJob.auditLog) {
              updatedJob.auditLog.push({
                timestamp: dayjs().toISOString(),
                action: '停止推理',
                user: '当前用户',
                details: '推理任务已被手动停止',
              })
            }
            
            return updatedJob
          }
          return job
        }))
        
        addNotification({
          type: 'error',
          title: '推理已停止',
          message: '推理任务已被手动停止',
        })
      },
    })
  }

  // 删除任务
  const handleDeleteJob = (jobId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个推理任务吗？此操作不可撤销。',
      onOk: () => {
        setJobs(prev => prev.filter(job => job.id !== jobId))
        addNotification({
          type: 'success',
          title: '任务已删除',
          message: '推理任务已成功删除',
        })
      },
    })
  }

  // 下载结果
  const handleDownloadResults = (job: InferenceJob) => {
    if (job.status !== 'completed') {
      addNotification({
        type: 'warning',
        title: '无法下载',
        message: '只有已完成的推理任务才能下载结果',
      })
      return
    }
    
    // 模拟下载
    const results = {
      jobInfo: {
        name: job.name,
        modelName: job.modelName,
        modelVersion: job.modelVersion,
        completedAt: job.completedAt,
      },
      metrics: {
        accuracy: job.accuracy,
        precision: job.precision,
        recall: job.recall,
        f1Score: job.f1Score,
      },
      results: job.results,
    }
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `inference_results_${job.id}.json`
    a.click()
    URL.revokeObjectURL(url)
    
    addNotification({
      type: 'success',
      title: '下载完成',
      message: '推理结果已下载到本地',
    })
  }

  // 状态配�?
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'pending':
        return { color: 'default', icon: <ClockCircleOutlined />, text: '等待中' }
      case 'running':
        return { color: 'processing', icon: <PlayCircleOutlined />, text: '推理中' }
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

  // 模型状态配�?
  const getModelStatusConfig = (status: string) => {
    switch (status) {
      case 'active':
        return { color: 'success', text: '活跃' }
      case 'deprecated':
        return { color: 'default', text: '已弃用' }
      case 'testing':
        return { color: 'processing', text: '测试中' }
      default:
        return { color: 'default', text: '未知' }
    }
  }

  // 表格列配�?
  const columns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: InferenceJob) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '模型',
      key: 'model',
      width: 150,
      render: (record: InferenceJob) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.modelName}</div>
          <Tag color="blue">{record.modelVersion}</Tag>
        </div>
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
            <Tag key={p} style={{ marginBottom: 2 }}>{p}</Tag>
          ))}
          {participants.length > 2 && (
            <Tag>+{participants.length - 2}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string, record: InferenceJob) => {
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
      render: (record: InferenceJob) => (
        <div>
          <Text style={{ fontSize: 12 }}>
            {record.processedSamples.toLocaleString()}/{record.totalSamples.toLocaleString()}
          </Text>
          {record.status === 'completed' && record.accuracy && (
            <div style={{ fontSize: 12, color: '#52c41a', marginTop: 2 }}>
              准确率 {(record.accuracy * 100).toFixed(1)}%
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
      render: (record: InferenceJob) => (
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
              onClick={() => handleStartInference(record.id)}
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
                onClick={() => handlePauseInference(record.id)}
              >
                暂停
              </Button>
              <Button
                type="text"
                size="small"
                icon={<StopOutlined />}
                danger
                onClick={() => handleStopInference(record.id)}
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
              onClick={() => handleStartInference(record.id)}
            >
              继续
            </Button>
          )}
          
          {record.status === 'completed' && (
            <Button
              type="text"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadResults(record)}
            >
              下载
            </Button>
          )}
          
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            danger
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

  return (
    <div style={{ padding: 24 }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0 }}>
          <RocketOutlined style={{ marginRight: 8 }} />
          联邦推理
        </Title>
        <Text type="secondary">联邦学习模型推理与预测服务</Text>
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
              title="推理中"
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

      {/* 推理任务列表 */}
      <Card
        title="推理任务"
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
        title="创建联邦推理任务"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={700}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
        >
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="请输入推理任务名称" />
          </Form.Item>
          
          <Form.Item
            name="description"
            label="任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
          >
            <TextArea rows={3} placeholder="请描述推理任务的目标和用途" />
          </Form.Item>
          
          <Form.Item
            name="modelId"
            label="选择模型"
            rules={[{ required: true, message: '请选择推理模型' }]}
          >
            <Select placeholder="请选择要使用的模型">
              {models.map(model => {
                const statusConfig = getModelStatusConfig(model.status)
                return (
                  <Option key={model.id} value={model.id}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div>{model.name} {model.version}</div>
                        <div style={{ fontSize: 12, color: '#666' }}>
                          {model.algorithm} | 准确率 {(model.accuracy * 100).toFixed(1)}%
                        </div>
                      </div>
                      <Tag color={statusConfig.color}>
                        {statusConfig.text}
                      </Tag>
                    </div>
                  </Option>
                )
              })}
            </Select>
          </Form.Item>
          
          <Form.Item
            name="participants"
            label="参与方"
            rules={[{ required: true, message: '请选择参与方' }]}
          >
            <Select
              mode="multiple"
              placeholder="请选择参与推理的机构"
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
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="totalSamples"
                label="样本总数"
                rules={[{ required: true, message: '请输入样本总数' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="请输入待推理的样本数量"
                  min={1}
                  max={1000000}
                  formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={value => parseInt(value!.replace(/\$\s?|(,*)/g, '')) as 1 | 1000000}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="batchSize"
                label="批次大小"
                rules={[{ required: true, message: '请输入批次大小' }]}
                initialValue={100}
              >
                <Select>
                  <Option value={50}>50</Option>
                  <Option value={100}>100</Option>
                  <Option value={200}>200</Option>
                  <Option value={500}>500</Option>
                  <Option value={1000}>1000</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="confidenceThreshold"
                label="置信度阈值"
                rules={[{ required: true, message: '请设置置信度阈值' }]}
                initialValue={0.7}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                  parser={value => (Number(value!.replace('%', '')) / 100) as 1 | 0.1}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="privacyLevel"
                label="隐私级别"
                rules={[{ required: true, message: '请选择隐私级别' }]}
                initialValue="medium"
              >
                <Select>
                  <Option value="low">低 - 基础隐私保护</Option>
                  <Option value="medium">中 - 标准隐私保护</Option>
                  <Option value="high">高 - 强化隐私保护</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="inputDataset" label="输入数据集">
            <Upload
              accept=".csv,.json,.parquet"
              beforeUpload={() => false}
              maxCount={1}
            >
              <Button icon={<UploadOutlined />}>选择数据文件</Button>
            </Upload>
          </Form.Item>
          
          <Form.Item
            name="outputPath"
            label="输出路径"
            initialValue="/results/"
          >
            <Input placeholder="请输入结果输出路径" />
          </Form.Item>
          
          <Form.Item
            name="enableAudit"
            label="启用审计"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch />
          </Form.Item>
          
          <div style={{ textAlign: 'right', marginTop: 24 }}>
            <Space>
              <Button onClick={() => {
                setCreateModalVisible(false)
                form.resetFields()
              }}>
                取消
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
              >
                创建任务
              </Button>
            </Space>
          </div>
        </Form>
      </Modal>

      {/* 任务详情弹窗 */}
      <Modal
        title="推理任务详情"
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
                  handleDownloadResults(selectedJob)
                  setDetailVisible(false)
                }
              }}
            >
              下载结果
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
                    <Descriptions.Item label="使用模型">
                      {selectedJob.modelName} {selectedJob.modelVersion}
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
                    <Descriptions.Item label="推理进度">
                      {selectedJob.processedSamples.toLocaleString()}/{selectedJob.totalSamples.toLocaleString()} ({selectedJob.progress}%)
                    </Descriptions.Item>
                    <Descriptions.Item label="输入数据集">
                      {selectedJob.inputDataset}
                    </Descriptions.Item>
                    <Descriptions.Item label="参与方" span={2}>
                      {selectedJob.participants.map(p => (
                        <Tag key={p} style={{ marginBottom: 4 }}>{p}</Tag>
                      ))}
                    </Descriptions.Item>
                    {selectedJob.status === 'completed' && selectedJob.accuracy && (
                      <>
                        <Descriptions.Item label="准确率">
                          <Text strong style={{ color: '#52c41a', fontSize: 16 }}>
                            {(selectedJob.accuracy * 100).toFixed(2)}%
                          </Text>
                        </Descriptions.Item>
                        <Descriptions.Item label="精确率">
                          <Text strong style={{ color: '#1890ff', fontSize: 16 }}>
                            {selectedJob.precision ? (selectedJob.precision * 100).toFixed(2) + '%' : 'N/A'}
                          </Text>
                        </Descriptions.Item>
                        <Descriptions.Item label="召回率">
                          <Text strong style={{ color: '#722ed1', fontSize: 16 }}>
                            {selectedJob.recall ? (selectedJob.recall * 100).toFixed(2) + '%' : 'N/A'}
                          </Text>
                        </Descriptions.Item>
                        <Descriptions.Item label="F1分数">
                          <Text strong style={{ color: '#fa8c16', fontSize: 16 }}>
                            {selectedJob.f1Score ? selectedJob.f1Score.toFixed(3) : 'N/A'}
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
                      <Descriptions.Item label="推理时长" span={2}>
                        {Math.floor(selectedJob.duration / 3600)}小时{Math.floor((selectedJob.duration % 3600) / 60)}分钟
                      </Descriptions.Item>
                    )}
                  </Descriptions>
                ),
              },
              {
                key: 'config',
                label: '推理配置',
                children: (
                  <Descriptions column={2} bordered>
                    <Descriptions.Item label="批次大小">
                      {selectedJob.config.batchSize}
                    </Descriptions.Item>
                    <Descriptions.Item label="置信度阈值">
                      {(selectedJob.config.confidenceThreshold * 100).toFixed(0)}%
                    </Descriptions.Item>
                    <Descriptions.Item label="隐私级别">
                      <Tag color={selectedJob.config.privacyLevel === 'high' ? 'red' : selectedJob.config.privacyLevel === 'medium' ? 'orange' : 'green'}>
                        {selectedJob.config.privacyLevel === 'high' ? '高' : selectedJob.config.privacyLevel === 'medium' ? '中' : '低'}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="审计功能">
                      <Tag color={selectedJob.config.enableAudit ? 'success' : 'default'}>
                        {selectedJob.config.enableAudit ? '已启用' : '未启用'}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="输出路径" span={2}>
                      {selectedJob.outputPath}
                    </Descriptions.Item>
                  </Descriptions>
                ),
              },
              ...(selectedJob.results ? [{
                key: 'results',
                label: '推理结果',
                children: (
                  <>
                    <Row gutter={16} style={{ marginBottom: 16 }}>
                      <Col span={6}>
                        <Card size="small">
                          <Statistic
                            title="预测总数"
                            value={selectedJob.results.predictions}
                            formatter={value => value!.toLocaleString()}
                          />
                        </Card>
                      </Col>
                      <Col span={6}>
                        <Card size="small">
                          <Statistic
                            title="正例率"
                            value={selectedJob.results.positiveRate * 100}
                            precision={1}
                            suffix="%"
                            valueStyle={{ color: '#cf1322' }}
                          />
                        </Card>
                      </Col>
                      <Col span={6}>
                        <Card size="small">
                          <Statistic
                            title="平均置信度"
                            value={selectedJob.results.averageConfidence * 100}
                            precision={1}
                            suffix="%"
                            valueStyle={{ color: '#1890ff' }}
                          />
                        </Card>
                      </Col>
                      <Col span={6}>
                        <Card size="small">
                          <Statistic
                            title="处理时长"
                            value={selectedJob.duration ? Math.floor(selectedJob.duration / 60) : 0}
                            suffix="分钟"
                            valueStyle={{ color: '#722ed1' }}
                          />
                        </Card>
                      </Col>
                    </Row>
                    
                    <Card title="风险分布" size="small">
                      <Pie
                        data={selectedJob.results.riskDistribution}
                        angleField="count"
                        colorField="level"
                        radius={0.8}
                        label={{
                          type: 'outer',
                          content: '{name}: {percentage}%',
                        }}
                        interactions={[{ type: 'element-active' }]}
                        height={300}
                      />
                    </Card>
                  </>
                ),
              }] : []),
              ...(selectedJob.auditLog && selectedJob.auditLog.length > 0 ? [{
                key: 'audit',
                label: '审计日志',
                children: (
                  <Timeline>
                    {selectedJob.auditLog.map((log, index) => (
                      <Timeline.Item
                        key={index}
                        color={log.action.includes('完成') ? 'green' : log.action.includes('失败') || log.action.includes('停止') ? 'red' : 'blue'}
                      >
                        <div>
                          <div style={{ fontWeight: 500 }}>{log.action}</div>
                          <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                            {log.details}
                          </div>
                          <div style={{ fontSize: 12, color: '#999', marginTop: 4 }}>
                            {dayjs(log.timestamp).format('YYYY-MM-DD HH:mm:ss')} | {log.user}
                          </div>
                        </div>
                      </Timeline.Item>
                    ))}
                  </Timeline>
                ),
              }] : []),
            ]}
          />
        )}
      </Modal>
    </div>
  )
}

export default InferencePage