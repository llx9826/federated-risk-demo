import React, { useState, useEffect } from 'react'
import {
  Button,
  Space,
  Tag,
  Modal,
  Descriptions,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Tabs,
  Upload,
  Switch,
  Timeline,
  message,
} from 'antd'
import {
  ProTable,
  PageContainer,
  ProForm,
  ProFormText,
  ProFormTextArea,
  ProFormSelect,
  ProFormDigit,
  ProFormUploadButton,
  ProFormSwitch,
} from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import ActionBar from '@/components/ActionBar'
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
import dayjs from 'dayjs'

const { Title, Text } = Typography

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
  const { addNotification } = useAppStore()
  const [form] = ProForm.useForm()
  const [jobs, setJobs] = useState<InferenceJob[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [detailVisible, setDetailVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<InferenceJob | null>(null)

  // 模拟数据
  const mockModels: ModelInfo[] = [
    {
      id: '1',
      name: '联邦风险评估模型',
      version: 'v1.2.0',
      algorithm: 'XGBoost',
      accuracy: 0.89,
      trainingDate: dayjs().subtract(7, 'day').toISOString(),
      status: 'active',
      description: '基于多方数据训练的风险评估模型',
    },
    {
      id: '2',
      name: '反欺诈检测模型',
      version: 'v2.1.0',
      algorithm: 'Random Forest',
      accuracy: 0.92,
      trainingDate: dayjs().subtract(3, 'day').toISOString(),
      status: 'active',
      description: '实时交易欺诈检测模型',
    },
    {
      id: '3',
      name: '客户流失预测模型',
      version: 'v1.0.0',
      algorithm: 'Neural Network',
      accuracy: 0.85,
      trainingDate: dayjs().subtract(14, 'day').toISOString(),
      status: 'testing',
      description: '客户流失风险预测模型',
    },
  ]

  const mockJobs: InferenceJob[] = [
    {
      id: '1',
      name: '风险评估批量推理',
      description: '对新客户数据进行批量风险评估',
      modelId: '1',
      modelName: '联邦风险评估模型',
      modelVersion: 'v1.2.0',
      status: 'completed',
      progress: 100,
      totalSamples: 10000,
      processedSamples: 10000,
      accuracy: 0.89,
      precision: 0.87,
      recall: 0.91,
      f1Score: 0.89,
      participants: ['银行A', '金融机构C'],
      createdAt: dayjs().subtract(2, 'hour').toISOString(),
      startedAt: dayjs().subtract(2, 'hour').add(5, 'minute').toISOString(),
      completedAt: dayjs().subtract(1, 'hour').toISOString(),
      duration: 3300,
      creator: '张三',
      inputDataset: 'customer_data_batch.csv',
      outputPath: '/results/risk_assessment_batch.json',
      config: {
        batchSize: 100,
        confidenceThreshold: 0.7,
        enableAudit: true,
        privacyLevel: 'high',
      },
      results: {
        predictions: 10000,
        positiveRate: 0.23,
        averageConfidence: 0.84,
        riskDistribution: [
          { level: '低风险', count: 7700, percentage: 77 },
          { level: '中风险', count: 1800, percentage: 18 },
          { level: '高风险', count: 500, percentage: 5 },
        ],
      },
    },
  ]

  useEffect(() => {
    setJobs(mockJobs)
    setModels(mockModels)
  }, [])

  // 状态配置
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

  // 模型状态配置
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
      }
      
      setJobs(prev => [newJob, ...prev])
      setCreateModalVisible(false)
      form.resetFields()
      
      message.success('推理任务创建成功')
    } catch (error) {
      message.error('创建失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 开始推理
  const handleStartInference = (jobId: string) => {
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
    
    message.success('推理已开始')
  }

  // 暂停推理
  const handlePauseInference = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return { ...job, status: 'paused' as const }
      }
      return job
    }))
    
    message.success('推理已暂停')
  }

  // 停止推理
  const handleStopInference = (jobId: string) => {
    setJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        return { ...job, status: 'failed' as const }
      }
      return job
    }))
    
    message.success('推理已停止')
  }

  // 删除任务
  const handleDeleteJob = (jobId: string) => {
    setJobs(prev => prev.filter(job => job.id !== jobId))
    message.success('任务已删除')
  }

  // 下载结果
  const handleDownloadResults = (job: InferenceJob) => {
    message.success('推理结果已下载到本地')
  }

  // 表格列配置
  const columns: ProColumns<InferenceJob>[] = [
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
      render: (_, record: InferenceJob) => (
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
      align: 'right',
      render: (_, record: InferenceJob) => (
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
      valueType: 'option',
      render: (_, record: InferenceJob) => (
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

  return (
    <PageContainer
      title="联邦推理"
      subTitle="基于联邦学习模型进行数据推理"
      extra={[
        <Button
          key="create"
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
        >
          创建推理任务
        </Button>,
      ]}
    >
      <ProTable<InferenceJob>
        columns={columns}
        dataSource={jobs}
        loading={loading}
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

      {/* 创建任务弹窗 */}
      <Modal
        title="创建推理任务"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={700}
      >
        <ProForm
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
          submitter={{
            render: () => (
              <ActionBar
                actions={[
                  {
                    key: 'cancel',
                    type: 'default',
                    children: '取消',
                    onClick: () => {
                      setCreateModalVisible(false)
                      form.resetFields()
                    }
                  },
                  {
                    key: 'create',
                    type: 'primary',
                    children: '创建任务',
                    htmlType: 'submit',
                    loading: loading
                  }
                ]}
              />
            )
          }}
        >
          <ProFormText
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
            placeholder="请输入推理任务名称"
          />
          
          <ProFormTextArea
            name="description"
            label="任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
            placeholder="请描述推理任务的目标和用途"
            fieldProps={{ rows: 3 }}
          />
          
          <ProFormSelect
            name="modelId"
            label="选择模型"
            rules={[{ required: true, message: '请选择推理模型' }]}
            placeholder="请选择要使用的模型"
            options={models.map(model => {
              const statusConfig = getModelStatusConfig(model.status)
              return {
                label: (
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
                ),
                value: model.id,
              }
            })}
          />
          
          <ProFormSelect
            name="participants"
            label="参与方"
            rules={[{ required: true, message: '请选择参与方' }]}
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
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="totalSamples"
                label="样本总数"
                rules={[{ required: true, message: '请输入样本总数' }]}
                placeholder="请输入待推理的样本数量"
                min={1}
                max={1000000}
                fieldProps={{
                  formatter: (value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ','),
                  parser: (value) => parseInt(value!.replace(/\$\s?|(,*)/g, '')) as any,
                }}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="batchSize"
                label="批次大小"
                rules={[{ required: true, message: '请输入批次大小' }]}
                initialValue={100}
                options={[
                  { label: '50', value: 50 },
                  { label: '100', value: 100 },
                  { label: '200', value: 200 },
                  { label: '500', value: 500 },
                  { label: '1000', value: 1000 },
                ]}
              />
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <ProFormDigit
                name="confidenceThreshold"
                label="置信度阈值"
                rules={[{ required: true, message: '请设置置信度阈值' }]}
                initialValue={0.7}
                min={0.1}
                max={1.0}
                fieldProps={{
                  step: 0.1,
                  formatter: (value) => `${(Number(value) * 100).toFixed(0)}%`,
                  parser: (value) => (Number(value!.replace('%', '')) / 100) as any,
                }}
              />
            </Col>
            <Col span={12}>
              <ProFormSelect
                name="privacyLevel"
                label="隐私级别"
                rules={[{ required: true, message: '请选择隐私级别' }]}
                initialValue="medium"
                options={[
                  { label: '低 - 基础隐私保护', value: 'low' },
                  { label: '中 - 标准隐私保护', value: 'medium' },
                  { label: '高 - 强化隐私保护', value: 'high' },
                ]}
              />
            </Col>
          </Row>
          
          <ProFormUploadButton
            name="inputDataset"
            label="输入数据集"
            title="选择数据文件"
            max={1}
            fieldProps={{
              accept: '.csv,.json,.parquet',
              beforeUpload: () => false,
            }}
          />
          
          <ProFormText
            name="outputPath"
            label="输出路径"
            initialValue="/results/"
            placeholder="请输入结果输出路径"
          />
          
          <ProFormSwitch
            name="enableAudit"
            label="启用审计"
            initialValue={true}
          />
        </ProForm>
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
              onClick={() => selectedJob && handleDownloadResults(selectedJob)}
            >
              下载结果
            </Button>
          ),
        ]}
        width={800}
      >
        {selectedJob && (
          <Descriptions column={2} bordered>
            <Descriptions.Item label="任务名称" span={2}>
              {selectedJob.name}
            </Descriptions.Item>
            <Descriptions.Item label="任务描述" span={2}>
              {selectedJob.description}
            </Descriptions.Item>
            <Descriptions.Item label="模型">
              {selectedJob.modelName} {selectedJob.modelVersion}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={getStatusConfig(selectedJob.status).color}>
                {getStatusConfig(selectedJob.status).text}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="进度">
              {selectedJob.progress}% ({selectedJob.processedSamples.toLocaleString()}/{selectedJob.totalSamples.toLocaleString()})
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {dayjs(selectedJob.createdAt).format('YYYY-MM-DD HH:mm:ss')}
            </Descriptions.Item>
            <Descriptions.Item label="参与方" span={2}>
              {selectedJob.participants.map(p => (
                <Tag key={p}>{p}</Tag>
              ))}
            </Descriptions.Item>
            {selectedJob.results && (
              <>
                <Descriptions.Item label="预测数量">
                  {selectedJob.results.predictions.toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="正例率">
                  {(selectedJob.results.positiveRate * 100).toFixed(1)}%
                </Descriptions.Item>
                <Descriptions.Item label="平均置信度">
                  {(selectedJob.results.averageConfidence * 100).toFixed(1)}%
                </Descriptions.Item>
              </>
            )}
          </Descriptions>
        )}
      </Modal>
    </PageContainer>
  )
}

export default InferencePage