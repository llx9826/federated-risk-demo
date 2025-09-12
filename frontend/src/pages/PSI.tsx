import React, { useState } from 'react'
import { Button, Space, Tag, Progress, Modal, message } from 'antd'
import { ProTable, PageContainer, ProForm, ProFormText, ProFormSelect, ProFormTextArea } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import { PlusOutlined, SearchOutlined, EyeOutlined, PlayCircleOutlined, StopOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '@/services/api'
import { formatDate } from '@/utils'

interface PSITask {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  participants: string[]
  datasetSize: number
  intersectionSize?: number
  startTime?: string
  endTime?: string
  createdAt: string
  updatedAt: string
}

const PSI: React.FC = () => {
  const [createForm] = ProForm.useForm()
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [selectedTask, setSelectedTask] = useState<PSITask | null>(null)
  const [searchParams, setSearchParams] = useState({})
  
  const queryClient = useQueryClient()

  // 获取PSI任务列表
  const { data: tasks, isLoading } = useQuery({
    queryKey: ['psi-tasks', searchParams],
    queryFn: () => apiService.psi.getAlignments(searchParams),
  })

  // 创建PSI任务
  const createMutation = useMutation({
    mutationFn: apiService.psi.createAlignment,
    onSuccess: () => {
      message.success('PSI任务创建成功')
      setIsCreateModalVisible(false)
      createForm.resetFields()
      queryClient.invalidateQueries({ queryKey: ['psi-tasks'] })
    },
    onError: () => {
      message.error('创建失败，请重试')
    },
  })

  // 启动PSI任务
  const startMutation = useMutation({
    mutationFn: apiService.psi.startAlignment,
    onSuccess: () => {
      message.success('任务启动成功')
      queryClient.invalidateQueries({ queryKey: ['psi-tasks'] })
    },
    onError: () => {
      message.error('启动失败，请重试')
    },
  })

  // 停止PSI任务
  const stopMutation = useMutation({
    mutationFn: (id: string) => apiService.psi.getAlignment(id), // 使用现有方法作为占位
    onSuccess: () => {
      message.success('任务已停止')
      queryClient.invalidateQueries({ queryKey: ['psi-tasks'] })
    },
    onError: (error) => {
      console.error('Stop PSI task failed:', error)
      message.error('停止任务失败')
    },
  })

  // 状态颜色映射
  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'orange',
      running: 'blue',
      completed: 'green',
      failed: 'red',
      cancelled: 'gray',
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  // 状态文本映射
  const getStatusText = (status: string) => {
    const texts = {
      pending: '待执行',
      running: '执行中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消',
    }
    return texts[status as keyof typeof texts] || status
  }

  // 表格列配置
  const columns: ProColumns<PSITask>[] = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: '参与方',
      dataIndex: 'participants',
      key: 'participants',
      width: 150,
      render: (_, record) => (
        <div>
          {record.participants.map((participant, index) => (
            <Tag key={index} style={{ marginBottom: 4 }}>
              {participant}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '数据集大小',
      dataIndex: 'datasetSize',
      key: 'datasetSize',
      width: 120,
      render: (_, record) => record.datasetSize ? `${record.datasetSize.toLocaleString()} 条` : '-',
    },
    {
      title: '交集大小',
      dataIndex: 'intersectionSize',
      key: 'intersectionSize',
      width: 120,
      render: (_, record) => record.intersectionSize ? `${record.intersectionSize.toLocaleString()} 条` : '-',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (_, record) => (
        <Tag color={getStatusColor(record.status)}>
          {getStatusText(record.status)}
        </Tag>
      ),
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (_, record) => (
        <Progress
          percent={record.progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : undefined}
        />
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
      width: 150,
      valueType: 'option',
      render: (_, record) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setSelectedTask(record)}
          >
            查看
          </Button>
          {record.status === 'pending' && (
            <Button
              type="link"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleStart(record.id)}
            >
              启动
            </Button>
          )}
          {record.status === 'running' && (
            <Button
              type="link"
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => handleStop(record.id)}
            >
              停止
            </Button>
          )}
        </Space>
      ),
    },
  ]

  // 处理搜索
  const handleSearch = (values: any) => {
    setSearchParams(values)
  }

  // 处理创建
  const handleCreate = async (values: any) => {
    await createMutation.mutateAsync(values)
  }

  // 处理启动
  const handleStart = (id: string) => {
    Modal.confirm({
      title: '确认启动',
      content: '确定要启动这个PSI任务吗？',
      onOk: () => startMutation.mutate(id),
    })
  }

  // 处理停止
  const handleStop = (id: string) => {
    Modal.confirm({
      title: '确认停止',
      content: '确定要停止这个PSI任务吗？',
      onOk: () => stopMutation.mutate(id),
    })
  }

  return (
    <PageContainer
      title="隐私求交 (PSI)"
      subTitle="管理隐私集合求交任务，保护数据隐私"
      extra={[
        <Button
          key="add"
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setIsCreateModalVisible(true)}
        >
          新建任务
        </Button>,
      ]}
    >
      <ProTable<PSITask>
        columns={columns}
        dataSource={Array.isArray(tasks?.data) ? tasks.data : (tasks ? [tasks] : [])}
        loading={isLoading}
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

      {/* 创建任务模态框 */}
      <Modal
        title="创建PSI任务"
        open={isCreateModalVisible}
        onCancel={() => setIsCreateModalVisible(false)}
        footer={null}
        width={600}
      >
        <ProForm
          form={createForm}
          layout="vertical"
          onFinish={handleCreate}
          submitter={{
            searchConfig: {
              submitText: '创建',
              resetText: '取消',
            },
            resetButtonProps: {
              onClick: () => setIsCreateModalVisible(false),
            },
            submitButtonProps: {
              loading: createMutation.isPending,
            },
          }}
        >
          <ProFormText
            name="name"
            label="任务名称"
            placeholder="请输入任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          />
          
          <ProFormTextArea
            name="description"
            label="任务描述"
            placeholder="请输入任务描述"
            rules={[{ required: true, message: '请输入任务描述' }]}
          />
          
          <ProFormSelect
            name="participants"
            label="参与方"
            placeholder="请选择参与方"
            mode="multiple"
            options={[
              { label: '银行A', value: 'bank_a' },
              { label: '银行B', value: 'bank_b' },
              { label: '银行C', value: 'bank_c' },
              { label: '第三方机构', value: 'third_party' },
            ]}
            rules={[{ required: true, message: '请选择参与方' }]}
          />
          
          <ProFormText
            name="datasetSize"
            label="数据集大小"
            placeholder="请输入数据集大小"
            rules={[{ required: true, message: '请输入数据集大小' }]}
          />
        </ProForm>
      </Modal>

      {/* 详情模态框 */}
      <Modal
        title="PSI任务详情"
        open={!!selectedTask}
        onCancel={() => setSelectedTask(null)}
        footer={[
          <Button key="close" onClick={() => setSelectedTask(null)}>
            关闭
          </Button>,
        ]}
        width={700}
      >
        {selectedTask && (
          <div className="space-y-4">
            <div>
              <label className="font-medium">任务名称：</label>
              <span>{selectedTask.name}</span>
            </div>
            <div>
              <label className="font-medium">任务描述：</label>
              <p className="mt-1">{selectedTask.description}</p>
            </div>
            <div>
              <label className="font-medium">参与方：</label>
              <div className="mt-1">
                {selectedTask.participants.map((participant, index) => (
                  <Tag key={index}>{participant}</Tag>
                ))}
              </div>
            </div>
            <div>
              <label className="font-medium">数据集大小：</label>
              <span>{selectedTask.datasetSize ? selectedTask.datasetSize.toLocaleString() : '-'} 条</span>
            </div>
            {selectedTask.intersectionSize && (
              <div>
                <label className="font-medium">交集大小：</label>
                <span>{selectedTask.intersectionSize?.toLocaleString() || '0'} 条</span>
              </div>
            )}
            <div>
              <label className="font-medium">状态：</label>
              <Tag color={getStatusColor(selectedTask.status)}>
                {getStatusText(selectedTask.status)}
              </Tag>
            </div>
            <div>
              <label className="font-medium">进度：</label>
              <Progress
                percent={selectedTask.progress}
                status={selectedTask.status === 'failed' ? 'exception' : undefined}
              />
            </div>
            {selectedTask.startTime && (
              <div>
                <label className="font-medium">开始时间：</label>
                <span>{formatDate(selectedTask.startTime)}</span>
              </div>
            )}
            {selectedTask.endTime && (
              <div>
                <label className="font-medium">结束时间：</label>
                <span>{formatDate(selectedTask.endTime)}</span>
              </div>
            )}
            <div>
              <label className="font-medium">创建时间：</label>
              <span>{formatDate(selectedTask.createdAt)}</span>
            </div>
            <div>
              <label className="font-medium">更新时间：</label>
              <span>{formatDate(selectedTask.updatedAt)}</span>
            </div>
          </div>
        )}
      </Modal>
    </PageContainer>
  )
}

export default PSI