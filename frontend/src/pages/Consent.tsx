import React, { useState, useEffect } from 'react'
import { Button, Space, Tag, Modal, message } from 'antd'
import { ProTable, PageContainer, ProForm, ProFormText, ProFormSelect, ProFormDateRangePicker, ProFormTextArea, ProFormDatePicker } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'
import { PlusOutlined, SearchOutlined, EyeOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '@/services/api'
import { formatDate } from '@/utils'

interface ConsentRecord {
  id: string
  title: string
  description: string
  status: 'pending' | 'approved' | 'rejected' | 'expired'
  requester: string
  approver?: string
  dataTypes: string[]
  purpose: string
  expiryDate: string
  createdAt: string
  updatedAt: string
}

const Consent: React.FC = () => {
  const [createForm] = ProForm.useForm()
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [selectedRecord, setSelectedRecord] = useState<ConsentRecord | null>(null)
  const [searchParams, setSearchParams] = useState({})
  
  const queryClient = useQueryClient()

  // 获取同意记录列表
  const { data: consents, isLoading } = useQuery({
    queryKey: ['consents', searchParams],
    queryFn: async () => {
      const response = await apiService.consent.getList(searchParams)
      return response.data.consents || []
    },
  })

  // 创建同意记录
  const createMutation = useMutation({
    mutationFn: apiService.consent.create,
    onSuccess: () => {
      message.success('同意记录创建成功')
      setIsCreateModalVisible(false)
      createForm.resetFields()
      queryClient.invalidateQueries({ queryKey: ['consents'] })
    },
    onError: () => {
      message.error('创建失败，请重试')
    },
  })

  // 删除同意记录
  const deleteMutation = useMutation({
    mutationFn: apiService.consent.delete,
    onSuccess: () => {
      message.success('删除成功')
      queryClient.invalidateQueries({ queryKey: ['consents'] })
    },
    onError: () => {
      message.error('删除失败，请重试')
    },
  })

  // 状态颜色映射
  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'orange',
      approved: 'green',
      rejected: 'red',
      expired: 'gray',
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  // 状态文本映射
  const getStatusText = (status: string) => {
    const texts = {
      pending: '待审批',
      approved: '已批准',
      rejected: '已拒绝',
      expired: '已过期',
    }
    return texts[status as keyof typeof texts] || status
  }

  // 表格列配置
  const columns: ProColumns<ConsentRecord>[] = [
    {
      title: '标题',
      dataIndex: 'title',
      key: 'title',
      width: 200,
    },
    {
      title: '申请人',
      dataIndex: 'requester',
      key: 'requester',
      width: 120,
    },
    {
      title: '数据类型',
      dataIndex: 'dataTypes',
      key: 'dataTypes',
      width: 200,
      render: (_, record) => (
        <div>
          {record.dataTypes.map((type, index) => (
            <Tag key={index} style={{ marginBottom: 4 }}>
              {type}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '使用目的',
      dataIndex: 'purpose',
      key: 'purpose',
      width: 200,
      ellipsis: true,
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
      title: '过期时间',
      dataIndex: 'expiryDate',
      key: 'expiryDate',
      width: 120,
      render: (_, record) => formatDate(record.expiryDate),
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
      key: 'action',
      width: 200,
      valueType: 'option',
      render: (_, record) => (
        <Space size="middle">
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => setSelectedRecord(record)}
          >
            查看
          </Button>
          <Button
            type="link"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record.id)}
          >
            编辑
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
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

  // 处理编辑
  const handleEdit = (id: string) => {
    // TODO: 实现编辑功能
    message.info('编辑功能待实现')
  }

  // 处理删除
  const handleDelete = (id: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这条同意记录吗？此操作不可撤销。',
      onOk: () => deleteMutation.mutate(id),
    })
  }

  return (
    <PageContainer
      title="数据使用同意管理"
      subTitle="管理数据使用同意记录，确保合规性"
      extra={[
        <Button
          key="add"
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setIsCreateModalVisible(true)}
        >
          新增同意
        </Button>,
      ]}
    >
      <ProTable<ConsentRecord>
        columns={columns}
        dataSource={consents || []}
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

      {/* 创建同意模态框 */}
      <Modal
        title="创建同意记录"
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
            name="title"
            label="标题"
            placeholder="请输入同意记录标题"
            rules={[{ required: true, message: '请输入标题' }]}
          />
          
          <ProFormTextArea
            name="description"
            label="描述"
            placeholder="请输入详细描述"
            rules={[{ required: true, message: '请输入描述' }]}
          />
          
          <ProFormText
            name="purpose"
            label="使用目的"
            placeholder="请输入数据使用目的"
            rules={[{ required: true, message: '请输入使用目的' }]}
          />
          
          <ProFormSelect
            name="dataTypes"
            label="数据类型"
            placeholder="请选择数据类型"
            mode="multiple"
            options={[
              { label: '用户信息', value: 'user_info' },
              { label: '交易记录', value: 'transaction' },
              { label: '行为数据', value: 'behavior' },
              { label: '设备信息', value: 'device' },
              { label: '位置信息', value: 'location' },
            ]}
            rules={[{ required: true, message: '请选择数据类型' }]}
          />
          
          <ProFormDatePicker
            name="expiryDate"
            label="过期时间"
            placeholder="请选择过期时间"
            rules={[{ required: true, message: '请选择过期时间' }]}
          />
        </ProForm>
      </Modal>

      {/* 详情模态框 */}
      <Modal
        title="同意记录详情"
        open={!!selectedRecord}
        onCancel={() => setSelectedRecord(null)}
        footer={[
          <Button key="close" onClick={() => setSelectedRecord(null)}>
            关闭
          </Button>,
        ]}
        width={600}
      >
        {selectedRecord && (
          <div className="space-y-4">
            <div>
              <label className="font-medium">标题：</label>
              <span>{selectedRecord.title}</span>
            </div>
            <div>
              <label className="font-medium">描述：</label>
              <p className="mt-1">{selectedRecord.description}</p>
            </div>
            <div>
              <label className="font-medium">申请人：</label>
              <span>{selectedRecord.requester}</span>
            </div>
            <div>
              <label className="font-medium">审批人：</label>
              <span>{selectedRecord.approver || '暂无'}</span>
            </div>
            <div>
              <label className="font-medium">数据类型：</label>
              <div className="mt-1">
                {selectedRecord.dataTypes.map((type, index) => (
                  <Tag key={index}>{type}</Tag>
                ))}
              </div>
            </div>
            <div>
              <label className="font-medium">使用目的：</label>
              <p className="mt-1">{selectedRecord.purpose}</p>
            </div>
            <div>
              <label className="font-medium">状态：</label>
              <Tag color={getStatusColor(selectedRecord.status)}>
                {getStatusText(selectedRecord.status)}
              </Tag>
            </div>
            <div>
              <label className="font-medium">过期时间：</label>
              <span>{formatDate(selectedRecord.expiryDate)}</span>
            </div>
            <div>
              <label className="font-medium">创建时间：</label>
              <span>{formatDate(selectedRecord.createdAt)}</span>
            </div>
            <div>
              <label className="font-medium">更新时间：</label>
              <span>{formatDate(selectedRecord.updatedAt)}</span>
            </div>
          </div>
        )}
      </Modal>
    </PageContainer>
  )
}

export default Consent