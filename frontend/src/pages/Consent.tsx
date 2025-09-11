import React, { useState, useEffect } from 'react'
import { Table, Button, Space, Tag, Card, Form, Input, Select, DatePicker, Modal, message } from 'antd'
import { PlusOutlined, SearchOutlined, EyeOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '@/services/api'
import { formatDate } from '@/utils'
import '@/styles/consent.css'

const { RangePicker } = DatePicker
const { Option } = Select

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
  const [searchForm] = Form.useForm()
  const [createForm] = Form.useForm()
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
  const columns = [
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
      width: 150,
      render: (dataTypes: string[]) => (
        <div>
          {dataTypes.map((type, index) => (
            <Tag key={index}>
              {type}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '用途',
      dataIndex: 'purpose',
      key: 'purpose',
      width: 150,
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      ),
    },
    {
      title: '过期时间',
      dataIndex: 'expiryDate',
      key: 'expiryDate',
      width: 120,
      render: (date: string) => formatDate(date),
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 120,
      render: (date: string) => formatDate(date),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_: any, record: ConsentRecord) => (
        <Space size="small">
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => setSelectedRecord(record)}
          />
          <Button
            type="text"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          />
          <Button
            type="text"
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record.id)}
          />
        </Space>
      ),
    },
  ]

  // 处理搜索
  const handleSearch = (values: any) => {
    setSearchParams(values)
  }

  // 处理创建
  const handleCreate = (values: any) => {
    createMutation.mutate(values)
  }

  // 处理编辑
  const handleEdit = (record: ConsentRecord) => {
    // TODO: 实现编辑功能
    message.info('编辑功能开发中')
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
    <div>
      {/* 页面标题和操作区域 */}
      <div className="bg-white px-6 py-4 border-b border-gray-100 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-1">同意管理</h1>
          <p className="text-sm text-gray-600">管理数据使用同意记录，确保合规性</p>
        </div>
        <Button
          type="primary"
          size="large"
          icon={<PlusOutlined />}
          onClick={() => setIsCreateModalVisible(true)}
          className="shadow-sm"
        >
          创建同意
        </Button>
      </div>

      <div className="p-6">
        {/* 搜索区域 */}
        <Card className="mb-6 shadow-sm">
          <Form
            form={searchForm}
            layout="inline"
            onFinish={handleSearch}
            className="search-form"
          >
            <Form.Item name="title" label="标题">
              <Input placeholder="请输入标题" allowClear style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="requester" label="申请人">
              <Input placeholder="请输入申请人" allowClear style={{ width: 150 }} />
            </Form.Item>
            <Form.Item name="status" label="状态">
              <Select placeholder="请选择状态" allowClear style={{ width: 120 }}>
                <Option value="pending">待审批</Option>
                <Option value="approved">已批准</Option>
                <Option value="rejected">已拒绝</Option>
                <Option value="expired">已过期</Option>
              </Select>
            </Form.Item>
            <Form.Item name="dateRange" label="创建时间">
              <RangePicker style={{ width: 240 }} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>
                  搜索
                </Button>
                <Button onClick={() => searchForm.resetFields()}>
                  重置
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Card>

      {/* 表格 */}
        <Card className="shadow-sm">
          <Table
            columns={columns}
            dataSource={consents || []}
            rowKey="id"
            loading={isLoading}
            pagination={{
              total: 0,
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 条记录`,
            }}
          />
        </Card>
      </div>

      {/* 创建同意模态框 */}
      <Modal
        title="创建同意记录"
        open={isCreateModalVisible}
        onCancel={() => setIsCreateModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={handleCreate}
        >
          <Form.Item
            name="title"
            label="标题"
            rules={[{ required: true, message: '请输入标题' }]}
          >
            <Input placeholder="请输入同意记录标题" />
          </Form.Item>
          
          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <Input.TextArea rows={3} placeholder="请输入详细描述" />
          </Form.Item>
          
          <Form.Item
            name="purpose"
            label="使用目的"
            rules={[{ required: true, message: '请输入使用目的' }]}
          >
            <Input placeholder="请输入数据使用目的" />
          </Form.Item>
          
          <Form.Item
            name="dataTypes"
            label="数据类型"
            rules={[{ required: true, message: '请选择数据类型' }]}
          >
            <Select
              mode="multiple"
              placeholder="请选择数据类型"
              options={[
                { label: '用户信息', value: 'user_info' },
                { label: '交易记录', value: 'transaction' },
                { label: '行为数据', value: 'behavior' },
                { label: '设备信息', value: 'device' },
                { label: '位置信息', value: 'location' },
              ]}
            />
          </Form.Item>
          
          <Form.Item
            name="expiryDate"
            label="过期时间"
            rules={[{ required: true, message: '请选择过期时间' }]}
          >
            <DatePicker
              style={{ width: '100%' }}
              placeholder="请选择过期时间"
            />
          </Form.Item>
          
          <Form.Item className="mb-0 text-right">
            <Space>
              <Button onClick={() => setIsCreateModalVisible(false)}>
                取消
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                loading={createMutation.isPending}
              >
                创建
              </Button>
            </Space>
          </Form.Item>
        </Form>
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
    </div>
  )
}

export default Consent