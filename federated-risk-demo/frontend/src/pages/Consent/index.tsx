import React, { useState, useEffect } from 'react'
import { Card, Table, Button, Space, Tag, Modal, Form, Input, Select, App } from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'

interface ConsentRecord {
  id: string
  partyId: string
  dataType: string
  purpose: string
  status: 'active' | 'revoked' | 'expired'
  createdAt: string
  expiresAt: string
}

const ConsentPage: React.FC = () => {
  const [consents, setConsents] = useState<ConsentRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [form] = Form.useForm()

  const columns = [
    {
      title: '参与方ID',
      dataIndex: 'partyId',
      key: 'partyId',
    },
    {
      title: '数据类型',
      dataIndex: 'dataType',
      key: 'dataType',
    },
    {
      title: '使用目的',
      dataIndex: 'purpose',
      key: 'purpose',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'active' ? 'green' : status === 'revoked' ? 'red' : 'orange'
        return <Tag color={color}>{status}</Tag>
      },
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
    },
    {
      title: '过期时间',
      dataIndex: 'expiresAt',
      key: 'expiresAt',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: ConsentRecord) => (
        <Space size="middle">
          <Button type="link" icon={<EditOutlined />} size="small">
            编辑
          </Button>
          <Button type="link" danger icon={<DeleteOutlined />} size="small">
            撤销
          </Button>
        </Space>
      ),
    },
  ]

  const handleAddConsent = () => {
    setModalVisible(true)
  }

  const handleSubmit = async (values: any) => {
    const { message } = App.useApp()
    try {
      // 这里应该调用API
      message.success('同意记录创建成功')
      setModalVisible(false)
      form.resetFields()
    } catch (error) {
      message.error('创建失败')
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Card 
        title="数据使用同意管理" 
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={handleAddConsent}>
            新增同意
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={consents}
          loading={loading}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
          }}
        />
      </Card>

      <Modal
        title="新增数据使用同意"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Form.Item
            name="partyId"
            label="参与方ID"
            rules={[{ required: true, message: '请输入参与方ID' }]}
          >
            <Input placeholder="请输入参与方ID" />
          </Form.Item>
          
          <Form.Item
            name="dataType"
            label="数据类型"
            rules={[{ required: true, message: '请选择数据类型' }]}
          >
            <Select placeholder="请选择数据类型">
              <Select.Option value="user_profile">用户画像</Select.Option>
              <Select.Option value="transaction">交易数据</Select.Option>
              <Select.Option value="behavior">行为数据</Select.Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="purpose"
            label="使用目的"
            rules={[{ required: true, message: '请输入使用目的' }]}
          >
            <Input.TextArea placeholder="请输入使用目的" rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default ConsentPage