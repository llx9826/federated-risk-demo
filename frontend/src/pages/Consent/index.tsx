import React, { useState } from 'react'
import { Button, Space, Tag, Modal, Input, Select, App } from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { ProTable, PageContainer, ProForm } from '@ant-design/pro-components'
import type { ProColumns } from '@ant-design/pro-components'

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
  const [consents] = useState<ConsentRecord[]>([])
  const [loading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [form] = ProForm.useForm()

  const columns: ProColumns<ConsentRecord>[] = [
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
      render: (_, record) => {
        const color = record.status === 'active' ? 'green' : record.status === 'revoked' ? 'red' : 'orange'
        return <Tag color={color}>{record.status}</Tag>
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
      render: (_: any) => (
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

  const handleSubmit = async () => {
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
    <PageContainer
      title="数据使用同意管理"
      extra={[
        <Button key="add" type="primary" icon={<PlusOutlined />} onClick={handleAddConsent}>
          新增同意
        </Button>,
      ]}
    >
      <ProTable<ConsentRecord>
        columns={columns}
        dataSource={consents}
        loading={loading}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
        }}
        search={false}
        toolBarRender={false}
      />

      <Modal
        title="新增数据使用同意"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <ProForm
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          submitter={false}
        >
          <ProForm.Item
            name="partyId"
            label="参与方ID"
            rules={[{ required: true, message: '请输入参与方ID' }]}
          >
            <Input placeholder="请输入参与方ID" />
          </ProForm.Item>
          
          <ProForm.Item
            name="dataType"
            label="数据类型"
            rules={[{ required: true, message: '请选择数据类型' }]}
          >
            <Select placeholder="请选择数据类型">
              <Select.Option value="user_profile">用户画像</Select.Option>
              <Select.Option value="transaction">交易数据</Select.Option>
              <Select.Option value="behavior">行为数据</Select.Option>
            </Select>
          </ProForm.Item>
          
          <ProForm.Item
            name="purpose"
            label="使用目的"
            rules={[{ required: true, message: '请输入使用目的' }]}
          >
            <Input.TextArea placeholder="请输入使用目的" rows={3} />
          </ProForm.Item>
        </ProForm>
      </Modal>
    </PageContainer>
  )
}

export default ConsentPage