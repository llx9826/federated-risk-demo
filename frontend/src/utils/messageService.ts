import { App } from 'antd'

// Message服务类，用于在非组件中使用antd的message
class MessageService {
  private messageApi: any = null

  // 设置message API实例
  setMessageApi(messageApi: any) {
    this.messageApi = messageApi
  }

  // 成功消息
  success(content: string, duration?: number) {
    if (this.messageApi) {
      this.messageApi.success({ content, duration })
    } else {
      console.warn('Message API not initialized')
    }
  }

  // 错误消息
  error(content: string, duration?: number) {
    if (this.messageApi) {
      this.messageApi.error({ content, duration })
    } else {
      console.warn('Message API not initialized')
    }
  }

  // 警告消息
  warning(content: string, duration?: number) {
    if (this.messageApi) {
      this.messageApi.warning({ content, duration })
    } else {
      console.warn('Message API not initialized')
    }
  }

  // 信息消息
  info(content: string, duration?: number) {
    if (this.messageApi) {
      this.messageApi.info({ content, duration })
    } else {
      console.warn('Message API not initialized')
    }
  }
}

// 导出单例实例
export const messageService = new MessageService()