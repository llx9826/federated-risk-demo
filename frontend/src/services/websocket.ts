interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
  id?: string
}

interface WebSocketConfig {
  url: string
  protocols?: string[]
  reconnectInterval?: number
  maxReconnectAttempts?: number
  heartbeatInterval?: number
  debug?: boolean
}

type MessageHandler = (message: WebSocketMessage) => void
type ConnectionHandler = (event: Event) => void
type ErrorHandler = (error: Event) => void

export class WebSocketService {
  private ws: WebSocket | null = null
  private config: WebSocketConfig
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map()
  private connectionHandlers: Set<ConnectionHandler> = new Set()
  private errorHandlers: Set<ErrorHandler> = new Set()
  private reconnectTimer: NodeJS.Timeout | null = null
  private heartbeatTimer: NodeJS.Timeout | null = null
  private reconnectAttempts = 0
  private isConnecting = false
  private isManualClose = false

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectInterval: 5000,
      maxReconnectAttempts: 5,
      heartbeatInterval: 30000,
      debug: false,
      ...config,
    }
  }

  // 连接WebSocket
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      if (this.isConnecting) {
        reject(new Error('Connection already in progress'))
        return
      }

      this.isConnecting = true
      this.isManualClose = false

      try {
        this.ws = new WebSocket(this.config.url, this.config.protocols)

        this.ws.onopen = (event) => {
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.startHeartbeat()
          this.log('WebSocket connected')
          
          this.connectionHandlers.forEach(handler => handler(event))
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            this.log('Failed to parse message:', error)
          }
        }

        this.ws.onclose = (event) => {
          this.isConnecting = false
          this.stopHeartbeat()
          this.log('WebSocket closed:', event.code, event.reason)

          if (!this.isManualClose && this.shouldReconnect()) {
            this.scheduleReconnect()
          }
        }

        this.ws.onerror = (event) => {
          this.isConnecting = false
          this.log('WebSocket error:', event)
          this.errorHandlers.forEach(handler => handler(event))
          reject(new Error('WebSocket connection failed'))
        }

      } catch (error) {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  // 断开连接
  disconnect(): void {
    this.isManualClose = true
    this.clearReconnectTimer()
    this.stopHeartbeat()

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect')
      this.ws = null
    }
  }

  // 发送消息
  send(type: string, data: any): boolean {
    if (!this.isConnected()) {
      this.log('Cannot send message: WebSocket not connected')
      return false
    }

    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now(),
      id: this.generateMessageId(),
    }

    try {
      this.ws!.send(JSON.stringify(message))
      this.log('Message sent:', message)
      return true
    } catch (error) {
      this.log('Failed to send message:', error)
      return false
    }
  }

  // 订阅消息类型
  subscribe(type: string, handler: MessageHandler): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set())
    }
    
    this.messageHandlers.get(type)!.add(handler)
    
    // 返回取消订阅函数
    return () => {
      const handlers = this.messageHandlers.get(type)
      if (handlers) {
        handlers.delete(handler)
        if (handlers.size === 0) {
          this.messageHandlers.delete(type)
        }
      }
    }
  }

  // 取消订阅
  unsubscribe(type: string, handler?: MessageHandler): void {
    if (!handler) {
      this.messageHandlers.delete(type)
      return
    }

    const handlers = this.messageHandlers.get(type)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.messageHandlers.delete(type)
      }
    }
  }

  // 监听连接事件
  onConnection(handler: ConnectionHandler): () => void {
    this.connectionHandlers.add(handler)
    return () => this.connectionHandlers.delete(handler)
  }

  // 监听错误事件
  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler)
    return () => this.errorHandlers.delete(handler)
  }

  // 检查连接状态
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  // 获取连接状态
  getReadyState(): number | null {
    return this.ws?.readyState ?? null
  }

  // 处理消息
  private handleMessage(message: WebSocketMessage): void {
    this.log('Message received:', message)

    // 处理心跳响应
    if (message.type === 'pong') {
      return
    }

    // 分发消息给订阅者
    const handlers = this.messageHandlers.get(message.type)
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message)
        } catch (error) {
          this.log('Message handler error:', error)
        }
      })
    }
  }

  // 开始心跳
  private startHeartbeat(): void {
    if (this.config.heartbeatInterval && this.config.heartbeatInterval > 0) {
      this.heartbeatTimer = setInterval(() => {
        this.send('ping', { timestamp: Date.now() })
      }, this.config.heartbeatInterval)
    }
  }

  // 停止心跳
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  // 是否应该重连
  private shouldReconnect(): boolean {
    return (
      this.config.maxReconnectAttempts! > 0 &&
      this.reconnectAttempts < this.config.maxReconnectAttempts!
    )
  }

  // 安排重连
  private scheduleReconnect(): void {
    this.clearReconnectTimer()
    
    this.reconnectAttempts++
    const delay = this.config.reconnectInterval! * Math.pow(2, this.reconnectAttempts - 1)
    
    this.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`)
    
    this.reconnectTimer = setTimeout(() => {
      this.log(`Reconnect attempt ${this.reconnectAttempts}`)
      this.connect().catch(error => {
        this.log('Reconnect failed:', error)
      })
    }, delay)
  }

  // 清除重连定时器
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  // 生成消息ID
  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  // 日志输出
  private log(...args: any[]): void {
    if (this.config.debug) {
      console.log('[WebSocket]', ...args)
    }
  }
}

// 创建WebSocket服务实例
const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'

export const websocketService = new WebSocketService({
  url: wsUrl,
  reconnectInterval: 5000,
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000,
  debug: process.env.NODE_ENV === 'development',
})

// 页面可见性变化处理
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // 页面隐藏时断开连接
    websocketService.disconnect()
  } else {
    // 页面显示时重新连接
    websocketService.connect().catch(error => {
      console.error('Failed to reconnect WebSocket:', error)
    })
  }
})

// 网络状态变化处理
window.addEventListener('online', () => {
  websocketService.connect().catch(error => {
    console.error('Failed to reconnect WebSocket:', error)
  })
})

window.addEventListener('offline', () => {
  websocketService.disconnect()
})

// 页面卸载时断开连接
window.addEventListener('beforeunload', () => {
  websocketService.disconnect()
})

export default websocketService