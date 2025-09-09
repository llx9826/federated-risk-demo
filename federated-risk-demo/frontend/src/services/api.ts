import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { message } from 'antd'
import { useAppStore } from '@store/app'

// API基础配置
const API_CONFIG = {
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
}

// 服务端点配置
export const SERVICE_ENDPOINTS = {
  PSI: '/api/psi',
  CONSENT: '/api/consent',
  TRAIN: '/api/train',
  SERVING: '/api/serving',
}

// 创建axios实例
const createApiInstance = (baseURL: string): AxiosInstance => {
  const instance = axios.create({
    baseURL,
    ...API_CONFIG,
  })

  // 请求拦截器
  instance.interceptors.request.use(
    (config) => {
      // 添加请求ID用于追踪
      config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      // 添加时间戳
      config.headers['X-Timestamp'] = Date.now().toString()
      
      // 如果有用户token，添加到请求头
      const { user } = useAppStore.getState()
      if (user?.id) {
        config.headers['X-User-ID'] = user.id
      }
      
      return config
    },
    (error) => {
      console.error('Request interceptor error:', error)
      return Promise.reject(error)
    }
  )

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse) => {
      // 记录成功响应
      console.log(`API Success [${response.config.method?.toUpperCase()}] ${response.config.url}:`, {
        status: response.status,
        data: response.data,
        requestId: response.config.headers['X-Request-ID'],
      })
      
      return response
    },
    (error) => {
      // 统一错误处理
      const { response, config } = error
      
      console.error(`API Error [${config?.method?.toUpperCase()}] ${config?.url}:`, {
        status: response?.status,
        data: response?.data,
        message: error.message,
        requestId: config?.headers['X-Request-ID'],
      })
      
      // 根据错误状态码显示不同消息
      if (response) {
        switch (response.status) {
          case 400:
            message.error('请求参数错误')
            break
          case 401:
            message.error('未授权访问')
            break
          case 403:
            message.error('访问被拒绝')
            break
          case 404:
            message.error('请求的资源不存在')
            break
          case 500:
            message.error('服务器内部错误')
            break
          case 502:
          case 503:
          case 504:
            message.error('服务暂时不可用')
            break
          default:
            message.error(`请求失败: ${response.status}`)
        }
      } else if (error.code === 'ECONNABORTED') {
        message.error('请求超时，请稍后重试')
      } else {
        message.error('网络错误，请检查网络连接')
      }
      
      return Promise.reject(error)
    }
  )

  return instance
}

// 创建各服务的API实例
export const psiApi = createApiInstance(SERVICE_ENDPOINTS.PSI)
export const consentApi = createApiInstance(SERVICE_ENDPOINTS.CONSENT)
export const trainApi = createApiInstance(SERVICE_ENDPOINTS.TRAIN)
export const servingApi = createApiInstance(SERVICE_ENDPOINTS.SERVING)

// 通用API响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
  timestamp?: number
  request_id?: string
}

// 分页响应类型
export interface PaginatedResponse<T = any> {
  items: T[]
  total: number
  page: number
  size: number
  pages: number
}

// 健康检查响应类型
export interface HealthResponse {
  status: 'healthy' | 'unhealthy'
  timestamp: string
  version: string
  uptime?: number
  dependencies?: Record<string, 'healthy' | 'unhealthy'>
}

// 通用请求方法
export class ApiService {
  private instance: AxiosInstance

  constructor(instance: AxiosInstance) {
    this.instance = instance
  }

  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.get<T>(url, config)
    return response.data
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.post<T>(url, data, config)
    return response.data
  }

  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.put<T>(url, data, config)
    return response.data
  }

  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.delete<T>(url, config)
    return response.data
  }

  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.patch<T>(url, data, config)
    return response.data
  }

  // 健康检查
  async healthCheck(): Promise<HealthResponse> {
    return this.get<HealthResponse>('/health')
  }

  // 获取指标
  async getMetrics(): Promise<Record<string, any>> {
    return this.get('/metrics')
  }
}

// 创建各服务的API服务实例
export const psiService = new ApiService(psiApi)
export const consentService = new ApiService(consentApi)
export const trainService = new ApiService(trainApi)
export const servingService = new ApiService(servingApi)

// 系统状态检查
export const checkSystemHealth = async (): Promise<Record<string, 'online' | 'offline' | 'error'>> => {
  const services = {
    psiService: psiService,
    consentService: consentService,
    trainService: trainService,
    servingService: servingService,
  }

  const results: Record<string, 'online' | 'offline' | 'error'> = {}

  await Promise.allSettled(
    Object.entries(services).map(async ([name, service]) => {
      try {
        const health = await service.healthCheck()
        results[name] = health.status === 'healthy' ? 'online' : 'error'
      } catch (error) {
        console.error(`Health check failed for ${name}:`, error)
        results[name] = 'offline'
      }
    })
  )

  return results
}

// 批量请求工具
export const batchRequest = async <T>(
  requests: Array<() => Promise<T>>,
  options: {
    concurrency?: number
    failFast?: boolean
  } = {}
): Promise<Array<{ success: boolean; data?: T; error?: any }>> => {
  const { concurrency = 3, failFast = false } = options
  const results: Array<{ success: boolean; data?: T; error?: any }> = []
  
  for (let i = 0; i < requests.length; i += concurrency) {
    const batch = requests.slice(i, i + concurrency)
    
    const batchResults = await Promise.allSettled(
      batch.map(request => request())
    )
    
    for (const result of batchResults) {
      if (result.status === 'fulfilled') {
        results.push({ success: true, data: result.value })
      } else {
        results.push({ success: false, error: result.reason })
        
        if (failFast) {
          throw result.reason
        }
      }
    }
  }
  
  return results
}

// 重试请求工具
export const retryRequest = async <T>(
  requestFn: () => Promise<T>,
  options: {
    maxRetries?: number
    delay?: number
    backoff?: boolean
  } = {}
): Promise<T> => {
  const { maxRetries = 3, delay = 1000, backoff = true } = options
  
  let lastError: any
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn()
    } catch (error) {
      lastError = error
      
      if (attempt === maxRetries) {
        break
      }
      
      const waitTime = backoff ? delay * Math.pow(2, attempt) : delay
      await new Promise(resolve => setTimeout(resolve, waitTime))
    }
  }
  
  throw lastError
}

export default {
  psiService,
  consentService,
  trainService,
  servingService,
  checkSystemHealth,
  batchRequest,
  retryRequest,
}