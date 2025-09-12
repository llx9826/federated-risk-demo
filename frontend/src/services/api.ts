import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { messageService } from '@/utils/messageService'

// API基础配置
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_TIMEOUT = 30000

// 服务端口配置
const SERVICE_PORTS = {
  consent: 8000,
  psi: 8001,
  train: 8002,
  explainer: 8003,
}

// 创建不同服务的axios实例
const createServiceInstance = (port: number): AxiosInstance => {
  return axios.create({
    baseURL: `http://localhost:${port}`,
    timeout: API_TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
    },
  })
}

// 主axios实例（同意授权服务）
const axiosInstance: AxiosInstance = createServiceInstance(SERVICE_PORTS.consent)

// PSI服务实例
const psiInstance: AxiosInstance = createServiceInstance(SERVICE_PORTS.psi)

// 训练服务实例
const trainInstance: AxiosInstance = createServiceInstance(SERVICE_PORTS.train)

// 解释服务实例
const explainerInstance: AxiosInstance = createServiceInstance(SERVICE_PORTS.explainer)

// 通用拦截器配置函数
const setupInterceptors = (instance: AxiosInstance) => {
  // 请求拦截器
  instance.interceptors.request.use(
    (config) => {
      // 添加认证token
      const token = localStorage.getItem('auth-token')
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      
      // 添加请求ID用于追踪
      config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse) => {
      return response
    },
    (error) => {
      // 统一错误处理
      if (error.response) {
        const { status, data } = error.response
        
        switch (status) {
          case 401:
            // 未授权，清除token并跳转登录
            localStorage.removeItem('auth-token')
            window.location.href = '/login'
            messageService.error('登录已过期，请重新登录')
            break
          case 403:
            messageService.error('权限不足')
            break
          case 404:
            messageService.error('请求的资源不存在')
            break
          case 500:
            messageService.error('服务器内部错误')
            break
          default:
            messageService.error(data?.message || '请求失败')
        }
      } else if (error.request) {
        messageService.error('网络连接失败，请检查网络')
      } else {
        messageService.error('请求配置错误')
      }
      
      return Promise.reject(error)
    }
  )
}

// 为所有实例设置拦截器
setupInterceptors(axiosInstance)
setupInterceptors(psiInstance)
setupInterceptors(trainInstance)
setupInterceptors(explainerInstance)

// API服务类
export class ApiService {
  private instance: AxiosInstance
  
  constructor(instance: AxiosInstance) {
    this.instance = instance
  }
  
  // 通用请求方法
  async request<T = any>(config: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.request<T>(config)
  }
  
  // GET请求
  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config)
  }
  
  // POST请求
  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, data, config)
  }
  
  // PUT请求
  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.put<T>(url, data, config)
  }
  
  // DELETE请求
  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.delete<T>(url, config)
  }
  
  // PATCH请求
  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.patch<T>(url, data, config)
  }
  
  // 文件上传
  async upload<T = any>(url: string, file: File, onProgress?: (progress: number) => void): Promise<AxiosResponse<T>> {
    const formData = new FormData()
    formData.append('file', file)
    
    return this.instance.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
  }
  
  // 文件下载
  async download(url: string, filename?: string): Promise<void> {
    const response = await this.instance.get(url, {
      responseType: 'blob',
    })
    
    const blob = new Blob([response.data])
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }
  
  // 设置认证token
  setAuthToken(token: string): void {
    this.instance.defaults.headers.common['Authorization'] = `Bearer ${token}`
    localStorage.setItem('auth-token', token)
  }
  
  // 清除认证token
  clearAuthToken(): void {
    delete this.instance.defaults.headers.common['Authorization']
    localStorage.removeItem('auth-token')
  }
  
  // 批量请求
  async batch<T = any>(requests: AxiosRequestConfig[]): Promise<AxiosResponse<T>[]> {
    const promises = requests.map(config => this.instance.request<T>(config))
    return Promise.all(promises)
  }
  
  // 重试请求
  async retry<T = any>(
    config: AxiosRequestConfig,
    maxRetries: number = 3,
    delay: number = 1000
  ): Promise<AxiosResponse<T>> {
    let lastError: any
    
    for (let i = 0; i <= maxRetries; i++) {
      try {
        return await this.instance.request<T>(config)
      } catch (error) {
        lastError = error
        if (i < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
        }
      }
    }
    
    throw lastError
  }
}

// 创建不同服务的API服务实例
const baseApiService = new ApiService(axiosInstance)
const psiApiService = new ApiService(psiInstance)
const trainApiService = new ApiService(trainInstance)
const explainerApiService = new ApiService(explainerInstance)

// 扩展apiService，添加业务模块
export const apiService = Object.assign(baseApiService, {
  consent: {
    getList: (params?: any) => baseApiService.get('/consent', { params }),
    get: (id: string) => baseApiService.get(`/consent/${id}`),
    create: (data: any) => baseApiService.post('/consent/issue', data),
    update: (id: string, data: any) => baseApiService.put(`/consent/${id}`, data),
    delete: (id: string) => baseApiService.delete(`/consent/${id}`),
    approve: (id: string) => baseApiService.post(`/consent/${id}/approve`),
    reject: (id: string, reason: string) => baseApiService.post(`/consent/${id}/reject`, { reason }),
  },
  psi: {
    getAlignments: (params?: any) => psiApiService.get('/psi/sessions', { params }),
    getAlignment: (id: string) => psiApiService.get(`/psi/sessions/${id}`),
    createAlignment: (data: any) => {
      // 转换前端数据格式为PSI服务期望的格式
      const psiRequest = {
        session_id: `psi_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        method: 'ecdh_psi',
        party_role: 'coordinator',
        party_id: 'frontend_client',
        other_parties: data.participants || [],
        timeout_seconds: 3600,
        metadata: {
          name: data.name,
          description: data.description,
          datasetSize: parseInt(data.datasetSize) || 0
        }
      }
      return psiApiService.post('/psi/session', psiRequest)
    },
    startAlignment: (id: string) => psiApiService.post(`/psi/compute`, { session_id: id, party_id: 'frontend_client', return_intersection: true }),
    uploadData: (sessionId: string, data: any) => psiApiService.post('/psi/upload', { session_id: sessionId, ...data }),
    getResults: (sessionId: string) => psiApiService.get(`/psi/results/${sessionId}`),
    getAlignmentResult: (id: string) => psiApiService.get(`/psi/results/${id}`),
    downloadAlignmentResult: (id: string) => psiApiService.download(`/psi/results/${id}/download`),
  },
  federated: {
    getTrainingJobs: (params?: any) => trainApiService.get('/api/train/training/jobs', { params }),
    getTrainingJob: (id: string) => trainApiService.get(`/api/train/training/jobs/${id}`),
    createTrainingJob: (data: any) => trainApiService.post('/api/train/training/jobs', data),
    startTraining: (id: string) => trainApiService.post(`/api/train/training/jobs/${id}/start`),
    stopTraining: (id: string) => trainApiService.post(`/api/train/training/jobs/${id}/stop`),
    getTrainingMetrics: (id: string) => trainApiService.get(`/api/train/training/jobs/${id}/metrics`),
    getTrainingLogs: (id: string) => trainApiService.get(`/api/train/training/jobs/${id}/logs`),
  },
  model: {
    getModels: (params?: any) => baseApiService.get('/api/serving/models', { params }),
    getModel: (id: string) => baseApiService.get(`/api/serving/models/${id}`),
    deployModel: (id: string, config: any) => baseApiService.post(`/api/serving/models/${id}/deploy`, config),
    undeployModel: (id: string) => baseApiService.post(`/api/serving/models/${id}/undeploy`),
    getModelMetrics: (id: string) => baseApiService.get(`/api/serving/models/${id}/metrics`),
    testModel: (id: string, data: any) => baseApiService.post(`/api/serving/models/${id}/test`, data),
    downloadModel: (id: string) => baseApiService.download(`/api/serving/models/${id}/download`),
  },
  explainer: {
    getModels: (params?: any) => explainerApiService.get('/api/explainer/models', { params }),
    getModel: (id: string) => explainerApiService.get(`/api/explainer/models/${id}`),
    explainModel: (id: string, data: any) => explainerApiService.post(`/api/explainer/models/${id}/explain`, data),
    getExplanation: (id: string) => explainerApiService.get(`/api/explainer/explanations/${id}`),
    getFeatureImportance: (modelId: string) => explainerApiService.get(`/api/explainer/models/${modelId}/feature-importance`),
  },
  audit: {
    getAuditLogs: (params?: any) => baseApiService.get('/audit/logs', { params }),
    getAuditLog: (id: string) => baseApiService.get(`/audit/logs/${id}`),
    exportAuditLogs: (params: any) => baseApiService.post('/audit/export', params),
    getComplianceReport: (params?: any) => baseApiService.get('/audit/compliance', { params }),
  },
  system: {
    getSystemStatus: () => baseApiService.get('/system/status'),
    getSystemMetrics: () => baseApiService.get('/system/metrics'),
    getSystemConfig: () => baseApiService.get('/system/config'),
    updateSystemConfig: (config: any) => baseApiService.put('/system/config', config),
    getSystemLogs: (params?: any) => baseApiService.get('/system/logs', { params }),
    healthCheck: () => baseApiService.get('/system/health'),
  },
  auth: {
    login: (credentials: { username: string; password: string }) => baseApiService.post('/auth/login', credentials),
    logout: () => baseApiService.post('/auth/logout'),
    register: (userData: any) => baseApiService.post('/auth/register', userData),
    refreshToken: () => baseApiService.post('/auth/refresh'),
    getProfile: () => baseApiService.get('/auth/me'),
    updateProfile: (data: any) => baseApiService.put('/auth/profile', data),
    changePassword: (data: { oldPassword: string; newPassword: string }) => baseApiService.put('/auth/password', data),
  },
})

// 系统健康检查
export const checkSystemHealth = async () => {
  return apiService.get('/api/system/health')
}

// 具体业务API
export const authAPI = {
  login: (credentials: { username: string; password: string }) =>
    apiService.post('/auth/login', credentials),
  logout: () => apiService.post('/auth/logout'),
  register: (userData: any) => apiService.post('/auth/register', userData),
  refreshToken: () => apiService.post('/auth/refresh'),
  getProfile: () => apiService.get('/auth/me'),
  updateProfile: (data: any) => apiService.put('/auth/profile', data),
  changePassword: (data: { oldPassword: string; newPassword: string }) =>
    apiService.put('/auth/password', data),
}

export const consentAPI = {
  getConsents: (params?: any) => apiService.get('/consent', { params }),
  getConsent: (id: string) => apiService.get(`/consent/${id}`),
  createConsent: (data: any) => apiService.post('/consent/issue', data),
  updateConsent: (id: string, data: any) => apiService.put(`/consent/${id}`, data),
  deleteConsent: (id: string) => apiService.delete(`/consent/${id}`),
  approveConsent: (id: string) => apiService.post(`/consent/${id}/approve`),
  rejectConsent: (id: string, reason: string) =>
    apiService.post(`/consent/${id}/reject`, { reason }),
}

export const psiAPI = {
  getAlignments: (params?: any) => apiService.get('/psi/sessions', { params }),
  getAlignment: (id: string) => apiService.get(`/psi/sessions/${id}`),
  createAlignment: (data: any) => apiService.post('/psi/session', data),
  startAlignment: (id: string) => apiService.post(`/psi/compute`, { session_id: id, party_id: 'frontend' }),
  getAlignmentResult: (id: string) => apiService.get(`/psi/results/${id}`),
  downloadAlignmentResult: (id: string) =>
    apiService.download(`/psi/results/${id}/download`),
}

export const federatedAPI = {
  getTrainingJobs: (params?: any) => apiService.get('/federated/jobs', { params }),
  getTrainingJob: (id: string) => apiService.get(`/federated/jobs/${id}`),
  createTrainingJob: (data: any) => apiService.post('/federated/jobs', data),
  startTraining: (id: string) => apiService.post(`/federated/jobs/${id}/start`),
  stopTraining: (id: string) => apiService.post(`/federated/jobs/${id}/stop`),
  getTrainingMetrics: (id: string) => apiService.get(`/federated/jobs/${id}/metrics`),
  getTrainingLogs: (id: string) => apiService.get(`/federated/jobs/${id}/logs`),
}

export const modelAPI = {
  getModels: (params?: any) => apiService.get('/models', { params }),
  getModel: (id: string) => apiService.get(`/models/${id}`),
  deployModel: (id: string, config: any) =>
    apiService.post(`/models/${id}/deploy`, config),
  undeployModel: (id: string) => apiService.post(`/models/${id}/undeploy`),
  getModelMetrics: (id: string) => apiService.get(`/models/${id}/metrics`),
  testModel: (id: string, data: any) => apiService.post(`/models/${id}/test`, data),
  downloadModel: (id: string) => apiService.download(`/models/${id}/download`),
}

export const auditAPI = {
  getAuditLogs: (params?: any) => apiService.get('/audit/logs', { params }),
  getAuditLog: (id: string) => apiService.get(`/audit/logs/${id}`),
  exportAuditLogs: (params: any) => apiService.post('/audit/export', params),
  getComplianceReport: (params?: any) =>
    apiService.get('/audit/compliance', { params }),
}

export const systemAPI = {
  getSystemStatus: () => apiService.get('/system/status'),
  getSystemMetrics: () => apiService.get('/system/metrics'),
  getSystemConfig: () => apiService.get('/system/config'),
  updateSystemConfig: (config: any) => apiService.put('/system/config', config),
  getSystemLogs: (params?: any) => apiService.get('/system/logs', { params }),
  healthCheck: () => apiService.get('/system/health'),
}