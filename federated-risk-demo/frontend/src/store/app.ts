import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface AppState {
  // 主题设置
  theme: 'light' | 'dark'
  setTheme: (theme: 'light' | 'dark') => void
  
  // 侧边栏状态
  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  
  // 用户信息
  user: {
    id?: string
    name?: string
    role?: string
    avatar?: string
  } | null
  setUser: (user: AppState['user']) => void
  
  // 系统设置
  settings: {
    autoRefresh: boolean
    refreshInterval: number // 秒
    showNotifications: boolean
    language: 'zh-CN' | 'en-US'
  }
  updateSettings: (settings: Partial<AppState['settings']>) => void
  
  // 全局加载状态
  globalLoading: boolean
  setGlobalLoading: (loading: boolean) => void
  
  // 通知消息
  notifications: Array<{
    id: string
    type: 'success' | 'error' | 'warning' | 'info'
    title: string
    message: string
    timestamp: number
    read: boolean
  }>
  addNotification: (notification: Omit<AppState['notifications'][0], 'id' | 'timestamp' | 'read'>) => void
  markNotificationRead: (id: string) => void
  clearNotifications: () => void
  
  // 系统状态
  systemStatus: {
    psiService: 'online' | 'offline' | 'error'
    consentService: 'online' | 'offline' | 'error'
    trainService: 'online' | 'offline' | 'error'
    servingService: 'online' | 'offline' | 'error'
    lastCheck: number
  }
  updateSystemStatus: (status: Partial<AppState['systemStatus']>) => void
  
  // 重置应用状态
  reset: () => void
}

const initialState = {
  theme: 'light' as const,
  sidebarCollapsed: false,
  user: null,
  settings: {
    autoRefresh: true,
    refreshInterval: 30,
    showNotifications: true,
    language: 'zh-CN' as const,
  },
  globalLoading: false,
  notifications: [],
  systemStatus: {
    psiService: 'offline' as const,
    consentService: 'offline' as const,
    trainService: 'offline' as const,
    servingService: 'offline' as const,
    lastCheck: 0,
  },
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      setTheme: (theme) => set({ theme }),
      
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
      
      setUser: (user) => set({ user }),
      
      updateSettings: (newSettings) => set((state) => ({
        settings: { ...state.settings, ...newSettings }
      })),
      
      setGlobalLoading: (globalLoading) => set({ globalLoading }),
      
      addNotification: (notification) => {
        const id = `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        const newNotification = {
          ...notification,
          id,
          timestamp: Date.now(),
          read: false,
        }
        
        set((state) => ({
          notifications: [newNotification, ...state.notifications].slice(0, 50) // 最多保留50条
        }))
      },
      
      markNotificationRead: (id) => set((state) => ({
        notifications: state.notifications.map(n => 
          n.id === id ? { ...n, read: true } : n
        )
      })),
      
      clearNotifications: () => set({ notifications: [] }),
      
      updateSystemStatus: (status) => set((state) => ({
        systemStatus: { 
          ...state.systemStatus, 
          ...status,
          lastCheck: Date.now()
        }
      })),
      
      reset: () => set(initialState),
    }),
    {
      name: 'federated-risk-app-store',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        user: state.user,
        settings: state.settings,
      }),
    }
  )
)

// 选择器函数
export const selectTheme = (state: AppState) => state.theme
export const selectUser = (state: AppState) => state.user
export const selectSettings = (state: AppState) => state.settings
export const selectSystemStatus = (state: AppState) => state.systemStatus
export const selectNotifications = (state: AppState) => state.notifications
export const selectUnreadNotifications = (state: AppState) => 
  state.notifications.filter(n => !n.read)
export const selectIsAllServicesOnline = (state: AppState) => {
  const { psiService, consentService, trainService, servingService } = state.systemStatus
  return [psiService, consentService, trainService, servingService].every(status => status === 'online')
}