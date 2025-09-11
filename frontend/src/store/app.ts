import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  content?: string
  duration?: number
  timestamp: number
  read?: boolean
}

interface AppState {
  // 主题
  theme: 'light' | 'dark'
  setTheme: (theme: 'light' | 'dark') => void
  toggleTheme: () => void

  // 语言
  locale: 'zh-CN' | 'en-US'
  setLocale: (locale: 'zh-CN' | 'en-US') => void

  // 通知
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void

  // 全局加载状态
  globalLoading: boolean
  setGlobalLoading: (loading: boolean) => void

  // 侧边栏
  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  toggleSidebar: () => void

  // 页面标题
  pageTitle: string
  setPageTitle: (title: string) => void

  // 面包屑
  breadcrumbs: Array<{ title: string; path?: string }>
  setBreadcrumbs: (breadcrumbs: Array<{ title: string; path?: string }>) => void

  // 系统状态
  isOnline: boolean
  lastHeartbeat: string | null
  setSystemStatus: (isOnline: boolean, lastHeartbeat?: string) => void
}

// 选择器
export const selectSystemStatus = (state: AppState) => ({
  isOnline: state.isOnline,
  lastHeartbeat: state.lastHeartbeat,
})

export const selectNotifications = (state: AppState) => state.notifications

export const selectUnreadNotifications = (state: AppState) => 
  state.notifications.filter(n => !n.read)

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // 主题
      theme: 'light',
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((state) => ({ 
        theme: state.theme === 'light' ? 'dark' : 'light' 
      })),

      // 语言
      locale: 'zh-CN',
      setLocale: (locale) => set({ locale }),

      // 通知
      notifications: [],
      addNotification: (notification) => {
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9)
        const newNotification: Notification = {
          ...notification,
          id,
          timestamp: Date.now(),
        }
        set((state) => ({
          notifications: [...state.notifications, newNotification]
        }))
        
        // 自动移除通知
        if (notification.duration !== 0) {
          setTimeout(() => {
            get().removeNotification(id)
          }, (notification.duration || 4.5) * 1000)
        }
      },
      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id)
      })),
      clearNotifications: () => set({ notifications: [] }),

      // 全局加载状态
      globalLoading: false,
      setGlobalLoading: (loading) => set({ globalLoading: loading }),

      // 侧边栏
      sidebarCollapsed: false,
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      toggleSidebar: () => set((state) => ({ 
        sidebarCollapsed: !state.sidebarCollapsed 
      })),

      // 页面标题
      pageTitle: '联邦风控系统',
      setPageTitle: (title) => set({ pageTitle: title }),

      // 面包屑管理
      breadcrumbs: [],
      setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs }),
      
      // 系统状态管理
      isOnline: true,
      lastHeartbeat: null,
      setSystemStatus: (isOnline, lastHeartbeat) => set({ 
        isOnline, 
        lastHeartbeat: lastHeartbeat || new Date().toISOString() 
      }),
    }),
    {
      name: 'app-store',
      partialize: (state) => ({
        theme: state.theme,
        locale: state.locale,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
)