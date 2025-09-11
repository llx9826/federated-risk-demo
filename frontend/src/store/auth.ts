import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { apiService } from '@services/api'

export interface User {
  id: string
  username: string
  email: string
  role: string
  permissions: string[]
  avatar?: string
  lastLoginTime?: string
}

interface AuthState {
  // 认证状态
  isAuthenticated: boolean
  token: string | null
  user: User | null
  
  // 权限
  permissions: string[]
  
  // 登录
  login: (credentials: { username: string; password: string }) => Promise<void>
  
  // 登出
  logout: () => void
  
  // 设置token
  setToken: (token: string) => void
  
  // 设置用户信息
  setUser: (user: User) => void
  
  // 检查认证状态
  checkAuthStatus: () => Promise<void>
  
  // 检查权限
  hasPermission: (permission: string) => boolean
  hasAnyPermission: (permissions: string[]) => boolean
  hasAllPermissions: (permissions: string[]) => boolean
  
  // 刷新token
  refreshToken: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // 初始状态
      isAuthenticated: false,
      token: null,
      user: null,
      permissions: [],
      
      // 登录
      login: async (credentials) => {
        try {
          const response = await apiService.post('/auth/login', credentials)
          const { access_token, user_id, username, role } = response.data
          
          // 构造用户对象
          const user: User = {
            id: user_id,
            username,
            email: `${username}@example.com`, // 临时邮箱
            role,
            permissions: [], // 根据角色设置权限
            lastLoginTime: new Date().toISOString()
          }
          
          set({
            isAuthenticated: true,
            token: access_token,
            user,
            permissions: user.permissions
          })
          
          // 设置API默认token
          apiService.setAuthToken(access_token)
          
        } catch (error) {
          console.error('登录失败:', error)
          throw error
        }
      },
      
      // 登出
      logout: () => {
        set({
          isAuthenticated: false,
          token: null,
          user: null,
          permissions: []
        })
        
        // 清除API token
        apiService.clearAuthToken()
        
        // 清除本地存储
        localStorage.removeItem('auth-store')
      },
      
      // 设置token
      setToken: (token) => {
        set({ token, isAuthenticated: !!token })
        if (token) {
          apiService.setAuthToken(token)
        } else {
          apiService.clearAuthToken()
        }
      },
      
      // 设置用户信息
      setUser: (user) => {
        set({ 
          user, 
          permissions: user.permissions || [] 
        })
      },
      
      // 检查认证状态
      checkAuthStatus: async () => {
        const { token } = get()
        if (!token) {
          return
        }
        
        try {
          // 验证token并获取用户信息
          const response = await apiService.get('/auth/me')
          const user = response.data
          
          set({
            isAuthenticated: true,
            user,
            permissions: user.permissions || []
          })
          
        } catch (error) {
          console.error('认证状态检查失败:', error)
          // token无效，清除认证状态
          get().logout()
        }
      },
      
      // 检查权限
      hasPermission: (permission) => {
        const { permissions } = get()
        return permissions.includes(permission) || permissions.includes('*')
      },
      
      hasAnyPermission: (permissionList) => {
        const { permissions } = get()
        if (permissions.includes('*')) return true
        return permissionList.some(permission => permissions.includes(permission))
      },
      
      hasAllPermissions: (permissionList) => {
        const { permissions } = get()
        if (permissions.includes('*')) return true
        return permissionList.every(permission => permissions.includes(permission))
      },
      
      // 刷新token
      refreshToken: async () => {
        try {
          const response = await apiService.post('/auth/refresh')
          const { token } = response.data
          
          get().setToken(token)
          
        } catch (error) {
          console.error('刷新token失败:', error)
          get().logout()
          throw error
        }
      },
    }),
    {
      name: 'auth-store',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)