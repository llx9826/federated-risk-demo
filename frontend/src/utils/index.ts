import { message } from 'antd'

// 日期时间工具
export const dateUtils = {
  // 格式化日期
  formatDate: (date: Date | string | number, format: string = 'YYYY-MM-DD'): string => {
    const d = new Date(date)
    if (isNaN(d.getTime())) return ''
    
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    const hours = String(d.getHours()).padStart(2, '0')
    const minutes = String(d.getMinutes()).padStart(2, '0')
    const seconds = String(d.getSeconds()).padStart(2, '0')
    
    return format
      .replace('YYYY', year.toString())
      .replace('MM', month)
      .replace('DD', day)
      .replace('HH', hours)
      .replace('mm', minutes)
      .replace('ss', seconds)
  },
  
  // 格式化相对时间
  formatRelativeTime: (date: Date | string | number): string => {
    const d = new Date(date)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)
    
    if (days > 0) return `${days}天前`
    if (hours > 0) return `${hours}小时前`
    if (minutes > 0) return `${minutes}分钟前`
    return '刚刚'
  },
  
  // 获取时间范围
  getTimeRange: (type: 'today' | 'week' | 'month' | 'year'): [Date, Date] => {
    const now = new Date()
    const start = new Date()
    
    switch (type) {
      case 'today':
        start.setHours(0, 0, 0, 0)
        break
      case 'week':
        start.setDate(now.getDate() - 7)
        break
      case 'month':
        start.setMonth(now.getMonth() - 1)
        break
      case 'year':
        start.setFullYear(now.getFullYear() - 1)
        break
    }
    
    return [start, now]
  },
}

// 数字工具
export const numberUtils = {
  // 格式化数字
  formatNumber: (num: number, decimals: number = 2): string => {
    return num.toLocaleString('zh-CN', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  },
  
  // 格式化百分比
  formatPercentage: (num: number, decimals: number = 2): string => {
    return `${(num * 100).toFixed(decimals)}%`
  },
  
  // 格式化文件大小
  formatFileSize: (bytes: number): string => {
    if (bytes === 0) return '0 B'
    
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
  },
  
  // 生成随机数
  random: (min: number, max: number): number => {
    return Math.floor(Math.random() * (max - min + 1)) + min
  },
  
  // 数字范围限制
  clamp: (num: number, min: number, max: number): number => {
    return Math.min(Math.max(num, min), max)
  },
}

// 字符串工具
export const stringUtils = {
  // 截断字符串
  truncate: (str: string, length: number, suffix: string = '...'): string => {
    if (str.length <= length) return str
    return str.substring(0, length) + suffix
  },
  
  // 首字母大写
  capitalize: (str: string): string => {
    return str.charAt(0).toUpperCase() + str.slice(1)
  },
  
  // 驼峰转下划线
  camelToSnake: (str: string): string => {
    return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`)
  },
  
  // 下划线转驼峰
  snakeToCamel: (str: string): string => {
    return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
  },
  
  // 生成随机字符串
  randomString: (length: number): string => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    let result = ''
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length))
    }
    return result
  },
  
  // 移除HTML标签
  stripHtml: (html: string): string => {
    const div = document.createElement('div')
    div.innerHTML = html
    return div.textContent || div.innerText || ''
  },
}

// 数组工具
export const arrayUtils = {
  // 数组去重
  unique: <T>(arr: T[]): T[] => {
    return [...new Set(arr)]
  },
  
  // 数组分组
  groupBy: <T, K extends keyof T>(arr: T[], key: K): Record<string, T[]> => {
    return arr.reduce((groups, item) => {
      const group = String(item[key])
      groups[group] = groups[group] || []
      groups[group].push(item)
      return groups
    }, {} as Record<string, T[]>)
  },
  
  // 数组分块
  chunk: <T>(arr: T[], size: number): T[][] => {
    const chunks: T[][] = []
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size))
    }
    return chunks
  },
  
  // 数组随机排序
  shuffle: <T>(arr: T[]): T[] => {
    const result = [...arr]
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[result[i], result[j]] = [result[j], result[i]]
    }
    return result
  },
  
  // 数组求和
  sum: (arr: number[]): number => {
    return arr.reduce((sum, num) => sum + num, 0)
  },
  
  // 数组平均值
  average: (arr: number[]): number => {
    return arr.length > 0 ? arrayUtils.sum(arr) / arr.length : 0
  },
}

// 对象工具
export const objectUtils = {
  // 深拷贝
  deepClone: <T>(obj: T): T => {
    if (obj === null || typeof obj !== 'object') return obj
    if (obj instanceof Date) return new Date(obj.getTime()) as unknown as T
    if (obj instanceof Array) return obj.map(item => objectUtils.deepClone(item)) as unknown as T
    
    const cloned = {} as T
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = objectUtils.deepClone(obj[key])
      }
    }
    return cloned
  },
  
  // 对象合并
  merge: <T extends Record<string, any>>(...objects: Partial<T>[]): T => {
    return objects.reduce((result, obj) => ({ ...result, ...obj }), {}) as T
  },
  
  // 获取嵌套属性
  get: (obj: any, path: string, defaultValue?: any): any => {
    const keys = path.split('.')
    let result = obj
    
    for (const key of keys) {
      if (result === null || result === undefined) {
        return defaultValue
      }
      result = result[key]
    }
    
    return result !== undefined ? result : defaultValue
  },
  
  // 设置嵌套属性
  set: (obj: any, path: string, value: any): void => {
    const keys = path.split('.')
    let current = obj
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i]
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {}
      }
      current = current[key]
    }
    
    current[keys[keys.length - 1]] = value
  },
  
  // 删除空值
  removeEmpty: (obj: Record<string, any>): Record<string, any> => {
    const result: Record<string, any> = {}
    
    for (const [key, value] of Object.entries(obj)) {
      if (value !== null && value !== undefined && value !== '') {
        result[key] = value
      }
    }
    
    return result
  },
}

// 验证工具
export const validationUtils = {
  // 邮箱验证
  isEmail: (email: string): boolean => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return regex.test(email)
  },
  
  // 手机号验证
  isPhone: (phone: string): boolean => {
    const regex = /^1[3-9]\d{9}$/
    return regex.test(phone)
  },
  
  // URL验证
  isUrl: (url: string): boolean => {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  },
  
  // 身份证验证
  isIdCard: (idCard: string): boolean => {
    const regex = /(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/
    return regex.test(idCard)
  },
  
  // 密码强度验证
  isStrongPassword: (password: string): boolean => {
    // 至少8位，包含大小写字母、数字和特殊字符
    const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/
    return regex.test(password)
  },
}

// 存储工具
export const storageUtils = {
  // localStorage
  local: {
    set: (key: string, value: any): void => {
      try {
        localStorage.setItem(key, JSON.stringify(value))
      } catch (error) {
        console.error('localStorage set error:', error)
      }
    },
    
    get: <T>(key: string, defaultValue?: T): T | null => {
      try {
        const item = localStorage.getItem(key)
        return item ? JSON.parse(item) : defaultValue || null
      } catch (error) {
        console.error('localStorage get error:', error)
        return defaultValue || null
      }
    },
    
    remove: (key: string): void => {
      try {
        localStorage.removeItem(key)
      } catch (error) {
        console.error('localStorage remove error:', error)
      }
    },
    
    clear: (): void => {
      try {
        localStorage.clear()
      } catch (error) {
        console.error('localStorage clear error:', error)
      }
    },
  },
  
  // sessionStorage
  session: {
    set: (key: string, value: any): void => {
      try {
        sessionStorage.setItem(key, JSON.stringify(value))
      } catch (error) {
        console.error('sessionStorage set error:', error)
      }
    },
    
    get: <T>(key: string, defaultValue?: T): T | null => {
      try {
        const item = sessionStorage.getItem(key)
        return item ? JSON.parse(item) : defaultValue || null
      } catch (error) {
        console.error('sessionStorage get error:', error)
        return defaultValue || null
      }
    },
    
    remove: (key: string): void => {
      try {
        sessionStorage.removeItem(key)
      } catch (error) {
        console.error('sessionStorage remove error:', error)
      }
    },
    
    clear: (): void => {
      try {
        sessionStorage.clear()
      } catch (error) {
        console.error('sessionStorage clear error:', error)
      }
    },
  },
}

// 防抖和节流
export const throttleUtils = {
  // 防抖
  debounce: <T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): ((...args: Parameters<T>) => void) => {
    let timeout: NodeJS.Timeout | null = null
    
    return (...args: Parameters<T>) => {
      if (timeout) clearTimeout(timeout)
      timeout = setTimeout(() => func(...args), wait)
    }
  },
  
  // 节流
  throttle: <T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): ((...args: Parameters<T>) => void) => {
    let lastTime = 0
    
    return (...args: Parameters<T>) => {
      const now = Date.now()
      if (now - lastTime >= wait) {
        lastTime = now
        func(...args)
      }
    }
  },
}

// 颜色工具
export const colorUtils = {
  // 十六进制转RGB
  hexToRgb: (hex: string): { r: number; g: number; b: number } | null => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null
  },
  
  // RGB转十六进制
  rgbToHex: (r: number, g: number, b: number): string => {
    return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`
  },
  
  // 生成随机颜色
  randomColor: (): string => {
    return `#${Math.floor(Math.random() * 16777215).toString(16)}`
  },
}

// 错误处理工具
export const errorUtils = {
  // 显示错误消息
  showError: (error: any, defaultMessage: string = '操作失败'): void => {
    const errorMessage = error?.response?.data?.message || error?.message || defaultMessage
    message.error(errorMessage)
  },
  
  // 显示成功消息
  showSuccess: (msg: string = '操作成功'): void => {
    message.success(msg)
  },
  
  // 显示警告消息
  showWarning: (msg: string): void => {
    message.warning(msg)
  },
  
  // 显示信息消息
  showInfo: (msg: string): void => {
    message.info(msg)
  },
}

// 导出所有工具
export default {
  dateUtils,
  numberUtils,
  stringUtils,
  arrayUtils,
  objectUtils,
  validationUtils,
  storageUtils,
  throttleUtils,
  colorUtils,
  errorUtils,
}

// 导出常用函数
export const formatDate = dateUtils.formatDate