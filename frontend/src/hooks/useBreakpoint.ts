import { useState, useEffect, useCallback, useMemo } from 'react'
import { Grid } from 'antd'

const { useBreakpoint: useAntdBreakpoint } = Grid

// 自定义断点配置
interface CustomBreakpoints {
  xs: number
  sm: number
  md: number
  lg: number
  xl: number
  xxl: number
}

// 设备类型
type DeviceType = 'mobile' | 'tablet' | 'desktop' | 'large-desktop'

// 屏幕方向
type ScreenOrientation = 'portrait' | 'landscape'

// 响应式状态接口
interface ResponsiveState {
  breakpoint: keyof CustomBreakpoints
  deviceType: DeviceType
  orientation: ScreenOrientation
  width: number
  height: number
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
  isLargeDesktop: boolean
  isPortrait: boolean
  isLandscape: boolean
  pixelRatio: number
  isRetina: boolean
  isTouchDevice: boolean
}

// 默认断点配置（与Antd保持一致）
const DEFAULT_BREAKPOINTS: CustomBreakpoints = {
  xs: 480,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
}

// 计算响应式状态的工具函数
const calculateResponsiveState = (
  width: number, 
  height: number, 
  bp: CustomBreakpoints
): ResponsiveState => {
  // 确定当前断点
  let breakpoint: keyof CustomBreakpoints = 'xs'
  if (width >= bp.xxl) breakpoint = 'xxl'
  else if (width >= bp.xl) breakpoint = 'xl'
  else if (width >= bp.lg) breakpoint = 'lg'
  else if (width >= bp.md) breakpoint = 'md'
  else if (width >= bp.sm) breakpoint = 'sm'
  else breakpoint = 'xs'

  // 确定设备类型
  let deviceType: DeviceType = 'mobile'
  if (width >= bp.xxl) deviceType = 'large-desktop'
  else if (width >= bp.lg) deviceType = 'desktop'
  else if (width >= bp.md) deviceType = 'tablet'
  else deviceType = 'mobile'

  // 确定屏幕方向
  const orientation: ScreenOrientation = width > height ? 'landscape' : 'portrait'

  // 检测设备特性
  const pixelRatio = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
  const isRetina = pixelRatio > 1
  
  // 检测触摸设备
  const isTouchDevice = typeof window !== 'undefined' ? 
    'ontouchstart' in window || navigator.maxTouchPoints > 0 : false

  return {
    breakpoint,
    deviceType,
    orientation,
    width,
    height,
    isMobile: deviceType === 'mobile',
    isTablet: deviceType === 'tablet',
    isDesktop: deviceType === 'desktop',
    isLargeDesktop: deviceType === 'large-desktop',
    isPortrait: orientation === 'portrait',
    isLandscape: orientation === 'landscape',
    pixelRatio,
    isRetina,
    isTouchDevice,
  }
}

/**
 * 响应式断点检查Hook
 * 提供屏幕尺寸检测、设备类型判断、方向检测等功能
 */
export const useBreakpoint = (customBreakpoints?: Partial<CustomBreakpoints>) => {
  const breakpoints = useMemo(() => ({ ...DEFAULT_BREAKPOINTS, ...customBreakpoints }), [customBreakpoints])
  const antdBreakpoints = useAntdBreakpoint()
  
  const [responsiveState, setResponsiveState] = useState<ResponsiveState>(() => {
    if (typeof window === 'undefined') {
      return {
        breakpoint: 'md',
        deviceType: 'desktop',
        orientation: 'landscape',
        width: 1024,
        height: 768,
        isMobile: false,
        isTablet: false,
        isDesktop: true,
        isLargeDesktop: false,
        isPortrait: false,
        isLandscape: true,
        pixelRatio: 1,
        isRetina: false,
        isTouchDevice: false,
      }
    }
    
    return calculateResponsiveState(window.innerWidth, window.innerHeight, breakpoints)
  })

  // 处理窗口大小变化
  const handleResize = useCallback(() => {
    if (typeof window === 'undefined') return
    
    const newState = calculateResponsiveState(
      window.innerWidth,
      window.innerHeight,
      breakpoints
    )
    
    setResponsiveState(newState)
  }, [breakpoints])

  // 处理方向变化
  const handleOrientationChange = useCallback(() => {
    // 延迟处理，因为orientationchange事件可能在尺寸更新前触发
    setTimeout(() => {
      handleResize()
    }, 100)
  }, [handleResize])

  // 监听窗口变化
  useEffect(() => {
    if (typeof window === 'undefined') return

    // 添加事件监听器
    window.addEventListener('resize', handleResize)
    window.addEventListener('orientationchange', handleOrientationChange)

    return () => {
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('orientationchange', handleOrientationChange)
    }
  }, [handleResize, handleOrientationChange])

  // 检查是否匹配指定断点
  const matches = useCallback(
    (breakpoint: keyof CustomBreakpoints | keyof CustomBreakpoints[]) => {
      if (Array.isArray(breakpoint)) {
        return breakpoint.includes(responsiveState.breakpoint)
      }
      return responsiveState.breakpoint === breakpoint
    },
    [responsiveState.breakpoint]
  )

  // 检查是否大于等于指定断点
  const isAtLeast = useCallback(
    (breakpoint: keyof CustomBreakpoints) => {
      const breakpointOrder: (keyof CustomBreakpoints)[] = ['xs', 'sm', 'md', 'lg', 'xl', 'xxl']
      const currentIndex = breakpointOrder.indexOf(responsiveState.breakpoint)
      const targetIndex = breakpointOrder.indexOf(breakpoint)
      return currentIndex >= targetIndex
    },
    [responsiveState.breakpoint]
  )

  // 检查是否小于等于指定断点
  const isAtMost = useCallback(
    (breakpoint: keyof CustomBreakpoints) => {
      const breakpointOrder: (keyof CustomBreakpoints)[] = ['xs', 'sm', 'md', 'lg', 'xl', 'xxl']
      const currentIndex = breakpointOrder.indexOf(responsiveState.breakpoint)
      const targetIndex = breakpointOrder.indexOf(breakpoint)
      return currentIndex <= targetIndex
    },
    [responsiveState.breakpoint]
  )

  // 检查是否在指定断点范围内
  const isBetween = useCallback(
    (min: keyof CustomBreakpoints, max: keyof CustomBreakpoints) => {
      return isAtLeast(min) && isAtMost(max)
    },
    [isAtLeast, isAtMost]
  )

  return {
    // 响应式状态
    ...responsiveState,
    
    // Antd断点状态（兼容性）
    antdBreakpoints,
    
    // 断点配置
    breakpoints,
    
    // 检查方法
    matches,
    isAtLeast,
    isAtMost,
    isBetween,
    
    // 便捷属性（别名）
    current: responsiveState.breakpoint,
    device: responsiveState.deviceType,
    screen: {
      width: responsiveState.width,
      height: responsiveState.height,
      orientation: responsiveState.orientation,
      pixelRatio: responsiveState.pixelRatio,
    },
  }
}

export type { ResponsiveState, DeviceType, ScreenOrientation, CustomBreakpoints }