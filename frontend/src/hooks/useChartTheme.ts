import { useContext } from 'react'
import { ConfigProvider } from 'antd'
import type { ThemeConfig } from 'antd'

// 图表主题配置接口
interface ChartThemeConfig {
  // 基础颜色
  primaryColor: string
  successColor: string
  warningColor: string
  errorColor: string
  infoColor: string
  
  // 渐变色配置
  gradients: {
    primary: string[]
    success: string[]
    warning: string[]
    error: string[]
    info: string[]
  }
  
  // 图表通用配置
  common: {
    backgroundColor: string
    textColor: string
    gridColor: string
    tooltipBackground: string
    tooltipTextColor: string
  }
  
  // Line图表配置
  line: {
    strokeWidth: number
    pointSize: number
    pointStroke: string
    pointStrokeWidth: number
  }
  
  // Pie图表配置
  pie: {
    strokeWidth: number
    labelColor: string
  }
}

// 默认图表主题配置
const defaultChartTheme: ChartThemeConfig = {
  primaryColor: '#1890ff',
  successColor: '#52c41a',
  warningColor: '#faad14',
  errorColor: '#ff4d4f',
  infoColor: '#13c2c2',
  
  gradients: {
    primary: ['#1677FF', '#40a9ff'],
    success: ['#56ab2f', '#a8e6cf'],
    warning: ['#f093fb', '#f5576c'],
    error: ['#ff6b6b', '#ee5a24'],
    info: ['#4facfe', '#00f2fe'],
  },
  
  common: {
    backgroundColor: '#ffffff',
    textColor: '#666666',
    gridColor: '#f0f0f0',
    tooltipBackground: 'rgba(0, 0, 0, 0.8)',
    tooltipTextColor: '#ffffff',
  },
  
  line: {
    strokeWidth: 3,
    pointSize: 4,
    pointStroke: '#ffffff',
    pointStrokeWidth: 2,
  },
  
  pie: {
    strokeWidth: 1,
    labelColor: '#666666',
  },
}

// 暗色主题配置
const darkChartTheme: ChartThemeConfig = {
  ...defaultChartTheme,
  
  common: {
    backgroundColor: '#1f1f1f',
    textColor: '#ffffff',
    gridColor: '#404040',
    tooltipBackground: 'rgba(255, 255, 255, 0.9)',
    tooltipTextColor: '#000000',
  },
  
  pie: {
    strokeWidth: 1,
    labelColor: '#ffffff',
  },
}

/**
 * 图表主题Hook
 * 根据当前Antd主题自动适配图表样式
 */
export const useChartTheme = () => {
  const { theme } = useContext(ConfigProvider.ConfigContext) || {}
  
  // 判断是否为暗色主题
  const isDark = (theme as any)?.token?.colorBgBase === '#000000' ||
    (Array.isArray(theme?.algorithm) ? 
      theme.algorithm.some((alg: any) => alg.toString().includes('dark')) :
      theme?.algorithm?.toString().includes('dark'))
  
  const chartTheme = isDark ? darkChartTheme : defaultChartTheme
  
  // 获取Line图表配置
  const getLineConfig = (color?: string) => ({
    smooth: true,
    color: color || {
      type: 'linear' as const,
      angle: 45,
      colorStops: [
        { offset: 0, color: chartTheme.gradients.primary[0] },
        { offset: 1, color: chartTheme.gradients.primary[1] },
      ],
    },
    point: {
      size: chartTheme.line.pointSize,
      shape: 'circle' as const,
      style: {
        fill: color || chartTheme.primaryColor,
        stroke: chartTheme.line.pointStroke,
        lineWidth: chartTheme.line.pointStrokeWidth,
      },
    },
    theme: {
      geometries: {
        line: {
          line: {
            style: {
              lineWidth: chartTheme.line.strokeWidth,
            },
          },
        },
      },
    },
    tooltip: {
      domStyles: {
        'g2-tooltip': {
          backgroundColor: chartTheme.common.tooltipBackground,
          color: chartTheme.common.tooltipTextColor,
          borderRadius: '6px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        },
      },
    },
  })
  
  // 获取Pie图表配置
  const getPieConfig = () => ({
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    innerRadius: 0.4,
    color: [
      chartTheme.primaryColor,
      chartTheme.successColor,
      chartTheme.warningColor,
      chartTheme.errorColor,
      chartTheme.infoColor,
    ],
    label: {
      type: 'outer' as const,
      content: '{name}: {percentage}',
      style: {
        fill: chartTheme.pie.labelColor,
        fontSize: 12,
      },
    },
    pieStyle: {
      stroke: chartTheme.common.backgroundColor,
      lineWidth: chartTheme.pie.strokeWidth,
    },
    tooltip: {
      domStyles: {
        'g2-tooltip': {
          backgroundColor: chartTheme.common.tooltipBackground,
          color: chartTheme.common.tooltipTextColor,
          borderRadius: '6px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        },
      },
    },
  })
  
  // 获取渐变色配置
  const getGradientColor = (type: keyof typeof chartTheme.gradients = 'primary') => ({
    type: 'linear' as const,
    angle: 45,
    colorStops: [
      { offset: 0, color: chartTheme.gradients[type][0] },
      { offset: 1, color: chartTheme.gradients[type][1] },
    ],
  })
  
  return {
    theme: chartTheme,
    isDark,
    getLineConfig,
    getPieConfig,
    getGradientColor,
    colors: {
      primary: chartTheme.primaryColor,
      success: chartTheme.successColor,
      warning: chartTheme.warningColor,
      error: chartTheme.errorColor,
      info: chartTheme.infoColor,
    },
  }
}

export default useChartTheme