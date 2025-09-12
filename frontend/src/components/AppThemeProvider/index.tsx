import React from 'react'
import { ConfigProvider, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { useAppStore } from '@/store/app'

interface AppThemeProviderProps {
  children: React.ReactNode
}

const AppThemeProvider: React.FC<AppThemeProviderProps> = ({ children }) => {
  const { theme: currentTheme } = useAppStore()

  // 银行级演示系统主题Token配置
  const themeConfig = {
    algorithm: currentTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      // 主色与状态色
      colorPrimary: '#1677FF',
      colorSuccess: '#52C41A',
      colorWarning: '#FAAD14',
      colorError: '#FF4D4F',
      colorInfo: '#1677FF',
      
      // 背景层次配置 - 银行级规范
      colorBgLayout: currentTheme === 'dark' ? '#141414' : '#F5F7FA',        // 全局基底浅灰
      colorBgContainer: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',     // 容器白底
      colorBgElevated: currentTheme === 'dark' ? '#262626' : '#FFFFFF',      // 浮层白底
      
      // 边框与分割线
      colorBorderSecondary: currentTheme === 'dark' ? '#434343' : '#EAECF0', // 分隔线
      colorSplit: currentTheme === 'dark' ? '#303030' : '#F0F2F5',           // 内容分割
      colorBorder: currentTheme === 'dark' ? '#595959' : '#D9D9D9',          // 主要边框
      
      // 文本颜色 - 确保对比度≥4.5:1
      colorText: currentTheme === 'dark' ? '#FFFFFF' : '#262626',            // 主文本
      colorTextSecondary: currentTheme === 'dark' ? '#A6A6A6' : '#8C8C8C',   // 次要文本
      colorTextTertiary: currentTheme === 'dark' ? '#737373' : '#BFBFBF',    // 三级文本
      colorTextQuaternary: currentTheme === 'dark' ? '#525252' : '#F0F0F0',  // 四级文本
      
      // 字体系统 - 支持中英文混排
      fontFamily: 'Inter, "SF Pro Text", system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
      fontFeatureSettings: '"tnum"',   // 等宽数字
      
      // 字号系统 - 银行级规范
      fontSizeHeading1: 22,            // H1
      fontSizeHeading2: 18,            // H2
      fontSizeHeading3: 16,            // H3
      fontSize: 14,                    // 正文
      fontSizeSM: 12,                  // 小字
      
      // 行高系统
      lineHeight: 1.6,                 // 正文行高
      lineHeightHeading1: 1.4,
      lineHeightHeading2: 1.4,
      lineHeightHeading3: 1.4,
      
      // 字重系统
      fontWeightStrong: 600,           // 标题
      fontWeight: 400,                 // 正文
      
      // 圆角系统 - 统一12px
      borderRadius: 12,                // 统一圆角
      borderRadiusLG: 12,
      borderRadiusSM: 8,
      borderRadiusXS: 4,
      
      // 间距系统 - 8px网格
      padding: 16,
      paddingLG: 24,                   // 页面内上下留白
      paddingMD: 16,                   // 卡片外间距
      paddingSM: 12,
      paddingXS: 8,
      paddingXXS: 4,
      
      margin: 16,
      marginLG: 24,
      marginMD: 16,
      marginSM: 12,
      marginXS: 8,
      marginXXS: 4,
      
      // 阴影系统 - 使用Ant Design默认轻阴影
      boxShadow: currentTheme === 'dark'
        ? '0 2px 8px 0 rgba(0, 0, 0, 0.15)'
        : '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
      boxShadowSecondary: currentTheme === 'dark'
        ? '0 6px 16px 0 rgba(0, 0, 0, 0.25)'
        : '0 6px 16px 0 rgba(0, 0, 0, 0.08), 0 3px 6px -4px rgba(0, 0, 0, 0.12), 0 9px 28px 8px rgba(0, 0, 0, 0.05)',
      
      // 控件尺寸
      controlHeight: 40,               // 按钮/输入框高度
      controlHeightLG: 48,
      controlHeightSM: 32,
      
      // 布局尺寸
      siderWidth: 220,                 // 侧边栏宽度
      headerHeight: 56,                // 头部高度
      
      // 动效配置
      motionDurationSlow: '0.3s',
      motionDurationMid: '0.2s',
      motionDurationFast: '0.1s',
    },
    components: {
      // Layout组件配置
      Layout: {
        bodyBg: currentTheme === 'dark' ? '#141414' : '#F5F7FA',     // 布局背景
        headerBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',   // 头部背景
        siderBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',    // 侧边栏背景
        triggerBg: currentTheme === 'dark' ? '#141414' : '#F5F7FA',
      },
      
      // Menu组件配置
      Menu: {
        itemBg: 'transparent',
        itemSelectedBg: currentTheme === 'dark' ? '#1677FF' : '#E6F4FF',
        itemHoverBg: currentTheme === 'dark' ? '#262626' : '#F5F5F5',
        itemActiveBg: currentTheme === 'dark' ? '#1677FF' : '#E6F4FF',
      },
      
      // Card组件配置
      Card: {
        headerBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',
        actionsBg: currentTheme === 'dark' ? '#262626' : '#FAFAFA',
      },
      
      // Table组件配置
      Table: {
        headerBg: currentTheme === 'dark' ? '#262626' : '#FAFAFA',
        rowHoverBg: currentTheme === 'dark' ? '#262626' : '#F5F5F5',
      },
      
      // Button组件配置
      Button: {
        borderRadius: 8,
        controlHeight: 40,
        paddingInline: 16,
      },
      
      // Input组件配置
      Input: {
        borderRadius: 8,
        controlHeight: 40,
        paddingInline: 12,
      },
      
      // Select组件配置
      Select: {
        borderRadius: 8,
        controlHeight: 40,
      },
      
      // DatePicker组件配置
      DatePicker: {
        borderRadius: 8,
        controlHeight: 40,
      },
      
      // Modal组件配置
      Modal: {
        borderRadius: 12,
        headerBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',
        contentBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',
        footerBg: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',
      },
      
      // Drawer组件配置
      Drawer: {
        borderRadius: 12,
        colorBgElevated: currentTheme === 'dark' ? '#1F1F1F' : '#FFFFFF',
      },
      
      // Statistic组件配置
      Statistic: {
        titleFontSize: 14,
        contentFontSize: 24,
        fontFamily: 'Inter, "SF Pro Text", system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
      },
      
      // Form组件配置
      Form: {
        itemMarginBottom: 24,
        verticalLabelPadding: '0 0 8px',
      },
    },
  }

  return (
    <ConfigProvider
      theme={themeConfig}
      locale={zhCN}
      componentSize="middle"
    >
      {children}
    </ConfigProvider>
  )
}

export default AppThemeProvider