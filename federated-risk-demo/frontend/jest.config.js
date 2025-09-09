module.exports = {
  // 测试环境
  testEnvironment: 'jsdom',
  
  // 根目录
  rootDir: '.',
  
  // 模块文件扩展名
  moduleFileExtensions: [
    'js',
    'jsx',
    'ts',
    'tsx',
    'json'
  ],
  
  // 测试文件匹配模式
  testMatch: [
    '<rootDir>/src/**/__tests__/**/*.(js|jsx|ts|tsx)',
    '<rootDir>/src/**/?(*.)(spec|test).(js|jsx|ts|tsx)'
  ],
  
  // 转换配置
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'babel-jest',
    '^.+\\.css$': 'jest-transform-css',
    '^.+\\.(png|jpg|jpeg|gif|webp|svg)$': 'jest-transform-file'
  },
  
  // 模块名映射
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@pages/(.*)$': '<rootDir>/src/pages/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1',
    '^@services/(.*)$': '<rootDir>/src/services/$1',
    '^@hooks/(.*)$': '<rootDir>/src/hooks/$1',
    '^@types/(.*)$': '<rootDir>/src/types/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy'
  },
  
  // 设置文件
  setupFilesAfterEnv: [
    '<rootDir>/src/setupTests.ts'
  ],
  
  // 覆盖率配置
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
    '!src/reportWebVitals.ts',
    '!src/setupTests.ts'
  ],
  
  // 覆盖率阈值
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  },
  
  // 覆盖率报告格式
  coverageReporters: [
    'text',
    'lcov',
    'html',
    'json-summary'
  ],
  
  // 覆盖率输出目录
  coverageDirectory: 'coverage',
  
  // 忽略的文件和目录
  testPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/build/',
    '<rootDir>/dist/'
  ],
  
  // 模块路径忽略模式
  modulePathIgnorePatterns: [
    '<rootDir>/build/',
    '<rootDir>/dist/'
  ],
  
  // 清除模拟
  clearMocks: true,
  
  // 恢复模拟
  restoreMocks: true,
  
  // 全局变量
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json'
    }
  },
  
  // 测试超时时间（毫秒）
  testTimeout: 10000,
  
  // 详细输出
  verbose: true,
  
  // 错误时停止
  bail: false,
  
  // 最大工作进程数
  maxWorkers: '50%',
  
  // 缓存目录
  cacheDirectory: '<rootDir>/node_modules/.cache/jest',
  
  // 监视插件
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ],
  
  // 模拟模块
  moduleNameMapping: {
    // 处理CSS模块
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    
    // 处理图片和其他静态资源
    '\\.(png|jpg|jpeg|gif|webp|svg)$': '<rootDir>/src/__mocks__/fileMock.js',
    
    // 处理字体文件
    '\\.(woff|woff2|eot|ttf|otf)$': '<rootDir>/src/__mocks__/fileMock.js',
    
    // 路径别名
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@pages/(.*)$': '<rootDir>/src/pages/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1',
    '^@services/(.*)$': '<rootDir>/src/services/$1',
    '^@hooks/(.*)$': '<rootDir>/src/hooks/$1',
    '^@types/(.*)$': '<rootDir>/src/types/$1'
  },
  
  // 转换忽略模式
  transformIgnorePatterns: [
    'node_modules/(?!(antd|@ant-design|rc-.+|@babel/runtime)/)',
    '^.+\\.module\\.(css|sass|scss)$'
  ],
  
  // 环境变量
  setupFiles: [
    '<rootDir>/src/setupEnv.js'
  ]
};