// 设置测试环境变量
process.env.NODE_ENV = 'test';
process.env.PUBLIC_URL = '';
process.env.REACT_APP_API_BASE_URL = 'http://localhost:8000';
process.env.REACT_APP_PSI_SERVICE_URL = 'http://localhost:8001';
process.env.REACT_APP_CONSENT_SERVICE_URL = 'http://localhost:8002';
process.env.REACT_APP_TRAINING_SERVICE_URL = 'http://localhost:8003';
process.env.REACT_APP_INFERENCE_SERVICE_URL = 'http://localhost:8004';

// 设置时区
process.env.TZ = 'Asia/Shanghai';

// 禁用警告
process.env.GENERATE_SOURCEMAP = 'false';
process.env.CI = 'true';

// Mock 环境特定的全局变量
global.process = {
  ...global.process,
  env: {
    ...global.process.env,
    ...process.env,
  },
};

// 设置默认的 fetch 超时
global.DEFAULT_TIMEOUT = 5000;

// Mock 浏览器 API
if (typeof window !== 'undefined') {
  // Mock crypto API
  Object.defineProperty(window, 'crypto', {
    value: {
      getRandomValues: (arr) => {
        for (let i = 0; i < arr.length; i++) {
          arr[i] = Math.floor(Math.random() * 256);
        }
        return arr;
      },
      subtle: {
        digest: jest.fn().mockResolvedValue(new ArrayBuffer(32)),
        encrypt: jest.fn().mockResolvedValue(new ArrayBuffer(32)),
        decrypt: jest.fn().mockResolvedValue(new ArrayBuffer(32)),
      },
    },
  });

  // Mock performance API
  Object.defineProperty(window, 'performance', {
    value: {
      now: jest.fn(() => Date.now()),
      mark: jest.fn(),
      measure: jest.fn(),
      getEntriesByName: jest.fn(() => []),
      getEntriesByType: jest.fn(() => []),
    },
  });

  // Mock requestAnimationFrame
  Object.defineProperty(window, 'requestAnimationFrame', {
    value: jest.fn((cb) => setTimeout(cb, 16)),
  });

  // Mock cancelAnimationFrame
  Object.defineProperty(window, 'cancelAnimationFrame', {
    value: jest.fn((id) => clearTimeout(id)),
  });

  // Mock requestIdleCallback
  Object.defineProperty(window, 'requestIdleCallback', {
    value: jest.fn((cb) => setTimeout(cb, 0)),
  });

  // Mock cancelIdleCallback
  Object.defineProperty(window, 'cancelIdleCallback', {
    value: jest.fn((id) => clearTimeout(id)),
  });

  // Mock MutationObserver
  Object.defineProperty(window, 'MutationObserver', {
    value: class MutationObserver {
      constructor(callback) {
        this.callback = callback;
      }
      observe() {}
      disconnect() {}
      takeRecords() {
        return [];
      }
    },
  });

  // Mock HTMLCanvasElement
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    value: jest.fn(() => ({
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn(() => ({ data: new Array(4) })),
      putImageData: jest.fn(),
      createImageData: jest.fn(() => ({ data: new Array(4) })),
      setTransform: jest.fn(),
      drawImage: jest.fn(),
      save: jest.fn(),
      fillText: jest.fn(),
      restore: jest.fn(),
      beginPath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      closePath: jest.fn(),
      stroke: jest.fn(),
      translate: jest.fn(),
      scale: jest.fn(),
      rotate: jest.fn(),
      arc: jest.fn(),
      fill: jest.fn(),
      measureText: jest.fn(() => ({ width: 0 })),
      transform: jest.fn(),
      rect: jest.fn(),
      clip: jest.fn(),
    })),
  });

  // Mock HTMLCanvasElement toDataURL
  Object.defineProperty(HTMLCanvasElement.prototype, 'toDataURL', {
    value: jest.fn(() => 'data:image/png;base64,mock'),
  });
}

// 设置全局错误处理
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

// 设置测试数据
global.TEST_DATA = {
  users: [
    {
      id: '1',
      username: 'admin',
      email: 'admin@example.com',
      role: 'admin',
      created_at: '2023-01-01T00:00:00Z',
    },
    {
      id: '2',
      username: 'user1',
      email: 'user1@example.com',
      role: 'user',
      created_at: '2023-01-02T00:00:00Z',
    },
  ],
  trainingJobs: [
    {
      id: '1',
      name: '风险评估模型训练',
      status: 'completed',
      algorithm: 'logistic_regression',
      participants: ['party_a', 'party_b'],
      created_at: '2023-01-01T00:00:00Z',
      completed_at: '2023-01-01T01:00:00Z',
    },
    {
      id: '2',
      name: '信用评分模型训练',
      status: 'running',
      algorithm: 'xgboost',
      participants: ['party_a', 'party_b', 'party_c'],
      created_at: '2023-01-02T00:00:00Z',
    },
  ],
  models: [
    {
      id: '1',
      name: '风险评估模型',
      version: '1.0.0',
      status: 'active',
      algorithm: 'logistic_regression',
      registered_at: '2023-01-01T01:00:00Z',
    },
    {
      id: '2',
      name: '信用评分模型',
      version: '0.9.0',
      status: 'inactive',
      algorithm: 'xgboost',
      registered_at: '2023-01-02T01:00:00Z',
    },
  ],
  auditLogs: [
    {
      id: '1',
      action: 'model_prediction',
      user_id: '1',
      resource: 'model_1',
      timestamp: '2023-01-01T02:00:00Z',
      details: { input_size: 100, prediction_count: 100 },
    },
    {
      id: '2',
      action: 'training_job_created',
      user_id: '2',
      resource: 'job_2',
      timestamp: '2023-01-02T00:00:00Z',
      details: { algorithm: 'xgboost', participants: 3 },
    },
  ],
};

console.log('Test environment setup completed');