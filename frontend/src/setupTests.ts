// Jest DOM matchers
import '@testing-library/jest-dom';
import { vi, beforeAll, afterAll, afterEach } from 'vitest';

// Extend global types
declare global {
  var testUtils: {
    createMockResponse: (data: any, status?: number) => any;
    createMockUser: (overrides?: any) => any;
    createMockTrainingJob: (overrides?: any) => any;
    createMockModel: (overrides?: any) => any;
    waitForAsync: () => Promise<void>;
    simulateUserInput: (element: HTMLElement, value: string) => Promise<void>;
  };
}

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  root = null;
  rootMargin = '';
  thresholds = [];
  constructor() {}
  observe() {
    return null;
  }
  disconnect() {
    return null;
  }
  unobserve() {
    return null;
  }
  takeRecords() {
    return [];
  }
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(_callback: ResizeObserverCallback) {}
  observe(_target: Element, _options?: ResizeObserverOptions) {}
  unobserve(_target: Element) {}
  disconnect() {}
} as any;

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock getComputedStyle
Object.defineProperty(window, 'getComputedStyle', {
  value: () => ({
    getPropertyValue: () => '',
  }),
});

// Mock scrollTo
Object.defineProperty(window, 'scrollTo', {
  value: vi.fn(),
  writable: true
});

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

// Mock sessionStorage
const sessionStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock
});

// Mock URL.createObjectURL
Object.defineProperty(URL, 'createObjectURL', {
  writable: true,
  value: vi.fn(() => 'mocked-url')
});

// Mock URL.revokeObjectURL
Object.defineProperty(URL, 'revokeObjectURL', {
  writable: true,
  value: vi.fn()
});

// Mock fetch
global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    blob: () => Promise.resolve(new Blob()),
  })
) as any;

// Mock console methods to reduce noise in tests
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render is no longer supported')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };

  console.warn = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      (args[0].includes('componentWillReceiveProps') ||
       args[0].includes('componentWillUpdate'))
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});

// Mock antd components that might cause issues
vi.mock('antd/lib/message', () => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
  loading: vi.fn(),
}));

vi.mock('antd/lib/notification', () => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
  open: vi.fn(),
}));

// Mock ECharts
vi.mock('echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
  })),
  dispose: vi.fn(),
  registerTheme: vi.fn(),
}));

// Mock @ant-design/plots
vi.mock('@ant-design/plots', () => ({
  Line: vi.fn(() => null),
  Column: vi.fn(() => null),
  Pie: vi.fn(() => null),
  Area: vi.fn(() => null),
  Bar: vi.fn(() => null),
  Scatter: vi.fn(() => null),
}));

// Mock dayjs
vi.mock('dayjs', async () => {
  const originalDayjs = await vi.importActual('dayjs') as any;
  const mockDayjs = vi.fn((date?: any) => {
    if (date) {
      return originalDayjs.default(date);
    }
    return originalDayjs.default('2023-01-01T00:00:00.000Z');
  });
  
  // Add extend method and other dayjs methods
  mockDayjs.extend = vi.fn();
  mockDayjs.locale = vi.fn();
  
  return {
    __esModule: true,
    default: mockDayjs,
  };
});

// Global test utilities
global.testUtils = {
  // Helper to create mock API responses
  createMockResponse: (data: any, status = 200) => ({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  }),
  
  // Helper to create mock user data
  createMockUser: (overrides = {}) => ({
    id: '1',
    username: 'testuser',
    email: 'test@example.com',
    role: 'user',
    ...overrides,
  }),
  
  // Helper to create mock training job data
  createMockTrainingJob: (overrides = {}) => ({
    id: '1',
    name: '测试训练任务',
    status: 'created',
    algorithm: 'logistic_regression',
    participants: ['party_a', 'party_b'],
    created_at: '2023-01-01T00:00:00Z',
    ...overrides,
  }),
  
  // Helper to create mock model data
  createMockModel: (overrides = {}) => ({
    id: '1',
    name: '测试模型',
    version: '1.0.0',
    status: 'registered',
    algorithm: 'logistic_regression',
    registered_at: '2023-01-01T00:00:00Z',
    ...overrides,
  }),
  
  // Helper to wait for async operations
  waitForAsync: () => new Promise(resolve => setTimeout(resolve, 0)),
  
  // Helper to simulate user interactions
  simulateUserInput: async (element: HTMLElement, value: string) => {
    const { fireEvent } = await import('@testing-library/react');
    fireEvent.change(element, { target: { value } });
    fireEvent.blur(element);
  },
};

// Global test utilities and Jest matchers are declared above

// Setup cleanup after each test
afterEach(() => {
  // Clear all mocks
  vi.clearAllMocks();
  
  // Clear localStorage and sessionStorage
  localStorageMock.clear();
  sessionStorageMock.clear();
  
  // Reset fetch mock
  (global.fetch as any).mockClear();
});