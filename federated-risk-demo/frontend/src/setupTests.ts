// Jest DOM matchers
import '@testing-library/jest-dom';

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
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
};

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(callback: ResizeObserverCallback) {}
  observe(target: Element, options?: ResizeObserverOptions) {}
  unobserve(target: Element) {}
  disconnect() {}
};

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
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
  value: jest.fn(),
  writable: true
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock
});

// Mock URL.createObjectURL
Object.defineProperty(URL, 'createObjectURL', {
  writable: true,
  value: jest.fn(() => 'mocked-url')
});

// Mock URL.revokeObjectURL
Object.defineProperty(URL, 'revokeObjectURL', {
  writable: true,
  value: jest.fn()
});

// Mock fetch
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    blob: () => Promise.resolve(new Blob()),
  })
) as jest.Mock;

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
jest.mock('antd/lib/message', () => ({
  success: jest.fn(),
  error: jest.fn(),
  warning: jest.fn(),
  info: jest.fn(),
  loading: jest.fn(),
}));

jest.mock('antd/lib/notification', () => ({
  success: jest.fn(),
  error: jest.fn(),
  warning: jest.fn(),
  info: jest.fn(),
  open: jest.fn(),
}));

// Mock ECharts
jest.mock('echarts', () => ({
  init: jest.fn(() => ({
    setOption: jest.fn(),
    resize: jest.fn(),
    dispose: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
  })),
  dispose: jest.fn(),
  registerTheme: jest.fn(),
}));

// Mock @ant-design/plots
jest.mock('@ant-design/plots', () => ({
  Line: jest.fn(() => null),
  Column: jest.fn(() => null),
  Pie: jest.fn(() => null),
  Area: jest.fn(() => null),
  Bar: jest.fn(() => null),
  Scatter: jest.fn(() => null),
}));

// Mock dayjs
jest.mock('dayjs', () => {
  const originalDayjs = jest.requireActual('dayjs');
  return {
    __esModule: true,
    default: jest.fn((date?: any) => {
      if (date) {
        return originalDayjs(date);
      }
      return originalDayjs('2023-01-01T00:00:00.000Z');
    }),
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

// Extend Jest matchers
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R;
      toHaveClass(className: string): R;
      toHaveStyle(style: Record<string, any>): R;
      toBeVisible(): R;
      toBeDisabled(): R;
      toHaveValue(value: string | number): R;
    }
  }
  
  var testUtils: {
    createMockResponse: (data: any, status?: number) => any;
    createMockUser: (overrides?: any) => any;
    createMockTrainingJob: (overrides?: any) => any;
    createMockModel: (overrides?: any) => any;
    waitForAsync: () => Promise<void>;
    simulateUserInput: (element: HTMLElement, value: string) => Promise<void>;
  };
}

// Setup cleanup after each test
afterEach(() => {
  // Clear all mocks
  jest.clearAllMocks();
  
  // Clear localStorage and sessionStorage
  localStorageMock.clear();
  sessionStorageMock.clear();
  
  // Reset fetch mock
  (global.fetch as jest.Mock).mockClear();
});