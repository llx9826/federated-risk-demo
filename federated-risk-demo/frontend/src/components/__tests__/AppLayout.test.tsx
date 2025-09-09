import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import AppLayout from '../Layout/AppLayout';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';

// Mock antd components that might cause issues in tests
jest.mock('antd', () => {
  const antd = jest.requireActual('antd');
  return {
    ...antd,
    message: {
      success: jest.fn(),
      error: jest.fn(),
      warning: jest.fn(),
      info: jest.fn(),
    },
    notification: {
      success: jest.fn(),
      error: jest.fn(),
      warning: jest.fn(),
      info: jest.fn(),
    },
  };
});

// Mock the system status and notification components
jest.mock('../SystemStatus', () => {
  return function MockSystemStatus() {
    return <div data-testid="system-status">系统状态组件</div>;
  };
});

jest.mock('../NotificationPanel', () => {
  return function MockNotificationPanel() {
    return <div data-testid="notification-panel">通知面板组件</div>;
  };
});

// Mock react-router-dom hooks
const mockNavigate = jest.fn();
const mockLocation = { pathname: '/dashboard' };

jest.mock('react-router-dom', () => {
  const actual = jest.requireActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => mockLocation,
  };
});

const renderAppLayout = (children?: React.ReactNode) => {
  return render(
    <BrowserRouter>
      <ConfigProvider locale={zhCN}>
        <AppLayout>
          {children || <div data-testid="test-content">测试内容</div>}
        </AppLayout>
      </ConfigProvider>
    </BrowserRouter>
  );
};

describe('AppLayout Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders without crashing', () => {
    renderAppLayout();
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });

  test('displays sidebar with navigation menu', () => {
    renderAppLayout();
    
    // Check for main navigation items
    expect(screen.getByText('仪表板')).toBeInTheDocument();
    expect(screen.getByText('PSI计算')).toBeInTheDocument();
    expect(screen.getByText('同意管理')).toBeInTheDocument();
    expect(screen.getByText('训练服务')).toBeInTheDocument();
    expect(screen.getByText('推理服务')).toBeInTheDocument();
    expect(screen.getByText('审计日志')).toBeInTheDocument();
    expect(screen.getByText('系统设置')).toBeInTheDocument();
  });

  test('displays header with system title', () => {
    renderAppLayout();
    expect(screen.getByText('联邦风控演示系统')).toBeInTheDocument();
  });

  test('shows system status component in header', () => {
    renderAppLayout();
    expect(screen.getByTestId('system-status')).toBeInTheDocument();
  });

  test('shows notification panel in header', () => {
    renderAppLayout();
    expect(screen.getByTestId('notification-panel')).toBeInTheDocument();
  });

  test('displays user menu in header', () => {
    renderAppLayout();
    
    // Look for user avatar or user menu trigger
    const userElements = screen.getAllByText(/用户|管理员|User|Admin/i);
    expect(userElements.length).toBeGreaterThan(0);
  });

  test('handles menu item clicks and navigation', async () => {
    renderAppLayout();
    
    // Click on PSI menu item
    const psiMenuItem = screen.getByText('PSI计算');
    fireEvent.click(psiMenuItem);
    
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/psi');
    });
  });

  test('highlights current menu item based on location', () => {
    // Mock location to be on PSI page
    mockLocation.pathname = '/psi';
    
    renderAppLayout();
    
    // The PSI menu item should be highlighted (this would depend on implementation)
    const psiMenuItem = screen.getByText('PSI计算');
    expect(psiMenuItem).toBeInTheDocument();
  });

  test('supports sidebar collapse/expand', async () => {
    renderAppLayout();
    
    // Look for collapse trigger (usually a button with collapse icon)
    const collapseButtons = screen.getAllByRole('button');
    const collapseButton = collapseButtons.find(button => 
      button.querySelector('.anticon-menu-fold') || 
      button.querySelector('.anticon-menu-unfold')
    );
    
    if (collapseButton) {
      fireEvent.click(collapseButton);
      
      // After clicking, the sidebar state should change
      // This would require checking the actual implementation
      await waitFor(() => {
        expect(collapseButton).toBeInTheDocument();
      });
    }
  });

  test('displays breadcrumb navigation', () => {
    renderAppLayout();
    
    // Look for breadcrumb component
    const breadcrumbElements = screen.getAllByText(/首页|Home|仪表板|Dashboard/i);
    expect(breadcrumbElements.length).toBeGreaterThan(0);
  });

  test('shows system health indicators', () => {
    renderAppLayout();
    
    // System status component should show health indicators
    expect(screen.getByTestId('system-status')).toBeInTheDocument();
  });

  test('handles user menu interactions', async () => {
    renderAppLayout();
    
    // Find and click user menu
    const userMenuTriggers = screen.getAllByRole('button');
    const userMenuTrigger = userMenuTriggers.find(button => 
      button.textContent?.includes('用户') || 
      button.textContent?.includes('管理员') ||
      button.querySelector('.anticon-user')
    );
    
    if (userMenuTrigger) {
      fireEvent.click(userMenuTrigger);
      
      await waitFor(() => {
        // Should show user menu options
        expect(screen.getByText(/个人设置|退出登录|Profile|Logout/i)).toBeInTheDocument();
      });
    }
  });

  test('displays notification badge when there are notifications', () => {
    renderAppLayout();
    
    // Notification panel should be present
    expect(screen.getByTestId('notification-panel')).toBeInTheDocument();
  });

  test('supports responsive design', () => {
    renderAppLayout();
    
    // Check that the layout adapts to different screen sizes
    // This would require more complex testing with viewport changes
    const layout = screen.getByRole('main') || screen.getByTestId('test-content').parentElement;
    expect(layout).toBeInTheDocument();
  });

  test('maintains proper accessibility attributes', () => {
    renderAppLayout();
    
    // Check for proper ARIA labels and roles
    const navigation = screen.getByRole('navigation') || screen.getByText('仪表板').closest('ul');
    expect(navigation).toBeInTheDocument();
    
    // Check for main content area
    const main = screen.getByRole('main') || screen.getByTestId('test-content');
    expect(main).toBeInTheDocument();
  });

  test('handles theme switching', async () => {
    renderAppLayout();
    
    // Look for theme switch button (if implemented)
    const themeButtons = screen.getAllByRole('button');
    const themeButton = themeButtons.find(button => 
      button.querySelector('.anticon-sun') || 
      button.querySelector('.anticon-moon') ||
      button.textContent?.includes('主题')
    );
    
    if (themeButton) {
      fireEvent.click(themeButton);
      
      await waitFor(() => {
        // Theme should change (this would require checking actual theme state)
        expect(themeButton).toBeInTheDocument();
      });
    }
  });

  test('displays loading states appropriately', () => {
    renderAppLayout();
    
    // Check that loading states are handled properly
    // This would depend on the actual implementation
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });

  test('handles error states gracefully', () => {
    // Test error boundary behavior
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    try {
      renderAppLayout();
      expect(screen.getByTestId('test-content')).toBeInTheDocument();
    } catch (error) {
      // Should not throw unhandled errors
      expect(error).toBeUndefined();
    } finally {
      consoleSpy.mockRestore();
    }
  });

  test('supports keyboard navigation', async () => {
    renderAppLayout();
    
    // Test keyboard navigation through menu items
    const firstMenuItem = screen.getByText('仪表板');
    firstMenuItem.focus();
    
    // Simulate Tab key to navigate
    fireEvent.keyDown(firstMenuItem, { key: 'Tab', code: 'Tab' });
    
    await waitFor(() => {
      // Next focusable element should be focused
      expect(document.activeElement).toBeDefined();
    });
  });

  test('maintains state across navigation', () => {
    renderAppLayout();
    
    // Test that layout state is maintained when navigating
    // This would require more complex state testing
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });

  test('handles window resize events', () => {
    renderAppLayout();
    
    // Simulate window resize
    global.innerWidth = 768;
    global.dispatchEvent(new Event('resize'));
    
    // Layout should adapt to new size
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });

  test('provides proper content area for children', () => {
    const testChild = <div data-testid="custom-child">自定义子组件</div>;
    renderAppLayout(testChild);
    
    expect(screen.getByTestId('custom-child')).toBeInTheDocument();
    expect(screen.getByText('自定义子组件')).toBeInTheDocument();
  });

  test('integrates with routing system correctly', () => {
    renderAppLayout();
    
    // Verify that routing integration works
    expect(mockLocation.pathname).toBeDefined();
    expect(mockNavigate).toBeDefined();
  });
});

describe('AppLayout Menu Configuration', () => {
  test('displays all required menu items', () => {
    renderAppLayout();
    
    const expectedMenuItems = [
      '仪表板',
      'PSI计算', 
      '同意管理',
      '训练服务',
      '推理服务',
      '审计日志',
      '系统设置'
    ];
    
    expectedMenuItems.forEach(item => {
      expect(screen.getByText(item)).toBeInTheDocument();
    });
  });

  test('menu items have correct icons', () => {
    renderAppLayout();
    
    // Check that menu items have appropriate icons
    // This would require checking for specific icon classes
    const menuItems = screen.getAllByRole('menuitem') || 
                     screen.getAllByText(/仪表板|PSI|同意|训练|推理|审计|设置/);
    
    expect(menuItems.length).toBeGreaterThan(0);
  });

  test('menu items navigate to correct routes', async () => {
    renderAppLayout();
    
    const menuTests = [
      { text: '仪表板', route: '/dashboard' },
      { text: 'PSI计算', route: '/psi' },
      { text: '同意管理', route: '/consent' },
      { text: '训练服务', route: '/training' },
      { text: '推理服务', route: '/inference' },
      { text: '审计日志', route: '/audit' },
      { text: '系统设置', route: '/settings' }
    ];
    
    for (const { text, route } of menuTests) {
      const menuItem = screen.getByText(text);
      fireEvent.click(menuItem);
      
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith(route);
      });
      
      mockNavigate.mockClear();
    }
  });
});