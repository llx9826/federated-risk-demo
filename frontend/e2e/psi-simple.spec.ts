import { test, expect } from '@playwright/test';

const PSI_SERVICE_URL = 'http://localhost:8001';

test.describe('PSI Service Simple Tests', () => {
  test.beforeEach(async ({ page }) => {
    // 添加延迟避免速率限制
    await page.waitForTimeout(3000);
  });
  
  test.afterEach(async ({ page }) => {
    // 测试间延迟避免速率限制
    await page.waitForTimeout(2000);
  });

  test('PSI服务健康检查', async ({ page }) => {
    const healthResponse = await page.request.get(`${PSI_SERVICE_URL}/health`);
    expect(healthResponse.ok()).toBeTruthy();
    
    const healthData = await healthResponse.json();
    expect(healthData.status).toBe('healthy');
  });

  test('PSI会话列表查询', async ({ page }) => {
    const sessionsResponse = await page.request.get(`${PSI_SERVICE_URL}/psi/sessions`);
    expect(sessionsResponse.ok()).toBeTruthy();
    
    const sessionsData = await sessionsResponse.json();
    expect(Array.isArray(sessionsData)).toBeTruthy();
  });

  test('PSI计算API连通性测试', async ({ page }) => {
    // 测试计算API是否可访问（预期会返回错误，但不应该是连接错误）
    const computeResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/compute`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: 'test_connectivity',
        operation: 'intersection'
      }
    });
    
    // 预期返回4xx错误（业务逻辑错误），而不是5xx或连接错误
    expect(computeResponse.status()).toBeGreaterThanOrEqual(400);
    expect(computeResponse.status()).toBeLessThan(500);
  });
});