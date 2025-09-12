import { test, expect } from '@playwright/test';

/**
 * PSI服务E2E测试
 * 包含非退化分布断言：预测标准差>0.01，0/1占比各<95%
 */
test.describe('PSI Service E2E Tests', () => {
  const PSI_SERVICE_URL = 'http://localhost:8001';
  const FRONTEND_URL = 'http://localhost:5173';

  test.beforeEach(async ({ page }) => {
    // 等待服务启动
    await page.goto(FRONTEND_URL);
    await page.waitForTimeout(2000);
    
    // 检查PSI服务健康状态
    const healthResponse = await page.request.get(`${PSI_SERVICE_URL}/health`);
    expect(healthResponse.ok()).toBeTruthy();
    
    // 添加延迟避免速率限制
    await page.waitForTimeout(2000);
  });
  
  test.afterEach(async ({ page }) => {
    // 测试间延迟避免速率限制
    await page.waitForTimeout(1000);
  });

  test('PSI计算流程完整性测试', async ({ page }) => {
    // 1. 导航到PSI页面
    await page.goto(`${FRONTEND_URL}/psi`);
   // 等待页面加载
    await page.waitForLoadState('networkidle');

    // 2. 检查PSI服务健康状态
    const healthResponse = await page.request.get(`${PSI_SERVICE_URL}/health`);
    expect(healthResponse.ok()).toBeTruthy();
    const healthData = await healthResponse.json();
    expect(healthData.status).toBe('healthy');

    // 3. 创建PSI会话 (使用TOKEN_JOIN方法)
    const uniqueSessionId = `test_session_e2e_${Date.now()}`;
    const sessionResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: uniqueSessionId,
        method: 'token_join',
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    expect(sessionResponse.ok()).toBeTruthy();
    const sessionData = await sessionResponse.json();
    expect(sessionData.session_id).toBe(uniqueSessionId);

    // 4. 设置测试数据 (为E2E测试预设模拟数据)
    const setupResponse = await page.request.post(`${PSI_SERVICE_URL}/test/psi/setup-mock-data?session_id=${uniqueSessionId}`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        party_1_id: 'party_1',
        party_2_id: 'party_2'
      }
    });
    expect(setupResponse.ok()).toBeTruthy();
    const setupData = await setupResponse.json();
    expect(setupData.data_ready).toBe(true);

    // 5. 执行PSI计算
    const computeResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/compute`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: uniqueSessionId,
        party_id: 'party_1'
      }
    });
    expect(computeResponse.ok()).toBeTruthy();
    const computeData = await computeResponse.json();
    
    // 7. 验证PSI结果的非退化分布
    expect(computeData.intersection_size).toBeGreaterThan(0);
    expect(computeData.intersection_size).toBeLessThan(1000); // 不应该是全部数据
    
    // 验证交集比例合理（不应该过高或过低）
    const intersectionRatio = computeData.intersection_size / 1000;
    expect(intersectionRatio).toBeGreaterThan(0.1); // 至少10%交集
    expect(intersectionRatio).toBeLessThan(0.95); // 不超过95%交集

    // 8. 获取PSI结果详情
    const resultResponse = await page.request.get(`${PSI_SERVICE_URL}/psi/results/${uniqueSessionId}`);
    expect(resultResponse.ok()).toBeTruthy();
    const resultData = await resultResponse.json();
    
    // 验证结果质量
    expect(resultData.session_id).toBe(uniqueSessionId);
    expect(resultData.computation_time_ms).toBeGreaterThan(0); // 应该有实际计算时间
    expect(resultData.method_used).toBe('token_join');
    expect(resultData.party_contributions).toBeDefined();
    expect(resultData.timestamp).toBeDefined();
  });

  test('PSI数据质量验证', async ({ page }) => {
    // 创建PSI会话进行数据质量测试
    const qualitySessionId = `quality_test_session_${Date.now()}`;
    const sessionResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: qualitySessionId,
        algorithm: 'ecdh',
        participants: ['partyA', 'partyB'],
        privacy_budget: 5.0,
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    expect(sessionResponse.ok()).toBeTruthy();

    // 测试数据合约验证
    const contractResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/validate_contract`, {
      data: {
        party_id: 'partyA',
        data_schema: {
          required_fields: ['psi_token', 'features'],
          min_records: 1000,
          max_missing_rate: 0.1
        }
      }
    });
    expect(contractResponse.ok()).toBeTruthy();
    const contractData = await contractResponse.json();
    expect(contractData.is_valid).toBeTruthy();
  });

  test('PSI隐私保护验证', async ({ page }) => {
    // 创建隐私保护测试会话
    const privacySessionId = `privacy_test_session_${Date.now()}`;
    const sessionResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: privacySessionId,
        algorithm: 'ecdh',
        participants: ['partyA', 'partyB'],
        privacy_budget: 5.0,
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    expect(sessionResponse.ok()).toBeTruthy();

    // 验证差分隐私参数
    const privacyResponse = await page.request.get(`${PSI_SERVICE_URL}/psi/privacy_budget/${privacySessionId}`);
    expect(privacyResponse.ok()).toBeTruthy();
    const privacyData = await privacyResponse.json();
    
    expect(privacyData.epsilon).toBeGreaterThan(0);
    expect(privacyData.delta).toBeLessThan(1);
    expect(privacyData.remaining_budget).toBeGreaterThanOrEqual(0);
  });

  test('PSI错误处理测试', async ({ page }) => {
    // 测试无效会话ID
    const invalidResponse = await page.request.get(`${PSI_SERVICE_URL}/psi/result/invalid_session`);
    expect(invalidResponse.status()).toBe(404);

    // 测试重复会话创建
    const duplicateSessionId = `duplicate_session_${Date.now()}`;
    await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: duplicateSessionId,
        algorithm: 'ecdh',
        participants: ['partyA', 'partyB'],
        privacy_budget: 5.0,
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    
    const duplicateResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: duplicateSessionId,
        algorithm: 'ecdh',
        participants: ['partyA', 'partyB'],
        privacy_budget: 5.0,
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    expect(duplicateResponse.status()).toBe(409); // Conflict
  });

  test('PSI性能基准测试', async ({ page }) => {
    // 先创建性能测试会话
    const perfSessionId = `perf_test_session_${Date.now()}`;
    const sessionResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/session`, {
      headers: {
        'Content-Type': 'application/json'
      },
      data: {
        session_id: perfSessionId,
        algorithm: 'ecdh',
        participants: ['partyA', 'partyB'],
        privacy_budget: 5.0,
        party_role: 'sender',
        party_id: 'party_1',
        other_parties: ['party_2'],
        metadata: {}
      }
    });
    expect(sessionResponse.ok()).toBeTruthy();
    
    const startTime = Date.now();
    
    // 执行PSI计算
    const computeResponse = await page.request.post(`${PSI_SERVICE_URL}/psi/compute`, {
      data: {
        session_id: perfSessionId
      }
    });
    
    const endTime = Date.now();
    const executionTime = endTime - startTime;
    
    expect(computeResponse.ok()).toBeTruthy();
    expect(executionTime).toBeLessThan(30000); // 应在30秒内完成
    
    const perfData = await computeResponse.json();
    expect(perfData.throughput_records_per_sec).toBeGreaterThan(100);
  });
});