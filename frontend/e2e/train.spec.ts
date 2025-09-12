import { test, expect } from '@playwright/test';

/**
 * 模型训练服务E2E测试
 * 包含非退化分布断言：预测标准差>0.01，0/1占比各<95%
 */
test.describe('Model Training Service E2E Tests', () => {
  const TRAINER_SERVICE_URL = 'http://localhost:8002';
  const FRONTEND_URL = 'http://localhost:5173';

  test.beforeEach(async ({ page }) => {
    // 等待服务启动
    await page.goto(FRONTEND_URL);
    await page.waitForTimeout(2000);
  });

  test('联邦学习训练流程完整性测试', async ({ page }) => {
    // 1. 导航到训练页面
    await page.goto(`${FRONTEND_URL}/training`);
    await expect(page.locator('h1')).toContainText('训练');

    // 2. 检查训练服务健康状态
    const healthResponse = await page.request.get(`${TRAINER_SERVICE_URL}/health`);
    expect(healthResponse.ok()).toBeTruthy();
    const healthData = await healthResponse.json();
    expect(healthData.status).toBe('healthy');

    // 3. 启动训练任务
    const trainingResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/start`, {
      data: {
        task_id: 'e2e_test_task',
        algorithm: 'SecureBoost',
        participants: [
          {
            party_id: 'partyA',
            role: 'guest',
            data_path: 'data/synth/partyA_bank.csv'
          },
          {
            party_id: 'partyB', 
            role: 'host',
            data_path: 'data/synth/partyB_ecom.csv'
          }
        ],
        config: {
          max_depth: 6,
          learning_rate: 0.1,
          n_estimators: 100,
          privacy_budget: 5.0,
          validation_split: 0.2
        }
      }
    });
    expect(trainingResponse.ok()).toBeTruthy();
    const trainingData = await trainingResponse.json();
    expect(trainingData.task_id).toBe('e2e_test_task');
    expect(trainingData.status).toBe('started');

    // 4. 等待训练完成
    let status = 'running';
    let attempts = 0;
    const maxAttempts = 60; // 最多等待5分钟
    
    while (status === 'running' && attempts < maxAttempts) {
      await page.waitForTimeout(5000); // 等待5秒
      const statusResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/status/e2e_test_task`);
      expect(statusResponse.ok()).toBeTruthy();
      const statusData = await statusResponse.json();
      status = statusData.status;
      attempts++;
    }

    expect(status).toBe('completed');

    // 5. 获取训练结果
    const resultResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/result/e2e_test_task`);
    expect(resultResponse.ok()).toBeTruthy();
    const resultData = await resultResponse.json();

    // 6. 验证训练质量 - 非退化分布断言
    const metrics = resultData.metrics;
    
    // AUC应该在合理范围内
    expect(metrics.auc).toBeGreaterThan(0.65);
    expect(metrics.auc).toBeLessThan(1.0);
    
    // KS统计量应该有意义
    expect(metrics.ks).toBeGreaterThan(0.20);
    expect(metrics.ks).toBeLessThan(1.0);
    
    // 验证预测分布非退化
    const predictions = resultData.predictions;
    expect(predictions).toBeDefined();
    expect(predictions.length).toBeGreaterThan(100);
    
    // 计算预测标准差
    const mean = predictions.reduce((sum: number, val: number) => sum + val, 0) / predictions.length;
    const variance = predictions.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / predictions.length;
    const stdDev = Math.sqrt(variance);
    
    // 预测标准差应该大于0.01（非退化分布）
    expect(stdDev).toBeGreaterThan(0.01);
    
    // 验证0/1占比各<95%（避免极端分布）
    const zeroCount = predictions.filter((p: number) => p < 0.1).length;
    const oneCount = predictions.filter((p: number) => p > 0.9).length;
    const zeroRatio = zeroCount / predictions.length;
    const oneRatio = oneCount / predictions.length;
    
    expect(zeroRatio).toBeLessThan(0.95);
    expect(oneRatio).toBeLessThan(0.95);
    
    // 验证训练时间合理
    expect(resultData.training_time_seconds).toBeGreaterThan(10); // 至少10秒
    expect(resultData.training_time_seconds).toBeLessThan(600); // 不超过10分钟
    
    // 验证模型文件存在
    expect(resultData.model_path).toBeDefined();
    expect(resultData.model_size_bytes).toBeGreaterThan(10000); // 至少10KB
  });

  test('差分隐私训练测试', async ({ page }) => {
    // 测试不同隐私预算下的训练
    const privacyBudgets = [3.0, 5.0, 10.0];
    
    for (const epsilon of privacyBudgets) {
      const trainingResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/start`, {
        data: {
          task_id: `dp_test_${epsilon}`,
          algorithm: 'SecureBoost',
          participants: [
            { party_id: 'partyA', role: 'guest', data_path: 'data/synth/partyA_bank.csv' },
            { party_id: 'partyB', role: 'host', data_path: 'data/synth/partyB_ecom.csv' }
          ],
          config: {
            privacy_budget: epsilon,
            max_depth: 4,
            n_estimators: 50
          }
        }
      });
      
      expect(trainingResponse.ok()).toBeTruthy();
      
      // 等待训练完成
      await page.waitForTimeout(30000);
      
      const resultResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/result/dp_test_${epsilon}`);
      expect(resultResponse.ok()).toBeTruthy();
      const resultData = await resultResponse.json();
      
      // 验证差分隐私效果
      expect(resultData.privacy_metrics.epsilon_used).toBeLessThanOrEqual(epsilon);
      expect(resultData.privacy_metrics.noise_added).toBeTruthy();
      
      // 较小的epsilon应该有更多噪声，性能可能略低
      if (epsilon < 5.0) {
        expect(resultData.metrics.auc).toBeGreaterThan(0.60); // 仍应有基本性能
      }
    }
  });

  test('训练数据质量验证', async ({ page }) => {
    // 验证训练前数据检查
    const validationResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/validate_data`, {
      data: {
        participants: [
          { party_id: 'partyA', data_path: 'data/synth/partyA_bank.csv' },
          { party_id: 'partyB', data_path: 'data/synth/partyB_ecom.csv' }
        ]
      }
    });
    
    expect(validationResponse.ok()).toBeTruthy();
    const validationData = await validationResponse.json();
    
    // 验证数据质量指标
    expect(validationData.intersection_size).toBeGreaterThan(1000);
    expect(validationData.label_distribution.bad_rate).toBeGreaterThan(0.05);
    expect(validationData.label_distribution.bad_rate).toBeLessThan(0.30);
    expect(validationData.feature_quality.missing_rate).toBeLessThan(0.40);
    expect(validationData.feature_quality.constant_features).toEqual([]);
  });

  test('模型解释性测试', async ({ page }) => {
    // 获取SHAP解释
    const shapResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/explanation/e2e_test_task`);
    expect(shapResponse.ok()).toBeTruthy();
    const shapData = await shapResponse.json();
    
    // 验证特征重要性
    expect(shapData.feature_importance).toBeDefined();
    expect(Object.keys(shapData.feature_importance).length).toBeGreaterThan(5);
    
    // 验证SHAP值分布
    const shapValues = Object.values(shapData.feature_importance) as number[];
    const totalImportance = shapValues.reduce((sum, val) => sum + Math.abs(val), 0);
    expect(totalImportance).toBeGreaterThan(0);
    
    // 验证解释的一致性
    expect(shapData.explanation_quality.consistency_score).toBeGreaterThan(0.7);
  });

  test('训练错误处理测试', async ({ page }) => {
    // 测试无效配置
    const invalidResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/start`, {
      data: {
        task_id: 'invalid_test',
        algorithm: 'InvalidAlgorithm',
        participants: []
      }
    });
    expect(invalidResponse.status()).toBe(400);

    // 测试数据不足
    const insufficientDataResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/start`, {
      data: {
        task_id: 'insufficient_data_test',
        algorithm: 'SecureBoost',
        participants: [
          { party_id: 'partyA', data_path: 'data/empty.csv' }
        ]
      }
    });
    expect(insufficientDataResponse.status()).toBe(400);
  });

  test('训练性能基准测试', async ({ page }) => {
    const startTime = Date.now();
    
    // 启动快速训练任务
    const quickTrainingResponse = await page.request.post(`${TRAINER_SERVICE_URL}/training/start`, {
      data: {
        task_id: 'perf_test_task',
        algorithm: 'SecureBoost',
        participants: [
          { party_id: 'partyA', role: 'guest', data_path: 'data/synth/partyA_bank.csv' },
          { party_id: 'partyB', role: 'host', data_path: 'data/synth/partyB_ecom.csv' }
        ],
        config: {
          max_depth: 3,
          n_estimators: 20,
          early_stopping: true
        }
      }
    });
    
    expect(quickTrainingResponse.ok()).toBeTruthy();
    
    // 等待完成并测量时间
    let completed = false;
    while (!completed && (Date.now() - startTime) < 120000) { // 最多2分钟
      await page.waitForTimeout(5000);
      const statusResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/status/perf_test_task`);
      const statusData = await statusResponse.json();
      completed = statusData.status === 'completed';
    }
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    
    expect(completed).toBeTruthy();
    expect(totalTime).toBeLessThan(120000); // 应在2分钟内完成
    
    // 验证性能指标
    const resultResponse = await page.request.get(`${TRAINER_SERVICE_URL}/training/result/perf_test_task`);
    const resultData = await resultResponse.json();
    
    expect(resultData.performance_metrics.samples_per_second).toBeGreaterThan(100);
    expect(resultData.performance_metrics.memory_usage_mb).toBeLessThan(2048); // 不超过2GB
  });
});