import { test, expect } from '@playwright/test';

/**
 * 模型服务E2E测试
 * 包含非退化分布断言：预测标准差>0.01，0/1占比各<95%
 */
test.describe('Model Serving Service E2E Tests', () => {
  const SERVING_SERVICE_URL = 'http://localhost:8003';
  const FRONTEND_URL = 'http://localhost:5173';

  test.beforeEach(async ({ page }) => {
    // 等待服务启动
    await page.goto(FRONTEND_URL);
    await page.waitForTimeout(2000);
  });

  test('模型服务完整性测试', async ({ page }) => {
    // 1. 导航到预测页面
    await page.goto(`${FRONTEND_URL}/prediction`);
    await expect(page.locator('h1')).toContainText('预测');

    // 2. 检查服务健康状态
    const healthResponse = await page.request.get(`${SERVING_SERVICE_URL}/health`);
    expect(healthResponse.ok()).toBeTruthy();
    const healthData = await healthResponse.json();
    expect(healthData.status).toBe('healthy');

    // 3. 获取可用模型列表
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    expect(modelsResponse.ok()).toBeTruthy();
    const modelsData = await modelsResponse.json();
    expect(modelsData.models.length).toBeGreaterThan(0);
    
    const testModel = modelsData.models[0];
    expect(testModel.model_id).toBeDefined();
    expect(testModel.status).toBe('active');

    // 4. 加载模型
    const loadResponse = await page.request.post(`${SERVING_SERVICE_URL}/models/${testModel.model_id}/load`);
    expect(loadResponse.ok()).toBeTruthy();
    const loadData = await loadResponse.json();
    expect(loadData.status).toBe('loaded');

    // 5. 批量预测测试
    const batchPredictResponse = await page.request.post(`${SERVING_SERVICE_URL}/predict/batch`, {
      data: {
        model_id: testModel.model_id,
        instances: [
          {
            // 银行侧特征
            annual_income: 75000,
            debt_to_income: 0.35,
            credit_score: 720,
            cc_utilization: 0.25,
            late_3m: 0,
            delinq_12m: 1,
            credit_len_yrs: 8,
            // 电商侧特征
            order_cnt_6m: 15,
            monetary_6m: 2500,
            return_rate: 0.1,
            recency_days: 7,
            midnight_orders_ratio: 0.05
          },
          {
            annual_income: 45000,
            debt_to_income: 0.55,
            credit_score: 650,
            cc_utilization: 0.85,
            late_3m: 2,
            delinq_12m: 3,
            credit_len_yrs: 3,
            order_cnt_6m: 5,
            monetary_6m: 800,
            return_rate: 0.3,
            recency_days: 45,
            midnight_orders_ratio: 0.2
          },
          {
            annual_income: 120000,
            debt_to_income: 0.15,
            credit_score: 800,
            cc_utilization: 0.1,
            late_3m: 0,
            delinq_12m: 0,
            credit_len_yrs: 15,
            order_cnt_6m: 25,
            monetary_6m: 5000,
            return_rate: 0.05,
            recency_days: 2,
            midnight_orders_ratio: 0.02
          }
        ]
      }
    });
    
    expect(batchPredictResponse.ok()).toBeTruthy();
    const batchPredictData = await batchPredictResponse.json();
    
    // 6. 验证预测结果的非退化分布
    const predictions = batchPredictData.predictions;
    expect(predictions.length).toBe(3);
    
    // 验证每个预测结果
    predictions.forEach((pred: any, index: number) => {
      expect(pred.probability).toBeGreaterThanOrEqual(0);
      expect(pred.probability).toBeLessThanOrEqual(1);
      expect(pred.risk_score).toBeDefined();
      expect(pred.confidence).toBeGreaterThan(0);
    });
    
    // 验证预测分布的多样性（非退化分布断言）
    const probabilities = predictions.map((p: any) => p.probability);
    const mean = probabilities.reduce((sum: number, val: number) => sum + val, 0) / probabilities.length;
    const variance = probabilities.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / probabilities.length;
    const stdDev = Math.sqrt(variance);
    
    // 预测标准差应该大于0.01（非退化分布）
    expect(stdDev).toBeGreaterThan(0.01);
    
    // 验证0/1占比各<95%（避免极端分布）
    const lowProbCount = probabilities.filter((p: number) => p < 0.1).length;
    const highProbCount = probabilities.filter((p: number) => p > 0.9).length;
    const lowProbRatio = lowProbCount / probabilities.length;
    const highProbRatio = highProbCount / probabilities.length;
    
    expect(lowProbRatio).toBeLessThan(0.95);
    expect(highProbRatio).toBeLessThan(0.95);
    
    // 验证预测结果的合理性（高风险用户应该有更高的违约概率）
    expect(predictions[1].probability).toBeGreaterThan(predictions[0].probability); // 高风险 > 中风险
    expect(predictions[0].probability).toBeGreaterThan(predictions[2].probability); // 中风险 > 低风险
  });

  test('单次预测测试', async ({ page }) => {
    // 获取可用模型
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    const modelsData = await modelsResponse.json();
    const testModel = modelsData.models[0];

    // 单次预测
    const singlePredictResponse = await page.request.post(`${SERVING_SERVICE_URL}/predict`, {
      data: {
        model_id: testModel.model_id,
        features: {
          annual_income: 60000,
          debt_to_income: 0.4,
          credit_score: 680,
          cc_utilization: 0.6,
          late_3m: 1,
          delinq_12m: 2,
          credit_len_yrs: 5,
          order_cnt_6m: 10,
          monetary_6m: 1500,
          return_rate: 0.15,
          recency_days: 20,
          midnight_orders_ratio: 0.1
        }
      }
    });
    
    expect(singlePredictResponse.ok()).toBeTruthy();
    const singlePredictData = await singlePredictResponse.json();
    
    expect(singlePredictData.probability).toBeGreaterThanOrEqual(0);
    expect(singlePredictData.probability).toBeLessThanOrEqual(1);
    expect(singlePredictData.risk_level).toMatch(/^(low|medium|high)$/);
    expect(singlePredictData.confidence).toBeGreaterThan(0);
    expect(singlePredictData.prediction_time_ms).toBeGreaterThan(0);
  });

  test('模型解释性测试', async ({ page }) => {
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    const modelsData = await modelsResponse.json();
    const testModel = modelsData.models[0];

    // 获取特征重要性
    const importanceResponse = await page.request.get(`${SERVING_SERVICE_URL}/models/${testModel.model_id}/feature_importance`);
    expect(importanceResponse.ok()).toBeTruthy();
    const importanceData = await importanceResponse.json();
    
    expect(importanceData.feature_importance).toBeDefined();
    expect(Object.keys(importanceData.feature_importance).length).toBeGreaterThan(5);
    
    // 验证重要性值的合理性
    const importanceValues = Object.values(importanceData.feature_importance) as number[];
    const totalImportance = importanceValues.reduce((sum, val) => sum + Math.abs(val), 0);
    expect(totalImportance).toBeGreaterThan(0);

    // SHAP解释
    const shapResponse = await page.request.post(`${SERVING_SERVICE_URL}/explain`, {
      data: {
        model_id: testModel.model_id,
        features: {
          annual_income: 60000,
          debt_to_income: 0.4,
          credit_score: 680,
          cc_utilization: 0.6,
          late_3m: 1,
          delinq_12m: 2,
          credit_len_yrs: 5,
          order_cnt_6m: 10,
          monetary_6m: 1500,
          return_rate: 0.15,
          recency_days: 20,
          midnight_orders_ratio: 0.1
        }
      }
    });
    
    expect(shapResponse.ok()).toBeTruthy();
    const shapData = await shapResponse.json();
    
    expect(shapData.shap_values).toBeDefined();
    expect(shapData.base_value).toBeDefined();
    expect(shapData.explanation_quality).toBeGreaterThan(0.5);
  });

  test('模型性能监控测试', async ({ page }) => {
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    const modelsData = await modelsResponse.json();
    const testModel = modelsData.models[0];

    // 获取模型性能指标
    const metricsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models/${testModel.model_id}/metrics`);
    expect(metricsResponse.ok()).toBeTruthy();
    const metricsData = await metricsResponse.json();
    
    expect(metricsData.prediction_count).toBeGreaterThanOrEqual(0);
    expect(metricsData.avg_prediction_time_ms).toBeGreaterThan(0);
    expect(metricsData.avg_prediction_time_ms).toBeLessThan(1000); // 应在1秒内
    expect(metricsData.error_rate).toBeLessThan(0.01); // 错误率应低于1%
    
    // 验证模型漂移检测
    if (metricsData.drift_detection) {
      expect(metricsData.drift_detection.feature_drift_score).toBeLessThan(0.5);
      expect(metricsData.drift_detection.prediction_drift_score).toBeLessThan(0.5);
    }
  });

  test('并发预测压力测试', async ({ page }) => {
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    const modelsData = await modelsResponse.json();
    const testModel = modelsData.models[0];

    // 并发发送多个预测请求
    const concurrentRequests = 10;
    const requests = [];
    
    for (let i = 0; i < concurrentRequests; i++) {
      requests.push(
        page.request.post(`${SERVING_SERVICE_URL}/predict`, {
          data: {
            model_id: testModel.model_id,
            features: {
              annual_income: 50000 + i * 1000,
              debt_to_income: 0.3 + i * 0.01,
              credit_score: 700 + i * 5,
              cc_utilization: 0.2 + i * 0.02,
              late_3m: i % 3,
              delinq_12m: i % 5,
              credit_len_yrs: 5 + i,
              order_cnt_6m: 10 + i,
              monetary_6m: 1000 + i * 100,
              return_rate: 0.1 + i * 0.01,
              recency_days: 10 + i,
              midnight_orders_ratio: 0.05 + i * 0.005
            }
          }
        })
      );
    }
    
    const startTime = Date.now();
    const responses = await Promise.all(requests);
    const endTime = Date.now();
    
    // 验证所有请求都成功
    responses.forEach(response => {
      expect(response.ok()).toBeTruthy();
    });
    
    // 验证响应时间
    const totalTime = endTime - startTime;
    const avgTimePerRequest = totalTime / concurrentRequests;
    expect(avgTimePerRequest).toBeLessThan(500); // 平均每个请求应在500ms内
    
    // 验证预测结果的一致性
    const predictions = await Promise.all(responses.map(r => r.json()));
    predictions.forEach(pred => {
      expect(pred.probability).toBeGreaterThanOrEqual(0);
      expect(pred.probability).toBeLessThanOrEqual(1);
    });
  });

  test('模型版本管理测试', async ({ page }) => {
    // 获取模型版本信息
    const versionsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models/versions`);
    expect(versionsResponse.ok()).toBeTruthy();
    const versionsData = await versionsResponse.json();
    
    expect(versionsData.versions.length).toBeGreaterThan(0);
    
    versionsData.versions.forEach((version: any) => {
      expect(version.version_id).toBeDefined();
      expect(version.created_at).toBeDefined();
      expect(version.model_metrics).toBeDefined();
      expect(version.status).toMatch(/^(active|inactive|deprecated)$/);
    });
  });

  test('错误处理测试', async ({ page }) => {
    // 测试无效模型ID
    const invalidModelResponse = await page.request.post(`${SERVING_SERVICE_URL}/predict`, {
      data: {
        model_id: 'invalid_model_id',
        features: { annual_income: 50000 }
      }
    });
    expect(invalidModelResponse.status()).toBe(404);

    // 测试缺失特征
    const modelsResponse = await page.request.get(`${SERVING_SERVICE_URL}/models`);
    const modelsData = await modelsResponse.json();
    const testModel = modelsData.models[0];
    
    const missingFeaturesResponse = await page.request.post(`${SERVING_SERVICE_URL}/predict`, {
      data: {
        model_id: testModel.model_id,
        features: { annual_income: 50000 } // 缺失其他必需特征
      }
    });
    expect(missingFeaturesResponse.status()).toBe(400);

    // 测试无效特征值
    const invalidFeaturesResponse = await page.request.post(`${SERVING_SERVICE_URL}/predict`, {
      data: {
        model_id: testModel.model_id,
        features: {
          annual_income: -1000, // 无效的负收入
          debt_to_income: 2.0,  // 无效的债务收入比
          credit_score: 1000    // 无效的信用分数
        }
      }
    });
    expect(invalidFeaturesResponse.status()).toBe(400);
  });
});