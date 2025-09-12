import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const predictionTime = new Trend('prediction_time');
const predictionRequests = new Counter('prediction_requests');
const successfulPredictions = new Counter('successful_predictions');
const modelLoadTime = new Trend('model_load_time');
const validPredictions = new Counter('valid_predictions');

// 推理服务专用测试配置
export const options = {
  stages: [
    { duration: '30s', target: 5 },   // 快速启动
    { duration: '2m', target: 20 },   // 中等负载
    { duration: '5m', target: 50 },   // 高负载
    { duration: '3m', target: 30 },   // 稳定负载
    { duration: '2m', target: 10 },   // 降低负载
    { duration: '30s', target: 0 },   // 停止
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 推理应该较快
    http_req_failed: ['rate<0.05'],    // 允许5%失败
    errors: ['rate<0.03'],             // 真正的错误应少于3%
    prediction_time: ['p(95)<1500'],   // 95%的预测应在1.5秒内完成
    valid_predictions: ['rate>0.9'],   // 90%以上的预测应有效
  },
};

const SERVING_URL = 'http://localhost:8003';

// 生成测试数据
function generatePredictionData() {
  const dataTypes = [
    {
      type: 'tabular',
      features: {
        age: Math.floor(Math.random() * 80) + 18,
        income: Math.floor(Math.random() * 100000) + 20000,
        credit_score: Math.floor(Math.random() * 300) + 500,
        employment_years: Math.floor(Math.random() * 40),
        debt_ratio: Math.random() * 0.8,
        has_mortgage: Math.random() > 0.5,
        education_level: Math.floor(Math.random() * 5) + 1
      }
    },
    {
      type: 'financial',
      features: {
        transaction_amount: Math.random() * 10000 + 100,
        account_balance: Math.random() * 50000 + 1000,
        transaction_frequency: Math.floor(Math.random() * 50) + 1,
        merchant_category: Math.floor(Math.random() * 20) + 1,
        time_of_day: Math.floor(Math.random() * 24),
        day_of_week: Math.floor(Math.random() * 7) + 1,
        location_risk_score: Math.random()
      }
    },
    {
      type: 'risk_assessment',
      features: {
        previous_defaults: Math.floor(Math.random() * 5),
        payment_history_score: Math.random() * 100,
        utilization_ratio: Math.random(),
        account_age_months: Math.floor(Math.random() * 240) + 6,
        recent_inquiries: Math.floor(Math.random() * 10),
        total_accounts: Math.floor(Math.random() * 20) + 1,
        current_balance: Math.random() * 25000
      }
    }
  ];
  
  const data = dataTypes[Math.floor(Math.random() * dataTypes.length)];
  data.metadata = {
    test_type: 'stress_test',
    timestamp: new Date().toISOString(),
    request_id: `stress_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`
  };
  
  return data;
}

// 验证预测结果
function validatePrediction(predictionData) {
  if (!predictionData) {
    return false;
  }
  
  // 检查基本结构
  if (!predictionData.prediction && predictionData.prediction !== 0) {
    return false;
  }
  
  // 检查概率值范围
  if (predictionData.probability !== undefined) {
    if (predictionData.probability < 0 || predictionData.probability > 1) {
      return false;
    }
  }
  
  // 检查置信度
  if (predictionData.confidence !== undefined) {
    if (predictionData.confidence < 0 || predictionData.confidence > 1) {
      return false;
    }
  }
  
  // 检查风险评分
  if (predictionData.risk_score !== undefined) {
    if (predictionData.risk_score < 0 || predictionData.risk_score > 100) {
      return false;
    }
  }
  
  return true;
}

// 主测试函数
export default function() {
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    testPrediction();
  } else if (scenario < 0.6) {
    testBatchPrediction();
  } else if (scenario < 0.8) {
    testModelInfo();
  } else {
    testHealthAndMetrics();
  }
  
  sleep(Math.random() * 1 + 0.2); // 0.2-1.2秒随机延迟
}

// 测试单个预测
function testPrediction() {
  const data = generatePredictionData();
  
  const startTime = Date.now();
  const response = http.post(`${SERVING_URL}/predict`, JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' },
  });
  const duration = Date.now() - startTime;
  
  predictionTime.add(duration);
  predictionRequests.add(1);
  
  const success = check(response, {
    'prediction request': (r) => r.status === 200,
  });
  
  if (success) {
    successfulPredictions.add(1);
    
    try {
      const predictionData = response.json();
      if (validatePrediction(predictionData)) {
        validPredictions.add(1);
      }
    } catch (e) {
      errorRate.add(1);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试批量预测
function testBatchPrediction() {
  const batchSize = Math.floor(Math.random() * 5) + 2; // 2-6个样本
  const batchData = {
    instances: [],
    metadata: {
      batch_id: `batch_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      test_type: 'stress_test'
    }
  };
  
  for (let i = 0; i < batchSize; i++) {
    batchData.instances.push(generatePredictionData());
  }
  
  const startTime = Date.now();
  const response = http.post(`${SERVING_URL}/predict/batch`, JSON.stringify(batchData), {
    headers: { 'Content-Type': 'application/json' },
  });
  const duration = Date.now() - startTime;
  
  predictionTime.add(duration);
  predictionRequests.add(batchSize);
  
  const success = check(response, {
    'batch prediction request': (r) => r.status === 200,
  });
  
  if (success) {
    successfulPredictions.add(batchSize);
    
    try {
      const batchResult = response.json();
      if (batchResult.predictions && Array.isArray(batchResult.predictions)) {
        let validCount = 0;
        for (const prediction of batchResult.predictions) {
          if (validatePrediction(prediction)) {
            validCount++;
          }
        }
        validPredictions.add(validCount);
      }
    } catch (e) {
      errorRate.add(1);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试模型信息
function testModelInfo() {
  // 获取可用模型列表
  let response = http.get(`${SERVING_URL}/models`);
  
  const success = check(response, {
    'models list': (r) => r.status === 200,
  });
  
  if (success) {
    try {
      const modelsData = response.json();
      
      if (modelsData.models && modelsData.models.length > 0) {
        // 随机选择一个模型获取详细信息
        const randomModel = modelsData.models[Math.floor(Math.random() * modelsData.models.length)];
        const modelId = randomModel.model_id || randomModel.id;
        
        if (modelId) {
          const startTime = Date.now();
          response = http.get(`${SERVING_URL}/models/${modelId}`);
          const loadTime = Date.now() - startTime;
          
          modelLoadTime.add(loadTime);
          
          check(response, {
            'model details': (r) => r.status === 200,
          }) || errorRate.add(1);
        }
      }
    } catch (e) {
      errorRate.add(1);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试健康检查和指标
function testHealthAndMetrics() {
  // 健康检查
  let response = http.get(`${SERVING_URL}/health`);
  check(response, {
    'health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 指标查询
  response = http.get(`${SERVING_URL}/metrics`);
  check(response, {
    'metrics endpoint': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 服务状态
  response = http.get(`${SERVING_URL}/status`);
  check(response, {
    'status endpoint': (r) => r.status === 200,
  }) || errorRate.add(1);
}

// 设置阶段
export function setup() {
  console.log('开始推理服务压力测试...');
  
  // 检查服务可用性
  const healthResponse = http.get(`${SERVING_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error('推理服务不可用');
  }
  
  // 预热模型（如果有预热接口）
  const warmupData = generatePredictionData();
  http.post(`${SERVING_URL}/predict`, JSON.stringify(warmupData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  return {
    startTime: new Date().toISOString()
  };
}

// 清理阶段
export function teardown(data) {
  console.log('推理服务压力测试完成');
  
  // 清理缓存（如果有清理接口）
  const cleanupResponse = http.delete(`${SERVING_URL}/cache/stress_test`);
  if (cleanupResponse.status === 200) {
    console.log('缓存清理完成');
  }
}

// 测试总结
export function handleSummary(data) {
  const summary = generateServingSummary(data);
  
  return {
    'stdout': summary,
    'serving_stress_results.json': JSON.stringify(data, null, 2),
    'serving_stress_summary.txt': summary,
  };
}

function generateServingSummary(data) {
  const totalRequests = data.metrics.http_reqs.values.count;
  const predictionRequests = data.metrics.prediction_requests ? data.metrics.prediction_requests.values.count : 0;
  const successfulPredictions = data.metrics.successful_predictions ? data.metrics.successful_predictions.values.count : 0;
  const validPredictions = data.metrics.valid_predictions ? data.metrics.valid_predictions.values.count : 0;
  const errorRate = data.metrics.errors.values.rate * 100;
  const successRate = predictionRequests > 0 ? (successfulPredictions / predictionRequests) * 100 : 0;
  const validRate = successfulPredictions > 0 ? (validPredictions / successfulPredictions) * 100 : 0;
  
  return `
=== 推理服务压力测试报告 ===

测试概况:
- 测试持续时间: ${(data.state.testRunDurationMs / 1000).toFixed(1)}秒
- 总请求数: ${totalRequests}
- 预测请求数: ${predictionRequests}
- 成功预测数: ${successfulPredictions}
- 有效预测数: ${validPredictions}
- 错误率: ${errorRate.toFixed(2)}%
- 预测成功率: ${successRate.toFixed(2)}%
- 预测有效率: ${validRate.toFixed(2)}%

响应时间统计:
- 平均响应时间: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 中位数响应时间: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms
- 95%分位响应时间: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99%分位响应时间: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

预测性能:
- 平均预测时间: ${data.metrics.prediction_time ? data.metrics.prediction_time.values.avg.toFixed(2) : 'N/A'}ms
- 95%预测时间: ${data.metrics.prediction_time ? data.metrics.prediction_time.values['p(95)'].toFixed(2) : 'N/A'}ms
- 平均模型加载时间: ${data.metrics.model_load_time ? data.metrics.model_load_time.values.avg.toFixed(2) : 'N/A'}ms
- 吞吐量: ${predictionRequests > 0 ? (predictionRequests / (data.state.testRunDurationMs / 1000)).toFixed(2) : 'N/A'} 预测/秒

性能阈值检查:
- 95%响应时间 < 2000ms: ${data.metrics.http_req_duration.values['p(95)'] < 2000 ? '✓ 通过' : '✗ 失败'}
- 95%预测时间 < 1500ms: ${data.metrics.prediction_time && data.metrics.prediction_time.values['p(95)'] < 1500 ? '✓ 通过' : '✗ 失败'}
- 错误率 < 3%: ${errorRate < 3 ? '✓ 通过' : '✗ 失败'}
- 预测有效率 > 90%: ${validRate > 90 ? '✓ 通过' : '✗ 失败'}

建议:
${generateServingRecommendations(data)}

=== 推理测试完成 ===
`;
}

function generateServingRecommendations(data) {
  const recommendations = [];
  const errorRate = data.metrics.errors.values.rate * 100;
  const avgResponseTime = data.metrics.http_req_duration.values.avg;
  const predictionRequests = data.metrics.prediction_requests ? data.metrics.prediction_requests.values.count : 0;
  const successfulPredictions = data.metrics.successful_predictions ? data.metrics.successful_predictions.values.count : 0;
  const validPredictions = data.metrics.valid_predictions ? data.metrics.valid_predictions.values.count : 0;
  const validRate = successfulPredictions > 0 ? (validPredictions / successfulPredictions) * 100 : 0;
  
  if (errorRate > 3) {
    recommendations.push('- 错误率过高，建议检查推理服务日志和模型状态');
  }
  
  if (avgResponseTime > 1500) {
    recommendations.push('- 平均响应时间较长，建议优化模型推理速度或增加服务实例');
  }
  
  if (validRate < 90) {
    recommendations.push('- 预测有效率较低，建议检查模型质量和输入数据验证');
  }
  
  if (predictionRequests === 0) {
    recommendations.push('- 未进行预测请求，建议确保推理服务正常工作');
  }
  
  const throughput = predictionRequests > 0 ? (predictionRequests / (data.state.testRunDurationMs / 1000)) : 0;
  if (throughput < 10) {
    recommendations.push('- 吞吐量较低，建议优化服务性能或扩展资源');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('- 推理服务性能表现良好，预测质量符合预期');
  }
  
  return recommendations.join('\n');
}