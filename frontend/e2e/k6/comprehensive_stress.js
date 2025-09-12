import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const successfulRequests = new Counter('successful_requests');
const psiSessionsCreated = new Counter('psi_sessions_created');
const trainingJobsStarted = new Counter('training_jobs_started');
const predictionsGenerated = new Counter('predictions_generated');

// 测试配置
export const options = {
  stages: [
    { duration: '1m', target: 5 },    // 预热阶段
    { duration: '3m', target: 15 },   // 逐步增加负载
    { duration: '5m', target: 30 },   // 中等负载
    { duration: '3m', target: 50 },   // 高负载
    { duration: '2m', target: 20 },   // 降低负载
    { duration: '1m', target: 0 },    // 冷却阶段
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95%的请求应在2秒内完成
    http_req_failed: ['rate<0.05'],    // 错误率应低于5%
    errors: ['rate<0.05'],             // 自定义错误率应低于5%
  },
};

// 服务端点
const CONSENT_URL = 'http://localhost:8000';
const PSI_URL = 'http://localhost:8001';
const TRAINER_URL = 'http://localhost:8002';
const SERVING_URL = 'http://localhost:8003';

// 测试数据生成器
function generateRandomUserId() {
  return `user_${Math.random().toString(36).substr(2, 9)}`;
}

function generateRandomSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
}

function generateRandomFeatures() {
  return {
    annual_income: Math.floor(Math.random() * 100000) + 30000,
    debt_to_income: Math.random() * 0.8 + 0.1,
    credit_score: Math.floor(Math.random() * 300) + 500,
    cc_utilization: Math.random() * 0.9 + 0.05,
    late_3m: Math.floor(Math.random() * 5),
    delinq_12m: Math.floor(Math.random() * 10),
    credit_len_yrs: Math.floor(Math.random() * 20) + 1,
    order_cnt_6m: Math.floor(Math.random() * 50) + 1,
    monetary_6m: Math.floor(Math.random() * 10000) + 100,
    return_rate: Math.random() * 0.5,
    recency_days: Math.floor(Math.random() * 365) + 1,
    midnight_orders_ratio: Math.random() * 0.3
  };
}

// 主测试函数
export default function() {
  // 随机选择测试场景
  const scenario = Math.random();
  
  if (scenario < 0.3) {
    testConsentService();
  } else if (scenario < 0.5) {
    testPSIService();
  } else if (scenario < 0.7) {
    testTrainingService();
  } else {
    testServingService();
  }
  
  sleep(Math.random() * 2 + 0.5); // 随机等待0.5-2.5秒
}

// 同意服务压力测试
function testConsentService() {
  const userId = generateRandomUserId();
  
  // 1. 健康检查
  let response = http.get(`${CONSENT_URL}/health`);
  check(response, {
    'consent health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 2. 创建同意记录
  const consentData = {
    user_id: userId,
    data_types: ['financial', 'behavioral'],
    purpose: 'risk_assessment',
    duration_days: 365,
    metadata: {
      source: 'stress_test',
      timestamp: new Date().toISOString()
    }
  };
  
  response = http.post(`${CONSENT_URL}/consent`, JSON.stringify(consentData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    'consent creation': (r) => r.status === 200 || r.status === 201,
  });
  
  if (success) {
    successfulRequests.add(1);
  } else {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

// PSI服务压力测试
function testPSIService() {
  const sessionId = generateRandomSessionId();
  
  // 1. 健康检查
  let response = http.get(`${PSI_URL}/health`);
  check(response, {
    'psi health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 2. 创建PSI会话
  const sessionData = {
    session_id: sessionId,
    party_role: 'sender',
    party_id: `party_${Math.floor(Math.random() * 10) + 1}`,
    other_parties: [`party_${Math.floor(Math.random() * 10) + 11}`],
    metadata: {
      test_type: 'stress_test',
      timestamp: new Date().toISOString()
    }
  };
  
  response = http.post(`${PSI_URL}/psi/session`, JSON.stringify(sessionData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    'psi session creation': (r) => r.status === 200 || r.status === 429, // 429是速率限制，也算正常
  });
  
  if (success && response.status === 200) {
    psiSessionsCreated.add(1);
    successfulRequests.add(1);
  } else if (response.status === 429) {
    // 速率限制不算错误
    successfulRequests.add(1);
  } else {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

// 训练服务压力测试
function testTrainingService() {
  // 1. 健康检查
  let response = http.get(`${TRAINER_URL}/health`);
  check(response, {
    'trainer health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 2. 获取训练状态
  response = http.get(`${TRAINER_URL}/training/status`);
  check(response, {
    'training status check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 3. 启动训练任务（较少频率）
  if (Math.random() < 0.1) { // 10%的概率启动训练
    const trainingConfig = {
      model_type: 'logistic_regression',
      participants: ['partyA', 'partyB'],
      privacy_budget: Math.random() * 5 + 1,
      max_iterations: Math.floor(Math.random() * 50) + 10,
      learning_rate: Math.random() * 0.1 + 0.01,
      metadata: {
        test_type: 'stress_test',
        timestamp: new Date().toISOString()
      }
    };
    
    response = http.post(`${TRAINER_URL}/training/start`, JSON.stringify(trainingConfig), {
      headers: { 'Content-Type': 'application/json' },
    });
    
    const success = check(response, {
      'training start': (r) => r.status === 200 || r.status === 202,
    });
    
    if (success) {
      trainingJobsStarted.add(1);
      successfulRequests.add(1);
    } else {
      errorRate.add(1);
    }
  }
  
  responseTime.add(response.timings.duration);
}

// 推理服务压力测试
function testServingService() {
  // 1. 健康检查
  let response = http.get(`${SERVING_URL}/health`);
  check(response, {
    'serving health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 2. 获取可用模型
  response = http.get(`${SERVING_URL}/models`);
  const modelsSuccess = check(response, {
    'models list': (r) => r.status === 200,
  });
  
  if (!modelsSuccess) {
    errorRate.add(1);
    return;
  }
  
  const modelsData = response.json();
  if (!modelsData.models || modelsData.models.length === 0) {
    return; // 没有可用模型，跳过预测测试
  }
  
  // 3. 执行预测
  const modelId = modelsData.models[0].model_id;
  const features = generateRandomFeatures();
  
  response = http.post(`${SERVING_URL}/predict`, JSON.stringify({
    model_id: modelId,
    features: features,
    metadata: {
      test_type: 'stress_test',
      timestamp: new Date().toISOString()
    }
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    'prediction request': (r) => r.status === 200,
  });
  
  if (success) {
    predictionsGenerated.add(1);
    successfulRequests.add(1);
    
    // 验证预测结果
    const predictionData = response.json();
    if (predictionData.prediction !== undefined) {
      check(predictionData, {
        'valid prediction': (data) => 
          typeof data.prediction === 'number' && 
          data.prediction >= 0 && 
          data.prediction <= 1,
      });
    }
  } else {
    errorRate.add(1);
  }
  
  responseTime.add(response.timings.duration);
}

// 测试总结
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'stress_test_results.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  const summary = `
=== 联邦风控系统压力测试报告 ===

测试持续时间: ${data.state.testRunDurationMs / 1000}秒
总请求数: ${data.metrics.http_reqs.values.count}
成功请求数: ${data.metrics.successful_requests ? data.metrics.successful_requests.values.count : 0}
错误率: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%

响应时间统计:
- 平均: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 中位数: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms
- 95%分位: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99%分位: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

业务指标:
- PSI会话创建: ${data.metrics.psi_sessions_created ? data.metrics.psi_sessions_created.values.count : 0}
- 训练任务启动: ${data.metrics.training_jobs_started ? data.metrics.training_jobs_started.values.count : 0}
- 预测生成: ${data.metrics.predictions_generated ? data.metrics.predictions_generated.values.count : 0}

性能阈值检查:
- 95%响应时间 < 2000ms: ${data.metrics.http_req_duration.values['p(95)'] < 2000 ? '✓ 通过' : '✗ 失败'}
- 错误率 < 5%: ${data.metrics.errors.values.rate < 0.05 ? '✓ 通过' : '✗ 失败'}

=== 测试完成 ===
`;
  
  return summary;
}