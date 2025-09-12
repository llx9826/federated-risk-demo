import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const predictionTime = new Trend('prediction_time');
const nonDegenerateRate = new Rate('non_degenerate_predictions');

// 测试配置
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // 逐步增加到10 RPS
    { duration: '5m', target: 10 },   // 保持10 RPS
    { duration: '2m', target: 25 },   // 增加到25 RPS
    { duration: '5m', target: 25 },   // 保持25 RPS
    { duration: '2m', target: 50 },   // 增加到50 RPS
    { duration: '5m', target: 50 },   // 保持50 RPS
    { duration: '2m', target: 100 },  // 增加到100 RPS
    { duration: '5m', target: 100 },  // 保持100 RPS
    { duration: '5m', target: 0 },    // 逐步减少到0
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95%的请求应在1秒内完成
    http_req_failed: ['rate<0.01'],    // 错误率应低于1%
    errors: ['rate<0.01'],             // 自定义错误率应低于1%
    non_degenerate_predictions: ['rate>0.95'], // 95%以上的预测应为非退化分布
  },
};

// 服务端点
const BASE_URL = 'http://localhost:8003';
const TRAINER_URL = 'http://localhost:8002';
const PSI_URL = 'http://localhost:8001';

// 测试数据生成器
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

// 验证预测结果是否为非退化分布
function isNonDegenerateDistribution(predictions) {
  if (!predictions || predictions.length < 2) {
    return false;
  }
  
  // 计算标准差
  const mean = predictions.reduce((sum, val) => sum + val, 0) / predictions.length;
  const variance = predictions.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / predictions.length;
  const stdDev = Math.sqrt(variance);
  
  // 检查标准差是否大于0.01
  if (stdDev <= 0.01) {
    return false;
  }
  
  // 检查0/1占比是否各小于95%
  const lowCount = predictions.filter(p => p < 0.1).length;
  const highCount = predictions.filter(p => p > 0.9).length;
  const lowRatio = lowCount / predictions.length;
  const highRatio = highCount / predictions.length;
  
  return lowRatio < 0.95 && highRatio < 0.95;
}

// 设置阶段 - 获取可用模型
export function setup() {
  const modelsResponse = http.get(`${BASE_URL}/models`);
  if (modelsResponse.status !== 200) {
    throw new Error('无法获取模型列表');
  }
  
  const modelsData = modelsResponse.json();
  if (!modelsData.models || modelsData.models.length === 0) {
    throw new Error('没有可用的模型');
  }
  
  return {
    modelId: modelsData.models[0].model_id
  };
}

// 主测试函数
export default function(data) {
  const { modelId } = data;
  
  // 测试场景1: 单次预测
  testSinglePrediction(modelId);
  
  // 测试场景2: 批量预测
  if (Math.random() < 0.3) { // 30%概率执行批量预测
    testBatchPrediction(modelId);
  }
  
  // 测试场景3: 模型解释
  if (Math.random() < 0.2) { // 20%概率执行模型解释
    testModelExplanation(modelId);
  }
  
  // 测试场景4: 健康检查
  if (Math.random() < 0.1) { // 10%概率执行健康检查
    testHealthCheck();
  }
  
  sleep(1); // 请求间隔
}

// 单次预测测试
function testSinglePrediction(modelId) {
  const features = generateRandomFeatures();
  const payload = {
    model_id: modelId,
    features: features
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/predict`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
  });
  const endTime = Date.now();
  
  const success = check(response, {
    '单次预测状态码为200': (r) => r.status === 200,
    '单次预测响应时间<1000ms': (r) => r.timings.duration < 1000,
    '单次预测包含概率字段': (r) => {
      try {
        const data = r.json();
        return data.probability !== undefined;
      } catch {
        return false;
      }
    },
    '单次预测概率在有效范围': (r) => {
      try {
        const data = r.json();
        return data.probability >= 0 && data.probability <= 1;
      } catch {
        return false;
      }
    }
  });
  
  if (!success) {
    errorRate.add(1);
  } else {
    errorRate.add(0);
    predictionTime.add(endTime - startTime);
    
    // 检查预测结果的非退化性
    try {
      const data = response.json();
      const isNonDegenerate = data.probability > 0.01 && data.probability < 0.99;
      nonDegenerateRate.add(isNonDegenerate ? 1 : 0);
    } catch {
      nonDegenerateRate.add(0);
    }
  }
}

// 批量预测测试
function testBatchPrediction(modelId) {
  const instances = [];
  const batchSize = Math.floor(Math.random() * 10) + 1; // 1-10个实例
  
  for (let i = 0; i < batchSize; i++) {
    instances.push(generateRandomFeatures());
  }
  
  const payload = {
    model_id: modelId,
    instances: instances
  };
  
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/predict/batch`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
  });
  const endTime = Date.now();
  
  const success = check(response, {
    '批量预测状态码为200': (r) => r.status === 200,
    '批量预测响应时间<3000ms': (r) => r.timings.duration < 3000,
    '批量预测返回正确数量': (r) => {
      try {
        const data = r.json();
        return data.predictions && data.predictions.length === batchSize;
      } catch {
        return false;
      }
    }
  });
  
  if (!success) {
    errorRate.add(1);
  } else {
    errorRate.add(0);
    predictionTime.add(endTime - startTime);
    
    // 检查批量预测的非退化分布
    try {
      const data = response.json();
      const probabilities = data.predictions.map(p => p.probability);
      const isNonDegenerate = isNonDegenerateDistribution(probabilities);
      nonDegenerateRate.add(isNonDegenerate ? 1 : 0);
    } catch {
      nonDegenerateRate.add(0);
    }
  }
}

// 模型解释测试
function testModelExplanation(modelId) {
  const features = generateRandomFeatures();
  const payload = {
    model_id: modelId,
    features: features
  };
  
  const response = http.post(`${BASE_URL}/explain`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    '模型解释状态码为200': (r) => r.status === 200,
    '模型解释响应时间<2000ms': (r) => r.timings.duration < 2000,
    '模型解释包含SHAP值': (r) => {
      try {
        const data = r.json();
        return data.shap_values !== undefined;
      } catch {
        return false;
      }
    }
  });
  
  if (!success) {
    errorRate.add(1);
  } else {
    errorRate.add(0);
  }
}

// 健康检查测试
function testHealthCheck() {
  // 测试所有服务的健康检查
  const services = [
    { name: '模型服务', url: `${BASE_URL}/health` },
    { name: '训练服务', url: `${TRAINER_URL}/health` },
    { name: 'PSI服务', url: `${PSI_URL}/health` }
  ];
  
  services.forEach(service => {
    const response = http.get(service.url);
    
    check(response, {
      [`${service.name}健康检查状态码为200`]: (r) => r.status === 200,
      [`${service.name}健康检查响应时间<500ms`]: (r) => r.timings.duration < 500,
      [`${service.name}健康状态为healthy`]: (r) => {
        try {
          const data = r.json();
          return data.status === 'healthy';
        } catch {
          return false;
        }
      }
    });
  });
}

// 清理阶段
export function teardown(data) {
  console.log('压力测试完成');
  console.log(`测试的模型ID: ${data.modelId}`);
}

// 处理摘要数据
export function handleSummary(data) {
  return {
    'reports/k6_stress_report.json': JSON.stringify(data, null, 2),
    'reports/k6_stress_summary.txt': textSummary(data, { indent: ' ', enableColors: true })
  };
}

// 生成文本摘要
function textSummary(data, options = {}) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let summary = `${indent}压力测试摘要\n`;
  summary += `${indent}================\n\n`;
  
  // 基本统计
  summary += `${indent}请求统计:\n`;
  summary += `${indent}  总请求数: ${data.metrics.http_reqs.values.count}\n`;
  summary += `${indent}  失败请求数: ${data.metrics.http_req_failed.values.fails}\n`;
  summary += `${indent}  成功率: ${((1 - data.metrics.http_req_failed.values.rate) * 100).toFixed(2)}%\n\n`;
  
  // 响应时间统计
  summary += `${indent}响应时间:\n`;
  summary += `${indent}  平均值: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms\n`;
  summary += `${indent}  P50: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms\n`;
  summary += `${indent}  P95: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += `${indent}  P99: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n\n`;
  
  // 自定义指标
  if (data.metrics.errors) {
    summary += `${indent}错误率: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%\n`;
  }
  
  if (data.metrics.non_degenerate_predictions) {
    summary += `${indent}非退化预测率: ${(data.metrics.non_degenerate_predictions.values.rate * 100).toFixed(2)}%\n`;
  }
  
  if (data.metrics.prediction_time) {
    summary += `${indent}预测时间平均值: ${data.metrics.prediction_time.values.avg.toFixed(2)}ms\n`;
  }
  
  summary += `\n${indent}测试完成时间: ${new Date().toISOString()}\n`;
  
  return summary;
}