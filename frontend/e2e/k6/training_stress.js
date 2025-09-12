import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const trainingStartTime = new Trend('training_start_time');
const trainingJobsStarted = new Counter('training_jobs_started');
const trainingJobsCompleted = new Counter('training_jobs_completed');
const modelQualityChecks = new Counter('model_quality_checks');
const nonDegenerateModels = new Counter('non_degenerate_models');

// 训练服务专用测试配置
export const options = {
  stages: [
    { duration: '1m', target: 3 },    // 缓慢启动
    { duration: '3m', target: 8 },    // 中等负载
    { duration: '5m', target: 15 },   // 高负载
    { duration: '3m', target: 10 },   // 稳定负载
    { duration: '2m', target: 5 },    // 降低负载
    { duration: '1m', target: 0 },    // 停止
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 训练操作可能较慢
    http_req_failed: ['rate<0.1'],     // 允许10%失败
    errors: ['rate<0.05'],             // 真正的错误应少于5%
    non_degenerate_models: ['rate>0.8'], // 80%以上的模型应为非退化
  },
};

const TRAINER_URL = 'http://localhost:8002';

// 生成训练配置
function generateTrainingConfig() {
  const configs = [
    {
      model_type: 'logistic_regression',
      participants: ['partyA', 'partyB'],
      privacy_budget: Math.random() * 3 + 1,
      max_iterations: Math.floor(Math.random() * 30) + 10,
      learning_rate: Math.random() * 0.05 + 0.01,
      convergence_threshold: Math.random() * 0.01 + 0.001
    },
    {
      model_type: 'linear_regression',
      participants: ['partyA', 'partyB', 'partyC'],
      privacy_budget: Math.random() * 4 + 2,
      max_iterations: Math.floor(Math.random() * 50) + 20,
      learning_rate: Math.random() * 0.1 + 0.005,
      regularization: Math.random() * 0.1 + 0.01
    },
    {
      model_type: 'neural_network',
      participants: ['partyA', 'partyB'],
      privacy_budget: Math.random() * 5 + 2,
      max_iterations: Math.floor(Math.random() * 100) + 50,
      learning_rate: Math.random() * 0.01 + 0.001,
      hidden_layers: [Math.floor(Math.random() * 50) + 10, Math.floor(Math.random() * 20) + 5]
    }
  ];
  
  const config = configs[Math.floor(Math.random() * configs.length)];
  config.metadata = {
    test_type: 'stress_test',
    timestamp: new Date().toISOString(),
    test_id: `stress_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`
  };
  
  return config;
}

// 验证模型质量
function validateModelQuality(modelData) {
  if (!modelData || !modelData.metrics) {
    return false;
  }
  
  const metrics = modelData.metrics;
  
  // 检查准确率不为0或1
  if (metrics.accuracy !== undefined) {
    if (metrics.accuracy <= 0.01 || metrics.accuracy >= 0.99) {
      return false;
    }
  }
  
  // 检查损失不为0
  if (metrics.loss !== undefined) {
    if (metrics.loss <= 0.001) {
      return false;
    }
  }
  
  // 检查权重分布
  if (modelData.weights && Array.isArray(modelData.weights)) {
    const weights = modelData.weights;
    const mean = weights.reduce((sum, w) => sum + w, 0) / weights.length;
    const variance = weights.reduce((sum, w) => sum + Math.pow(w - mean, 2), 0) / weights.length;
    const stdDev = Math.sqrt(variance);
    
    // 权重标准差应大于0.01
    if (stdDev <= 0.01) {
      return false;
    }
  }
  
  return true;
}

// 主测试函数
export default function() {
  const scenario = Math.random();
  
  if (scenario < 0.3) {
    testTrainingStart();
  } else if (scenario < 0.5) {
    testTrainingStatus();
  } else if (scenario < 0.7) {
    testModelRetrieval();
  } else if (scenario < 0.9) {
    testHealthAndMetrics();
  } else {
    testTrainingHistory();
  }
  
  sleep(Math.random() * 2 + 0.5); // 0.5-2.5秒随机延迟
}

// 测试训练启动
function testTrainingStart() {
  const config = generateTrainingConfig();
  
  const startTime = Date.now();
  const response = http.post(`${TRAINER_URL}/training/start`, JSON.stringify(config), {
    headers: { 'Content-Type': 'application/json' },
  });
  const duration = Date.now() - startTime;
  
  trainingStartTime.add(duration);
  
  const success = check(response, {
    'training start request': (r) => r.status === 200 || r.status === 202,
  });
  
  if (success) {
    trainingJobsStarted.add(1);
    
    const responseData = response.json();
    if (responseData.job_id || responseData.training_id) {
      // 记录训练任务ID以便后续查询
      console.log(`训练任务启动: ${responseData.job_id || responseData.training_id}`);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试训练状态查询
function testTrainingStatus() {
  // 查询全局训练状态
  let response = http.get(`${TRAINER_URL}/training/status`);
  
  const success = check(response, {
    'training status': (r) => r.status === 200,
  });
  
  if (success) {
    const statusData = response.json();
    
    // 如果有活跃的训练任务，查询具体状态
    if (statusData.active_jobs && statusData.active_jobs.length > 0) {
      const jobId = statusData.active_jobs[0].job_id || statusData.active_jobs[0].id;
      if (jobId) {
        response = http.get(`${TRAINER_URL}/training/status/${jobId}`);
        check(response, {
          'specific job status': (r) => r.status === 200 || r.status === 404,
        }) || errorRate.add(1);
      }
    }
    
    // 检查是否有完成的训练任务
    if (statusData.completed_jobs && statusData.completed_jobs.length > 0) {
      trainingJobsCompleted.add(statusData.completed_jobs.length);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试模型检索和质量验证
function testModelRetrieval() {
  // 获取可用模型列表
  let response = http.get(`${TRAINER_URL}/models`);
  
  const success = check(response, {
    'models list': (r) => r.status === 200,
  });
  
  if (success) {
    const modelsData = response.json();
    
    if (modelsData.models && modelsData.models.length > 0) {
      // 随机选择一个模型进行详细检查
      const randomModel = modelsData.models[Math.floor(Math.random() * modelsData.models.length)];
      const modelId = randomModel.model_id || randomModel.id;
      
      if (modelId) {
        response = http.get(`${TRAINER_URL}/models/${modelId}`);
        const modelSuccess = check(response, {
          'model details': (r) => r.status === 200,
        });
        
        if (modelSuccess) {
          modelQualityChecks.add(1);
          const modelData = response.json();
          
          // 验证模型质量
          if (validateModelQuality(modelData)) {
            nonDegenerateModels.add(1);
          }
        } else {
          errorRate.add(1);
        }
      }
    }
  } else {
    errorRate.add(1);
  }
}

// 测试健康检查和指标
function testHealthAndMetrics() {
  // 健康检查
  let response = http.get(`${TRAINER_URL}/health`);
  check(response, {
    'health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 指标查询
  response = http.get(`${TRAINER_URL}/metrics`);
  check(response, {
    'metrics endpoint': (r) => r.status === 200,
  }) || errorRate.add(1);
}

// 测试训练历史
function testTrainingHistory() {
  const response = http.get(`${TRAINER_URL}/training/history`);
  
  const success = check(response, {
    'training history': (r) => r.status === 200,
  });
  
  if (success) {
    const historyData = response.json();
    
    // 验证历史数据结构
    check(historyData, {
      'valid history structure': (data) => 
        Array.isArray(data.history) || Array.isArray(data.jobs),
    }) || errorRate.add(1);
  } else {
    errorRate.add(1);
  }
}

// 设置阶段
export function setup() {
  console.log('开始训练服务压力测试...');
  
  // 检查服务可用性
  const healthResponse = http.get(`${TRAINER_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error('训练服务不可用');
  }
  
  return {
    startTime: new Date().toISOString()
  };
}

// 清理阶段
export function teardown(data) {
  console.log('训练服务压力测试完成');
  
  // 尝试清理测试产生的训练任务（如果有清理接口）
  const cleanupResponse = http.delete(`${TRAINER_URL}/training/cleanup/stress_test`);
  if (cleanupResponse.status === 200) {
    console.log('测试数据清理完成');
  }
}

// 测试总结
export function handleSummary(data) {
  const summary = generateTrainingSummary(data);
  
  return {
    'stdout': summary,
    'training_stress_results.json': JSON.stringify(data, null, 2),
    'training_stress_summary.txt': summary,
  };
}

function generateTrainingSummary(data) {
  const totalRequests = data.metrics.http_reqs.values.count;
  const jobsStarted = data.metrics.training_jobs_started ? data.metrics.training_jobs_started.values.count : 0;
  const jobsCompleted = data.metrics.training_jobs_completed ? data.metrics.training_jobs_completed.values.count : 0;
  const qualityChecks = data.metrics.model_quality_checks ? data.metrics.model_quality_checks.values.count : 0;
  const nonDegenerateModels = data.metrics.non_degenerate_models ? data.metrics.non_degenerate_models.values.count : 0;
  const errorRate = data.metrics.errors.values.rate * 100;
  const nonDegenerateRate = qualityChecks > 0 ? (nonDegenerateModels / qualityChecks) * 100 : 0;
  
  return `
=== 训练服务压力测试报告 ===

测试概况:
- 测试持续时间: ${(data.state.testRunDurationMs / 1000).toFixed(1)}秒
- 总请求数: ${totalRequests}
- 启动训练任务: ${jobsStarted}
- 完成训练任务: ${jobsCompleted}
- 模型质量检查: ${qualityChecks}
- 非退化模型: ${nonDegenerateModels}
- 错误率: ${errorRate.toFixed(2)}%

响应时间统计:
- 平均响应时间: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 中位数响应时间: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms
- 95%分位响应时间: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99%分位响应时间: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

训练性能:
- 平均训练启动时间: ${data.metrics.training_start_time ? data.metrics.training_start_time.values.avg.toFixed(2) : 'N/A'}ms
- 95%训练启动时间: ${data.metrics.training_start_time ? data.metrics.training_start_time.values['p(95)'].toFixed(2) : 'N/A'}ms
- 训练完成率: ${jobsStarted > 0 ? ((jobsCompleted / jobsStarted) * 100).toFixed(2) : 'N/A'}%
- 非退化模型率: ${nonDegenerateRate.toFixed(2)}%

性能阈值检查:
- 95%响应时间 < 5000ms: ${data.metrics.http_req_duration.values['p(95)'] < 5000 ? '✓ 通过' : '✗ 失败'}
- 错误率 < 5%: ${errorRate < 5 ? '✓ 通过' : '✗ 失败'}
- 非退化模型率 > 80%: ${nonDegenerateRate > 80 ? '✓ 通过' : '✗ 失败'}

建议:
${generateTrainingRecommendations(data)}

=== 训练测试完成 ===
`;
}

function generateTrainingRecommendations(data) {
  const recommendations = [];
  const errorRate = data.metrics.errors.values.rate * 100;
  const avgResponseTime = data.metrics.http_req_duration.values.avg;
  const qualityChecks = data.metrics.model_quality_checks ? data.metrics.model_quality_checks.values.count : 0;
  const nonDegenerateModels = data.metrics.non_degenerate_models ? data.metrics.non_degenerate_models.values.count : 0;
  const nonDegenerateRate = qualityChecks > 0 ? (nonDegenerateModels / qualityChecks) * 100 : 0;
  
  if (errorRate > 5) {
    recommendations.push('- 错误率过高，建议检查训练服务日志和配置');
  }
  
  if (avgResponseTime > 3000) {
    recommendations.push('- 平均响应时间较长，建议优化训练算法或增加计算资源');
  }
  
  if (nonDegenerateRate < 80) {
    recommendations.push('- 非退化模型比例较低，建议检查训练数据质量和算法参数');
  }
  
  if (qualityChecks === 0) {
    recommendations.push('- 未进行模型质量检查，建议确保有可用的训练模型');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('- 训练服务性能表现良好，模型质量符合预期');
  }
  
  return recommendations.join('\n');
}