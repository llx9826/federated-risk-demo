import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const rateLimitRate = new Rate('rate_limited');
const sessionCreationTime = new Trend('session_creation_time');
const sessionsCreated = new Counter('sessions_created');
const rateLimitedRequests = new Counter('rate_limited_requests');

// PSI专用测试配置 - 考虑速率限制
export const options = {
  stages: [
    { duration: '30s', target: 2 },   // 缓慢启动
    { duration: '2m', target: 5 },    // 低负载测试
    { duration: '3m', target: 8 },    // 中等负载
    { duration: '2m', target: 12 },   // 接近限制
    { duration: '1m', target: 5 },    // 降低负载
    { duration: '30s', target: 0 },   // 停止
  ],
  thresholds: {
    http_req_duration: ['p(95)<3000'], // PSI操作可能较慢
    http_req_failed: ['rate<0.1'],     // 允许10%失败（主要是速率限制）
    rate_limited: ['rate<0.5'],        // 速率限制应少于50%
    errors: ['rate<0.05'],             // 真正的错误应少于5%
  },
};

const PSI_URL = 'http://localhost:8001';

// 生成唯一会话ID
function generateSessionId() {
  return `stress_test_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
}

// 生成随机参与方ID
function generatePartyId() {
  return `party_${Math.floor(Math.random() * 1000) + 1}`;
}

// 主测试函数
export default function() {
  // 随机选择测试场景
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    testSessionCreation();
  } else if (scenario < 0.6) {
    testSessionQuery();
  } else if (scenario < 0.8) {
    testHealthAndMetrics();
  } else {
    testPSICompute();
  }
  
  // 添加随机延迟以模拟真实使用模式
  sleep(Math.random() * 3 + 1); // 1-4秒随机延迟
}

// 测试PSI会话创建
function testSessionCreation() {
  const sessionId = generateSessionId();
  const partyId = generatePartyId();
  const otherParty = generatePartyId();
  
  const sessionData = {
    session_id: sessionId,
    party_role: Math.random() < 0.5 ? 'sender' : 'receiver',
    party_id: partyId,
    other_parties: [otherParty],
    timeout_seconds: Math.floor(Math.random() * 1800) + 600, // 10-30分钟
    metadata: {
      test_type: 'stress_test',
      timestamp: new Date().toISOString(),
      batch_id: `batch_${Math.floor(Math.random() * 100)}`
    }
  };
  
  const startTime = Date.now();
  const response = http.post(`${PSI_URL}/psi/session`, JSON.stringify(sessionData), {
    headers: { 'Content-Type': 'application/json' },
  });
  const duration = Date.now() - startTime;
  
  sessionCreationTime.add(duration);
  
  if (response.status === 429) {
    // 速率限制
    rateLimitRate.add(1);
    rateLimitedRequests.add(1);
    check(response, {
      'rate limit response': (r) => r.status === 429,
    });
  } else if (response.status === 200) {
    // 成功创建
    sessionsCreated.add(1);
    check(response, {
      'session created successfully': (r) => {
        const data = r.json();
        return data.session_id === sessionId;
      },
    }) || errorRate.add(1);
  } else {
    // 其他错误
    errorRate.add(1);
    check(response, {
      'unexpected error': (r) => false, // 记录意外错误
    });
  }
}

// 测试会话查询
function testSessionQuery() {
  // 查询所有会话
  let response = http.get(`${PSI_URL}/psi/sessions`);
  
  const success = check(response, {
    'sessions list': (r) => r.status === 200,
  });
  
  if (success) {
    const sessionsData = response.json();
    
    // 如果有会话，随机查询一个具体会话
    if (sessionsData && sessionsData.length > 0) {
      const randomSession = sessionsData[Math.floor(Math.random() * sessionsData.length)];
      if (randomSession.session_id) {
        response = http.get(`${PSI_URL}/psi/sessions/${randomSession.session_id}`);
        check(response, {
          'specific session query': (r) => r.status === 200 || r.status === 404,
        }) || errorRate.add(1);
      }
    }
  } else {
    errorRate.add(1);
  }
}

// 测试健康检查和指标
function testHealthAndMetrics() {
  // 健康检查
  let response = http.get(`${PSI_URL}/health`);
  check(response, {
    'health check': (r) => r.status === 200 && r.json().status === 'healthy',
  }) || errorRate.add(1);
  
  // 指标查询
  response = http.get(`${PSI_URL}/metrics`);
  check(response, {
    'metrics endpoint': (r) => r.status === 200,
  }) || errorRate.add(1);
}

// 测试PSI计算（模拟）
function testPSICompute() {
  const sessionId = `compute_test_${Date.now()}`;
  
  // 尝试计算（预期会失败，因为没有有效会话）
  const computeData = {
    session_id: sessionId,
    operation: 'intersection',
    metadata: {
      test_type: 'compute_stress_test'
    }
  };
  
  const response = http.post(`${PSI_URL}/psi/compute`, JSON.stringify(computeData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  // 预期返回4xx错误（会话不存在）
  check(response, {
    'compute endpoint accessible': (r) => r.status >= 400 && r.status < 500,
  }) || errorRate.add(1);
}

// 设置阶段 - 清理旧的测试会话
export function setup() {
  console.log('开始PSI压力测试...');
  
  // 检查服务可用性
  const healthResponse = http.get(`${PSI_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error('PSI服务不可用');
  }
  
  return {
    startTime: new Date().toISOString()
  };
}

// 清理阶段
export function teardown(data) {
  console.log('PSI压力测试完成');
  console.log(`测试开始时间: ${data.startTime}`);
  console.log(`测试结束时间: ${new Date().toISOString()}`);
}

// 测试总结
export function handleSummary(data) {
  const summary = generatePSISummary(data);
  
  return {
    'stdout': summary,
    'psi_stress_results.json': JSON.stringify(data, null, 2),
    'psi_stress_summary.txt': summary,
  };
}

function generatePSISummary(data) {
  const totalRequests = data.metrics.http_reqs.values.count;
  const sessionsCreated = data.metrics.sessions_created ? data.metrics.sessions_created.values.count : 0;
  const rateLimited = data.metrics.rate_limited_requests ? data.metrics.rate_limited_requests.values.count : 0;
  const errorRate = data.metrics.errors.values.rate * 100;
  const rateLimitRate = data.metrics.rate_limited.values.rate * 100;
  
  return `
=== PSI服务压力测试报告 ===

测试概况:
- 测试持续时间: ${(data.state.testRunDurationMs / 1000).toFixed(1)}秒
- 总请求数: ${totalRequests}
- 成功创建会话: ${sessionsCreated}
- 速率限制请求: ${rateLimited}
- 速率限制比例: ${rateLimitRate.toFixed(2)}%
- 错误率: ${errorRate.toFixed(2)}%

响应时间统计:
- 平均响应时间: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 中位数响应时间: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms
- 95%分位响应时间: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99%分位响应时间: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

会话创建性能:
- 平均创建时间: ${data.metrics.session_creation_time ? data.metrics.session_creation_time.values.avg.toFixed(2) : 'N/A'}ms
- 95%创建时间: ${data.metrics.session_creation_time ? data.metrics.session_creation_time.values['p(95)'].toFixed(2) : 'N/A'}ms

性能阈值检查:
- 95%响应时间 < 3000ms: ${data.metrics.http_req_duration.values['p(95)'] < 3000 ? '✓ 通过' : '✗ 失败'}
- 错误率 < 5%: ${errorRate < 5 ? '✓ 通过' : '✗ 失败'}
- 速率限制 < 50%: ${rateLimitRate < 50 ? '✓ 通过' : '✗ 失败'}

建议:
${generateRecommendations(data)}

=== PSI测试完成 ===
`;
}

function generateRecommendations(data) {
  const recommendations = [];
  const errorRate = data.metrics.errors.values.rate * 100;
  const rateLimitRate = data.metrics.rate_limited.values.rate * 100;
  const avgResponseTime = data.metrics.http_req_duration.values.avg;
  
  if (errorRate > 5) {
    recommendations.push('- 错误率过高，建议检查PSI服务日志和配置');
  }
  
  if (rateLimitRate > 50) {
    recommendations.push('- 速率限制过于频繁，建议调整限流配置或增加服务实例');
  }
  
  if (avgResponseTime > 1000) {
    recommendations.push('- 平均响应时间较长，建议优化PSI算法或增加计算资源');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('- PSI服务性能表现良好，可以考虑逐步增加负载');
  }
  
  return recommendations.join('\n');
}