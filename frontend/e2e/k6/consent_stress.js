import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const consentCreationTime = new Trend('consent_creation_time');
const consentRequests = new Counter('consent_requests');
const consentApprovals = new Counter('consent_approvals');
const consentRevocations = new Counter('consent_revocations');
const validConsents = new Counter('valid_consents');
const auditLogEntries = new Counter('audit_log_entries');

// 同意服务专用测试配置
export const options = {
  stages: [
    { duration: '30s', target: 8 },   // 快速启动
    { duration: '2m', target: 25 },   // 中等负载
    { duration: '4m', target: 60 },   // 高负载
    { duration: '3m', target: 40 },   // 稳定负载
    { duration: '2m', target: 15 },   // 降低负载
    { duration: '30s', target: 0 },   // 停止
  ],
  thresholds: {
    http_req_duration: ['p(95)<1500'], // 同意操作应该快速
    http_req_failed: ['rate<0.05'],    // 允许5%失败
    errors: ['rate<0.02'],             // 真正的错误应少于2%
    consent_creation_time: ['p(95)<1000'], // 95%的同意创建应在1秒内完成
    valid_consents: ['rate>0.95'],     // 95%以上的同意应有效
  },
};

const CONSENT_URL = 'http://localhost:8000';

// 生成同意请求数据
function generateConsentData() {
  const consentTypes = [
    {
      type: 'data_sharing',
      purpose: 'federated_learning',
      data_categories: ['financial_records', 'transaction_history'],
      retention_period: Math.floor(Math.random() * 365) + 30, // 30-395天
      sharing_scope: 'consortium_members'
    },
    {
      type: 'model_training',
      purpose: 'risk_assessment',
      data_categories: ['credit_score', 'payment_history', 'demographic_info'],
      retention_period: Math.floor(Math.random() * 730) + 90, // 90-820天
      sharing_scope: 'training_participants'
    },
    {
      type: 'analytics',
      purpose: 'fraud_detection',
      data_categories: ['transaction_patterns', 'device_info', 'location_data'],
      retention_period: Math.floor(Math.random() * 180) + 60, // 60-240天
      sharing_scope: 'security_partners'
    },
    {
      type: 'research',
      purpose: 'algorithm_improvement',
      data_categories: ['aggregated_statistics', 'model_parameters'],
      retention_period: Math.floor(Math.random() * 1095) + 180, // 180-1275天
      sharing_scope: 'research_consortium'
    }
  ];
  
  const consent = consentTypes[Math.floor(Math.random() * consentTypes.length)];
  
  return {
    user_id: `user_${Math.random().toString(36).substr(2, 8)}`,
    consent_id: `consent_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
    ...consent,
    granted_at: new Date().toISOString(),
    expires_at: new Date(Date.now() + consent.retention_period * 24 * 60 * 60 * 1000).toISOString(),
    granularity: Math.random() > 0.5 ? 'fine_grained' : 'coarse_grained',
    revocable: Math.random() > 0.1, // 90%可撤销
    metadata: {
      test_type: 'stress_test',
      timestamp: new Date().toISOString(),
      ip_address: `192.168.1.${Math.floor(Math.random() * 254) + 1}`,
      user_agent: 'k6-stress-test/1.0'
    }
  };
}

// 验证同意数据
function validateConsent(consentData) {
  if (!consentData) {
    return false;
  }
  
  // 检查必需字段
  const requiredFields = ['consent_id', 'user_id', 'type', 'purpose', 'granted_at'];
  for (const field of requiredFields) {
    if (!consentData[field]) {
      return false;
    }
  }
  
  // 检查时间戳格式
  if (consentData.granted_at) {
    const grantedDate = new Date(consentData.granted_at);
    if (isNaN(grantedDate.getTime())) {
      return false;
    }
  }
  
  // 检查过期时间
  if (consentData.expires_at) {
    const expiresDate = new Date(consentData.expires_at);
    const grantedDate = new Date(consentData.granted_at);
    if (isNaN(expiresDate.getTime()) || expiresDate <= grantedDate) {
      return false;
    }
  }
  
  // 检查数据类别
  if (consentData.data_categories && !Array.isArray(consentData.data_categories)) {
    return false;
  }
  
  return true;
}

// 主测试函数
export default function() {
  const scenario = Math.random();
  
  if (scenario < 0.3) {
    testConsentCreation();
  } else if (scenario < 0.5) {
    testConsentQuery();
  } else if (scenario < 0.65) {
    testConsentUpdate();
  } else if (scenario < 0.8) {
    testConsentRevocation();
  } else if (scenario < 0.9) {
    testAuditLog();
  } else {
    testHealthAndCompliance();
  }
  
  sleep(Math.random() * 0.8 + 0.1); // 0.1-0.9秒随机延迟
}

// 测试同意创建
function testConsentCreation() {
  const consentData = generateConsentData();
  
  const startTime = Date.now();
  const response = http.post(`${CONSENT_URL}/consent`, JSON.stringify(consentData), {
    headers: { 'Content-Type': 'application/json' },
  });
  const duration = Date.now() - startTime;
  
  consentCreationTime.add(duration);
  consentRequests.add(1);
  
  const success = check(response, {
    'consent creation': (r) => r.status === 201 || r.status === 200,
  });
  
  if (success) {
    try {
      const responseData = response.json();
      if (validateConsent(responseData)) {
        validConsents.add(1);
        
        // 记录同意ID以便后续操作
        if (responseData.consent_id) {
          console.log(`同意创建成功: ${responseData.consent_id}`);
        }
      }
    } catch (e) {
      errorRate.add(1);
    }
  } else {
    errorRate.add(1);
  }
}

// 测试同意查询
function testConsentQuery() {
  const userId = `user_${Math.random().toString(36).substr(2, 8)}`;
  
  // 查询用户的所有同意
  let response = http.get(`${CONSENT_URL}/consent/user/${userId}`);
  
  const success = check(response, {
    'consent query by user': (r) => r.status === 200 || r.status === 404,
  });
  
  if (success && response.status === 200) {
    try {
      const consentsData = response.json();
      
      if (consentsData.consents && Array.isArray(consentsData.consents)) {
        // 如果有同意记录，查询具体同意详情
        if (consentsData.consents.length > 0) {
          const randomConsent = consentsData.consents[Math.floor(Math.random() * consentsData.consents.length)];
          const consentId = randomConsent.consent_id || randomConsent.id;
          
          if (consentId) {
            response = http.get(`${CONSENT_URL}/consent/${consentId}`);
            check(response, {
              'specific consent query': (r) => r.status === 200 || r.status === 404,
            }) || errorRate.add(1);
          }
        }
      }
    } catch (e) {
      errorRate.add(1);
    }
  } else if (!success) {
    errorRate.add(1);
  }
  
  // 查询活跃同意
  response = http.get(`${CONSENT_URL}/consent/active`);
  check(response, {
    'active consents query': (r) => r.status === 200,
  }) || errorRate.add(1);
}

// 测试同意更新
function testConsentUpdate() {
  // 首先创建一个同意
  const consentData = generateConsentData();
  let response = http.post(`${CONSENT_URL}/consent`, JSON.stringify(consentData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  if (response.status === 201 || response.status === 200) {
    try {
      const createdConsent = response.json();
      const consentId = createdConsent.consent_id || createdConsent.id;
      
      if (consentId) {
        // 更新同意
        const updateData = {
          data_categories: [...consentData.data_categories, 'additional_category'],
          retention_period: consentData.retention_period + 30,
          updated_at: new Date().toISOString(),
          update_reason: 'scope_expansion'
        };
        
        response = http.put(`${CONSENT_URL}/consent/${consentId}`, JSON.stringify(updateData), {
          headers: { 'Content-Type': 'application/json' },
        });
        
        check(response, {
          'consent update': (r) => r.status === 200,
        }) || errorRate.add(1);
      }
    } catch (e) {
      errorRate.add(1);
    }
  }
}

// 测试同意撤销
function testConsentRevocation() {
  // 首先创建一个可撤销的同意
  const consentData = generateConsentData();
  consentData.revocable = true;
  
  let response = http.post(`${CONSENT_URL}/consent`, JSON.stringify(consentData), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  if (response.status === 201 || response.status === 200) {
    try {
      const createdConsent = response.json();
      const consentId = createdConsent.consent_id || createdConsent.id;
      
      if (consentId) {
        // 撤销同意
        const revocationData = {
          revoked_at: new Date().toISOString(),
          revocation_reason: 'user_request',
          effective_immediately: Math.random() > 0.5
        };
        
        response = http.post(`${CONSENT_URL}/consent/${consentId}/revoke`, JSON.stringify(revocationData), {
          headers: { 'Content-Type': 'application/json' },
        });
        
        const success = check(response, {
          'consent revocation': (r) => r.status === 200,
        });
        
        if (success) {
          consentRevocations.add(1);
        } else {
          errorRate.add(1);
        }
      }
    } catch (e) {
      errorRate.add(1);
    }
  }
}

// 测试审计日志
function testAuditLog() {
  // 查询审计日志
  const response = http.get(`${CONSENT_URL}/audit/log`);
  
  const success = check(response, {
    'audit log query': (r) => r.status === 200,
  });
  
  if (success) {
    try {
      const auditData = response.json();
      
      if (auditData.entries && Array.isArray(auditData.entries)) {
        auditLogEntries.add(auditData.entries.length);
        
        // 验证审计日志条目结构
        for (const entry of auditData.entries.slice(0, 5)) { // 只检查前5个
          check(entry, {
            'valid audit entry': (e) => 
              e.timestamp && e.action && e.user_id,
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

// 测试健康检查和合规性
function testHealthAndCompliance() {
  // 健康检查
  let response = http.get(`${CONSENT_URL}/health`);
  check(response, {
    'health check': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 合规性检查
  response = http.get(`${CONSENT_URL}/compliance/status`);
  check(response, {
    'compliance status': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  // 数据保护影响评估
  response = http.get(`${CONSENT_URL}/dpia/summary`);
  check(response, {
    'DPIA summary': (r) => r.status === 200 || r.status === 404,
  }) || errorRate.add(1);
  
  // 同意统计
  response = http.get(`${CONSENT_URL}/consent/statistics`);
  check(response, {
    'consent statistics': (r) => r.status === 200,
  }) || errorRate.add(1);
}

// 设置阶段
export function setup() {
  console.log('开始同意服务压力测试...');
  
  // 检查服务可用性
  const healthResponse = http.get(`${CONSENT_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error('同意服务不可用');
  }
  
  return {
    startTime: new Date().toISOString()
  };
}

// 清理阶段
export function teardown(data) {
  console.log('同意服务压力测试完成');
  
  // 清理测试数据（如果有清理接口）
  const cleanupResponse = http.delete(`${CONSENT_URL}/consent/cleanup/stress_test`);
  if (cleanupResponse.status === 200) {
    console.log('测试数据清理完成');
  }
}

// 测试总结
export function handleSummary(data) {
  const summary = generateConsentSummary(data);
  
  return {
    'stdout': summary,
    'consent_stress_results.json': JSON.stringify(data, null, 2),
    'consent_stress_summary.txt': summary,
  };
}

function generateConsentSummary(data) {
  const totalRequests = data.metrics.http_reqs.values.count;
  const consentRequests = data.metrics.consent_requests ? data.metrics.consent_requests.values.count : 0;
  const validConsents = data.metrics.valid_consents ? data.metrics.valid_consents.values.count : 0;
  const consentRevocations = data.metrics.consent_revocations ? data.metrics.consent_revocations.values.count : 0;
  const auditEntries = data.metrics.audit_log_entries ? data.metrics.audit_log_entries.values.count : 0;
  const errorRate = data.metrics.errors.values.rate * 100;
  const validRate = consentRequests > 0 ? (validConsents / consentRequests) * 100 : 0;
  
  return `
=== 同意服务压力测试报告 ===

测试概况:
- 测试持续时间: ${(data.state.testRunDurationMs / 1000).toFixed(1)}秒
- 总请求数: ${totalRequests}
- 同意请求数: ${consentRequests}
- 有效同意数: ${validConsents}
- 同意撤销数: ${consentRevocations}
- 审计日志条目: ${auditEntries}
- 错误率: ${errorRate.toFixed(2)}%
- 同意有效率: ${validRate.toFixed(2)}%

响应时间统计:
- 平均响应时间: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
- 中位数响应时间: ${data.metrics.http_req_duration.values.med.toFixed(2)}ms
- 95%分位响应时间: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
- 99%分位响应时间: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

同意管理性能:
- 平均同意创建时间: ${data.metrics.consent_creation_time ? data.metrics.consent_creation_time.values.avg.toFixed(2) : 'N/A'}ms
- 95%同意创建时间: ${data.metrics.consent_creation_time ? data.metrics.consent_creation_time.values['p(95)'].toFixed(2) : 'N/A'}ms
- 同意处理吞吐量: ${consentRequests > 0 ? (consentRequests / (data.state.testRunDurationMs / 1000)).toFixed(2) : 'N/A'} 请求/秒
- 撤销成功率: ${consentRequests > 0 ? ((consentRevocations / consentRequests) * 100).toFixed(2) : 'N/A'}%

性能阈值检查:
- 95%响应时间 < 1500ms: ${data.metrics.http_req_duration.values['p(95)'] < 1500 ? '✓ 通过' : '✗ 失败'}
- 95%同意创建时间 < 1000ms: ${data.metrics.consent_creation_time && data.metrics.consent_creation_time.values['p(95)'] < 1000 ? '✓ 通过' : '✗ 失败'}
- 错误率 < 2%: ${errorRate < 2 ? '✓ 通过' : '✗ 失败'}
- 同意有效率 > 95%: ${validRate > 95 ? '✓ 通过' : '✗ 失败'}

合规性指标:
- 审计日志完整性: ${auditEntries > 0 ? '✓ 正常记录' : '⚠ 无记录'}
- 同意撤销功能: ${consentRevocations > 0 ? '✓ 正常工作' : '⚠ 未测试'}
- 数据保护合规: ${data.metrics.http_req_failed.values.rate < 0.05 ? '✓ 符合要求' : '⚠ 需要关注'}

建议:
${generateConsentRecommendations(data)}

=== 同意管理测试完成 ===
`;
}

function generateConsentRecommendations(data) {
  const recommendations = [];
  const errorRate = data.metrics.errors.values.rate * 100;
  const avgResponseTime = data.metrics.http_req_duration.values.avg;
  const consentRequests = data.metrics.consent_requests ? data.metrics.consent_requests.values.count : 0;
  const validConsents = data.metrics.valid_consents ? data.metrics.valid_consents.values.count : 0;
  const auditEntries = data.metrics.audit_log_entries ? data.metrics.audit_log_entries.values.count : 0;
  const validRate = consentRequests > 0 ? (validConsents / consentRequests) * 100 : 0;
  
  if (errorRate > 2) {
    recommendations.push('- 错误率过高，建议检查同意服务日志和数据库连接');
  }
  
  if (avgResponseTime > 1000) {
    recommendations.push('- 平均响应时间较长，建议优化数据库查询和缓存策略');
  }
  
  if (validRate < 95) {
    recommendations.push('- 同意有效率较低，建议检查数据验证逻辑和输入格式');
  }
  
  if (auditEntries === 0) {
    recommendations.push('- 未记录审计日志，建议确保审计功能正常工作');
  }
  
  if (consentRequests === 0) {
    recommendations.push('- 未处理同意请求，建议确保同意服务正常接收请求');
  }
  
  const throughput = consentRequests > 0 ? (consentRequests / (data.state.testRunDurationMs / 1000)) : 0;
  if (throughput < 20) {
    recommendations.push('- 处理吞吐量较低，建议优化服务性能或扩展资源');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('- 同意服务性能表现良好，合规性功能正常');
  }
  
  return recommendations.join('\n');
}