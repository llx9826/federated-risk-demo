import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// 自定义指标
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const successfulRequests = new Counter('successful_requests');
const failedRequests = new Counter('failed_requests');

// 测试配置
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8003';
const TEST_DURATION = __ENV.TEST_DURATION || '60s';
const RAMP_UP_TIME = __ENV.RAMP_UP_TIME || '30s';
const RAMP_DOWN_TIME = __ENV.RAMP_DOWN_TIME || '30s';

// 测试数据生成
function generateTestData() {
    return {
        features: {
            credit_score: Math.floor(Math.random() * 400) + 300, // 300-700
            annual_income: Math.floor(Math.random() * 150000) + 30000, // 30k-180k
            debt_to_income_ratio: Math.random() * 0.6 + 0.1, // 0.1-0.7
            employment_length: Math.floor(Math.random() * 20) + 1, // 1-20 years
            loan_amount: Math.floor(Math.random() * 50000) + 5000, // 5k-55k
            loan_purpose: ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'][Math.floor(Math.random() * 4)],
            home_ownership: ['RENT', 'OWN', 'MORTGAGE'][Math.floor(Math.random() * 3)],
            verification_status: ['Verified', 'Source Verified', 'Not Verified'][Math.floor(Math.random() * 3)],
            loan_term: [36, 60][Math.floor(Math.random() * 2)],
            interest_rate: Math.random() * 20 + 5, // 5-25%
            grade: ['A', 'B', 'C', 'D', 'E', 'F', 'G'][Math.floor(Math.random() * 7)],
            sub_grade: Math.floor(Math.random() * 5) + 1, // 1-5
            inquiries_last_6m: Math.floor(Math.random() * 10), // 0-9
            delinq_2yrs: Math.floor(Math.random() * 5), // 0-4
            pub_rec: Math.floor(Math.random() * 3), // 0-2
            revol_bal: Math.floor(Math.random() * 50000), // 0-50k
            revol_util: Math.random() * 100, // 0-100%
            total_acc: Math.floor(Math.random() * 50) + 5, // 5-55
            initial_list_status: ['w', 'f'][Math.floor(Math.random() * 2)],
            application_type: ['Individual', 'Joint App'][Math.floor(Math.random() * 2)],
            mort_acc: Math.floor(Math.random() * 10), // 0-9
            pub_rec_bankruptcies: Math.floor(Math.random() * 3) // 0-2
        },
        model_version: 'federated_risk_model_latest',
        request_id: `bench_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
}

// 批量评分数据生成
function generateBatchTestData(batchSize = 64) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
        batch.push(generateTestData());
    }
    return {
        batch: batch,
        batch_id: `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
}

// 测试场景配置
export const options = {
    scenarios: {
        // 梯度压力测试：10 -> 100 -> 300 RPS
        ramp_up_test: {
            executor: 'ramping-vus',
            stages: [
                { duration: '30s', target: 10 },   // 10 RPS
                { duration: '60s', target: 10 },   // 保持 10 RPS
                { duration: '30s', target: 50 },   // 升至 50 RPS
                { duration: '60s', target: 50 },   // 保持 50 RPS
                { duration: '30s', target: 100 },  // 升至 100 RPS
                { duration: '60s', target: 100 },  // 保持 100 RPS
                { duration: '30s', target: 200 },  // 升至 200 RPS
                { duration: '60s', target: 200 },  // 保持 200 RPS
                { duration: '30s', target: 300 },  // 升至 300 RPS
                { duration: '60s', target: 300 },  // 保持 300 RPS
                { duration: '30s', target: 0 },    // 降至 0
            ],
            gracefulRampDown: '30s',
        },
        
        // 稳定性测试
        stability_test: {
            executor: 'constant-vus',
            vus: 50,
            duration: '300s', // 5分钟稳定性测试
            startTime: '600s', // 在梯度测试后开始
        },
        
        // 峰值测试
        spike_test: {
            executor: 'ramping-vus',
            stages: [
                { duration: '10s', target: 100 },
                { duration: '30s', target: 500 }, // 突然峰值
                { duration: '10s', target: 100 },
            ],
            startTime: '900s', // 在稳定性测试后开始
        }
    },
    
    thresholds: {
        http_req_duration: ['p(50)<100', 'p(95)<500', 'p(99)<1000'], // 响应时间阈值
        http_req_failed: ['rate<0.05'], // 错误率 < 5%
        error_rate: ['rate<0.05'],
        response_time: ['p(95)<500'],
    },
};

// 单次评分测试
export function singleScore() {
    const testData = generateTestData();
    
    const params = {
        headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': testData.request_id,
        },
        timeout: '30s',
    };
    
    const response = http.post(`${BASE_URL}/score`, JSON.stringify(testData), params);
    
    // 检查响应
    const success = check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 1000ms': (r) => r.timings.duration < 1000,
        'has score field': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.hasOwnProperty('score');
            } catch (e) {
                return false;
            }
        },
        'score is valid': (r) => {
            try {
                const body = JSON.parse(r.body);
                const score = body.score;
                return typeof score === 'number' && score >= 0 && score <= 1;
            } catch (e) {
                return false;
            }
        },
        'has audit receipt': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.hasOwnProperty('audit_receipt');
            } catch (e) {
                return false;
            }
        }
    });
    
    // 记录指标
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    
    if (success) {
        successfulRequests.add(1);
    } else {
        failedRequests.add(1);
        console.error(`Request failed: ${response.status} - ${response.body}`);
    }
    
    // 解析响应以获取详细指标
    if (response.status === 200) {
        try {
            const body = JSON.parse(response.body);
            
            // 检查审计回执完整性
            if (body.audit_receipt) {
                const receipt = body.audit_receipt;
                const requiredFields = ['request_id', 'model_version', 'timestamp', 'score_hash'];
                const missingFields = requiredFields.filter(field => !receipt.hasOwnProperty(field));
                
                if (missingFields.length > 0) {
                    console.warn(`Missing audit fields: ${missingFields.join(', ')}`);
                }
            }
            
        } catch (e) {
            console.error(`Failed to parse response: ${e.message}`);
        }
    }
    
    sleep(0.1); // 100ms 间隔
}

// 批量评分测试
export function batchScore() {
    const batchData = generateBatchTestData(64); // 64个样本的批次
    
    const params = {
        headers: {
            'Content-Type': 'application/json',
            'X-Batch-ID': batchData.batch_id,
        },
        timeout: '60s', // 批量请求需要更长超时
    };
    
    const response = http.post(`${BASE_URL}/score/batch`, JSON.stringify(batchData), params);
    
    const success = check(response, {
        'batch status is 200': (r) => r.status === 200,
        'batch response time < 5000ms': (r) => r.timings.duration < 5000,
        'has batch results': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.hasOwnProperty('results') && Array.isArray(body.results);
            } catch (e) {
                return false;
            }
        },
        'correct batch size': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.results && body.results.length === 64;
            } catch (e) {
                return false;
            }
        }
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    
    if (success) {
        successfulRequests.add(1);
    } else {
        failedRequests.add(1);
    }
    
    sleep(0.5); // 批量请求间隔更长
}

// 健康检查测试
export function healthCheck() {
    const response = http.get(`${BASE_URL}/health`);
    
    check(response, {
        'health check status is 200': (r) => r.status === 200,
        'health check response time < 100ms': (r) => r.timings.duration < 100,
    });
    
    sleep(5); // 每5秒检查一次
}

// 默认测试函数（单次评分）
export default function() {
    // 90% 单次评分，10% 批量评分
    if (Math.random() < 0.9) {
        singleScore();
    } else {
        batchScore();
    }
}

// 测试设置函数
export function setup() {
    console.log('开始评分性能基准测试...');
    console.log(`目标服务: ${BASE_URL}`);
    
    // 预热请求
    console.log('执行预热请求...');
    const warmupData = generateTestData();
    const warmupResponse = http.post(`${BASE_URL}/score`, JSON.stringify(warmupData), {
        headers: { 'Content-Type': 'application/json' },
        timeout: '30s'
    });
    
    if (warmupResponse.status !== 200) {
        console.error(`预热失败: ${warmupResponse.status} - ${warmupResponse.body}`);
        throw new Error('服务预热失败，请检查服务状态');
    }
    
    console.log('预热成功，开始正式测试...');
    return { startTime: Date.now() };
}

// 测试清理函数
export function teardown(data) {
    const duration = Date.now() - data.startTime;
    console.log(`测试完成，总耗时: ${duration}ms`);
}

// 自定义摘要报告
export function handleSummary(data) {
    const summary = {
        timestamp: new Date().toISOString(),
        test_duration: data.state.testRunDurationMs,
        scenarios: {},
        metrics: {
            http_reqs: data.metrics.http_reqs,
            http_req_duration: data.metrics.http_req_duration,
            http_req_failed: data.metrics.http_req_failed,
            vus: data.metrics.vus,
            vus_max: data.metrics.vus_max,
            iterations: data.metrics.iterations,
            data_received: data.metrics.data_received,
            data_sent: data.metrics.data_sent
        },
        custom_metrics: {
            error_rate: data.metrics.error_rate,
            response_time: data.metrics.response_time,
            successful_requests: data.metrics.successful_requests,
            failed_requests: data.metrics.failed_requests
        },
        thresholds: data.thresholds
    };
    
    // 计算关键指标
    const httpReqs = data.metrics.http_reqs;
    const httpReqDuration = data.metrics.http_req_duration;
    const httpReqFailed = data.metrics.http_req_failed;
    
    if (httpReqs && httpReqDuration) {
        summary.performance = {
            total_requests: httpReqs.count,
            requests_per_second: httpReqs.rate,
            avg_response_time: httpReqDuration.avg,
            p50_response_time: httpReqDuration.p50,
            p95_response_time: httpReqDuration.p95,
            p99_response_time: httpReqDuration.p99,
            max_response_time: httpReqDuration.max,
            error_rate: httpReqFailed ? httpReqFailed.rate : 0,
            throughput_mb_per_sec: data.metrics.data_received ? data.metrics.data_received.rate / 1024 / 1024 : 0
        };
    }
    
    // 生成文本摘要
    const textReport = textSummary(data, { indent: ' ', enableColors: true });
    
    // 保存详细报告
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    return {
        'stdout': textReport,
        [`reports/bench/score/k6_summary_${timestamp}.json`]: JSON.stringify(summary, null, 2),
        [`reports/bench/score/k6_detailed_${timestamp}.json`]: JSON.stringify(data, null, 2),
        'reports/bench/score/latest.json': JSON.stringify(summary, null, 2)
    };
}