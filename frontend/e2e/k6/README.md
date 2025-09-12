# 联邦风险评估系统 - K6 压力测试套件

本目录包含了针对联邦风险评估系统各个服务的完整压力测试套件，使用 [k6](https://k6.io/) 性能测试工具。

## 📋 测试概览

### 服务架构

```
联邦风险评估系统
├── 同意服务 (Consent Service)     - 端口 8000
├── PSI服务 (Private Set Intersection) - 端口 8001
├── 训练服务 (Training Service)     - 端口 8002
└── 推理服务 (Serving Service)      - 端口 8003
```

### 测试文件

| 测试文件 | 目标服务 | 主要测试场景 |
|---------|---------|-------------|
| `consent_stress.js` | 同意服务 | 同意创建、查询、更新、撤销、审计日志 |
| `psi_stress.js` | PSI服务 | 会话创建、PSI计算、健康检查 |
| `training_stress.js` | 训练服务 | 训练启动、状态查询、模型检索、质量验证 |
| `serving_stress.js` | 推理服务 | 单个预测、批量预测、模型信息查询 |
| `comprehensive_stress.js` | 全服务 | 跨服务协调、端到端流程 |
| `score_stress.js` | 评分服务 | 现有评分相关测试 |

## 🚀 快速开始

### 前置条件

1. **安装 k6**
   ```bash
   # macOS
   brew install k6
   
   # 其他平台请参考: https://k6.io/docs/getting-started/installation/
   ```

2. **启动所有服务**
   确保以下服务正在运行:
   - 同意服务: `http://localhost:8000`
   - PSI服务: `http://localhost:8001`
   - 训练服务: `http://localhost:8002`
   - 推理服务: `http://localhost:8003`

### 运行测试

#### 方式一: 运行所有测试 (推荐)

```bash
# 运行完整的压力测试套件
./run_all_stress_tests.sh
```

这将:
- 检查所有服务的健康状态
- 依次运行所有压力测试
- 生成综合测试报告
- 在 `stress_test_results_YYYYMMDD_HHMMSS/` 目录中保存结果

#### 方式二: 运行单个测试

```bash
# 测试同意服务
k6 run consent_stress.js

# 测试PSI服务
k6 run psi_stress.js

# 测试训练服务
k6 run training_stress.js

# 测试推理服务
k6 run serving_stress.js

# 测试全服务协调
k6 run comprehensive_stress.js
```

## 📊 性能目标

### 同意服务 (consent_stress.js)
- **响应时间**: 95% < 1500ms
- **错误率**: < 2%
- **同意有效率**: > 95%
- **特殊指标**: 审计日志完整性、撤销功能正常

### PSI服务 (psi_stress.js)
- **响应时间**: 95% < 3000ms
- **错误率**: < 5%
- **会话成功率**: > 90%
- **特殊指标**: 处理速率限制、计算准确性

### 训练服务 (training_stress.js)
- **响应时间**: 95% < 5000ms
- **错误率**: < 5%
- **非退化模型率**: > 80%
- **特殊指标**: 训练完成率、模型质量验证

### 推理服务 (serving_stress.js)
- **响应时间**: 95% < 2000ms
- **错误率**: < 3%
- **预测有效率**: > 90%
- **特殊指标**: 吞吐量、预测准确性

## 📈 测试配置

### 负载模式

所有测试都采用渐进式负载模式:

```javascript
stages: [
  { duration: '30s-1m', target: 5-8 },    // 启动阶段
  { duration: '2m-3m', target: 20-25 },   // 中等负载
  { duration: '4m-5m', target: 50-60 },   // 高负载
  { duration: '3m', target: 30-40 },      // 稳定负载
  { duration: '2m', target: 10-15 },      // 降低负载
  { duration: '30s-1m', target: 0 },      // 停止
]
```

### 自定义指标

每个测试都包含特定的业务指标:

- **错误率** (errors): 真正的业务错误
- **响应时间** (custom_duration): 特定操作的耗时
- **成功率** (success_rate): 业务操作成功比例
- **质量指标** (quality_metrics): 服务特定的质量度量

## 📋 测试报告

### 输出文件

运行测试后，会生成以下文件:

```
stress_test_results_YYYYMMDD_HHMMSS/
├── comprehensive_stress_report.md     # 综合测试报告
├── consent_stress_results.json        # 同意服务详细数据
├── consent_stress_summary.txt         # 同意服务摘要
├── psi_stress_results.json           # PSI服务详细数据
├── psi_stress_summary.txt            # PSI服务摘要
├── training_stress_results.json      # 训练服务详细数据
├── training_stress_summary.txt       # 训练服务摘要
├── serving_stress_results.json       # 推理服务详细数据
├── serving_stress_summary.txt        # 推理服务摘要
└── *_output.log                      # 各测试的控制台输出
```

### 报告内容

每个测试报告包含:

1. **测试概况**: 持续时间、请求数、成功率、错误率
2. **响应时间统计**: 平均值、中位数、95%/99%分位数
3. **业务指标**: 服务特定的性能指标
4. **阈值检查**: 是否达到预设的性能目标
5. **优化建议**: 基于测试结果的改进建议

## 🔧 自定义配置

### 修改测试参数

在各测试文件的 `options` 对象中调整:

```javascript
export const options = {
  stages: [
    // 修改负载阶段
  ],
  thresholds: {
    // 修改性能阈值
  },
};
```

### 修改服务端点

在各测试文件顶部修改服务URL:

```javascript
const SERVICE_URL = 'http://localhost:PORT';
```

### 添加自定义指标

```javascript
import { Counter, Rate, Trend } from 'k6/metrics';

const customMetric = new Counter('custom_metric');
// 在测试中使用: customMetric.add(1);
```

## 🐛 故障排除

### 常见问题

1. **服务不可用**
   ```
   错误: 服务健康检查失败
   解决: 确保所有服务正在运行并可访问
   ```

2. **速率限制**
   ```
   错误: HTTP 429 Too Many Requests
   解决: 调整测试负载或增加服务间延迟
   ```

3. **内存不足**
   ```
   错误: k6 内存溢出
   解决: 减少并发用户数或缩短测试时间
   ```

4. **网络超时**
   ```
   错误: 请求超时
   解决: 增加超时时间或检查网络连接
   ```

### 调试技巧

1. **启用详细日志**
   ```bash
   k6 run --verbose test.js
   ```

2. **减少负载测试**
   ```bash
   k6 run --vus 1 --duration 30s test.js
   ```

3. **检查服务日志**
   查看各服务的控制台输出以识别问题

## 📚 参考资料

- [k6 官方文档](https://k6.io/docs/)
- [k6 性能测试最佳实践](https://k6.io/docs/testing-guides/)
- [联邦学习性能优化指南](https://federated-learning.org/)

## 🤝 贡献

如需添加新的测试场景或改进现有测试:

1. 遵循现有的代码结构和命名约定
2. 添加适当的错误处理和验证
3. 更新相应的文档
4. 确保测试的可重复性和稳定性

---

*最后更新: $(date '+%Y-%m-%d')*