# 联邦风控系统测试指南

## 概述

本文档提供了联邦风控系统的完整测试指南，包括自动化测试、手动测试、性能测试和安全测试等。

## 快速开始

### 1. 运行完整测试套件

#### 使用新的测试运行器（推荐）

```bash
# 快速测试（开发时使用）
python scripts/test_runner.py quick

# 完整测试（发布前使用）
python scripts/test_runner.py full

# 仅健康检查
python scripts/test_runner.py health

# 仅API测试
python scripts/test_runner.py api

# 仅性能测试
python scripts/test_runner.py perf
```

#### 使用传统脚本

```bash
# 运行所有测试
./test.sh all

# 或者分步执行
./test.sh start-services    # 启动服务
./test.sh self-test         # 自动化测试
./test.sh code-quality      # 代码质量检查
./test.sh security-check    # 安全检查
./test.sh performance-test  # 性能测试
./test.sh report           # 生成报告
```

### 2. 查看服务状态

```bash
./test.sh status
```

### 3. 获取帮助

```bash
./test.sh help
```

## 测试类型

### 1. 自动化测试

系统提供了多个测试工具，适用于不同的测试场景：

#### 1.1 测试运行器 (test_runner.py)

统一的测试入口，支持多种测试模式：

```bash
# 快速测试（开发时使用，< 30秒）
python scripts/test_runner.py quick

# 完整测试（发布前使用，2-5分钟）
python scripts/test_runner.py full

# 健康检查（仅检查服务状态）
python scripts/test_runner.py health

# API测试（仅测试API功能）
python scripts/test_runner.py api

# 性能测试（仅测试性能基准）
python scripts/test_runner.py perf
```

#### 1.2 快速测试 (quick_test.py)

适合开发过程中的快速验证：

```bash
python scripts/quick_test.py
```

#### 1.3 综合测试 (comprehensive_test.py)

完整的系统测试套件：

```bash
python scripts/comprehensive_test.py
```

#### 1.4 传统自测脚本 (self_test.py)

```bash
# 基本模式
./test.sh self-test

# 详细模式
./test.sh self-test -v

# 直接运行
python scripts/self_test.py
```# 使用Python直接运行
python3 scripts/self_test.py --verbose
```

#### 1.2 运行特定模块测试

```bash
# 健康检查
./test.sh self-test health

# 同意服务测试
./test.sh self-test consent

# PSI服务测试
./test.sh self-test psi

# 模型训练服务测试
./test.sh self-test trainer

# 模型解释服务测试
./test.sh self-test explainer

# 数据文件检查
./test.sh self-test data

# 依赖包检查
./test.sh self-test deps

# 性能测试
./test.sh self-test performance
```

#### 1.3 测试覆盖范围

- **服务健康检查**: 验证所有微服务是否正常运行
- **API接口测试**: 测试各服务的核心API功能
- **数据文件验证**: 检查合成数据文件完整性
- **依赖包检查**: 验证Python依赖包是否正确安装
- **性能基准测试**: 检查API响应时间

### 2. 单元测试

#### 2.1 运行单元测试

```bash
# 使用测试脚本
./test.sh unit-test

# 直接使用pytest
python3 -m pytest tests/ -v

# 生成覆盖率报告
python3 -m pytest tests/ --cov=services --cov-report=html
```

#### 2.2 单元测试结构

```
tests/
├── test_consent_service.py      # 同意服务测试
├── test_psi_service.py          # PSI服务测试
├── test_model_trainer.py        # 模型训练服务测试
├── test_model_explainer.py      # 模型解释服务测试
├── test_utils.py                # 工具函数测试
└── conftest.py                  # 测试配置
```

#### 2.3 编写单元测试

```python
# 示例: test_consent_service.py
import pytest
from fastapi.testclient import TestClient
from services.consent_service.app import app

client = TestClient(app)

def test_create_consent():
    """测试创建同意记录"""
    consent_data = {
        "user_id": "test_user",
        "data_types": ["profile"],
        "purposes": ["risk_assessment"],
        "retention_period": 365
    }
    
    response = client.post("/consent", json=consent_data)
    assert response.status_code == 201
    assert "consent_id" in response.json()

def test_get_consent_status():
    """测试查询同意状态"""
    response = client.get("/consent/test_user/status")
    assert response.status_code == 200
    assert "status" in response.json()
```

### 3. 集成测试

#### 3.1 运行集成测试

```bash
# 确保所有服务已启动
./test.sh start-services

# 运行集成测试
./test.sh integration-test
```

#### 3.2 集成测试场景

- **端到端工作流**: 测试完整的联邦风控流程
- **服务间通信**: 验证微服务之间的交互
- **数据一致性**: 确保跨服务的数据一致性
- **错误处理**: 测试异常情况的处理

### 4. 性能测试

#### 4.1 运行性能测试

```bash
# 使用测试脚本
./test.sh performance-test

# 直接使用locust
cd logs
locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s --html performance_report.html --headless
```

#### 4.2 性能指标

- **响应时间**: API平均响应时间 < 2秒
- **吞吐量**: 每秒处理请求数 > 100
- **并发用户**: 支持至少50个并发用户
- **资源使用**: CPU使用率 < 80%, 内存使用 < 2GB

#### 4.3 性能测试场景

```python
# 示例: 性能测试脚本
from locust import HttpUser, task, between

class FederatedRiskUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def test_consent_api(self):
        """测试同意服务性能"""
        self.client.post("/consent", json={
            "user_id": f"user_{self.environment.runner.user_count}",
            "data_types": ["profile"],
            "purposes": ["risk_assessment"],
            "retention_period": 365
        })
    
    @task(2)
    def test_psi_api(self):
        """测试PSI服务性能"""
        self.client.post("/psi/session", json={
            "session_id": f"session_{self.environment.runner.user_count}",
            "method": "ecdh_psi",
            "party_role": "sender"
        })
```

### 5. 安全测试

#### 5.1 运行安全测试

```bash
# 使用测试脚本
./test.sh security-check

# 代码安全扫描
python3 -m bandit -r services/ -f json -o logs/security_report.json

# 依赖安全检查
python3 -m safety check --json --output logs/dependency_security.json
```

#### 5.2 安全测试内容

- **代码安全扫描**: 使用bandit检查代码安全问题
- **依赖安全检查**: 使用safety检查依赖包漏洞
- **输入验证测试**: 测试API输入验证
- **认证授权测试**: 验证访问控制机制
- **数据加密测试**: 确保敏感数据加密

#### 5.3 安全测试清单

- [ ] SQL注入防护
- [ ] XSS攻击防护
- [ ] CSRF攻击防护
- [ ] 输入验证和清理
- [ ] 认证和授权机制
- [ ] 数据传输加密
- [ ] 敏感数据存储加密
- [ ] 日志安全（不记录敏感信息）

### 6. 代码质量检查

#### 6.1 运行代码质量检查

```bash
# 使用测试脚本
./test.sh code-quality

# 单独运行各项检查
python3 -m black --check services/ scripts/
python3 -m isort services/ scripts/ --check-only
python3 -m flake8 services/ scripts/
python3 -m mypy services/ --ignore-missing-imports
```

#### 6.2 代码质量标准

- **代码格式**: 使用black进行代码格式化
- **导入排序**: 使用isort进行导入排序
- **代码风格**: 遵循PEP8规范，使用flake8检查
- **类型注解**: 使用mypy进行类型检查
- **文档字符串**: 所有公共函数和类都有docstring

#### 6.3 代码质量配置

```ini
# setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist

[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

## 手动测试

### 1. API测试

#### 1.1 使用Swagger UI

访问各服务的API文档进行交互式测试：

- 同意服务: http://localhost:8000/docs
- PSI服务: http://localhost:8001/docs
- 模型训练服务: http://localhost:8002/docs
- 模型解释服务: http://localhost:8003/docs

#### 1.2 使用curl命令

```bash
# 测试同意服务
curl -X POST "http://localhost:8000/consent" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "test_user",
       "data_types": ["profile"],
       "purposes": ["risk_assessment"],
       "retention_period": 365
     }'

# 测试PSI服务
curl -X POST "http://localhost:8001/psi/session" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "test_session",
       "method": "ecdh_psi",
       "party_role": "sender"
     }'
```

#### 1.3 使用Postman

导入API文档到Postman进行测试：

1. 访问 http://localhost:8000/openapi.json
2. 复制JSON内容
3. 在Postman中导入OpenAPI规范
4. 创建测试集合和环境变量

### 2. 前端测试

#### 2.1 功能测试

访问 http://localhost:5173 进行前端功能测试：

- [ ] 页面加载正常
- [ ] 导航功能正常
- [ ] 表单提交功能
- [ ] 数据展示功能
- [ ] 错误处理显示

#### 2.2 兼容性测试

在不同浏览器和设备上测试：

- [ ] Chrome (最新版本)
- [ ] Firefox (最新版本)
- [ ] Safari (最新版本)
- [ ] 移动设备浏览器
- [ ] 不同屏幕分辨率

### 3. 端到端测试

#### 3.1 完整工作流测试

1. **用户注册和同意管理**
   - 创建用户账户
   - 设置数据使用同意
   - 查看同意状态
   - 修改同意设置

2. **数据准备和PSI**
   - 上传数据文件
   - 配置PSI参数
   - 执行隐私集合求交
   - 验证结果正确性

3. **联邦学习训练**
   - 创建训练任务
   - 配置参与方
   - 监控训练进度
   - 评估模型性能

4. **模型解释和推理**
   - 生成模型解释
   - 执行模型推理
   - 查看解释结果
   - 导出报告

## 测试数据管理

### 1. 合成数据生成

```bash
# 生成测试数据
python3 scripts/generate_synthetic_data.py

# 验证数据质量
python3 scripts/validate_data.py
```

### 2. 测试数据清理

```bash
# 清理测试数据
python3 scripts/cleanup_test_data.py

# 重置数据库
python3 scripts/reset_database.py
```

### 3. 数据隐私保护

- 测试环境使用完全合成的数据
- 不包含任何真实个人信息
- 定期清理测试数据
- 遵循数据最小化原则

## 持续集成测试

### 1. GitHub Actions配置

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        ./test.sh all
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### 2. 测试报告

- 自动生成测试报告
- 覆盖率统计
- 性能基准对比
- 安全扫描结果

## 故障排除

### 1. 常见问题

#### 1.1 服务启动失败

```bash
# 检查端口占用
lsof -i :8000

# 查看服务日志
tail -f logs/consent-service.log

# 重启服务
./test.sh stop-services
./test.sh start-services
```

#### 1.2 测试失败

```bash
# 查看详细错误信息
./test.sh self-test -v

# 检查依赖包
pip3 list | grep -E "fastapi|uvicorn|pandas"

# 重新安装依赖
pip3 install -r requirements.txt
```

#### 1.3 性能问题

```bash
# 监控系统资源
top -p $(pgrep -f uvicorn)

# 检查数据库连接
python3 -c "import redis; r=redis.Redis(); print(r.ping())"

# 清理缓存
redis-cli FLUSHALL
```

### 2. 调试技巧

#### 2.1 日志分析

```bash
# 查看所有服务日志
tail -f logs/*.log

# 过滤错误日志
grep -i error logs/*.log

# 分析API调用
grep -E "POST|GET|PUT|DELETE" logs/*.log
```

#### 2.2 网络调试

```bash
# 测试服务连通性
curl -I http://localhost:8000/health

# 检查网络延迟
ping localhost

# 监控网络流量
netstat -an | grep :800
```

### 3. 性能优化

#### 3.1 数据库优化

- 添加适当的索引
- 优化查询语句
- 使用连接池
- 实现查询缓存

#### 3.2 应用优化

- 使用异步处理
- 实现响应缓存
- 优化算法复杂度
- 减少内存使用

#### 3.3 系统优化

- 调整系统参数
- 优化网络配置
- 使用负载均衡
- 实现水平扩展

## 测试最佳实践

### 1. 测试设计原则

- **独立性**: 测试之间相互独立
- **可重复性**: 测试结果可重复
- **快速反馈**: 测试执行快速
- **全面覆盖**: 覆盖主要功能和边界情况
- **易于维护**: 测试代码易于理解和维护

### 2. 测试数据管理

- 使用合成数据进行测试
- 保护真实数据隐私
- 定期清理测试数据
- 版本化测试数据集

### 3. 测试环境管理

- 隔离测试环境
- 自动化环境搭建
- 版本化环境配置
- 监控环境状态

### 4. 测试文档维护

- 及时更新测试文档
- 记录测试用例变更
- 分享测试经验
- 培训团队成员

## 参考资料

- [pytest文档](https://docs.pytest.org/)
- [FastAPI测试指南](https://fastapi.tiangolo.com/tutorial/testing/)
- [Locust性能测试](https://locust.io/)
- [Bandit安全扫描](https://bandit.readthedocs.io/)
- [联邦学习测试最佳实践](https://federated-learning.org/testing/)

---

**注意**: 本测试指南应根据项目发展持续更新和完善。如有问题或建议，请提交Issue或Pull Request。