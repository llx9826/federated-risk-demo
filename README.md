# 联邦风控演示系统

基于联邦学习的金融风险控制演示平台，展示隐私保护机器学习在金融风控场景中的应用。

## 🌟 项目特色

- **隐私保护**: 采用差分隐私和安全多方计算技术，确保数据隐私
- **联邦学习**: 支持多方协作训练，无需共享原始数据
- **现代化架构**: 基于微服务架构，支持容器化部署
- **完整工作流**: 从数据预处理到模型部署的完整流程
- **可视化界面**: 直观的Web界面，实时监控训练过程
- **安全审计**: 完整的操作审计和权限管理

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   API网关       │    │   后端服务      │
│   React + AntD  │◄──►│   Nginx         │◄──►│   FastAPI       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │   数据存储      │◄────────────┘
                       │   PostgreSQL    │
                       │   Redis         │
                       └─────────────────┘
```

### 核心组件

1. **PSI服务** - 隐私集合求交
2. **同意管理服务** - 数据使用授权
3. **训练服务** - 联邦学习训练
4. **推理服务** - 模型推理预测
5. **前端界面** - 用户交互界面

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Docker & Docker Compose
- 8GB+ 内存

### 一键启动

```bash
# 克隆项目
git clone <repository-url>
cd federated-risk-demo

# 启动所有服务
docker-compose up -d

# 等待服务启动完成（约2-3分钟）
docker-compose logs -f
```

### 访问地址

- **前端界面**: http://localhost:3000
- **API文档**: http://localhost:8000/docs
- **PSI服务**: http://localhost:8001
- **同意管理**: http://localhost:8002
- **训练服务**: http://localhost:8003
- **推理服务**: http://localhost:8004

### 默认账户

- **用户名**: admin
- **密码**: admin123

## 📁 项目结构

```
federated-risk-demo/
├── backend/                    # 后端服务
│   ├── psi_service/           # PSI隐私集合求交服务
│   ├── consent_service/       # 同意管理服务
│   ├── training_service/      # 训练服务
│   └── inference_service/     # 推理服务
├── frontend/                  # 前端界面
│   ├── src/
│   │   ├── components/        # 通用组件
│   │   ├── pages/            # 页面组件
│   │   ├── services/         # API服务
│   │   └── store/            # 状态管理
│   └── public/               # 静态资源
├── data/                     # 数据文件
│   ├── synth/               # 合成数据
│   └── models/              # 模型文件
├── docs/                    # 文档
├── scripts/                 # 脚本工具
└── docker-compose.yml       # 容器编排
```

## 🔧 开发指南

### 后端开发

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动PSI服务
cd psi_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 启动其他服务（新终端）
cd consent_service
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

### 前端开发

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 运行测试
npm test

# 构建生产版本
npm run build
```

#### 技术栈

- **框架**: React 18 + TypeScript
- **构建工具**: Vite
- **测试框架**: Vitest + React Testing Library
- **UI组件**: Ant Design
- **状态管理**: Zustand
- **路由**: React Router

### 数据准备

```bash
# 生成合成数据
cd data/synth
python generate_data.py

# 查看生成的数据
ls -la *.csv
```

## 📊 功能特性

### 1. 隐私集合求交 (PSI)

- **功能**: 在不泄露各方数据的情况下计算交集
- **算法**: 基于哈希的PSI协议
- **应用**: 客户匹配、风险名单比对

```python
# PSI使用示例
from psi_service.client import PSIClient

client = PSIClient("http://localhost:8001")
result = client.compute_intersection(
    party_a_data=["user1", "user2", "user3"],
    party_b_data=["user2", "user3", "user4"]
)
print(result)  # ["user2", "user3"]
```

### 2. 同意管理

- **功能**: 管理数据使用授权和隐私偏好
- **特性**: 细粒度权限控制、审计追踪
- **合规**: 支持GDPR、CCPA等隐私法规

### 3. 联邦学习训练

- **算法**: FedAvg、FedProx、SCAFFOLD
- **隐私保护**: 差分隐私、安全聚合
- **监控**: 实时训练指标、收敛分析

```python
# 训练任务配置示例
training_config = {
    "algorithm": "fedavg",
    "rounds": 10,
    "participants": ["bank_a", "bank_b"],
    "privacy_budget": 1.0,
    "noise_multiplier": 1.1
}
```

### 4. 模型推理

- **部署**: 支持多种模型格式
- **监控**: 推理性能、准确率追踪
- **安全**: 输入验证、输出脱敏

## 🔒 安全特性

### 隐私保护技术

1. **差分隐私**: 在模型训练中添加校准噪声
2. **安全多方计算**: PSI协议保护数据隐私
3. **同态加密**: 支持加密状态下的计算
4. **联邦学习**: 数据不出本地的协作学习

### 安全措施

- JWT身份认证
- RBAC权限控制
- API限流保护
- 操作审计日志
- 数据加密存储

## 📈 监控与运维

### 系统监控

- **健康检查**: 服务状态实时监控
- **性能指标**: CPU、内存、网络使用率
- **业务指标**: 训练进度、模型性能

### 日志管理

```bash
# 查看服务日志
docker-compose logs -f psi-service
docker-compose logs -f training-service

# 查看错误日志
docker-compose logs --tail=100 | grep ERROR
```

### 数据备份

```bash
# 备份数据库
docker-compose exec postgres pg_dump -U federated_user federated_db > backup.sql

# 恢复数据库
docker-compose exec -T postgres psql -U federated_user federated_db < backup.sql
```

## 🧪 测试

### 单元测试

```bash
# 后端测试
cd backend
pytest tests/ -v

# 前端测试
cd frontend
npm test
```

### 测试框架

- **后端**: pytest + FastAPI TestClient
- **前端**: Vitest + React Testing Library
- **E2E**: Playwright (可选)

### 集成测试

```bash
# 端到端测试
python scripts/e2e_test.py
```

### 性能测试

```bash
# 压力测试
cd scripts
python load_test.py --concurrent=10 --requests=1000
```

## 🔧 配置说明

### 环境变量

```bash
# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379

# 安全配置
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# 联邦学习配置
PRIVACY_BUDGET=1.0
NOISE_MULTIPLIER=1.1
MAX_GRAD_NORM=1.0
```

### 服务配置

每个服务的配置文件位于 `backend/{service}/config.yaml`：

```yaml
# training_service/config.yaml
server:
  host: 0.0.0.0
  port: 8003
  workers: 4

federated_learning:
  default_algorithm: fedavg
  max_rounds: 100
  min_participants: 2
  
privacy:
  enable_dp: true
  privacy_budget: 1.0
  noise_multiplier: 1.1
```

## 🚀 部署指南

### 生产环境部署

1. **准备环境**
```bash
# 安装Docker和Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置
vim .env
```

3. **启动服务**
```bash
# 生产环境启动
docker-compose -f docker-compose.prod.yml up -d

# 检查服务状态
docker-compose ps
```

### Kubernetes部署

```bash
# 应用Kubernetes配置
kubectl apply -f k8s/

# 检查部署状态
kubectl get pods -n federated-risk
```

## 🤝 贡献指南

### 开发流程

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范

- **Python**: 遵循PEP 8规范
- **TypeScript**: 使用ESLint和Prettier
- **提交信息**: 使用Conventional Commits格式

### 测试要求

- 新功能必须包含单元测试
- 测试覆盖率不低于80%
- 通过所有CI检查

## 📚 文档

- [API文档](docs/api.md)
- [架构设计](docs/architecture.md)
- [部署指南](docs/deployment.md)
- [开发指南](docs/development.md)
- [故障排除](docs/troubleshooting.md)

## 🔍 故障排除

### 常见问题

**Q: 服务启动失败**
```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 检查Docker状态
docker-compose ps
docker-compose logs service-name
```

**Q: 前端无法连接后端**
```bash
# 检查网络连接
curl http://localhost:8000/health

# 检查防火墙设置
sudo ufw status
```

**Q: 训练任务失败**
```bash
# 查看训练日志
docker-compose logs training-service

# 检查数据格式
python scripts/validate_data.py
```

### 性能优化

1. **数据库优化**
   - 添加适当索引
   - 调整连接池大小
   - 启用查询缓存

2. **服务优化**
   - 调整worker数量
   - 启用异步处理
   - 使用连接池

3. **前端优化**
   - 启用代码分割
   - 使用CDN加速
   - 优化图片资源

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 🙏 致谢

感谢以下开源项目的支持：

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Python Web框架
- [React](https://reactjs.org/) - 用户界面构建库
- [Ant Design](https://ant.design/) - 企业级UI设计语言
- [SecretFlow](https://www.secretflow.org.cn/) - 隐私保护计算框架
- [PostgreSQL](https://www.postgresql.org/) - 开源关系型数据库
- [Redis](https://redis.io/) - 内存数据结构存储

## 📞 联系我们

- **项目主页**: https://github.com/your-org/federated-risk-demo
- **问题反馈**: https://github.com/your-org/federated-risk-demo/issues
- **邮箱**: contact@your-org.com

---

**注意**: 本项目仅用于演示和学习目的，生产环境使用前请进行充分的安全评估和测试。