# 系统架构设计

## 总体架构

### 架构原则
- **数据不出域**：所有原始数据保留在各方本地
- **最小权限**：基于用途和时效的精细化授权
- **可观测性**：全链路监控和审计追踪
- **弹性扩展**：支持水平扩展和多方动态加入
- **安全第一**：端到端加密和隐私保护

### 系统全景图

```mermaid
architecture-beta
    group api(cloud)[API Gateway]
    group consent(cloud)[Consent Layer]
    group psi(cloud)[PSI Layer] 
    group fed(cloud)[Federation Layer]
    group serving(cloud)[Serving Layer]
    group audit(cloud)[Audit Layer]
    
    service gateway(internet)[API Gateway] in api
    service auth(server)[Auth Service] in consent
    service policy(server)[Policy Engine] in consent
    service consent_svc(server)[Consent Service] in consent
    
    service psi_svc(server)[PSI Service] in psi
    service token_join(server)[Token Join] in psi
    
    service orchestrator(server)[Fed Orchestrator] in fed
    service trainer(server)[Model Trainer] in fed
    service explainer(server)[SHAP Explainer] in fed
    
    service feature_store(database)[Feature Store] in serving
    service model_serving(server)[Model Serving] in serving
    service he_verifier(server)[HE Verifier] in serving
    
    service audit_ledger(database)[Audit Ledger] in audit
    service monitoring(server)[Monitoring] in audit
    
    gateway:R --> L:auth
    auth:R --> L:policy
    policy:R --> L:consent_svc
    consent_svc:R --> L:psi_svc
    psi_svc:R --> L:token_join
    psi_svc:R --> L:orchestrator
    orchestrator:R --> L:trainer
    trainer:R --> L:explainer
    trainer:R --> L:feature_store
    feature_store:R --> L:model_serving
    model_serving:R --> L:he_verifier
    model_serving:R --> L:audit_ledger
    audit_ledger:R --> L:monitoring
```

## 数据流设计

### 训练阶段数据流

```mermaid
sequenceDiagram
    participant A as 银行A
    participant B as 电商B
    participant C as Consent Gateway
    participant P as PSI Service
    participant O as Fed Orchestrator
    participant T as Model Trainer
    participant S as Model Serving
    participant L as Audit Ledger
    
    Note over A,L: 1. 同意授权阶段
    A->>C: 申请训练授权
    C->>C: 验证身份和策略
    C->>A: 返回consent_jwt
    
    Note over A,L: 2. 数据对齐阶段
    A->>P: 上传PSI tokens (带consent_jwt)
    B->>P: 上传PSI tokens (带consent_jwt)
    P->>P: ECDH-PSI计算交集
    P->>A: 返回交集映射
    P->>B: 返回交集映射
    
    Note over A,L: 3. 联合训练阶段
    A->>O: 注册训练任务
    B->>O: 注册训练任务
    O->>T: 启动SecureBoost训练
    
    loop 训练迭代
        T->>A: 请求梯度计算
        A->>T: 返回加密梯度
        T->>B: 请求梯度计算
        B->>T: 返回加密梯度
        T->>T: SecAgg聚合 + DP噪声
    end
    
    T->>S: 注册训练完成的模型
    T->>L: 记录训练审计日志
```

### 推理阶段数据流

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as 银行A
    participant C as Consent Gateway
    participant F as Feature Store
    participant S as Model Serving
    participant H as HE Verifier
    participant L as Audit Ledger
    
    Note over U,L: 1. 授权验证
    U->>A: 申请风控评分
    A->>C: 验证用户同意
    C->>A: 返回consent_jwt
    
    Note over U,L: 2. 特征获取
    A->>F: 获取用户特征 (带consent_jwt)
    F->>F: 验证授权范围
    F->>A: 返回授权特征
    
    Note over U,L: 3. 模型推理
    A->>S: 请求风控评分
    S->>S: 模型推理计算
    S->>H: HE验证 (可选)
    H->>S: 返回验证结果
    S->>A: 返回评分结果
    
    Note over U,L: 4. 审计记录
    S->>L: 记录推理审计
    L->>L: 生成审计回执
    A->>U: 返回评分 + 审计回执
```

## 威胁建模

### 威胁分析矩阵

| 威胁类型 | 威胁描述 | 影响等级 | 缓解措施 | 责任方 |
|----------|----------|----------|----------|--------|
| **数据泄露** | 训练/推理过程中原始数据泄露 | 高 | SecAgg+DP+同态加密 | 技术团队 |
| **模型窃取** | 通过推理API逆向工程模型 | 中 | 访问频率限制+差分隐私 | 安全团队 |
| **身份伪造** | 恶意方伪造身份参与联邦 | 高 | mTLS+数字证书+身份验证 | 安全团队 |
| **同意绕过** | 绕过用户同意直接访问数据 | 高 | 强制同意验证+审计日志 | 合规团队 |
| **拒绝服务** | 恶意请求导致服务不可用 | 中 | 限流+负载均衡+监控 | 运维团队 |
| **数据投毒** | 恶意方注入有害训练数据 | 中 | 数据质量检查+异常检测 | 算法团队 |
| **梯度泄露** | 通过梯度信息推断原始数据 | 中 | SecAgg+差分隐私 | 技术团队 |
| **重放攻击** | 重复使用历史请求进行攻击 | 低 | 时间戳+nonce+签名 | 安全团队 |

### 安全边界

```mermaid
flowchart TB
    subgraph "外部边界"
        subgraph "DMZ区域"
            LB[负载均衡器]
            WAF[Web应用防火墙]
        end
        
        subgraph "应用区域"
            API[API网关]
            AUTH[认证服务]
            CONSENT[同意服务]
        end
        
        subgraph "计算区域"
            PSI[PSI服务]
            FED[联邦训练]
            SERVE[模型服务]
        end
        
        subgraph "数据区域"
            FS[特征存储]
            AUDIT[审计存储]
            MODEL[模型存储]
        end
    end
    
    Internet --> LB
    LB --> WAF
    WAF --> API
    API --> AUTH
    AUTH --> CONSENT
    CONSENT --> PSI
    PSI --> FED
    FED --> SERVE
    SERVE --> FS
    SERVE --> AUDIT
    FED --> MODEL
    
    classDef dmz fill:#ffcccc
    classDef app fill:#ccffcc  
    classDef compute fill:#ccccff
    classDef data fill:#ffffcc
    
    class LB,WAF dmz
    class API,AUTH,CONSENT app
    class PSI,FED,SERVE compute
    class FS,AUDIT,MODEL data
```

## 微服务架构

### 服务依赖图

```mermaid
graph TD
    subgraph "前端层"
        UI[React前端]
    end
    
    subgraph "网关层"
        GW[API网关]
        LB[负载均衡]
    end
    
    subgraph "业务层"
        CONSENT[同意服务]
        PSI[PSI服务]
        ORCH[联邦编排]
        TRAIN[模型训练]
        SERVE[模型服务]
        FEAT[特征存储]
    end
    
    subgraph "基础层"
        AUDIT[审计服务]
        MONITOR[监控服务]
        POLICY[策略引擎]
        REGISTRY[服务注册]
    end
    
    subgraph "存储层"
        DB[(数据库)]
        CACHE[(缓存)]
        FILES[(文件存储)]
    end
    
    UI --> GW
    GW --> LB
    LB --> CONSENT
    LB --> PSI
    LB --> SERVE
    
    CONSENT --> POLICY
    CONSENT --> AUDIT
    PSI --> ORCH
    ORCH --> TRAIN
    TRAIN --> SERVE
    SERVE --> FEAT
    SERVE --> AUDIT
    
    POLICY --> DB
    AUDIT --> DB
    FEAT --> CACHE
    TRAIN --> FILES
    SERVE --> FILES
    
    MONITOR --> REGISTRY
    REGISTRY --> DB
```

### 服务清单

| 服务名称 | 端口 | 职责 | 技术栈 | 扩展性 |
|----------|------|------|--------|--------|
| **consent-gateway** | 7002 | 同意管理、策略验证 | FastAPI+Casbin+JWT | 水平扩展 |
| **psi-service** | 7001 | 隐私集合交集计算 | FastAPI+ECDH | 水平扩展 |
| **federated-orchestrator** | 7003 | 联邦训练编排 | FastAPI+Celery | 水平扩展 |
| **model-trainer** | 7004 | 模型训练执行 | FATE/Flower+GPU | 垂直扩展 |
| **model-serving** | 7005 | 在线推理服务 | FastAPI+ONNX | 水平扩展 |
| **feature-store** | 7006 | 特征存储管理 | Feast+Redis | 水平扩展 |
| **monitoring** | 7007 | 监控指标收集 | Prometheus+Grafana | 水平扩展 |
| **audit-ledger** | 7008 | 审计日志存储 | FastAPI+PostgreSQL | 水平扩展 |
| **policy** | 7009 | 策略决策引擎 | OPA+Rego | 水平扩展 |

## 数据架构

### 数据分类

```mermaid
flowchart LR
    subgraph "原始数据 (不出域)"
        A1[银行客户数据]
        A2[电商行为数据]
        A3[运营商通信数据]
    end
    
    subgraph "对齐数据 (PSI Token)"
        B1[哈希化标识符]
        B2[交集映射表]
        B3[对齐统计信息]
    end
    
    subgraph "训练数据 (加密梯度)"
        C1[本地特征向量]
        C2[加密梯度信息]
        C3[聚合模型参数]
    end
    
    subgraph "推理数据 (授权特征)"
        D1[用户授权特征]
        D2[模型预测结果]
        D3[审计回执信息]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    
    C3 --> D2
    B2 --> D1
    D1 --> D2
    D2 --> D3
```

### 存储策略

| 数据类型 | 存储方式 | 加密方式 | 保留期限 | 访问控制 |
|----------|----------|----------|----------|----------|
| **用户同意** | PostgreSQL | AES-256-GCM | 7年 | 基于角色 |
| **PSI Token** | Redis | 内存加密 | 24小时 | 基于会话 |
| **模型参数** | 文件存储 | AES-256-GCM | 永久 | 基于版本 |
| **审计日志** | PostgreSQL | AES-256-GCM | 10年 | 只读访问 |
| **特征数据** | Redis | 内存加密 | 1小时 | 基于授权 |
| **监控指标** | InfluxDB | TLS传输 | 1年 | 基于角色 |

## 部署架构

### Kubernetes部署拓扑

```mermaid
flowchart TB
    subgraph "K8s集群"
        subgraph "kube-system"
            DNS[CoreDNS]
            PROXY[kube-proxy]
        end
        
        subgraph "ingress-nginx"
            INGRESS[Nginx Ingress]
        end
        
        subgraph "federated-risk namespace"
            subgraph "前端层"
                UI_POD[UI Pod]
            end
            
            subgraph "网关层"
                GW_POD[Gateway Pod]
            end
            
            subgraph "业务层"
                CONSENT_POD[Consent Pod]
                PSI_POD[PSI Pod]
                TRAIN_POD[Training Pod]
                SERVE_POD[Serving Pod]
            end
            
            subgraph "存储层"
                DB_POD[PostgreSQL Pod]
                REDIS_POD[Redis Pod]
            end
        end
        
        subgraph "monitoring namespace"
            PROM[Prometheus]
            GRAFANA[Grafana]
        end
    end
    
    Internet --> INGRESS
    INGRESS --> UI_POD
    INGRESS --> GW_POD
    GW_POD --> CONSENT_POD
    GW_POD --> PSI_POD
    GW_POD --> SERVE_POD
    PSI_POD --> TRAIN_POD
    TRAIN_POD --> SERVE_POD
    
    CONSENT_POD --> DB_POD
    PSI_POD --> REDIS_POD
    SERVE_POD --> REDIS_POD
    
    PROM --> CONSENT_POD
    PROM --> PSI_POD
    PROM --> TRAIN_POD
    PROM --> SERVE_POD
    GRAFANA --> PROM
```

### 网络策略

```yaml
# 示例网络策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: federated-risk-network-policy
  namespace: federated-risk
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  - from:
    - podSelector:
        matchLabels:
          app: consent-service
    ports:
    - protocol: TCP
      port: 7002
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## 扩展性设计

### 水平扩展策略

1. **无状态服务**：所有API服务设计为无状态，支持任意副本数扩展
2. **数据分片**：PSI计算支持数据分片并行处理
3. **负载均衡**：基于一致性哈希的智能路由
4. **缓存策略**：多级缓存减少数据库压力

### 垂直扩展策略

1. **GPU加速**：模型训练支持GPU集群
2. **内存优化**：大数据集的流式处理
3. **存储优化**：冷热数据分离存储
4. **计算优化**：算法并行化和向量化

### 多方扩展

```mermaid
flowchart LR
    subgraph "初始状态 (2方)"
        A1[银行A]
        B1[电商B]
    end
    
    subgraph "扩展状态 (N方)"
        A2[银行A]
        B2[电商B]
        C2[运营商C]
        D2[互金D]
        E2[...更多方]
    end
    
    A1 -.-> A2
    B1 -.-> B2
    
    A2 <--> PSI2[PSI-N方]
    B2 <--> PSI2
    C2 <--> PSI2
    D2 <--> PSI2
    E2 <--> PSI2
    
    PSI2 --> FED2[联邦训练-N方]
```

## 性能指标

### 关键性能指标 (KPI)

| 指标类型 | 指标名称 | 目标值 | 监控方式 |
|----------|----------|--------|----------|
| **吞吐量** | PSI对齐TPS | >1000 req/s | Prometheus |
| **延迟** | 推理响应时间 | <100ms P95 | APM |
| **可用性** | 服务可用率 | >99.9% | 健康检查 |
| **准确性** | 模型AUC | >0.75 | 模型监控 |
| **安全性** | 隐私预算消耗 | <ε=5 | 差分隐私监控 |
| **合规性** | 审计覆盖率 | 100% | 审计日志 |

### 容量规划

| 资源类型 | 当前配置 | 峰值配置 | 扩展策略 |
|----------|----------|----------|----------|
| **CPU** | 16核 | 64核 | HPA自动扩展 |
| **内存** | 32GB | 128GB | 垂直扩展 |
| **存储** | 1TB SSD | 10TB SSD | 存储扩容 |
| **网络** | 1Gbps | 10Gbps | 带宽升级 |
| **GPU** | 2张V100 | 8张A100 | GPU集群 |

## 技术债务管理

### 已知技术债务

1. **PSI算法优化**：当前使用简化版ECDH-PSI，需要升级到更高效的算法
2. **模型格式统一**：支持更多模型格式 (ONNX, TensorFlow, PyTorch)
3. **监控完善**：增加业务指标监控和告警
4. **文档完善**：API文档和运维文档需要持续更新

### 技术演进路线

```mermaid
timeline
    title 技术演进路线图
    
    Q1 2024 : 基础功能
           : PSI基础实现
           : 联邦训练MVP
           : 基础监控
    
    Q2 2024 : 性能优化
           : PSI算法优化
           : GPU加速训练
           : 缓存优化
    
    Q3 2024 : 安全增强
           : 同态加密集成
           : 零知识证明
           : 安全多方计算
    
    Q4 2024 : 生产就绪
           : 高可用部署
           : 灾备方案
           : 性能调优
```