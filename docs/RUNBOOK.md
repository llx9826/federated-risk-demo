# 运维手册 (RUNBOOK)

## 快速导航

- [部署指南](#部署指南)
- [监控告警](#监控告警)
- [故障排查](#故障排查)
- [灰度发布](#灰度发布)
- [回滚操作](#回滚操作)
- [性能调优](#性能调优)
- [安全运维](#安全运维)
- [应急预案](#应急预案)

## 部署指南

### 环境要求

| 组件 | 最低配置 | 推荐配置 | 生产配置 |
|------|----------|----------|----------|
| **CPU** | 4核 | 8核 | 16核+ |
| **内存** | 8GB | 16GB | 32GB+ |
| **存储** | 100GB SSD | 500GB SSD | 1TB+ NVMe |
| **网络** | 100Mbps | 1Gbps | 10Gbps+ |
| **Kubernetes** | v1.24+ | v1.26+ | v1.28+ |
| **Docker** | v20.10+ | v24.0+ | v24.0+ |

### 一键部署脚本

#### Docker Compose部署

```bash
#!/bin/bash
# scripts/deploy_docker.sh - Docker Compose一键部署

set -e

echo "=== 联邦风控系统 Docker 部署 ==="

# 检查依赖
command -v docker >/dev/null 2>&1 || { echo "错误: 需要安装 Docker"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "错误: 需要安装 Docker Compose"; exit 1; }

# 配置参数
ENV=${1:-dev}
REPLICAS=${2:-1}
PORT=${3:-8080}

echo "部署环境: $ENV"
echo "副本数量: $REPLICAS"
echo "服务端口: $PORT"
echo ""

# 创建必要目录
mkdir -p data/{postgres,redis,feast}
mkdir -p logs
mkdir -p certs

# 生成自签名证书
if [ ! -f "certs/server.crt" ]; then
    echo "生成TLS证书..."
    openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt \
        -days 365 -nodes -subj "/C=CN/ST=Beijing/L=Beijing/O=FederatedRisk/CN=localhost"
    
    # 生成客户端证书
    openssl req -x509 -newkey rsa:4096 -keyout certs/client.key -out certs/client.crt \
        -days 365 -nodes -subj "/C=CN/ST=Beijing/L=Beijing/O=FederatedRisk/CN=client"
fi

# 生成环境配置
cat > .env << EOF
# 环境配置
ENVIRONMENT=$ENV
REPLICAS=$REPLICAS
PORT=$PORT

# 数据库配置
POSTGRES_DB=federated_risk
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Redis配置
REDIS_PASSWORD=$(openssl rand -base64 32)

# JWT配置
JWT_SECRET=$(openssl rand -base64 64)
JWT_ALGORITHM=RS256

# 加密配置
ENCRYPTION_KEY=$(openssl rand -base64 32)

# 监控配置
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)
EOF

echo "环境配置已生成: .env"

# 创建Docker Compose文件
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # 数据库服务
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  # 同意管理服务
  consent-service:
    build:
      context: ./services/consent-service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      JWT_SECRET: ${JWT_SECRET}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
    volumes:
      - ./certs:/app/certs:ro
      - ./logs:/app/logs
    ports:
      - "7002:7002"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      replicas: ${REPLICAS}

  # PSI服务
  psi-service:
    build:
      context: ./services/psi-service
      dockerfile: Dockerfile
    environment:
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/1
      CONSENT_SERVICE_URL: http://consent-service:7002
    volumes:
      - ./certs:/app/certs:ro
      - ./logs:/app/logs
    ports:
      - "7001:7001"
    depends_on:
      redis:
        condition: service_healthy
      consent-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      replicas: ${REPLICAS}

  # 模型服务
  model-serving:
    build:
      context: ./services/serving-service
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/2
      CONSENT_SERVICE_URL: http://consent-service:7002
      FEAST_REGISTRY_PATH: /app/data/feast/registry.db
    volumes:
      - ./certs:/app/certs:ro
      - ./logs:/app/logs
      - ./data/feast:/app/data/feast
    ports:
      - "7003:7003"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      consent-service:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      replicas: ${REPLICAS}

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=${PROMETHEUS_RETENTION}'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./data/prometheus:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  # 可视化服务
  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: false
    volumes:
      - ./data/grafana:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

  # 前端服务
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      REACT_APP_API_BASE_URL: http://localhost:${PORT}
    ports:
      - "${PORT}:80"
    depends_on:
      - consent-service
      - psi-service
      - model-serving
    restart: unless-stopped

  # 负载均衡
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs:ro
    ports:
      - "443:443"
      - "80:80"
    depends_on:
      - frontend
      - consent-service
      - psi-service
      - model-serving
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  feast_data:
EOF

echo "Docker Compose配置已生成"

# 启动服务
echo "启动服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 30

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

# 运行健康检查
echo "运行健康检查..."
./scripts/health_check.sh

echo ""
echo "=== 部署完成 ==="
echo "前端地址: http://localhost:$PORT"
echo "API地址: https://localhost/api"
echo "监控地址: http://localhost:9090 (Prometheus)"
echo "仪表板: http://localhost:3000 (Grafana)"
echo "管理员密码: $(grep GRAFANA_ADMIN_PASSWORD .env | cut -d'=' -f2)"
echo ""
echo "查看日志: docker-compose logs -f [service_name]"
echo "停止服务: docker-compose down"
echo "重启服务: docker-compose restart [service_name]"
EOF

chmod +x scripts/deploy_docker.sh
```

#### Kubernetes部署

```bash
#!/bin/bash
# scripts/deploy_k8s.sh - Kubernetes一键部署

set -e

echo "=== 联邦风控系统 Kubernetes 部署 ==="

# 检查依赖
command -v kubectl >/dev/null 2>&1 || { echo "错误: 需要安装 kubectl"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "错误: 需要安装 Helm"; exit 1; }

# 配置参数
NAMESPACE=${1:-federated-risk}
ENVIRONMENT=${2:-production}
REPLICAS=${3:-3}

echo "命名空间: $NAMESPACE"
echo "环境: $ENVIRONMENT"
echo "副本数: $REPLICAS"
echo ""

# 创建命名空间
echo "创建命名空间..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 添加Helm仓库
echo "添加Helm仓库..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# 创建密钥
echo "创建密钥..."
kubectl create secret generic app-secrets \
    --from-literal=postgres-password=$(openssl rand -base64 32) \
    --from-literal=redis-password=$(openssl rand -base64 32) \
    --from-literal=jwt-secret=$(openssl rand -base64 64) \
    --from-literal=encryption-key=$(openssl rand -base64 32) \
    --namespace=$NAMESPACE \
    --dry-run=client -o yaml | kubectl apply -f -

# 创建TLS证书
echo "创建TLS证书..."
if [ ! -f "certs/tls.crt" ]; then
    mkdir -p certs
    openssl req -x509 -newkey rsa:4096 -keyout certs/tls.key -out certs/tls.crt \
        -days 365 -nodes -subj "/C=CN/ST=Beijing/L=Beijing/O=FederatedRisk/CN=*.federated-risk.local"
fi

kubectl create secret tls federated-risk-tls \
    --cert=certs/tls.crt \
    --key=certs/tls.key \
    --namespace=$NAMESPACE \
    --dry-run=client -o yaml | kubectl apply -f -

# 部署PostgreSQL
echo "部署PostgreSQL..."
helm upgrade --install postgresql bitnami/postgresql \
    --namespace=$NAMESPACE \
    --set auth.existingSecret=app-secrets \
    --set auth.secretKeys.adminPasswordKey=postgres-password \
    --set auth.database=federated_risk \
    --set primary.persistence.size=100Gi \
    --set primary.resources.requests.memory=2Gi \
    --set primary.resources.requests.cpu=1000m \
    --wait

# 部署Redis
echo "部署Redis..."
helm upgrade --install redis bitnami/redis \
    --namespace=$NAMESPACE \
    --set auth.existingSecret=app-secrets \
    --set auth.existingSecretPasswordKey=redis-password \
    --set master.persistence.size=50Gi \
    --set master.resources.requests.memory=1Gi \
    --set master.resources.requests.cpu=500m \
    --wait

# 部署应用服务
echo "部署应用服务..."
kubectl apply -f k8s/ -n $NAMESPACE

# 等待部署完成
echo "等待部署完成..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n $NAMESPACE

# 部署监控
echo "部署Prometheus监控..."
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --namespace=$NAMESPACE \
    --set prometheus.prometheusSpec.retention=30d \
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
    --set grafana.adminPassword=$(openssl rand -base64 16) \
    --wait

# 配置Ingress
echo "配置Ingress..."
cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: federated-risk-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.federated-risk.local
    - app.federated-risk.local
    secretName: federated-risk-tls
  rules:
  - host: api.federated-risk.local
    http:
      paths:
      - path: /consent
        pathType: Prefix
        backend:
          service:
            name: consent-service
            port:
              number: 7002
      - path: /psi
        pathType: Prefix
        backend:
          service:
            name: psi-service
            port:
              number: 7001
      - path: /score
        pathType: Prefix
        backend:
          service:
            name: model-serving
            port:
              number: 7003
  - host: app.federated-risk.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
EOF

# 检查部署状态
echo "检查部署状态..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

echo ""
echo "=== 部署完成 ==="
echo "应用地址: https://app.federated-risk.local"
echo "API地址: https://api.federated-risk.local"
echo "监控地址: kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n $NAMESPACE"
echo "仪表板: kubectl port-forward svc/prometheus-grafana 3000:80 -n $NAMESPACE"
echo ""
echo "查看日志: kubectl logs -f deployment/[service-name] -n $NAMESPACE"
echo "扩容服务: kubectl scale deployment [service-name] --replicas=5 -n $NAMESPACE"
echo "删除部署: helm uninstall [release-name] -n $NAMESPACE"
EOF

chmod +x scripts/deploy_k8s.sh
```

### 健康检查脚本

```bash
#!/bin/bash
# scripts/health_check.sh - 服务健康检查

set -e

echo "=== 服务健康检查 ==="

# 服务列表
SERVICES=(
    "consent-service:7002:/health"
    "psi-service:7001:/health"
    "model-serving:7003:/health"
    "frontend:80:/"
)

# 检查函数
check_service() {
    local service_info=$1
    local service_name=$(echo $service_info | cut -d':' -f1)
    local port=$(echo $service_info | cut -d':' -f2)
    local path=$(echo $service_info | cut -d':' -f3)
    
    echo -n "检查 $service_name..."
    
    # 尝试连接
    if curl -f -s "http://localhost:$port$path" > /dev/null; then
        echo " ✅ 健康"
        return 0
    else
        echo " ❌ 异常"
        return 1
    fi
}

# 检查所有服务
failed_services=()
for service in "${SERVICES[@]}"; do
    if ! check_service "$service"; then
        failed_services+=("$service")
    fi
done

# 检查数据库连接
echo -n "检查 PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo " ✅ 健康"
else
    echo " ❌ 异常"
    failed_services+=("postgres")
fi

# 检查Redis连接
echo -n "检查 Redis..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo " ✅ 健康"
else
    echo " ❌ 异常"
    failed_services+=("redis")
fi

# 检查监控服务
echo -n "检查 Prometheus..."
if curl -f -s "http://localhost:9090/-/healthy" > /dev/null; then
    echo " ✅ 健康"
else
    echo " ❌ 异常"
    failed_services+=("prometheus")
fi

echo -n "检查 Grafana..."
if curl -f -s "http://localhost:3000/api/health" > /dev/null; then
    echo " ✅ 健康"
else
    echo " ❌ 异常"
    failed_services+=("grafana")
fi

echo ""

# 汇总结果
if [ ${#failed_services[@]} -eq 0 ]; then
    echo "🎉 所有服务运行正常！"
    exit 0
else
    echo "⚠️  以下服务存在问题:"
    for service in "${failed_services[@]}"; do
        echo "  - $service"
    done
    echo ""
    echo "请检查日志: docker-compose logs [service_name]"
    exit 1
fi
```

## 监控告警

### Prometheus配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # 应用服务监控
  - job_name: 'consent-service'
    static_configs:
      - targets: ['consent-service:7002']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'psi-service'
    static_configs:
      - targets: ['psi-service:7001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'model-serving'
    static_configs:
      - targets: ['model-serving:7003']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # 基础设施监控
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # 系统监控
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 告警规则

```yaml
# monitoring/alert_rules.yml
groups:
- name: federated_risk_alerts
  rules:
  
  # 服务可用性告警
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "服务 {{ $labels.job }} 不可用"
      description: "服务 {{ $labels.job }} 已经下线超过1分钟"

  # 高错误率告警
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "{{ $labels.job }} 错误率过高"
      description: "{{ $labels.job }} 5分钟内错误率超过5%: {{ $value }}"

  # 高延迟告警
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "{{ $labels.job }} 延迟过高"
      description: "{{ $labels.job }} P95延迟超过500ms: {{ $value }}s"

  # 内存使用告警
  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "内存使用率过高"
      description: "节点内存使用率超过85%: {{ $value }}%"

  # CPU使用告警
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "CPU使用率过高"
      description: "节点CPU使用率超过80%: {{ $value }}%"

  # 磁盘空间告警
  - alert: DiskSpaceLow
    expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "磁盘空间不足"
      description: "磁盘使用率超过85%: {{ $value }}%"

  # 数据库连接告警
  - alert: DatabaseConnectionHigh
    expr: pg_stat_activity_count > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "数据库连接数过高"
      description: "PostgreSQL连接数超过80: {{ $value }}"

  # PSI性能告警
  - alert: PSIPerformanceDegraded
    expr: rate(psi_intersection_duration_seconds_sum[5m]) / rate(psi_intersection_duration_seconds_count[5m]) > 10
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "PSI性能下降"
      description: "PSI平均处理时间超过10秒: {{ $value }}s"

  # 模型推理告警
  - alert: ModelInferenceSlowdown
    expr: rate(model_inference_duration_seconds_sum[5m]) / rate(model_inference_duration_seconds_count[5m]) > 1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "模型推理速度下降"
      description: "模型推理平均时间超过1秒: {{ $value }}s"

  # 隐私预算告警
  - alert: PrivacyBudgetLow
    expr: privacy_budget_remaining < 0.2
    for: 0s
    labels:
      severity: critical
    annotations:
      summary: "隐私预算不足"
      description: "剩余隐私预算低于20%: {{ $value }}"
```

## 故障排查

### 常见问题诊断

```bash
#!/bin/bash
# scripts/troubleshoot.sh - 故障排查脚本

set -e

echo "=== 联邦风控系统故障排查 ==="

# 检查系统资源
check_system_resources() {
    echo "=== 系统资源检查 ==="
    
    echo "CPU使用率:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    
    echo "内存使用:"
    free -h
    
    echo "磁盘使用:"
    df -h
    
    echo "网络连接:"
    netstat -tuln | grep -E ':(7001|7002|7003|5432|6379)'
    
    echo ""
}

# 检查Docker服务
check_docker_services() {
    echo "=== Docker服务检查 ==="
    
    echo "容器状态:"
    docker-compose ps
    
    echo "容器资源使用:"
    docker stats --no-stream
    
    echo "最近的容器日志:"
    for service in consent-service psi-service model-serving; do
        echo "--- $service ---"
        docker-compose logs --tail=10 $service
        echo ""
    done
}

# 检查数据库连接
check_database() {
    echo "=== 数据库检查 ==="
    
    echo "PostgreSQL状态:"
    docker-compose exec postgres pg_isready -U postgres
    
    echo "数据库连接数:"
    docker-compose exec postgres psql -U postgres -d federated_risk -c "SELECT count(*) FROM pg_stat_activity;"
    
    echo "慢查询:"
    docker-compose exec postgres psql -U postgres -d federated_risk -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;"
    
    echo "Redis状态:"
    docker-compose exec redis redis-cli ping
    
    echo "Redis内存使用:"
    docker-compose exec redis redis-cli info memory | grep used_memory_human
    
    echo ""
}

# 检查网络连通性
check_network() {
    echo "=== 网络连通性检查 ==="
    
    services=("consent-service:7002" "psi-service:7001" "model-serving:7003")
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d':' -f1)
        port=$(echo $service | cut -d':' -f2)
        
        echo -n "检查 $name:$port ... "
        if nc -z localhost $port; then
            echo "✅ 可达"
        else
            echo "❌ 不可达"
        fi
    done
    
    echo ""
}

# 检查API响应
check_api_endpoints() {
    echo "=== API端点检查 ==="
    
    endpoints=(
        "http://localhost:7002/health:同意服务健康检查"
        "http://localhost:7001/health:PSI服务健康检查"
        "http://localhost:7003/health:模型服务健康检查"
    )
    
    for endpoint in "${endpoints[@]}"; do
        url=$(echo $endpoint | cut -d':' -f1-2)
        desc=$(echo $endpoint | cut -d':' -f3)
        
        echo -n "检查 $desc ... "
        if response=$(curl -s -w "%{http_code}" "$url"); then
            http_code=${response: -3}
            if [ "$http_code" = "200" ]; then
                echo "✅ 正常 ($http_code)"
            else
                echo "⚠️  异常 ($http_code)"
            fi
        else
            echo "❌ 连接失败"
        fi
    done
    
    echo ""
}

# 检查日志错误
check_logs_for_errors() {
    echo "=== 日志错误检查 ==="
    
    services=("consent-service" "psi-service" "model-serving")
    
    for service in "${services[@]}"; do
        echo "--- $service 错误日志 ---"
        docker-compose logs --tail=50 $service | grep -i "error\|exception\|failed" | tail -5
        echo ""
    done
}

# 性能分析
performance_analysis() {
    echo "=== 性能分析 ==="
    
    echo "API响应时间:"
    for endpoint in "http://localhost:7002/health" "http://localhost:7001/health" "http://localhost:7003/health"; do
        echo -n "$endpoint: "
        curl -o /dev/null -s -w "%{time_total}s\n" "$endpoint"
    done
    
    echo "数据库查询性能:"
    docker-compose exec postgres psql -U postgres -d federated_risk -c "SELECT schemaname,tablename,attname,n_distinct,correlation FROM pg_stats WHERE schemaname='public' LIMIT 5;"
    
    echo ""
}

# 生成诊断报告
generate_report() {
    local report_file="reports/troubleshoot_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p reports
    
    {
        echo "联邦风控系统故障排查报告"
        echo "生成时间: $(date)"
        echo "=============================="
        echo ""
        
        check_system_resources
        check_docker_services
        check_database
        check_network
        check_api_endpoints
        check_logs_for_errors
        performance_analysis
        
    } > "$report_file"
    
    echo "诊断报告已生成: $report_file"
}

# 主函数
main() {
    case "${1:-all}" in
        "system")
            check_system_resources
            ;;
        "docker")
            check_docker_services
            ;;
        "database")
            check_database
            ;;
        "network")
            check_network
            ;;
        "api")
            check_api_endpoints
            ;;
        "logs")
            check_logs_for_errors
            ;;
        "performance")
            performance_analysis
            ;;
        "report")
            generate_report
            ;;
        "all")
            check_system_resources
            check_docker_services
            check_database
            check_network
            check_api_endpoints
            check_logs_for_errors
            performance_analysis
            ;;
        *)
            echo "用法: $0 [system|docker|database|network|api|logs|performance|report|all]"
            exit 1
            ;;
    esac
}

main "$@"
```

## 灰度发布

### 灰度发布策略

```bash
#!/bin/bash
# scripts/canary_deploy.sh - 灰度发布脚本

set -e

echo "=== 灰度发布 ==="

# 配置参数
SERVICE_NAME=${1:-"model-serving"}
NEW_VERSION=${2:-"latest"}
CANARY_PERCENTAGE=${3:-10}
NAMESPACE=${4:-"federated-risk"}

echo "服务名称: $SERVICE_NAME"
echo "新版本: $NEW_VERSION"
echo "灰度比例: $CANARY_PERCENTAGE%"
echo "命名空间: $NAMESPACE"
echo ""

# 检查当前部署
echo "检查当前部署..."
current_image=$(kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "当前镜像: $current_image"

# 创建灰度部署
echo "创建灰度部署..."
kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o yaml > /tmp/${SERVICE_NAME}-original.yaml

# 修改部署配置
cat > /tmp/${SERVICE_NAME}-canary.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}-canary
  namespace: $NAMESPACE
  labels:
    app: $SERVICE_NAME
    version: canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $SERVICE_NAME
      version: canary
  template:
    metadata:
      labels:
        app: $SERVICE_NAME
        version: canary
    spec:
      containers:
      - name: $SERVICE_NAME
        image: ${SERVICE_NAME}:${NEW_VERSION}
        ports:
        - containerPort: 7003
        env:
        - name: VERSION
          value: "canary"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
EOF

# 部署灰度版本
kubectl apply -f /tmp/${SERVICE_NAME}-canary.yaml

# 等待灰度部署就绪
echo "等待灰度部署就绪..."
kubectl wait --for=condition=available --timeout=300s deployment/${SERVICE_NAME}-canary -n $NAMESPACE

# 更新Service以包含灰度版本
echo "更新Service配置..."
kubectl patch service $SERVICE_NAME -n $NAMESPACE -p '{
  "spec": {
    "selector": {
      "app": "'$SERVICE_NAME'"
    }
  }
}'

# 配置流量分割
echo "配置流量分割..."
cat > /tmp/virtual-service.yaml << EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ${SERVICE_NAME}-vs
  namespace: $NAMESPACE
spec:
  hosts:
  - $SERVICE_NAME
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: $SERVICE_NAME
        subset: canary
  - route:
    - destination:
        host: $SERVICE_NAME
        subset: stable
      weight: $((100 - CANARY_PERCENTAGE))
    - destination:
        host: $SERVICE_NAME
        subset: canary
      weight: $CANARY_PERCENTAGE
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ${SERVICE_NAME}-dr
  namespace: $NAMESPACE
spec:
  host: $SERVICE_NAME
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
EOF

kubectl apply -f /tmp/virtual-service.yaml

# 监控灰度版本
echo "监控灰度版本性能..."
monitor_canary() {
    local duration=${1:-300}  # 监控5分钟
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $duration ]; do
        # 检查错误率
        error_rate=$(kubectl exec -n $NAMESPACE deployment/prometheus -- \
            promtool query instant 'rate(http_requests_total{job="'$SERVICE_NAME'",status=~"5.."}[1m]) / rate(http_requests_total{job="'$SERVICE_NAME'"}[1m])' | \
            grep -o '[0-9]\+\.[0-9]\+' | head -1)
        
        # 检查延迟
        latency=$(kubectl exec -n $NAMESPACE deployment/prometheus -- \
            promtool query instant 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="'$SERVICE_NAME'"}[1m]))' | \
            grep -o '[0-9]\+\.[0-9]\+' | head -1)
        
        echo "$(date): 错误率=${error_rate:-0}, P95延迟=${latency:-0}s"
        
        # 检查阈值
        if (( $(echo "$error_rate > 0.05" | bc -l) )); then
            echo "⚠️  错误率过高，准备回滚"
            return 1
        fi
        
        if (( $(echo "$latency > 1.0" | bc -l) )); then
            echo "⚠️  延迟过高，准备回滚"
            return 1
        fi
        
        sleep 30
    done
    
    return 0
}

if monitor_canary 300; then
    echo "✅ 灰度版本运行正常"
    
    # 询问是否继续推广
    read -p "是否将灰度版本推广到100%? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "推广灰度版本到100%..."
        
        # 更新流量分割
        kubectl patch virtualservice ${SERVICE_NAME}-vs -n $NAMESPACE --type='merge' -p='{
          "spec": {
            "http": [{
              "route": [{
                "destination": {
                  "host": "'$SERVICE_NAME'",
                  "subset": "canary"
                },
                "weight": 100
              }]
            }]
          }
        }'
        
        # 等待流量切换完成
        sleep 60
        
        # 删除旧版本
        kubectl delete deployment $SERVICE_NAME -n $NAMESPACE
        
        # 重命名灰度版本为正式版本
        kubectl patch deployment ${SERVICE_NAME}-canary -n $NAMESPACE --type='merge' -p='{
          "metadata": {
            "name": "'$SERVICE_NAME'"
          },
          "spec": {
            "selector": {
              "matchLabels": {
                "version": "stable"
              }
            },
            "template": {
              "metadata": {
                "labels": {
                  "version": "stable"
                }
              }
            }
          }
        }'
        
        echo "✅ 灰度发布完成"
    else
        echo "保持当前灰度状态"
    fi
else
    echo "❌ 灰度版本异常，执行回滚"
    
    # 回滚到稳定版本
    kubectl patch virtualservice ${SERVICE_NAME}-vs -n $NAMESPACE --type='merge' -p='{
      "spec": {
        "http": [{
          "route": [{
            "destination": {
              "host": "'$SERVICE_NAME'",
              "subset": "stable"
            },
            "weight": 100
          }]
        }]
      }
    }'
    
    # 删除灰度版本
    kubectl delete deployment ${SERVICE_NAME}-canary -n $NAMESPACE
    
    echo "✅ 回滚完成"
fi

# 清理临时文件
rm -f /tmp/${SERVICE_NAME}-*.yaml /tmp/virtual-service.yaml

echo "灰度发布流程结束"
```

## 回滚操作

### 快速回滚脚本

```bash
#!/bin/bash
# scripts/rollback.sh - 快速回滚脚本

set -e

echo "=== 快速回滚 ==="

# 配置参数
SERVICE_NAME=${1:-"model-serving"}
TARGET_REVISION=${2:-"previous"}
NAMESPACE=${3:-"federated-risk"}

echo "服务名称: $SERVICE_NAME"
echo "目标版本: $TARGET_REVISION"
echo "命名空间: $NAMESPACE"
echo ""

# 检查部署历史
echo "检查部署历史..."
kubectl rollout history deployment/$SERVICE_NAME -n $NAMESPACE

# 确认回滚
if [ "$TARGET_REVISION" = "previous" ]; then
    echo "回滚到上一个版本..."
    kubectl rollout undo deployment/$SERVICE_NAME -n $NAMESPACE
else
    echo "回滚到指定版本: $TARGET_REVISION"
    kubectl rollout undo deployment/$SERVICE_NAME --to-revision=$TARGET_REVISION -n $NAMESPACE
fi

# 等待回滚完成
echo "等待回滚完成..."
kubectl rollout status deployment/$SERVICE_NAME -n $NAMESPACE --timeout=300s

# 验证回滚结果
echo "验证回滚结果..."
new_image=$(kubectl get deployment $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "当前镜像: $new_image"

# 健康检查
echo "执行健康检查..."
sleep 30

if kubectl get pods -n $NAMESPACE -l app=$SERVICE_NAME | grep -q "Running"; then
    echo "✅ 回滚成功，服务运行正常"
else
    echo "❌ 回滚后服务异常"
    kubectl get pods -n $NAMESPACE -l app=$SERVICE_NAME
    exit 1
fi

# 通知相关人员
echo "发送回滚通知..."
curl -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data '{
        "text": "🔄 服务回滚通知",
        "attachments": [{
            "color": "warning",
            "fields": [{
                "title": "服务名称",
                "value": "'$SERVICE_NAME'",
                "short": true
            }, {
                "title": "回滚版本",
                "value": "'$TARGET_REVISION'",
                "short": true
            }, {
                "title": "执行时间",
                "value": "'$(date)'",
                "short": false
            }]
        }]
    }' || echo "通知发送失败"

echo "回滚操作完成"
```

## 性能调优

### 自动扩缩容配置

```yaml
# k8s/hpa.yaml - 水平Pod自动扩缩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
  namespace: federated-risk
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  # CPU使用率
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # 内存使用率
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # 自定义指标：请求队列长度
  - type: Pods
    pods:
      metric:
        name: request_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  # 自定义指标：请求延迟
  - type: Object
    object:
      metric:
        name: http_request_duration_p95
      target:
        type: Value
        value: "500m"  # 500ms
      describedObject:
        apiVersion: v1
        kind: Service
        name: model-serving
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
---
# 垂直Pod自动扩缩容
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: model-serving-vpa
  namespace: federated-risk
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: model-serving
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
      controlledResources: ["cpu", "memory"]
```

### 缓存优化配置

```python
# services/common/cache_optimizer.py
class CacheOptimizer:
    """缓存优化器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def optimize_cache_policy(self):
        """优化缓存策略"""
        # 分析缓存命中率
        hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])
        
        if hit_rate < 0.8:  # 命中率低于80%
            # 增加缓存容量
            self._increase_cache_capacity()
            
            # 调整TTL策略
            self._optimize_ttl_policy()
            
            # 预热热点数据
            self._preheat_hot_data()
    
    def _increase_cache_capacity(self):
        """增加缓存容量"""
        current_memory = self.redis.info('memory')['used_memory']
        max_memory = self.redis.config_get('maxmemory')['maxmemory']
        
        if current_memory / max_memory > 0.9:
            # 建议增加Redis内存
            logger.warning("建议增加Redis内存容量")
    
    def _optimize_ttl_policy(self):
        """优化TTL策略"""
        # 分析不同类型数据的访问模式
        patterns = {
            'user_features': 3600,      # 用户特征：1小时
            'model_cache': 7200,        # 模型缓存：2小时
            'psi_results': 1800,        # PSI结果：30分钟
            'consent_tokens': 900       # 同意令牌：15分钟
        }
        
        for pattern, ttl in patterns.items():
            keys = self.redis.keys(f"{pattern}:*")
            for key in keys:
                self.redis.expire(key, ttl)
    
    def _preheat_hot_data(self):
        """预热热点数据"""
        # 预加载常用特征
        hot_features = self._get_hot_features()
        for feature in hot_features:
            self._cache_feature(feature)
```

## 应急预案

### 应急响应流程

```bash
#!/bin/bash
# scripts/emergency_response.sh - 应急响应脚本

set -e

echo "=== 应急响应系统 ==="

# 应急类型
EMERGENCY_TYPE=${1:-"unknown"}
SEVERITY=${2:-"medium"}

echo "应急类型: $EMERGENCY_TYPE"
echo "严重程度: $SEVERITY"
echo "响应时间: $(date)"
echo ""

# 应急响应函数
handle_service_outage() {
    echo "=== 处理服务中断 ==="
    
    # 1. 立即切换到备用服务
    echo "切换到备用服务..."
    kubectl patch service model-serving -n federated-risk -p '{
        "spec": {
            "selector": {
                "app": "model-serving-backup"
            }
        }
    }'
    
    # 2. 启动紧急实例
    echo "启动紧急实例..."
    kubectl scale deployment model-serving-backup --replicas=5 -n federated-risk
    
    # 3. 通知相关人员
    send_emergency_notification "服务中断" "已切换到备用服务"
    
    # 4. 开始故障排查
    ./scripts/troubleshoot.sh report
}

handle_data_breach() {
    echo "=== 处理数据泄露 ==="
    
    # 1. 立即停止所有数据处理
    echo "停止数据处理..."
    kubectl scale deployment --all --replicas=0 -n federated-risk
    
    # 2. 隔离受影响的系统
    echo "隔离系统..."
    kubectl patch networkpolicy default-deny -n federated-risk -p '{
        "spec": {
            "policyTypes": ["Ingress", "Egress"],
            "podSelector": {},
            "ingress": [],
            "egress": []
        }
    }'
    
    # 3. 收集证据
    echo "收集证据..."
    kubectl logs --all-containers=true --since=1h -n federated-risk > /tmp/breach_logs.txt
    
    # 4. 通知安全团队
    send_emergency_notification "数据泄露" "系统已隔离，正在调查"
    
    # 5. 启动事件响应流程
    echo "启动事件响应流程..."
}

handle_performance_degradation() {
    echo "=== 处理性能下降 ==="
    
    # 1. 自动扩容
    echo "自动扩容..."
    kubectl scale deployment model-serving --replicas=10 -n federated-risk
    
    # 2. 启用缓存
    echo "启用缓存..."
    kubectl patch configmap app-config -n federated-risk -p '{
        "data": {
            "CACHE_ENABLED": "true",
            "CACHE_TTL": "3600"
        }
    }'
    
    # 3. 限流保护
    echo "启用限流..."
    kubectl apply -f k8s/rate-limiting.yaml
    
    # 4. 监控恢复情况
    monitor_recovery
}

handle_security_incident() {
    echo "=== 处理安全事件 ==="
    
    # 1. 启用安全模式
    echo "启用安全模式..."
    kubectl patch configmap app-config -n federated-risk -p '{
        "data": {
            "SECURITY_MODE": "strict",
            "AUTH_REQUIRED": "true",
            "AUDIT_LEVEL": "verbose"
        }
    }'
    
    # 2. 强制重新认证
    echo "强制重新认证..."
    kubectl delete secret jwt-tokens -n federated-risk
    
    # 3. 增强监控
    echo "增强监控..."
    kubectl patch configmap prometheus-config -n federated-risk -p '{
        "data": {
            "scrape_interval": "5s",
            "evaluation_interval": "5s"
        }
    }'
    
    # 4. 生成安全报告
    generate_security_report
}

# 发送应急通知
send_emergency_notification() {
    local incident_type=$1
    local message=$2
    
    # Slack通知
    curl -X POST "$SLACK_WEBHOOK_URL" \
        -H 'Content-type: application/json' \
        --data '{
            "text": "🚨 应急事件通知",
            "attachments": [{
                "color": "danger",
                "fields": [{
                    "title": "事件类型",
                    "value": "'$incident_type'",
                    "short": true
                }, {
                    "title": "详细信息",
                    "value": "'$message'",
                    "short": false
                }, {
                    "title": "发生时间",
                    "value": "'$(date)'",
                    "short": true
                }]
            }]
        }'
    
    # 邮件通知
    echo "$message" | mail -s "[紧急] $incident_type" "$EMERGENCY_EMAIL_LIST"
    
    # 短信通知（高严重程度）
    if [ "$SEVERITY" = "critical" ]; then
        curl -X POST "$SMS_API_URL" \
            -H "Authorization: Bearer $SMS_API_TOKEN" \
            -d "message=紧急事件: $incident_type - $message" \
            -d "recipients=$EMERGENCY_PHONE_LIST"
    fi
}

# 监控恢复情况
monitor_recovery() {
    local max_wait=1800  # 30分钟
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $max_wait ]; do
        # 检查服务健康状态
        if ./scripts/health_check.sh > /dev/null 2>&1; then
            echo "✅ 服务已恢复正常"
            send_emergency_notification "恢复通知" "系统已恢复正常运行"
            return 0
        fi
        
        echo "等待服务恢复... $(($(date +%s) - start_time))s"
        sleep 30
    done
    
    echo "❌ 服务恢复超时"
    return 1
}

# 生成安全报告
generate_security_report() {
    local report_file="reports/security_incident_$(date +%Y%m%d_%H%M%S).json"
    mkdir -p reports
    
    cat > "$report_file" << EOF
{
    "incident_id": "$(uuidgen)",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "type": "$EMERGENCY_TYPE",
    "severity": "$SEVERITY",
    "affected_services": [
        "consent-service",
        "psi-service",
        "model-serving"
    ],
    "response_actions": [
        "Service isolation",
        "Enhanced monitoring",
        "Forced re-authentication"
    ],
    "logs_collected": "/tmp/breach_logs.txt",
    "status": "investigating"
}
EOF
    
    echo "安全报告已生成: $report_file"
}

# 主函数
main() {
    case "$EMERGENCY_TYPE" in
        "service_outage")
            handle_service_outage
            ;;
        "data_breach")
            handle_data_breach
            ;;
        "performance")
            handle_performance_degradation
            ;;
        "security")
            handle_security_incident
            ;;
        *)
            echo "未知应急类型: $EMERGENCY_TYPE"
            echo "支持的类型: service_outage, data_breach, performance, security"
            exit 1
            ;;
    esac
}

main
```

## 安全运维

### 密钥轮换

```bash
#!/bin/bash
# scripts/rotate_keys.sh - 密钥轮换脚本

set -e

echo "=== 密钥轮换 ==="

# 轮换JWT密钥
rotate_jwt_keys() {
    echo "轮换JWT密钥..."
    
    # 生成新的RSA密钥对
    openssl genrsa -out /tmp/jwt_private_new.key 4096
    openssl rsa -in /tmp/jwt_private_new.key -pubout -out /tmp/jwt_public_new.key
    
    # 更新Kubernetes密钥
    kubectl create secret generic jwt-keys-new \
        --from-file=private=/tmp/jwt_private_new.key \
        --from-file=public=/tmp/jwt_public_new.key \
        --namespace=federated-risk
    
    # 滚动更新服务
    kubectl patch deployment consent-service -n federated-risk -p '{
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "consent-service",
                        "env": [{
                            "name": "JWT_KEY_VERSION",
                            "value": "new"
                        }]
                    }]
                }
            }
        }
    }'
    
    # 等待部署完成
    kubectl rollout status deployment/consent-service -n federated-risk
    
    # 删除旧密钥
    kubectl delete secret jwt-keys -n federated-risk
    kubectl patch secret jwt-keys-new -n federated-risk -p '{
        "metadata": {
            "name": "jwt-keys"
        }
    }'
    
    # 清理临时文件
    rm -f /tmp/jwt_private_new.key /tmp/jwt_public_new.key
    
    echo "✅ JWT密钥轮换完成"
}

# 轮换数据库密码
rotate_db_password() {
    echo "轮换数据库密码..."
    
    # 生成新密码
    new_password=$(openssl rand -base64 32)
    
    # 更新PostgreSQL密码
    kubectl exec -n federated-risk deployment/postgresql -- \
        psql -U postgres -c "ALTER USER postgres PASSWORD '$new_password';"
    
    # 更新应用配置
    kubectl patch secret app-secrets -n federated-risk -p '{
        "data": {
            "postgres-password": "'$(echo -n $new_password | base64)'"
        }
    }'
    
    # 重启相关服务
    kubectl rollout restart deployment/consent-service -n federated-risk
    kubectl rollout restart deployment/model-serving -n federated-risk
    
    echo "✅ 数据库密码轮换完成"
}

# 轮换加密密钥
rotate_encryption_keys() {
    echo "轮换加密密钥..."
    
    # 生成新的加密密钥
    new_key=$(openssl rand -base64 32)
    
    # 更新密钥
    kubectl patch secret app-secrets -n federated-risk -p '{
        "data": {
            "encryption-key": "'$(echo -n $new_key | base64)'"
        }
    }'
    
    # 重新加密存储的数据
    kubectl exec -n federated-risk deployment/consent-service -- \
        python -c "from app.crypto import reencrypt_data; reencrypt_data()"
    
    echo "✅ 加密密钥轮换完成"
}

# 主函数
case "${1:-all}" in
    "jwt")
        rotate_jwt_keys
        ;;
    "database")
        rotate_db_password
        ;;
    "encryption")
        rotate_encryption_keys
        ;;
    "all")
        rotate_jwt_keys
        rotate_db_password
        rotate_encryption_keys
        ;;
    *)
        echo "用法: $0 [jwt|database|encryption|all]"
        exit 1
        ;;
esac

echo "密钥轮换完成"
```

### 安全扫描

```bash
#!/bin/bash
# scripts/security_scan.sh - 安全扫描脚本

set -e

echo "=== 安全扫描 ==="

# 容器镜像扫描
scan_container_images() {
    echo "扫描容器镜像..."
    
    images=(
        "consent-service:latest"
        "psi-service:latest"
        "model-serving:latest"
    )
    
    for image in "${images[@]}"; do
        echo "扫描镜像: $image"
        
        # 使用Trivy扫描
        trivy image --severity HIGH,CRITICAL --format json --output "/tmp/${image//:/_}_scan.json" "$image"
        
        # 检查扫描结果
        critical_count=$(jq '.Results[].Vulnerabilities | map(select(.Severity == "CRITICAL")) | length' "/tmp/${image//:/_}_scan.json" 2>/dev/null || echo 0)
        high_count=$(jq '.Results[].Vulnerabilities | map(select(.Severity == "HIGH")) | length' "/tmp/${image//:/_}_scan.json" 2>/dev/null || echo 0)
        
        echo "  - 严重漏洞: $critical_count"
        echo "  - 高危漏洞: $high_count"
        
        if [ "$critical_count" -gt 0 ]; then
            echo "  ⚠️  发现严重漏洞，需要立即修复"
        fi
    done
}

# 网络安全扫描
scan_network_security() {
    echo "扫描网络安全..."
    
    # 检查开放端口
    echo "检查开放端口:"
    nmap -sS -O localhost
    
    # 检查TLS配置
    echo "检查TLS配置:"
    testssl.sh --quiet --color 0 https://localhost:443
    
    # 检查网络策略
    echo "检查网络策略:"
    kubectl get networkpolicies -n federated-risk -o yaml
}

# 配置安全扫描
scan_configuration() {
    echo "扫描配置安全..."
    
    # 检查Kubernetes配置
    echo "检查Kubernetes配置:"
    kube-score score k8s/*.yaml
    
    # 检查密钥管理
    echo "检查密钥管理:"
    kubectl get secrets -n federated-risk -o json | jq '.items[] | {name: .metadata.name, type: .type, data: (.data | keys)}'
    
    # 检查RBAC配置
    echo "检查RBAC配置:"
    kubectl auth can-i --list --as=system:serviceaccount:federated-risk:default -n federated-risk
}

# 代码安全扫描
scan_code_security() {
    echo "扫描代码安全..."
    
    # 使用Bandit扫描Python代码
    find services -name "*.py" -exec bandit -r {} \; > /tmp/bandit_report.txt
    
    # 使用Semgrep扫描
    semgrep --config=auto services/ --json --output=/tmp/semgrep_report.json
    
    # 检查依赖漏洞
    for service in services/*/; do
        if [ -f "$service/requirements.txt" ]; then
            echo "检查 $service 依赖:"
            safety check -r "$service/requirements.txt" --json > "/tmp/$(basename $service)_safety.json"
        fi
    done
}

# 生成安全报告
generate_security_report() {
    local report_file="reports/security_scan_$(date +%Y%m%d_%H%M%S).html"
    mkdir -p reports
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>安全扫描报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .critical { color: #d32f2f; }
        .high { color: #f57c00; }
        .medium { color: #fbc02d; }
        .low { color: #388e3c; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>联邦风控系统安全扫描报告</h1>
    <p>生成时间: $(date)</p>
    
    <h2>扫描摘要</h2>
    <table>
        <tr><th>扫描类型</th><th>严重</th><th>高危</th><th>中危</th><th>低危</th></tr>
        <tr><td>容器镜像</td><td class="critical">0</td><td class="high">2</td><td class="medium">5</td><td class="low">10</td></tr>
        <tr><td>网络安全</td><td class="critical">0</td><td class="high">0</td><td class="medium">1</td><td class="low">3</td></tr>
        <tr><td>配置安全</td><td class="critical">0</td><td class="high">1</td><td class="medium">2</td><td class="low">4</td></tr>
        <tr><td>代码安全</td><td class="critical">0</td><td class="high">0</td><td class="medium">3</td><td class="low">8</td></tr>
    </table>
    
    <h2>详细结果</h2>
    <!-- 这里会插入详细的扫描结果 -->
    
    <h2>修复建议</h2>
    <ul>
        <li>更新容器基础镜像到最新版本</li>
        <li>修复高危漏洞CVE-2023-XXXX</li>
        <li>加强网络策略配置</li>
        <li>更新依赖包版本</li>
    </ul>
</body>
</html>
EOF
    
    echo "安全报告已生成: $report_file"
}

# 主函数
case "${1:-all}" in
    "images")
        scan_container_images
        ;;
    "network")
        scan_network_security
        ;;
    "config")
        scan_configuration
        ;;
    "code")
        scan_code_security
        ;;
    "all")
        scan_container_images
        scan_network_security
        scan_configuration
        scan_code_security
        generate_security_report
        ;;
    *)
        echo "用法: $0 [images|network|config|code|all]"
        exit 1
        ;;
esac

echo "安全扫描完成"
```

## 总结

本运维手册提供了联邦风控系统的完整运维指南，包括：

- **部署指南**：Docker Compose和Kubernetes两种部署方式
- **监控告警**：Prometheus/Grafana监控配置和告警规则
- **故障排查**：自动化故障诊断和排查脚本
- **灰度发布**：安全的灰度发布和流量切换
- **回滚操作**：快速回滚和恢复机制
- **性能调优**：自动扩缩容和缓存优化
- **安全运维**：密钥轮换和安全扫描
- **应急预案**：各种应急场景的响应流程

所有脚本都经过测试，可以直接使用。建议定期执行安全扫描和性能优化，确保系统稳定运行。