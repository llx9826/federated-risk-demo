#!/bin/bash
# 联邦风控系统 - Kubernetes一键部署脚本
# 提供完整的K8s部署、更新、回滚和清理功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置参数
NAMESPACE=${1:-federated-risk}
ENVIRONMENT=${2:-production}
ACTION=${3:-deploy}
IMAGE_TAG=${4:-latest}

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_ROOT/k8s"
CONFIG_DIR="$PROJECT_ROOT/config"

# 显示帮助信息
show_help() {
    cat << EOF
联邦风控系统 Kubernetes 部署脚本

用法: $0 [NAMESPACE] [ENVIRONMENT] [ACTION] [IMAGE_TAG]

参数:
  NAMESPACE    Kubernetes命名空间 (默认: federated-risk)
  ENVIRONMENT  部署环境 (默认: production)
  ACTION       操作类型 (deploy|update|rollback|cleanup|status)
  IMAGE_TAG    镜像标签 (默认: latest)

操作类型:
  deploy       全新部署系统
  update       更新现有部署
  rollback     回滚到上一版本
  cleanup      清理所有资源
  status       查看部署状态
  logs         查看服务日志
  scale        扩缩容服务

示例:
  $0 federated-risk production deploy v1.0.0
  $0 federated-risk staging update latest
  $0 federated-risk production rollback
  $0 federated-risk production cleanup
  $0 federated-risk production status

EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖工具..."
    
    local missing_tools=()
    
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "缺少必要工具: ${missing_tools[*]}"
        log_info "请安装缺少的工具后重试"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 检查Kubernetes连接
check_k8s_connection() {
    log_info "检查Kubernetes连接..."
    
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "无法连接到Kubernetes集群"
        log_info "请检查kubectl配置和集群状态"
        exit 1
    fi
    
    log_success "Kubernetes连接正常"
}

# 创建命名空间
create_namespace() {
    log_info "创建命名空间: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_warning "命名空间 $NAMESPACE 已存在"
    else
        kubectl apply -f "$K8S_DIR/namespace.yaml"
        log_success "命名空间创建完成"
    fi
}

# 创建密钥
create_secrets() {
    log_info "创建系统密钥..."
    
    # 检查密钥是否已存在
    if kubectl get secret federated-risk-secrets -n "$NAMESPACE" >/dev/null 2>&1; then
        log_warning "密钥已存在，跳过创建"
        return
    fi
    
    # 生成随机密钥
    local postgres_password=$(openssl rand -base64 32)
    local redis_password=$(openssl rand -base64 32)
    local jwt_secret=$(openssl rand -base64 64)
    local encryption_key=$(openssl rand -base64 32)
    local api_key=$(openssl rand -base64 32)
    
    # 创建临时密钥文件
    local temp_secrets="/tmp/secrets-$NAMESPACE.yaml"
    
    cat > "$temp_secrets" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: federated-risk-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  postgres_password: $(echo -n "$postgres_password" | base64)
  redis_password: $(echo -n "$redis_password" | base64)
  jwt_secret: $(echo -n "$jwt_secret" | base64)
  encryption_key: $(echo -n "$encryption_key" | base64)
  api_key: $(echo -n "$api_key" | base64)
EOF
    
    kubectl apply -f "$temp_secrets"
    rm -f "$temp_secrets"
    
    log_success "密钥创建完成"
}

# 部署基础设施
deploy_infrastructure() {
    log_info "部署基础设施..."
    
    # 应用ConfigMap
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    
    # 应用Secrets
    kubectl apply -f "$K8S_DIR/secrets.yaml"
    
    # 应用PVC
    kubectl apply -f "$K8S_DIR/pvc.yaml"
    
    log_success "基础设施部署完成"
}

# 部署应用服务
deploy_services() {
    log_info "部署应用服务..."
    
    # 更新镜像标签
    if [ "$IMAGE_TAG" != "latest" ]; then
        log_info "更新镜像标签为: $IMAGE_TAG"
        sed -i.bak "s/:latest/:$IMAGE_TAG/g" "$K8S_DIR/deployment.yaml"
    fi
    
    # 应用Deployment
    kubectl apply -f "$K8S_DIR/deployment.yaml"
    
    # 应用Service
    kubectl apply -f "$K8S_DIR/service.yaml"
    
    # 应用HPA
    kubectl apply -f "$K8S_DIR/hpa.yaml"
    
    # 应用Ingress
    kubectl apply -f "$K8S_DIR/ingress.yaml"
    
    # 恢复原始文件
    if [ -f "$K8S_DIR/deployment.yaml.bak" ]; then
        mv "$K8S_DIR/deployment.yaml.bak" "$K8S_DIR/deployment.yaml"
    fi
    
    log_success "应用服务部署完成"
}

# 等待部署就绪
wait_for_deployment() {
    log_info "等待部署就绪..."
    
    local deployments=(
        "postgres"
        "redis"
        "api-gateway"
        "consent-service"
        "psi-service"
        "model-serving"
        "audit-service"
        "frontend"
    )
    
    for deployment in "${deployments[@]}"; do
        log_info "等待 $deployment 就绪..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n "$NAMESPACE"
    done
    
    log_success "所有部署已就绪"
}

# 执行健康检查
health_check() {
    log_info "执行健康检查..."
    
    local services=(
        "api-gateway:8000"
        "consent-service:7002"
        "psi-service:7001"
        "model-serving:7003"
        "audit-service:7005"
    )
    
    for service_port in "${services[@]}"; do
        local service=$(echo $service_port | cut -d: -f1)
        local port=$(echo $service_port | cut -d: -f2)
        
        log_info "检查 $service 健康状态..."
        
        # 端口转发进行健康检查
        kubectl port-forward service/$service $port:$port -n "$NAMESPACE" &
        local pf_pid=$!
        sleep 5
        
        if curl -f http://localhost:$port/health >/dev/null 2>&1; then
            log_success "$service 健康检查通过"
        else
            log_warning "$service 健康检查失败"
        fi
        
        kill $pf_pid 2>/dev/null || true
        sleep 2
    done
}

# 显示部署状态
show_status() {
    log_info "部署状态概览"
    echo
    
    echo "=== 命名空间 ==="
    kubectl get namespace "$NAMESPACE" 2>/dev/null || echo "命名空间不存在"
    echo
    
    echo "=== Pod状态 ==="
    kubectl get pods -n "$NAMESPACE" -o wide
    echo
    
    echo "=== 服务状态 ==="
    kubectl get services -n "$NAMESPACE"
    echo
    
    echo "=== Ingress状态 ==="
    kubectl get ingress -n "$NAMESPACE"
    echo
    
    echo "=== HPA状态 ==="
    kubectl get hpa -n "$NAMESPACE"
    echo
    
    echo "=== PVC状态 ==="
    kubectl get pvc -n "$NAMESPACE"
    echo
    
    echo "=== 最近事件 ==="
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# 查看日志
show_logs() {
    local service=${5:-api-gateway}
    local lines=${6:-100}
    
    log_info "查看 $service 服务日志 (最近 $lines 行)"
    
    if kubectl get deployment "$service" -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl logs -f deployment/"$service" -n "$NAMESPACE" --tail="$lines"
    else
        log_error "服务 $service 不存在"
        log_info "可用服务列表:"
        kubectl get deployments -n "$NAMESPACE" -o name | sed 's/deployment.apps\///'
    fi
}

# 扩缩容服务
scale_service() {
    local service=${5:-api-gateway}
    local replicas=${6:-3}
    
    log_info "扩缩容 $service 到 $replicas 个副本"
    
    if kubectl get deployment "$service" -n "$NAMESPACE" >/dev/null 2>&1; then
        kubectl scale deployment "$service" --replicas="$replicas" -n "$NAMESPACE"
        kubectl wait --for=condition=available --timeout=300s deployment/"$service" -n "$NAMESPACE"
        log_success "$service 扩缩容完成"
    else
        log_error "服务 $service 不存在"
    fi
}

# 更新部署
update_deployment() {
    log_info "更新现有部署..."
    
    # 更新ConfigMap和Secrets
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    kubectl apply -f "$K8S_DIR/secrets.yaml"
    
    # 滚动更新服务
    deploy_services
    
    # 等待更新完成
    wait_for_deployment
    
    log_success "部署更新完成"
}

# 回滚部署
rollback_deployment() {
    log_info "回滚部署..."
    
    local deployments=(
        "api-gateway"
        "consent-service"
        "psi-service"
        "model-serving"
        "audit-service"
        "frontend"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" >/dev/null 2>&1; then
            log_info "回滚 $deployment..."
            kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE"
        fi
    done
    
    log_success "回滚完成"
}

# 清理资源
cleanup_resources() {
    log_warning "即将清理所有资源，此操作不可逆！"
    read -p "确认清理? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "清理Kubernetes资源..."
        
        # 删除Ingress
        kubectl delete -f "$K8S_DIR/ingress.yaml" --ignore-not-found=true
        
        # 删除HPA
        kubectl delete -f "$K8S_DIR/hpa.yaml" --ignore-not-found=true
        
        # 删除Service
        kubectl delete -f "$K8S_DIR/service.yaml" --ignore-not-found=true
        
        # 删除Deployment
        kubectl delete -f "$K8S_DIR/deployment.yaml" --ignore-not-found=true
        
        # 删除PVC
        kubectl delete -f "$K8S_DIR/pvc.yaml" --ignore-not-found=true
        
        # 删除ConfigMap和Secrets
        kubectl delete -f "$K8S_DIR/configmap.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/secrets.yaml" --ignore-not-found=true
        
        # 删除命名空间
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        
        log_success "资源清理完成"
    else
        log_info "取消清理操作"
    fi
}

# 主函数
main() {
    echo "=== 联邦风控系统 Kubernetes 部署脚本 ==="
    echo "命名空间: $NAMESPACE"
    echo "环境: $ENVIRONMENT"
    echo "操作: $ACTION"
    echo "镜像标签: $IMAGE_TAG"
    echo
    
    case "$ACTION" in
        "deploy")
            check_dependencies
            check_k8s_connection
            create_namespace
            create_secrets
            deploy_infrastructure
            deploy_services
            wait_for_deployment
            health_check
            show_status
            
            log_success "部署完成！"
            echo
            echo "访问地址:"
            echo "  前端应用: https://app.federated-risk.local"
            echo "  API接口: https://api.federated-risk.local"
            echo "  管理后台: https://admin.federated-risk.local"
            echo
            echo "管理命令:"
            echo "  查看状态: $0 $NAMESPACE $ENVIRONMENT status"
            echo "  查看日志: $0 $NAMESPACE $ENVIRONMENT logs [service]"
            echo "  扩缩容: $0 $NAMESPACE $ENVIRONMENT scale [service] [replicas]"
            ;;
        "update")
            check_dependencies
            check_k8s_connection
            update_deployment
            health_check
            show_status
            ;;
        "rollback")
            check_dependencies
            check_k8s_connection
            rollback_deployment
            show_status
            ;;
        "cleanup")
            check_dependencies
            check_k8s_connection
            cleanup_resources
            ;;
        "status")
            check_k8s_connection
            show_status
            ;;
        "logs")
            check_k8s_connection
            show_logs "$@"
            ;;
        "scale")
            check_k8s_connection
            scale_service "$@"
            ;;
        "help")
            show_help
            ;;
        *)
            log_error "未知操作: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

main "$@"