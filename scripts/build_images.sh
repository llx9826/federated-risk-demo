#!/bin/bash
# 联邦风控系统 - Docker镜像构建脚本
# 为所有服务构建和推送Docker镜像

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
REGISTRY=${1:-localhost:5000}
TAG=${2:-latest}
SERVICE=${3:-all}
ACTION=${4:-build}

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICES_DIR="$PROJECT_ROOT/services"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# 服务列表
SERVICES=(
    "api-gateway"
    "consent-service"
    "psi-service"
    "train-service"
    "serving-service"
    "audit-service"
    "consent-gateway"
    "federated-orchestrator"
    "model-trainer"
    "model-explainer"
    "model-deployment"
    "model-serving"
    "feature-store"
)

# 显示帮助信息
show_help() {
    cat << EOF
联邦风控系统 Docker 镜像构建脚本

用法: $0 [REGISTRY] [TAG] [SERVICE] [ACTION]

参数:
  REGISTRY     Docker镜像仓库地址 (默认: localhost:5000)
  TAG          镜像标签 (默认: latest)
  SERVICE      服务名称 (默认: all)
  ACTION       操作类型 (build|push|build-push|clean)

服务列表:
  all                    构建所有服务
  frontend               前端应用
  api-gateway            API网关
  consent-service        同意服务
  psi-service           PSI服务
  train-service         训练服务
  serving-service       推理服务
  audit-service         审计服务
  consent-gateway       同意网关
  federated-orchestrator 联邦编排器
  model-trainer         模型训练器
  model-explainer       模型解释器
  model-deployment      模型部署
  model-serving         模型服务
  feature-store         特征存储

操作类型:
  build        仅构建镜像
  push         仅推送镜像
  build-push   构建并推送镜像
  clean        清理本地镜像
  list         列出所有镜像

示例:
  $0 localhost:5000 v1.0.0 all build-push
  $0 registry.example.com latest api-gateway build
  $0 localhost:5000 latest frontend push
  $0 localhost:5000 latest all clean

EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖工具..."
    
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker未安装或不在PATH中"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker守护进程未运行"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 创建基础Dockerfile
create_base_dockerfile() {
    local service_dir="$1"
    local service_name="$2"
    
    if [ ! -f "$service_dir/Dockerfile" ]; then
        log_info "为 $service_name 创建Dockerfile"
        
        cat > "$service_dir/Dockerfile" << EOF
# $service_name Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "app.py"]
EOF
    fi
}

# 创建前端Dockerfile
create_frontend_dockerfile() {
    if [ ! -f "$FRONTEND_DIR/Dockerfile" ]; then
        log_info "为前端创建Dockerfile"
        
        cat > "$FRONTEND_DIR/Dockerfile" << EOF
# 前端应用 Dockerfile
# 多阶段构建
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 生产阶段
FROM nginx:alpine

# 复制构建结果
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 创建nginx配置文件
RUN echo 'server {' > /etc/nginx/conf.d/default.conf && \\
    echo '    listen 80;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    server_name localhost;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    root /usr/share/nginx/html;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    index index.html;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    location / {' >> /etc/nginx/conf.d/default.conf && \\
    echo '        try_files \$uri \$uri/ /index.html;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    }' >> /etc/nginx/conf.d/default.conf && \\
    echo '    location /api/ {' >> /etc/nginx/conf.d/default.conf && \\
    echo '        proxy_pass http://api-gateway:8000/;' >> /etc/nginx/conf.d/default.conf && \\
    echo '        proxy_set_header Host \$host;' >> /etc/nginx/conf.d/default.conf && \\
    echo '        proxy_set_header X-Real-IP \$remote_addr;' >> /etc/nginx/conf.d/default.conf && \\
    echo '    }' >> /etc/nginx/conf.d/default.conf && \\
    echo '}' >> /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost/ || exit 1

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
EOF
    fi
}

# 构建单个服务
build_service() {
    local service="$1"
    local image_name="$REGISTRY/federated-risk/$service:$TAG"
    
    log_info "构建服务: $service"
    
    if [ "$service" = "frontend" ]; then
        local build_dir="$FRONTEND_DIR"
        create_frontend_dockerfile
    else
        local build_dir="$SERVICES_DIR/$service"
        if [ ! -d "$build_dir" ]; then
            log_error "服务目录不存在: $build_dir"
            return 1
        fi
        create_base_dockerfile "$build_dir" "$service"
    fi
    
    # 构建镜像
    log_info "构建镜像: $image_name"
    
    if docker build -t "$image_name" "$build_dir"; then
        log_success "$service 构建成功"
        
        # 添加latest标签
        if [ "$TAG" != "latest" ]; then
            docker tag "$image_name" "$REGISTRY/federated-risk/$service:latest"
        fi
        
        return 0
    else
        log_error "$service 构建失败"
        return 1
    fi
}

# 推送单个服务
push_service() {
    local service="$1"
    local image_name="$REGISTRY/federated-risk/$service:$TAG"
    
    log_info "推送镜像: $image_name"
    
    if docker push "$image_name"; then
        log_success "$service 推送成功"
        
        # 推送latest标签
        if [ "$TAG" != "latest" ]; then
            docker push "$REGISTRY/federated-risk/$service:latest"
        fi
        
        return 0
    else
        log_error "$service 推送失败"
        return 1
    fi
}

# 清理镜像
clean_images() {
    local service="$1"
    
    if [ "$service" = "all" ]; then
        log_info "清理所有镜像..."
        
        # 清理所有federated-risk镜像
        docker images "$REGISTRY/federated-risk/*" -q | xargs -r docker rmi -f
        
        # 清理前端镜像
        docker images "$REGISTRY/federated-risk/frontend" -q | xargs -r docker rmi -f
        
        # 清理悬空镜像
        docker image prune -f
        
        log_success "镜像清理完成"
    else
        log_info "清理 $service 镜像..."
        
        docker images "$REGISTRY/federated-risk/$service" -q | xargs -r docker rmi -f
        
        log_success "$service 镜像清理完成"
    fi
}

# 列出镜像
list_images() {
    log_info "联邦风控系统镜像列表"
    echo
    
    echo "=== 服务镜像 ==="
    docker images "$REGISTRY/federated-risk/*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo
    
    echo "=== 镜像统计 ==="
    local total_images=$(docker images "$REGISTRY/federated-risk/*" -q | wc -l)
    local total_size=$(docker images "$REGISTRY/federated-risk/*" --format "{{.Size}}" | sed 's/[^0-9.]//g' | awk '{sum += $1} END {print sum "MB"}')
    
    echo "总镜像数: $total_images"
    echo "总大小: $total_size"
}

# 构建所有服务
build_all_services() {
    log_info "构建所有服务..."
    
    local failed_services=()
    local success_count=0
    
    # 构建前端
    if build_service "frontend"; then
        ((success_count++))
    else
        failed_services+=("frontend")
    fi
    
    # 构建后端服务
    for service in "${SERVICES[@]}"; do
        if build_service "$service"; then
            ((success_count++))
        else
            failed_services+=("$service")
        fi
    done
    
    echo
    log_info "构建完成统计:"
    echo "成功: $success_count"
    echo "失败: ${#failed_services[@]}"
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_warning "失败的服务: ${failed_services[*]}"
        return 1
    else
        log_success "所有服务构建成功"
        return 0
    fi
}

# 推送所有服务
push_all_services() {
    log_info "推送所有服务..."
    
    local failed_services=()
    local success_count=0
    
    # 推送前端
    if push_service "frontend"; then
        ((success_count++))
    else
        failed_services+=("frontend")
    fi
    
    # 推送后端服务
    for service in "${SERVICES[@]}"; do
        if push_service "$service"; then
            ((success_count++))
        else
            failed_services+=("$service")
        fi
    done
    
    echo
    log_info "推送完成统计:"
    echo "成功: $success_count"
    echo "失败: ${#failed_services[@]}"
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_warning "失败的服务: ${failed_services[*]}"
        return 1
    else
        log_success "所有服务推送成功"
        return 0
    fi
}

# 主函数
main() {
    echo "=== 联邦风控系统 Docker 镜像构建脚本 ==="
    echo "镜像仓库: $REGISTRY"
    echo "镜像标签: $TAG"
    echo "服务: $SERVICE"
    echo "操作: $ACTION"
    echo
    
    check_dependencies
    
    case "$ACTION" in
        "build")
            if [ "$SERVICE" = "all" ]; then
                build_all_services
            else
                build_service "$SERVICE"
            fi
            ;;
        "push")
            if [ "$SERVICE" = "all" ]; then
                push_all_services
            else
                push_service "$SERVICE"
            fi
            ;;
        "build-push")
            if [ "$SERVICE" = "all" ]; then
                if build_all_services; then
                    push_all_services
                fi
            else
                if build_service "$SERVICE"; then
                    push_service "$SERVICE"
                fi
            fi
            ;;
        "clean")
            clean_images "$SERVICE"
            ;;
        "list")
            list_images
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