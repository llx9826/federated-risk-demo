#!/bin/bash

# Ray集群停止脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ray分布式集群停止 ===${NC}"

# 显示当前集群状态
echo -e "${YELLOW}检查集群状态...${NC}"
if curl -s http://localhost:8265/api/cluster_status >/dev/null 2>&1; then
    echo -e "${YELLOW}Ray集群正在运行，准备停止...${NC}"
else
    echo -e "${YELLOW}Ray集群未运行或无法访问${NC}"
fi

# 停止容器
echo -e "${YELLOW}停止Ray容器...${NC}"
docker-compose -f docker-compose.ray.yml down --remove-orphans

# 清理Ray临时文件
echo -e "${YELLOW}清理临时文件...${NC}"
rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf data/ray_temp/* 2>/dev/null || true

# 清理Docker资源
echo -e "${YELLOW}清理Docker资源...${NC}"
docker system prune -f --filter "label=com.docker.compose.project=federated-risk-demo" 2>/dev/null || true

# 检查端口释放
echo -e "${YELLOW}检查端口释放...${NC}"
sleep 2

check_port_free() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}警告: 端口 $port 仍被占用${NC}"
        return 1
    else
        echo -e "${GREEN}✓ 端口 $port 已释放${NC}"
        return 0
    fi
}

check_port_free 6379
check_port_free 8265
check_port_free 10001

echo -e "${GREEN}✓ Ray集群已停止${NC}"
echo -e "${BLUE}使用 './cluster_up.sh [workers]' 重新启动集群${NC}"