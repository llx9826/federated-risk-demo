#!/bin/bash

# Ray集群启动脚本
# 用法: ./cluster_up.sh [workers_count]

set -e

# 默认worker数量
WORKERS=${1:-16}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ray分布式集群启动 ===${NC}"
echo -e "${YELLOW}Worker数量: ${WORKERS}${NC}"

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}错误: Docker未运行，请先启动Docker${NC}"
    exit 1
fi

# 检查端口占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}错误: 端口 $port 已被占用${NC}"
        echo "请检查是否有其他Ray集群在运行"
        exit 1
    fi
}

echo -e "${YELLOW}检查端口占用...${NC}"
check_port 6379
check_port 8265

# 清理旧容器
echo -e "${YELLOW}清理旧容器...${NC}"
docker-compose -f docker-compose.ray.yml down --remove-orphans 2>/dev/null || true

# 创建必要目录
echo -e "${YELLOW}创建目录结构...${NC}"
mkdir -p reports/{psi,train,plots}
mkdir -p docs/assets
mkdir -p data/ray_temp

# 设置系统参数优化
echo -e "${YELLOW}优化系统参数...${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux系统优化
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf > /dev/null || true
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf > /dev/null || true
    
    # 启用大页内存（如果可用）
    if [ -f /proc/sys/vm/nr_hugepages ]; then
        echo 1024 | sudo tee /proc/sys/vm/nr_hugepages > /dev/null || true
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS系统优化
    ulimit -n 65536 2>/dev/null || true
fi

# 启动Ray集群
echo -e "${YELLOW}启动Ray集群...${NC}"
export WORKERS=$WORKERS
docker-compose -f docker-compose.ray.yml up -d

# 等待集群就绪
echo -e "${YELLOW}等待集群就绪...${NC}"
sleep 10

# 检查集群状态
echo -e "${YELLOW}检查集群状态...${NC}"
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:8265/api/cluster_status | grep -q '"status":"running"'; then
        echo -e "${GREEN}✓ Ray集群启动成功${NC}"
        break
    fi
    
    retry_count=$((retry_count + 1))
    echo -e "${YELLOW}等待集群就绪... ($retry_count/$max_retries)${NC}"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo -e "${RED}错误: 集群启动超时${NC}"
    docker-compose -f docker-compose.ray.yml logs
    exit 1
fi

# 安装Python依赖
echo -e "${YELLOW}安装Python依赖...${NC}"
docker exec ray-head pip install -q \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    scikit-learn==1.3.0 \
    xgboost==1.7.6 \
    pybloom_live==4.0.0 \
    cryptography==41.0.3 \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    aiohttp==3.8.5 \
    loguru==0.7.0 \
    tqdm==4.66.1 \
    psutil==5.9.5

# 显示集群信息
echo -e "${GREEN}=== Ray集群信息 ===${NC}"
echo -e "${BLUE}Dashboard: http://localhost:8265${NC}"
echo -e "${BLUE}Client端口: localhost:6379${NC}"
echo -e "${BLUE}Worker数量: ${WORKERS}${NC}"

# 显示资源状态
echo -e "${YELLOW}获取资源状态...${NC}"
docker exec ray-head python -c "
import ray
ray.init(address='ray://localhost:10001')
print(f'集群节点数: {len(ray.nodes())}')
print(f'可用CPU核心: {ray.cluster_resources().get("CPU", 0)}')
print(f'可用内存: {ray.cluster_resources().get("memory", 0) / 1e9:.1f} GB')
ray.shutdown()
" 2>/dev/null || echo -e "${YELLOW}暂时无法获取资源状态，集群可能仍在初始化${NC}"

echo -e "${GREEN}✓ Ray集群启动完成！${NC}"
echo -e "${BLUE}使用 './cluster_down.sh' 停止集群${NC}"