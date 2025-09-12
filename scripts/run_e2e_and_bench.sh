#!/bin/bash
# 一键全链路自测脚本：E2E + 基准测试 + 文档生成
# 顺序：reset → 造数 → 授权 → PSI对齐 → 特征处理 → 联邦训练 → 上线评分 → 审计回执校验 → 基准测试 → 生成图表与报告

set -e  # 失败即停

# 颜色输出
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

# 错误处理函数
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "脚本在第 $line_number 行失败，退出码: $exit_code"
    
    case $line_number in
        *)
            log_error "排障建议："
            log_error "1. 检查服务是否正常启动: curl http://localhost:8000/health"
            log_error "2. 检查依赖是否安装: pip install -r requirements.txt"
            log_error "3. 检查端口是否被占用: lsof -i :8000-8003"
            log_error "4. 查看详细日志: tail -f logs/self_test.log"
            ;;
    esac
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# 创建必要目录
mkdir -p reports/bench/psi reports/bench/train reports/bench/score reports/plots logs traces

log_info "=== 第1步: 系统重置与环境检查 ==="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    log_error "Python3 未安装"
    exit 1
fi

# 检查Node.js环境
if ! command -v node &> /dev/null; then
    log_error "Node.js 未安装"
    exit 1
fi

# 检查k6
if ! command -v k6 &> /dev/null; then
    log_warning "k6 未安装，将跳过压力测试"
    SKIP_K6=true
else
    SKIP_K6=false
fi

# 停止可能运行的服务
log_info "停止现有服务..."
pkill -f "uvicorn.*app:app" || true
sleep 2

# 清理旧数据
log_info "清理旧数据..."
rm -rf data/psi/e2e_test_* data/synth/e2e_* models/e2e_* reports/e2e_*

log_info "=== 第2步: 数据生成 ==="

# 生成纵向联邦数据
log_info "生成两方数据 (n=10000, overlap=0.3)..."
python3 tools/seed/synth_vertical_benchmark.py \
    --n 10000 \
    --overlap 0.3 \
    --parties 2 \
    --bad_rate 0.15 \
    --noise 0.05 \
    --seed 42 \
    --output data/synth/e2e_two_party

log_info "生成多方数据 (n=5000, parties=3)..."
python3 tools/seed/synth_vertical_benchmark.py \
    --n 5000 \
    --overlap 0.25 \
    --parties 3 \
    --bad_rate 0.12 \
    --noise 0.03 \
    --seed 123 \
    --output data/synth/e2e_multi_party

log_success "数据生成完成"

log_info "=== 第3步: 启动服务 ==="

# 启动所有服务
log_info "启动同意管理服务..."
python3 -m uvicorn services.consent-service.app:app --host 0.0.0.0 --port 8000 --reload > logs/consent_service.log 2>&1 &
CONSENT_PID=$!

log_info "启动PSI服务..."
MAX_CONCURRENT_PSI=50 python3 -m uvicorn services.psi-service.app:app --host 0.0.0.0 --port 8001 --reload > logs/psi_service.log 2>&1 &
PSI_PID=$!

log_info "启动训练服务..."
python3 -m uvicorn services.train-service.app:app --host 0.0.0.0 --port 8002 --reload > logs/train_service.log 2>&1 &
TRAIN_PID=$!

log_info "启动模型服务..."
python3 -m uvicorn services.serving-service.app:app --host 0.0.0.0 --port 8003 --reload > logs/serving_service.log 2>&1 &
SERVING_PID=$!

# 等待服务启动
log_info "等待服务启动..."
sleep 10

# 健康检查
log_info "执行健康检查..."
for port in 8000 8001 8002 8003; do
    if ! curl -s http://localhost:$port/health > /dev/null; then
        log_error "服务 localhost:$port 启动失败"
        exit 1
    fi
done

log_success "所有服务启动成功"

log_info "=== 第4步: 数据合约校验 ==="

# 校验数据质量
log_info "校验两方数据合约..."
python3 tools/contract/data_contract.py \
    --input data/synth/e2e_two_party \
    --output reports/e2e_two_party_contract.json

log_info "校验多方数据合约..."
python3 tools/contract/data_contract.py \
    --input data/synth/e2e_multi_party \
    --output reports/e2e_multi_party_contract.json

log_success "数据合约校验完成"

log_info "=== 第5步: 授权测试 ==="

# 签发同意授权
log_info "签发同意授权..."
CONSENT_RESPONSE=$(curl -s -X POST "http://localhost:8000/consent/issue" \
    -H "Content-Type: application/json" \
    -d '{
        "subject": "e2e_test_customer",
        "purpose": "credit_scoring",
        "scope_features": ["credit_score", "annual_income", "debt_ratio", "has_mortgage"],
        "ttl_hours": 24,
        "issuer": "e2e_test_bank"
    }')

CONSENT_JWT=$(echo $CONSENT_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['consent_jwt'])")

if [ -z "$CONSENT_JWT" ]; then
    log_error "授权签发失败"
    exit 1
fi

log_success "授权签发成功: ${CONSENT_JWT:0:50}..."

# 测试授权验证
log_info "测试授权验证..."
VERIFY_RESPONSE=$(curl -s -X POST "http://localhost:8000/consent/verify" \
    -H "Content-Type: application/json" \
    -d "{
        \"consent_jwt\": \"$CONSENT_JWT\",
        \"requested_purpose\": \"credit_scoring\",
        \"requested_features\": [\"credit_score\", \"annual_income\"]
    }")

VALID=$(echo $VERIFY_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['valid'])")

if [ "$VALID" != "True" ]; then
    log_error "授权验证失败"
    exit 1
fi

log_success "授权验证通过"

log_info "=== 第6步: PSI数据对齐 ==="

# 执行PSI对齐
log_info "执行ECDH-PSI对齐..."
PSI_RESPONSE=$(curl -s -X POST "http://localhost:8001/psi/compute" \
    -H "Authorization: Bearer $CONSENT_JWT" \
    -H "Content-Type: application/json" \
    -d '{
        "party_a_data": ["e2e_001", "e2e_002", "e2e_003", "e2e_004", "e2e_005"],
        "party_b_data": ["e2e_003", "e2e_004", "e2e_005", "e2e_006", "e2e_007"],
        "algorithm": "ecdh_psi"
    }')

INTERSECTION_SIZE=$(echo $PSI_RESPONSE | python3 -c "import sys, json; print(len(json.load(sys.stdin)['intersection']))")

if [ "$INTERSECTION_SIZE" -eq 0 ]; then
    log_error "PSI对齐失败，交集为空"
    exit 1
fi

log_success "PSI对齐完成，交集大小: $INTERSECTION_SIZE"

# 保存PSI结果
echo $PSI_RESPONSE > reports/e2e_psi_result.json

log_info "=== 第7步: 联邦训练 ==="

# 启动联邦训练 (ε=∞)
log_info "启动联邦训练 (ε=∞)..."
TRAIN_RESPONSE_INF=$(curl -s -X POST "http://localhost:8002/train/federated" \
    -H "Content-Type: application/json" \
    -d '{
        "task_name": "e2e_test_inf",
        "participants": ["party_a", "party_b"],
        "algorithm": "secure_boost",
        "privacy_config": {"epsilon": "infinity"},
        "max_rounds": 15
    }')

TASK_ID_INF=$(echo $TRAIN_RESPONSE_INF | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")

# 等待训练完成
log_info "等待训练完成..."
sleep 30

# 获取训练结果
RESULT_RESPONSE_INF=$(curl -s "http://localhost:8002/train/result/$TASK_ID_INF")
AUC_INF=$(echo $RESULT_RESPONSE_INF | python3 -c "import sys, json; print(json.load(sys.stdin)['performance_metrics']['auc'])")

log_success "联邦训练完成 (ε=∞), AUC: $AUC_INF"

# 启动联邦训练 (ε=5)
log_info "启动联邦训练 (ε=5)..."
TRAIN_RESPONSE_5=$(curl -s -X POST "http://localhost:8002/train/federated" \
    -H "Content-Type: application/json" \
    -d '{
        "task_name": "e2e_test_eps5",
        "participants": ["party_a", "party_b"],
        "algorithm": "secure_boost",
        "privacy_config": {"epsilon": 5.0},
        "max_rounds": 20
    }')

TASK_ID_5=$(echo $TRAIN_RESPONSE_5 | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")

# 等待训练完成
sleep 35

# 获取训练结果
RESULT_RESPONSE_5=$(curl -s "http://localhost:8002/train/result/$TASK_ID_5")
AUC_5=$(echo $RESULT_RESPONSE_5 | python3 -c "import sys, json; print(json.load(sys.stdin)['performance_metrics']['auc'])")

log_success "联邦训练完成 (ε=5), AUC: $AUC_5"

# 保存训练结果
echo $RESULT_RESPONSE_INF > reports/e2e_train_result_inf.json
echo $RESULT_RESPONSE_5 > reports/e2e_train_result_eps5.json

log_info "=== 第8步: 模型评分测试 ==="

# 测试模型评分
log_info "测试模型评分..."
SCORE_RESPONSE=$(curl -s -X POST "http://localhost:8003/score" \
    -H "Authorization: Bearer $CONSENT_JWT" \
    -H "Content-Type: application/json" \
    -d '{
        "features": {
            "credit_score": 720,
            "annual_income": 80000,
            "debt_ratio": 0.35,
            "has_mortgage": true
        },
        "model_version": "e2e_test_inf"
    }')

SCORE=$(echo $SCORE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['score'])")
DECISION=$(echo $SCORE_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['decision'])")

log_success "模型评分完成, Score: $SCORE, Decision: $DECISION"

# 保存评分结果
echo $SCORE_RESPONSE > reports/e2e_score_result.json

log_info "=== 第9步: 审计回执校验 ==="

# 检查回执字段完整性
log_info "校验审计回执字段..."
python3 -c "
import json
with open('reports/e2e_score_result.json') as f:
    result = json.load(f)
required_fields = ['request_id', 'score', 'decision', 'model_hash', 'timestamp', 'consent_fingerprint']
missing = [f for f in required_fields if f not in result]
if missing:
    print(f'缺失字段: {missing}')
    exit(1)
else:
    print('审计回执字段完整')
"

log_success "审计回执校验通过"

log_info "=== 第10步: 基准测试 ==="

# PSI基准测试
log_info "执行PSI基准测试..."
node bench/psi/psi_bench.js --n 10000 --workers 4 --shards 8 --output reports/bench/psi/e2e_psi_bench.json

# 训练基准测试
log_info "执行训练基准测试..."
node bench/train/train_bench.js --n 5000 --epsilon 5 --participants 2 --output reports/bench/train/e2e_train_bench.json

# 评分压力测试
if [ "$SKIP_K6" = false ]; then
    log_info "执行评分压力测试..."
    k6 run bench/score/score_k6.js --out json=reports/bench/score/e2e_score_bench.json
else
    log_warning "跳过k6压力测试"
fi

log_success "基准测试完成"

log_info "=== 第11步: 统计分析与外推 ==="

# 执行统计外推
log_info "执行性能外推分析..."
python3 tools/stats/extrapolate.py \
    --psi_data reports/bench/psi/e2e_psi_bench.json \
    --train_data reports/bench/train/e2e_train_bench.json \
    --output reports/e2e_extrapolation.json

log_success "统计分析完成"

log_info "=== 第12步: 生成图表与报告 ==="

# 生成性能图表
log_info "生成性能图表..."
python3 tools/plots/generate_charts.py \
    --input_dir reports \
    --output_dir reports/plots

# 更新展示文档
log_info "更新展示文档..."
python3 tools/docs/update_showcase.py \
    --template docs/Hackathon_Showcase_PABank.md \
    --data_dir reports \
    --output docs/Hackathon_Showcase_PABank.md

log_success "文档更新完成"

# 清理进程
log_info "清理服务进程..."
kill $CONSENT_PID $PSI_PID $TRAIN_PID $SERVING_PID 2>/dev/null || true

log_success "=== E2E测试与基准测试全部完成 ==="
log_info "报告文件:"
log_info "  - 展示文档: docs/Hackathon_Showcase_PABank.md"
log_info "  - 基准报告: reports/bench/"
log_info "  - 性能图表: reports/plots/"
log_info "  - 外推分析: reports/e2e_extrapolation.json"

echo "E2E测试成功完成！"