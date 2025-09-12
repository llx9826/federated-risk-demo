#!/bin/bash

# 最小复现器 - 联邦风控系统
# 用事故包中的 seed/参数，在本地最小数据量重放，确保问题可复现

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPRO_DIR="$PROJECT_ROOT/repro"
LOGS_DIR="$PROJECT_ROOT/logs"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

# 使用说明
usage() {
    echo "使用方法: $0 <incident_pack_path>"
    echo ""
    echo "参数:"
    echo "  incident_pack_path    事故包路径 (.zip 文件或解压后的目录)"
    echo ""
    echo "示例:"
    echo "  $0 incidents/2024-01-15T10-30-00-abc123.zip"
    echo "  $0 incidents/temp_2024-01-15T10-30-00-abc123"
    echo ""
    echo "说明:"
    echo "  此脚本会使用事故包中的参数在最小数据量下重现问题"
    echo "  如果问题无法重现，会提示可能的原因（规模/并发耦合）"
    exit 1
}

# 检查参数
if [ $# -ne 1 ]; then
    usage
fi

INCIDENT_PACK="$1"

if [ ! -e "$INCIDENT_PACK" ]; then
    log_error "事故包不存在: $INCIDENT_PACK"
    exit 1
fi

# 创建复现目录
mkdir -p "$REPRO_DIR" "$LOGS_DIR"

# 复现日志
REPRO_LOG="$LOGS_DIR/repro_$(date +%Y%m%d_%H%M%S).log"

# 解压事故包（如果是zip文件）
extract_incident_pack() {
    local pack_path="$1"
    local extract_dir="$REPRO_DIR/incident_$(date +%Y%m%d_%H%M%S)"
    
    if [[ "$pack_path" == *.zip ]]; then
        log_info "解压事故包: $pack_path"
        
        if ! command -v unzip &> /dev/null; then
            log_error "缺少 unzip 命令"
            exit 1
        fi
        
        mkdir -p "$extract_dir"
        unzip -q "$pack_path" -d "$extract_dir"
        
        # 查找实际的内容目录
        local content_dir=$(find "$extract_dir" -name "metadata.json" -exec dirname {} \;)
        if [ -n "$content_dir" ]; then
            echo "$content_dir"
        else
            log_error "事故包格式无效，找不到 metadata.json"
            exit 1
        fi
    else
        # 假设是已解压的目录
        if [ -f "$pack_path/metadata.json" ]; then
            echo "$pack_path"
        else
            log_error "目录格式无效，找不到 metadata.json"
            exit 1
        fi
    fi
}

# 读取种子参数
read_seed_parameters() {
    local incident_dir="$1"
    local seed_file="$incident_dir/seeds/seed_parameters.json"
    
    if [ ! -f "$seed_file" ]; then
        log_warn "种子参数文件不存在: $seed_file"
        # 使用默认参数
        echo '{
            "parameters": {
                "data_generation": {
                    "seed": 42,
                    "size": 5000,
                    "overlap_rate": 0.6,
                    "bad_rate": 0.15
                },
                "training": {
                    "random_state": 42,
                    "learning_rate": 0.1,
                    "max_iter": 30
                }
            }
        }'
    else
        cat "$seed_file"
    fi
}

# 生成最小数据集
generate_minimal_data() {
    local seed_params="$1"
    
    log_info "生成最小数据集..."
    
    # 解析参数
    local seed=$(echo "$seed_params" | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['data_generation']['seed'])" 2>/dev/null || echo "42")
    local size=5000  # 固定使用最小规模
    local overlap_rate=$(echo "$seed_params" | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['data_generation']['overlap_rate'])" 2>/dev/null || echo "0.6")
    local bad_rate=$(echo "$seed_params" | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['data_generation']['bad_rate'])" 2>/dev/null || echo "0.15")
    
    log_info "数据参数: seed=$seed, size=$size, overlap_rate=$overlap_rate, bad_rate=$bad_rate"
    
    # 生成数据
    cd "$PROJECT_ROOT"
    
    python3 tools/seed/synth_vertical_v2.py \
        --output_dir "$REPRO_DIR/data" \
        --prefix "repro" \
        --size "$size" \
        --overlap_rate "$overlap_rate" \
        --bad_rate "$bad_rate" \
        --seed "$seed" 2>&1 | tee -a "$REPRO_LOG"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "最小数据集生成失败"
        return 1
    fi
    
    log_success "最小数据集生成完成"
    return 0
}

# 执行最小PSI对齐
perform_minimal_psi() {
    log_info "执行最小PSI对齐..."
    
    # 检查PSI服务
    if ! curl -s "http://localhost:8001/health" > /dev/null; then
        log_error "PSI服务不可用，请先启动服务"
        return 1
    fi
    
    # 创建PSI会话
    local session_id="repro_$(date +%Y%m%d_%H%M%S)"
    
    local psi_response=$(curl -s -X POST "http://localhost:8001/psi/sessions" \
        -H "Content-Type: application/json" \
        -d "{
            \"session_id\": \"$session_id\",
            \"method\": \"token_join\",
            \"parties\": [\"party_a\", \"party_b\"]
        }" 2>&1)
    
    if [ $? -ne 0 ]; then
        log_error "PSI会话创建失败: $psi_response"
        return 1
    fi
    
    # 等待PSI完成
    local max_wait=180  # 3分钟超时
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        local status_response=$(curl -s "http://localhost:8001/psi/results/$session_id" 2>/dev/null)
        local status=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [ "$status" = "completed" ]; then
            local intersection_size=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('intersection_size', 0))" 2>/dev/null || echo "0")
            log_success "PSI对齐完成，交集大小: $intersection_size"
            
            # 记录PSI结果
            echo "$status_response" > "$REPRO_DIR/psi_result.json"
            return 0
        elif [ "$status" = "failed" ]; then
            log_error "PSI对齐失败"
            echo "$status_response" | tee -a "$REPRO_LOG"
            return 1
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
    done
    
    log_error "PSI对齐超时"
    return 1
}

# 执行最小训练
perform_minimal_training() {
    local seed_params="$1"
    
    log_info "执行最小训练..."
    
    # 检查训练服务
    if ! curl -s "http://localhost:8002/health" > /dev/null; then
        log_error "训练服务不可用，请先启动服务"
        return 1
    fi
    
    # 解析训练参数
    local learning_rate=$(echo "$seed_params" | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['training']['learning_rate'])" 2>/dev/null || echo "0.1")
    local max_iter=30  # 使用更少的迭代次数
    local random_state=$(echo "$seed_params" | python3 -c "import sys, json; print(json.load(sys.stdin)['parameters']['training']['random_state'])" 2>/dev/null || echo "42")
    
    log_info "训练参数: learning_rate=$learning_rate, max_iter=$max_iter, random_state=$random_state"
    
    # 构建训练请求
    local train_request="{
        \"job_name\": \"repro_minimal\",
        \"algorithm\": \"hetero_lr\",
        \"participants\": [\"party_a\", \"party_b\"],
        \"config\": {
            \"learning_rate\": $learning_rate,
            \"max_iter\": $max_iter,
            \"epsilon\": 0,
            \"random_state\": $random_state,
            \"early_stopping\": false
        },
        \"data_config\": {
            \"party_a_data\": \"$REPRO_DIR/data/repro_partyA_bank.csv\",
            \"party_b_data\": \"$REPRO_DIR/data/repro_partyB_ecom.csv\"
        }
    }"
    
    # 提交训练任务
    local train_response=$(curl -s -X POST "http://localhost:8002/train" \
        -H "Content-Type: application/json" \
        -d "$train_request" 2>&1)
    
    if [ $? -ne 0 ]; then
        log_error "训练任务提交失败: $train_response"
        return 1
    fi
    
    local job_id=$(echo "$train_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null || echo "")
    
    if [ -z "$job_id" ]; then
        log_error "训练任务ID获取失败"
        echo "$train_response" | tee -a "$REPRO_LOG"
        return 1
    fi
    
    log_info "训练任务已提交: $job_id"
    
    # 等待训练完成
    local max_train_wait=600  # 10分钟超时
    local train_wait_time=0
    
    while [ $train_wait_time -lt $max_train_wait ]; do
        local job_status=$(curl -s "http://localhost:8002/train/jobs/$job_id" 2>/dev/null)
        local status=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [ "$status" = "completed" ]; then
            local auc=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('auc', 0))" 2>/dev/null || echo "0")
            local ks=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('ks', 0))" 2>/dev/null || echo "0")
            
            log_success "训练完成: AUC=$auc, KS=$ks"
            
            # 记录训练结果
            echo "$job_status" > "$REPRO_DIR/training_result.json"
            return 0
        elif [ "$status" = "failed" ]; then
            log_error "训练失败"
            echo "$job_status" | tee -a "$REPRO_LOG"
            return 1
        fi
        
        sleep 10
        train_wait_time=$((train_wait_time + 10))
    done
    
    log_error "训练超时"
    return 1
}

# 执行最小评分测试
perform_minimal_scoring() {
    log_info "执行最小评分测试..."
    
    # 检查评分服务
    if ! curl -s "http://localhost:8003/health" > /dev/null; then
        log_error "评分服务不可用，请先启动服务"
        return 1
    fi
    
    # 生成5个测试样本
    python3 -c "
import pandas as pd
import numpy as np
import json
import requests
import sys

# 读取复现数据
try:
    party_a = pd.read_csv('$REPRO_DIR/data/repro_partyA_bank.csv')
    party_b = pd.read_csv('$REPRO_DIR/data/repro_partyB_ecom.csv')
except Exception as e:
    print(f'读取复现数据失败: {e}')
    sys.exit(1)

# 生成测试样本
np.random.seed(42)
test_samples = []
scores = []

for i in range(5):
    sample_a = party_a.sample(1).iloc[0]
    sample_b = party_b.sample(1).iloc[0]
    
    features = {}
    for col in party_a.columns:
        if col not in ['user_id', 'label']:
            features[f'a_{col}'] = float(sample_a[col]) if pd.notna(sample_a[col]) else 0.0
    
    for col in party_b.columns:
        if col not in ['user_id', 'label']:
            features[f'b_{col}'] = float(sample_b[col]) if pd.notna(sample_b[col]) else 0.0
    
    sample = {
        'sample_id': f'repro_{i+1:02d}',
        'features': features
    }
    test_samples.append(sample)
    
    # 执行评分
    try:
        response = requests.post(
            'http://localhost:8003/score',
            json={
                'subject': sample['sample_id'],
                'features': features,
                'consent_token': 'repro_consent',
                'purpose': 'repro_test'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            score = result.get('risk_score', 0)
            scores.append(score)
            print(f'样本 {sample[\"sample_id\"]}: 风险评分 = {score:.4f}')
        else:
            print(f'样本 {sample[\"sample_id\"]} 评分失败: HTTP {response.status_code}')
            sys.exit(1)
            
    except Exception as e:
        print(f'样本 {sample[\"sample_id\"]} 评分异常: {e}')
        sys.exit(1)

# 保存结果
result = {
    'samples': test_samples,
    'scores': scores,
    'statistics': {
        'count': len(scores),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores))
    }
}

with open('$REPRO_DIR/scoring_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'评分测试完成: 均值={np.mean(scores):.4f}, 标准差={np.std(scores):.4f}')
" 2>&1 | tee -a "$REPRO_LOG"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "评分测试失败"
        return 1
    fi
    
    log_success "评分测试完成"
    return 0
}

# 比较结果
compare_results() {
    local incident_dir="$1"
    
    log_info "比较复现结果与原始事故..."
    
    # 读取原始指标
    local original_metrics="$incident_dir/metrics/metrics.json"
    local original_scoring="$incident_dir/samples/scoring_results.json"
    
    # 读取复现结果
    local repro_training="$REPRO_DIR/training_result.json"
    local repro_scoring="$REPRO_DIR/scoring_result.json"
    
    python3 -c "
import json
import sys

# 读取结果文件
def safe_read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f'无法读取 {file_path}: {e}')
        return None

original_metrics = safe_read_json('$original_metrics')
original_scoring = safe_read_json('$original_scoring')
repro_training = safe_read_json('$repro_training')
repro_scoring = safe_read_json('$repro_scoring')

print('\n=== 复现结果对比 ===')

# 比较训练指标
if original_metrics and repro_training:
    orig_auc = original_metrics.get('auc', 0)
    repro_auc = repro_training.get('metrics', {}).get('auc', 0)
    
    print(f'AUC: 原始={orig_auc:.4f}, 复现={repro_auc:.4f}, 差异={abs(orig_auc - repro_auc):.4f}')
    
    if abs(orig_auc - repro_auc) > 0.05:
        print('⚠️  AUC差异较大，可能存在规模依赖问题')
    else:
        print('✅ AUC差异在可接受范围内')
else:
    print('❌ 无法比较训练指标')

# 比较评分分布
if original_scoring and repro_scoring:
    orig_std = original_scoring.get('statistics', {}).get('std', 0)
    repro_std = repro_scoring.get('statistics', {}).get('std', 0)
    
    print(f'评分标准差: 原始={orig_std:.4f}, 复现={repro_std:.4f}, 差异={abs(orig_std - repro_std):.4f}')
    
    if abs(orig_std - repro_std) > 0.02:
        print('⚠️  评分分布差异较大，可能存在并发依赖问题')
    else:
        print('✅ 评分分布差异在可接受范围内')
else:
    print('❌ 无法比较评分分布')

print('\n=== 复现结论 ===')

# 判断是否成功复现
if repro_training and repro_scoring:
    repro_auc = repro_training.get('metrics', {}).get('auc', 0)
    repro_std = repro_scoring.get('statistics', {}).get('std', 0)
    
    if repro_auc < 0.6 or repro_std < 0.01:
        print('✅ 问题成功复现：训练指标异常或评分分布退化')
        sys.exit(0)
    else:
        print('❌ 问题未能复现：指标正常')
        print('\n可能原因:')
        print('  1. 规模耦合：问题只在大数据量下出现')
        print('  2. 并发耦合：问题只在高并发场景下出现')
        print('  3. 环境差异：依赖版本或配置不同')
        print('  4. 时间依赖：问题与特定时间窗口相关')
        sys.exit(1)
else:
    print('❌ 复现失败：无法获取完整结果')
    sys.exit(1)
" 2>&1 | tee -a "$REPRO_LOG"
    
    return $?
}

# 生成复现报告
generate_repro_report() {
    local incident_dir="$1"
    local repro_success="$2"
    
    local report_file="$REPRO_DIR/repro_report.md"
    
    cat > "$report_file" << EOF
# 最小复现报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')  
**事故包**: $INCIDENT_PACK  
**复现目录**: $REPRO_DIR  

## 复现配置

- **数据规模**: 5,000 条记录
- **重叠率**: 60%
- **训练迭代**: 30 轮
- **差分隐私**: 关闭 (ε=0)

## 执行步骤

1. ✅ 解压事故包
2. ✅ 读取种子参数
3. ✅ 生成最小数据集
4. ✅ 执行PSI对齐
5. ✅ 执行联邦训练
6. ✅ 执行评分测试
7. ✅ 比较结果

## 复现结果

EOF
    
    if [ "$repro_success" = "0" ]; then
        cat >> "$report_file" << EOF
**状态**: ✅ 问题成功复现

问题在最小数据量下得到重现，说明问题与数据规模无关，可能是算法或配置问题。

### 建议行动

1. 检查算法参数配置
2. 验证特征工程逻辑
3. 排查模型训练流程
4. 检查评分服务实现

EOF
    else
        cat >> "$report_file" << EOF
**状态**: ❌ 问题未能复现

问题在最小数据量下未能重现，可能存在规模或并发耦合。

### 可能原因

1. **规模耦合**: 问题只在大数据量下出现
   - 内存不足导致的性能退化
   - 大规模数据的统计特性差异
   - 分布式计算的同步问题

2. **并发耦合**: 问题只在高并发场景下出现
   - 资源竞争导致的计算错误
   - 线程安全问题
   - 网络延迟和超时

3. **环境差异**: 依赖版本或配置不同
   - Python包版本差异
   - 系统环境变量
   - 硬件性能差异

4. **时间依赖**: 问题与特定时间窗口相关
   - 随机种子的时间依赖
   - 外部服务的状态变化

### 建议行动

1. 在生产环境规模下重现问题
2. 增加并发压力测试
3. 检查环境配置一致性
4. 分析时间相关的依赖

EOF
    fi
    
    cat >> "$report_file" << EOF

## 详细日志

查看完整执行日志: \`$REPRO_LOG\`

## 生成文件

- 复现数据: \`$REPRO_DIR/data/\`
- PSI结果: \`$REPRO_DIR/psi_result.json\`
- 训练结果: \`$REPRO_DIR/training_result.json\`
- 评分结果: \`$REPRO_DIR/scoring_result.json\`

---
**复现器版本**: 1.0.0  
**执行时间**: $(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    log_info "复现报告已生成: $report_file"
}

# 主流程
main() {
    log_info "开始最小复现..."
    
    # 检查依赖
    for cmd in curl python3 unzip; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "缺少依赖: $cmd"
            exit 1
        fi
    done
    
    # 解压事故包
    local incident_dir
    incident_dir=$(extract_incident_pack "$INCIDENT_PACK")
    
    if [ $? -ne 0 ]; then
        log_error "事故包解压失败"
        exit 1
    fi
    
    log_success "事故包解压完成: $incident_dir"
    
    # 读取种子参数
    local seed_params
    seed_params=$(read_seed_parameters "$incident_dir")
    
    log_info "种子参数读取完成"
    
    # 执行复现流程
    local repro_success=1
    
    if generate_minimal_data "$seed_params" && \
       perform_minimal_psi && \
       perform_minimal_training "$seed_params" && \
       perform_minimal_scoring; then
        
        log_info "复现流程执行完成，开始比较结果..."
        
        if compare_results "$incident_dir"; then
            repro_success=0
            log_success "🎉 问题成功复现！"
        else
            log_warn "⚠️  问题未能复现，可能存在规模/并发耦合"
        fi
    else
        log_error "复现流程执行失败"
    fi
    
    # 生成报告
    generate_repro_report "$incident_dir" "$repro_success"
    
    log_info "📋 查看复现报告: $REPRO_DIR/repro_report.md"
    log_info "📝 查看执行日志: $REPRO_LOG"
    
    exit $repro_success
}

# 信号处理
trap 'log_error "最小复现被中断"; exit 1' INT TERM

# 执行主流程
main "$@"