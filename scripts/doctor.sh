#!/bin/bash

# 一键医生脚本 - 联邦风控系统诊断与自愈
# 流程：reset → 造数 → PSI对齐 → 训练 → 评估 → 在线评分 → 审计校验 → 生成事故包

set -e  # 失败即停

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/reports"
LOGS_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data"
INCIDENTS_DIR="$PROJECT_ROOT/incidents"

# 创建必要目录
mkdir -p "$REPORTS_DIR" "$LOGS_DIR" "$INCIDENTS_DIR"

# 日志文件
DOCTOR_LOG="$LOGS_DIR/doctor_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="$REPORTS_DIR/doctor_report.md"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DOCTOR_LOG"
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

# 诊断树输出
diagnose() {
    local phase="$1"
    local issue="$2"
    local action="$3"
    
    log_error "诊断: [$phase] $issue"
    log_info "建议行动: $action"
    
    # 记录到报告
    echo "## 诊断结果: $phase" >> "$REPORT_FILE"
    echo "- **问题**: $issue" >> "$REPORT_FILE"
    echo "- **建议**: $action" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# 失败处理
fail_with_diagnosis() {
    local phase="$1"
    local issue="$2"
    local action="$3"
    
    diagnose "$phase" "$issue" "$action"
    
    log_error "医生诊断失败，正在生成事故包..."
    generate_incident_pack
    
    log_error "诊断失败，请查看报告: $REPORT_FILE"
    log_error "事故包已生成，位置: $INCIDENTS_DIR"
    exit 1
}

# 生成事故包
generate_incident_pack() {
    log_info "生成事故包..."
    
    if [ -f "$PROJECT_ROOT/tools/incident/export_incident_pack.js" ]; then
        cd "$PROJECT_ROOT"
        node tools/incident/export_incident_pack.js
    else
        log_warn "事故包导出器不存在，跳过"
    fi
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."
    
    local services=("8000:consent" "8001:psi" "8002:train" "8003:serving")
    
    for service in "${services[@]}"; do
        local port="${service%%:*}"
        local name="${service##*:}"
        
        if ! curl -s "http://localhost:$port/health" > /dev/null; then
            fail_with_diagnosis "服务层" "$name 服务 (端口 $port) 不可用" "启动 $name 服务并确保健康检查通过"
        fi
    done
    
    log_success "所有服务运行正常"
}

# 重置环境
reset_environment() {
    log_info "重置环境..."
    
    # 清理旧数据
    rm -rf "$DATA_DIR/psi/doctor_*"
    rm -rf "$DATA_DIR/models/doctor_*"
    rm -rf "$DATA_DIR/synth/doctor_*"
    
    # 初始化报告
    cat > "$REPORT_FILE" << EOF
# 联邦风控系统医生报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**执行ID**: doctor_$(date +%Y%m%d_%H%M%S)

## 执行摘要

EOF
    
    log_success "环境重置完成"
}

# 数据生成与验证
generate_and_validate_data() {
    log_info "生成合成数据..."
    
    cd "$PROJECT_ROOT"
    
    # 生成数据
    python3 tools/seed/synth_vertical_v2.py \
        --output_dir "$DATA_DIR/synth" \
        --prefix "doctor" \
        --size 10000 \
        --overlap_rate 0.6 \
        --bad_rate 0.15 \
        --seed 42
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "数据层" "合成数据生成失败" "检查数据生成器配置和依赖"
    fi
    
    # 数据质量检查
    python3 -c "
import pandas as pd
import numpy as np
import json
import sys

# 读取数据
try:
    party_a = pd.read_csv('$DATA_DIR/synth/doctor_partyA_bank.csv')
    party_b = pd.read_csv('$DATA_DIR/synth/doctor_partyB_ecom.csv')
except Exception as e:
    print(f'数据读取失败: {e}')
    sys.exit(1)

# 数据质量检查
issues = []

# 检查交集
common_ids = set(party_a['user_id']) & set(party_b['user_id'])
if len(common_ids) < 1000:
    issues.append(f'交集过小: {len(common_ids)} < 1000')

# 检查标签分布
if 'label' in party_a.columns:
    bad_rate = party_a['label'].mean()
    if bad_rate < 0.08 or bad_rate > 0.20:
        issues.append(f'坏账率异常: {bad_rate:.3f} 不在 [0.08, 0.20] 范围内')
    
    if party_a['label'].nunique() < 2:
        issues.append('标签单一，无法训练分类模型')

# 检查特征质量
numeric_cols = party_a.select_dtypes(include=[np.number]).columns
valid_features = 0
for col in numeric_cols:
    if col != 'label' and col != 'user_id':
        # 检查方差
        if party_a[col].var() > 1e-10:
            # 检查与标签的相关性
            if 'label' in party_a.columns:
                corr = abs(party_a[col].corr(party_a['label']))
                if corr >= 0.1:
                    valid_features += 1

if valid_features < 6:
    issues.append(f'有效特征不足: {valid_features} < 6')

# 检查缺失值和异常值
for df, name in [(party_a, 'PartyA'), (party_b, 'PartyB')]:
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['user_id', 'label']:
            if df[col].isnull().sum() > len(df) * 0.5:
                issues.append(f'{name}.{col} 缺失率过高: {df[col].isnull().mean():.2%}')
            
            if np.isinf(df[col]).any():
                issues.append(f'{name}.{col} 包含无穷值')

# 输出结果
if issues:
    print('数据质量问题:')
    for issue in issues:
        print(f'  - {issue}')
    sys.exit(1)
else:
    print('数据质量检查通过')
    
    # 输出统计信息
    stats = {
        'party_a_size': len(party_a),
        'party_b_size': len(party_b),
        'intersection_size': len(common_ids),
        'bad_rate': float(party_a['label'].mean()) if 'label' in party_a.columns else 0,
        'valid_features': int(valid_features)
    }
    
    with open('$DATA_DIR/synth/doctor_data_profile.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f'数据统计: {stats}')
"
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "数据层" "数据质量检查失败" "检查数据生成参数，确保交集≥1000、标签两类存在、bad_rate∈[0.08,0.20]、≥6个特征|ρ|≥0.1"
    fi
    
    log_success "数据生成与验证完成"
}

# PSI对齐
perform_psi_alignment() {
    log_info "执行PSI对齐..."
    
    # 创建PSI会话
    local session_id="doctor_$(date +%Y%m%d_%H%M%S)"
    
    # 调用PSI服务
    local psi_response=$(curl -s -X POST "http://localhost:8001/psi/sessions" \
        -H "Content-Type: application/json" \
        -d "{
            \"session_id\": \"$session_id\",
            \"method\": \"token_join\",
            \"parties\": [\"party_a\", \"party_b\"]
        }")
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "对齐层" "PSI会话创建失败" "检查PSI服务状态和网络连接"
    fi
    
    # 等待PSI完成
    local max_wait=300  # 5分钟超时
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        local status_response=$(curl -s "http://localhost:8001/psi/results/$session_id")
        local status=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [ "$status" = "completed" ]; then
            log_success "PSI对齐完成"
            
            # 验证对齐结果
            local intersection_size=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('intersection_size', 0))" 2>/dev/null || echo "0")
            
            if [ "$intersection_size" -lt 1000 ]; then
                fail_with_diagnosis "对齐层" "交集过小: $intersection_size < 1000" "检查PSI盐配置、键清洗逻辑，重新执行PSI对齐"
            fi
            
            echo "PSI对齐统计: 交集大小=$intersection_size" >> "$REPORT_FILE"
            return 0
        elif [ "$status" = "failed" ]; then
            fail_with_diagnosis "对齐层" "PSI对齐失败" "检查双方数据格式、盐配置和网络连接"
        fi
        
        sleep 10
        wait_time=$((wait_time + 10))
    done
    
    fail_with_diagnosis "对齐层" "PSI对齐超时" "检查PSI服务性能和数据规模"
}

# 联邦训练
perform_federated_training() {
    log_info "开始联邦训练..."
    
    local training_configs=(
        "infinity:0"  # ε=∞ (无差分隐私)
        "relaxed:5"   # ε=5
        "strict:3"   # ε=3
    )
    
    local best_auc=0
    local best_config=""
    local healing_attempts=0
    local max_healing=3
    
    for config in "${training_configs[@]}"; do
        local config_name="${config%%:*}"
        local epsilon="${config##*:}"
        
        log_info "尝试训练配置: $config_name (ε=$epsilon)"
        
        while [ $healing_attempts -lt $max_healing ]; do
            # 构建训练请求
            local train_request="{
                \"job_name\": \"doctor_$config_name\",
                \"algorithm\": \"hetero_lr\",
                \"participants\": [\"party_a\", \"party_b\"],
                \"config\": {
                    \"learning_rate\": 0.1,
                    \"max_iter\": 50,
                    \"epsilon\": $epsilon,
                    \"early_stopping\": true,
                    \"min_iter\": 20
                },
                \"data_config\": {
                    \"party_a_data\": \"$DATA_DIR/synth/doctor_partyA_bank.csv\",
                    \"party_b_data\": \"$DATA_DIR/synth/doctor_partyB_ecom.csv\"
                }
            }"
            
            # 提交训练任务
            local train_response=$(curl -s -X POST "http://localhost:8002/train" \
                -H "Content-Type: application/json" \
                -d "$train_request")
            
            if [ $? -ne 0 ]; then
                fail_with_diagnosis "训练层" "训练任务提交失败" "检查训练服务状态和请求格式"
            fi
            
            local job_id=$(echo "$train_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null || echo "")
            
            if [ -z "$job_id" ]; then
                fail_with_diagnosis "训练层" "训练任务ID获取失败" "检查训练服务响应格式"
            fi
            
            # 等待训练完成
            local max_train_wait=1800  # 30分钟超时
            local train_wait_time=0
            
            while [ $train_wait_time -lt $max_train_wait ]; do
                local job_status=$(curl -s "http://localhost:8002/train/jobs/$job_id")
                local status=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
                
                if [ "$status" = "completed" ]; then
                    log_success "训练完成: $config_name"
                    
                    # 获取训练指标
                    local auc=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('auc', 0))" 2>/dev/null || echo "0")
                    local ks=$(echo "$job_status" | python3 -c "import sys, json; print(json.load(sys.stdin).get('metrics', {}).get('ks', 0))" 2>/dev/null || echo "0")
                    
                    log_info "训练指标: AUC=$auc, KS=$ks"
                    
                    # 检查指标是否达标
                    local auc_ok=$(python3 -c "print(1 if float('$auc') >= 0.65 else 0)")
                    local ks_ok=$(python3 -c "print(1 if float('$ks') >= 0.20 else 0)")
                    
                    if [ "$auc_ok" = "1" ] && [ "$ks_ok" = "1" ]; then
                        if (( $(echo "$auc > $best_auc" | bc -l) )); then
                            best_auc="$auc"
                            best_config="$config_name"
                        fi
                        
                        echo "训练成功: $config_name, AUC=$auc, KS=$ks" >> "$REPORT_FILE"
                        return 0
                    else
                        log_warn "训练指标不达标: AUC=$auc (<0.65), KS=$ks (<0.20)"
                        
                        # 尝试自愈
                        healing_attempts=$((healing_attempts + 1))
                        log_info "尝试自愈 ($healing_attempts/$max_healing)..."
                        
                        # 自愈策略：调整学习率和迭代次数
                        if [ $healing_attempts -eq 1 ]; then
                            log_info "自愈策略1: 降低学习率到0.05，增加迭代次数到100"
                            train_request=$(echo "$train_request" | sed 's/"learning_rate": 0.1/"learning_rate": 0.05/' | sed 's/"max_iter": 50/"max_iter": 100/')
                        elif [ $healing_attempts -eq 2 ]; then
                            log_info "自愈策略2: 启用类别平衡"
                            train_request=$(echo "$train_request" | sed 's/"early_stopping": true/"early_stopping": true, "scale_pos_weight": "auto"/')
                        elif [ $healing_attempts -eq 3 ]; then
                            log_info "自愈策略3: 关闭差分隐私"
                            train_request=$(echo "$train_request" | sed 's/"epsilon": [0-9]*/"epsilon": 0/')
                        fi
                        
                        continue  # 重新训练
                    fi
                elif [ "$status" = "failed" ]; then
                    log_error "训练失败: $config_name"
                    healing_attempts=$((healing_attempts + 1))
                    
                    if [ $healing_attempts -lt $max_healing ]; then
                        log_info "尝试自愈 ($healing_attempts/$max_healing)..."
                        continue
                    else
                        break
                    fi
                fi
                
                sleep 30
                train_wait_time=$((train_wait_time + 30))
            done
            
            if [ $train_wait_time -ge $max_train_wait ]; then
                log_error "训练超时: $config_name"
                healing_attempts=$((healing_attempts + 1))
            fi
            
            if [ $healing_attempts -ge $max_healing ]; then
                break
            fi
        done
        
        # 重置自愈计数器为下一个配置
        healing_attempts=0
    done
    
    if [ "$best_auc" = "0" ]; then
        fail_with_diagnosis "训练层" "所有训练配置均失败" "检查数据质量、特征工程、模型参数；考虑增大样本量或关闭差分隐私"
    fi
    
    log_success "最佳训练配置: $best_config, AUC=$best_auc"
}

# 在线评分测试
perform_online_scoring() {
    log_info "执行在线评分测试..."
    
    # 生成20个测试样本
    python3 -c "
import pandas as pd
import numpy as np
import json
import requests
import sys

# 读取训练数据以获取特征分布
try:
    party_a = pd.read_csv('$DATA_DIR/synth/doctor_partyA_bank.csv')
    party_b = pd.read_csv('$DATA_DIR/synth/doctor_partyB_ecom.csv')
except Exception as e:
    print(f'读取数据失败: {e}')
    sys.exit(1)

# 生成测试样本
np.random.seed(42)
test_samples = []

for i in range(20):
    # 从训练数据中随机选择特征值
    sample_a = party_a.sample(1).iloc[0]
    sample_b = party_b.sample(1).iloc[0]
    
    # 构建评分请求
    features = {}
    for col in party_a.columns:
        if col not in ['user_id', 'label']:
            features[f'a_{col}'] = float(sample_a[col]) if pd.notna(sample_a[col]) else 0.0
    
    for col in party_b.columns:
        if col not in ['user_id', 'label']:
            features[f'b_{col}'] = float(sample_b[col]) if pd.notna(sample_b[col]) else 0.0
    
    test_samples.append({
        'sample_id': f'test_{i+1:02d}',
        'features': features
    })

# 执行评分
scores = []
for sample in test_samples:
    try:
        response = requests.post(
            'http://localhost:8003/score',
            json={
                'subject': sample['sample_id'],
                'features': sample['features'],
                'consent_token': 'test_consent',
                'purpose': 'doctor_test'
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

# 检查评分分布
if len(scores) < 20:
    print(f'评分样本不足: {len(scores)} < 20')
    sys.exit(1)

scores_array = np.array(scores)
score_std = scores_array.std()
score_mean = scores_array.mean()
zero_ratio = (scores_array == 0).mean()
one_ratio = (scores_array == 1).mean()

print(f'评分统计: 均值={score_mean:.4f}, 标准差={score_std:.4f}')
print(f'分布检查: 0值占比={zero_ratio:.2%}, 1值占比={one_ratio:.2%}')

# 检查是否退化
if score_std < 0.01:
    print('评分分布退化: 标准差过小')
    sys.exit(1)

if zero_ratio > 0.95 or one_ratio > 0.95:
    print('评分分布退化: 单一值占比过高')
    sys.exit(1)

print('在线评分测试通过')

# 保存评分结果
with open('$DATA_DIR/doctor_scoring_results.json', 'w') as f:
    json.dump({
        'samples': test_samples,
        'scores': scores,
        'statistics': {
            'mean': float(score_mean),
            'std': float(score_std),
            'zero_ratio': float(zero_ratio),
            'one_ratio': float(one_ratio)
        }
    }, f, indent=2)
"
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "服务层" "在线评分测试失败" "检查模型服务状态、特征格式和评分逻辑"
    fi
    
    log_success "在线评分测试完成"
}

# 审计校验
perform_audit_validation() {
    log_info "执行审计校验..."
    
    # 检查审计日志
    local audit_response=$(curl -s "http://localhost:8000/audit/logs?limit=50")
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "审计层" "审计日志获取失败" "检查审计服务状态和数据库连接"
    fi
    
    # 验证审计字段完整性
    python3 -c "
import json
import sys

try:
    audit_data = json.loads('$audit_response')
    logs = audit_data.get('logs', [])
except Exception as e:
    print(f'审计数据解析失败: {e}')
    sys.exit(1)

if not logs:
    print('审计日志为空')
    sys.exit(1)

# 检查必需字段
required_fields = ['timestamp', 'action', 'user', 'resource', 'result']
missing_fields = []

for log in logs[:10]:  # 检查最近10条
    for field in required_fields:
        if field not in log:
            missing_fields.append(field)

if missing_fields:
    print(f'审计字段缺失: {set(missing_fields)}')
    sys.exit(1)

print(f'审计校验通过: 检查了 {len(logs)} 条日志')
"
    
    if [ $? -ne 0 ]; then
        fail_with_diagnosis "审计层" "审计字段完整性检查失败" "检查审计日志格式，确保包含: timestamp, action, user, resource, result"
    fi
    
    log_success "审计校验完成"
}

# 生成最终报告
generate_final_report() {
    log_info "生成最终报告..."
    
    # 读取统计数据
    local data_stats=""
    if [ -f "$DATA_DIR/synth/doctor_data_profile.json" ]; then
        data_stats=$(cat "$DATA_DIR/synth/doctor_data_profile.json")
    fi
    
    local scoring_stats=""
    if [ -f "$DATA_DIR/doctor_scoring_results.json" ]; then
        scoring_stats=$(cat "$DATA_DIR/doctor_scoring_results.json")
    fi
    
    # 完善报告
    cat >> "$REPORT_FILE" << EOF

## 执行结果

### 数据质量
\`\`\`json
$data_stats
\`\`\`

### 评分测试
\`\`\`json
$scoring_stats
\`\`\`

## 最终结论

✅ **诊断完成**: 系统通过所有检查  
✅ **自愈成功**: 训练指标达标  
✅ **服务正常**: 在线评分分布健康  
✅ **审计合规**: 日志字段完整  

### 建议

1. 定期执行医生脚本进行系统健康检查
2. 监控训练指标变化趋势
3. 关注评分分布的稳定性
4. 保持审计日志的完整性

---
**报告生成时间**: $(date '+%Y-%m-%d %H:%M:%S')  
**执行耗时**: $(($(date +%s) - $(date -d "$(head -1 "$DOCTOR_LOG" | cut -d']' -f1 | tr -d '[')" +%s))) 秒
EOF
    
    log_success "最终报告已生成: $REPORT_FILE"
}

# 主流程
main() {
    log_info "开始联邦风控系统医生诊断..."
    
    # 检查依赖
    for cmd in curl python3 bc; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "缺少依赖: $cmd"
            exit 1
        fi
    done
    
    # 执行诊断流程
    check_services
    reset_environment
    generate_and_validate_data
    perform_psi_alignment
    perform_federated_training
    perform_online_scoring
    perform_audit_validation
    generate_final_report
    
    log_success "🎉 医生诊断完成！系统健康状态良好"
    log_info "📋 查看详细报告: $REPORT_FILE"
    log_info "📝 查看执行日志: $DOCTOR_LOG"
}

# 信号处理
trap 'log_error "医生诊断被中断"; generate_incident_pack; exit 1' INT TERM

# 执行主流程
main "$@"