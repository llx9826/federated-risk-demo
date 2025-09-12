#!/bin/bash

# 联邦风险评估系统 - 全服务压力测试脚本
# 此脚本将依次运行所有服务的压力测试

set -e  # 遇到错误时退出

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

# 检查k6是否安装
check_k6() {
    if ! command -v k6 &> /dev/null; then
        log_error "k6 未安装。请先安装 k6:"
        echo "  macOS: brew install k6"
        echo "  其他平台: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    log_info "k6 版本: $(k6 version)"
}

# 检查服务健康状态
check_service_health() {
    local service_name=$1
    local health_url=$2
    
    log_info "检查 ${service_name} 健康状态..."
    
    if curl -s -f "$health_url" > /dev/null 2>&1; then
        log_success "${service_name} 服务正常"
        return 0
    else
        log_warning "${service_name} 服务不可用 (${health_url})"
        return 1
    fi
}

# 运行单个压力测试
run_stress_test() {
    local test_name=$1
    local test_file=$2
    local output_dir=$3
    
    log_info "开始运行 ${test_name} 压力测试..."
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行测试
    local start_time=$(date +%s)
    if k6 run "$test_file" --out json="${output_dir}/${test_name}_results.json" 2>&1 | tee "${output_dir}/${test_name}_output.log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "${test_name} 测试完成 (耗时: ${duration}秒)"
        return 0
    else
        log_error "${test_name} 测试失败"
        return 1
    fi
}

# 生成综合报告
generate_summary_report() {
    local output_dir=$1
    local report_file="${output_dir}/comprehensive_stress_report.md"
    
    log_info "生成综合测试报告..."
    
    cat > "$report_file" << EOF
# 联邦风险评估系统 - 压力测试综合报告

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 测试概览

本次压力测试涵盖了联邦风险评估系统的所有核心服务:

- **同意服务** (Consent Service) - 端口 8000
- **PSI服务** (Private Set Intersection) - 端口 8001  
- **训练服务** (Training Service) - 端口 8002
- **推理服务** (Serving Service) - 端口 8003

## 服务健康状态

EOF

    # 检查各服务状态并写入报告
    local services=("同意服务:http://localhost:8000/health" "PSI服务:http://localhost:8001/health" "训练服务:http://localhost:8002/health" "推理服务:http://localhost:8003/health")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service"
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "- ✅ ${name}: 正常运行" >> "$report_file"
        else
            echo "- ❌ ${name}: 服务不可用" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## 测试结果摘要

### 同意服务压力测试
- 测试文件: consent_stress.js
- 重点测试: 同意创建、查询、更新、撤销、审计日志
- 性能目标: 95%响应时间 < 1500ms, 错误率 < 2%

### PSI服务压力测试  
- 测试文件: psi_stress.js
- 重点测试: 会话创建、PSI计算、健康检查
- 性能目标: 95%响应时间 < 3000ms, 错误率 < 5%

### 训练服务压力测试
- 测试文件: training_stress.js  
- 重点测试: 训练启动、状态查询、模型检索、质量验证
- 性能目标: 95%响应时间 < 5000ms, 非退化模型率 > 80%

### 推理服务压力测试
- 测试文件: serving_stress.js
- 重点测试: 单个预测、批量预测、模型信息查询
- 性能目标: 95%响应时间 < 2000ms, 预测有效率 > 90%

### 综合压力测试
- 测试文件: comprehensive_stress.js
- 重点测试: 跨服务协调、端到端流程
- 性能目标: 整体系统稳定性和一致性

## 详细结果

各服务的详细测试结果请查看对应的输出文件:

EOF

    # 列出所有结果文件
    if ls "${output_dir}"/*_results.json > /dev/null 2>&1; then
        echo "### 测试结果文件" >> "$report_file"
        for file in "${output_dir}"/*_results.json; do
            if [[ -f "$file" ]]; then
                local basename=$(basename "$file" .json)
                echo "- \`${basename}.json\` - JSON格式详细数据" >> "$report_file"
                echo "- \`${basename/results/summary}.txt\` - 文本格式摘要报告" >> "$report_file"
            fi
        done
    fi
    
    cat >> "$report_file" << EOF

## 性能建议

基于压力测试结果，建议关注以下方面:

1. **响应时间优化**: 如果某些服务的95%响应时间超过阈值，考虑:
   - 优化数据库查询
   - 增加缓存层
   - 扩展服务实例

2. **错误率控制**: 如果错误率过高，检查:
   - 服务日志中的具体错误
   - 资源限制和配置
   - 网络连接稳定性

3. **模型质量保证**: 对于训练和推理服务:
   - 确保模型非退化
   - 验证预测结果有效性
   - 监控模型性能指标

4. **合规性维护**: 对于同意服务:
   - 确保审计日志完整
   - 验证撤销功能正常
   - 保持数据保护合规

## 后续行动

- [ ] 分析性能瓶颈并制定优化计划
- [ ] 根据测试结果调整服务配置
- [ ] 建立持续性能监控
- [ ] 定期重复压力测试以验证改进效果

---

*此报告由自动化压力测试脚本生成*
EOF

    log_success "综合报告已生成: $report_file"
}

# 主函数
main() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local output_dir="${script_dir}/stress_test_results_${timestamp}"
    
    log_info "=== 联邦风险评估系统压力测试 ==="
    log_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "输出目录: $output_dir"
    
    # 检查k6
    check_k6
    
    # 检查服务健康状态
    log_info "\n=== 服务健康检查 ==="
    local services_ok=true
    
    check_service_health "同意服务" "http://localhost:8000/health" || services_ok=false
    check_service_health "PSI服务" "http://localhost:8001/health" || services_ok=false
    check_service_health "训练服务" "http://localhost:8002/health" || services_ok=false
    check_service_health "推理服务" "http://localhost:8003/health" || services_ok=false
    
    if [[ "$services_ok" != "true" ]]; then
        log_warning "部分服务不可用，但将继续运行测试"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "测试已取消"
            exit 1
        fi
    fi
    
    # 运行压力测试
    log_info "\n=== 开始压力测试 ==="
    
    local tests_passed=0
    local tests_total=0
    
    # 定义测试列表
    local tests=(
        "consent_stress:${script_dir}/consent_stress.js"
        "psi_stress:${script_dir}/psi_stress.js"
        "training_stress:${script_dir}/training_stress.js"
        "serving_stress:${script_dir}/serving_stress.js"
        "comprehensive_stress:${script_dir}/comprehensive_stress.js"
    )
    
    for test in "${tests[@]}"; do
        IFS=':' read -r test_name test_file <<< "$test"
        
        if [[ -f "$test_file" ]]; then
            ((tests_total++))
            if run_stress_test "$test_name" "$test_file" "$output_dir"; then
                ((tests_passed++))
            fi
            
            # 测试间隔，避免服务过载
            if [[ $tests_total -lt ${#tests[@]} ]]; then
                log_info "等待30秒后继续下一个测试..."
                sleep 30
            fi
        else
            log_warning "测试文件不存在: $test_file"
        fi
    done
    
    # 生成综合报告
    log_info "\n=== 生成测试报告 ==="
    generate_summary_report "$output_dir"
    
    # 测试总结
    log_info "\n=== 测试完成 ==="
    log_info "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "测试通过: ${tests_passed}/${tests_total}"
    log_info "结果目录: $output_dir"
    
    if [[ $tests_passed -eq $tests_total ]]; then
        log_success "所有压力测试已成功完成!"
        exit 0
    else
        log_warning "部分测试失败，请检查详细日志"
        exit 1
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi