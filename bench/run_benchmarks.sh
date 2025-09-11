#!/bin/bash

# 联邦风控系统性能基准测试运行脚本

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

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker 未安装，某些测试可能无法运行"
    fi
    
    # 检查服务是否运行
    check_service "PSI服务" "http://localhost:8003/health"
    check_service "联邦编排服务" "http://localhost:8002/health"
    
    log_success "依赖检查完成"
}

check_service() {
    local service_name="$1"
    local health_url="$2"
    
    if curl -s "$health_url" > /dev/null 2>&1; then
        log_success "$service_name 运行正常"
    else
        log_warning "$service_name 未运行或不可访问"
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装测试依赖..."
    
    # 数据生成工具依赖
    if [ -f "data-gen/requirements.txt" ]; then
        log_info "安装数据生成工具依赖..."
        cd data-gen
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    # PSI测试依赖
    if [ -f "psi-bench/requirements.txt" ]; then
        log_info "安装PSI测试依赖..."
        cd psi-bench
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    # 联邦训练测试依赖
    if [ -f "train-bench/requirements.txt" ]; then
        log_info "安装联邦训练测试依赖..."
        cd train-bench
        python3 -m pip install -r requirements.txt
        cd ..
    fi
    
    log_success "依赖安装完成"
}

# 生成测试数据
generate_test_data() {
    log_info "生成测试数据..."
    
    cd data-gen
    
    # 生成小规模测试数据
    python3 generate_data.py --scenario quick --output-dir ../results/data/quick
    
    # 生成标准测试数据
    python3 generate_data.py --scenario standard --output-dir ../results/data/standard
    
    cd ..
    
    log_success "测试数据生成完成"
}

# 运行PSI性能测试
run_psi_benchmark() {
    log_info "运行PSI性能基准测试..."
    
    cd psi-bench
    
    # 快速测试
    if [ "$1" = "quick" ]; then
        log_info "运行PSI快速测试..."
        python3 psi_benchmark.py \
            --data-sizes 1000 5000 \
            --algorithms ecdh \
            --iterations 1
    # 标准测试
    elif [ "$1" = "standard" ]; then
        log_info "运行PSI标准测试..."
        python3 psi_benchmark.py \
            --data-sizes 1000 5000 10000 50000 \
            --algorithms ecdh token_join \
            --iterations 3
    # 压力测试
    elif [ "$1" = "stress" ]; then
        log_info "运行PSI压力测试..."
        python3 psi_benchmark.py \
            --data-sizes 100000 500000 1000000 \
            --algorithms ecdh token_join \
            --iterations 5
    else
        log_info "运行PSI完整测试套件..."
        python3 psi_benchmark.py
    fi
    
    cd ..
    
    log_success "PSI性能测试完成"
}

# 运行联邦训练性能测试
run_training_benchmark() {
    log_info "运行联邦训练性能基准测试..."
    
    cd train-bench
    
    # 快速测试
    if [ "$1" = "quick" ]; then
        log_info "运行联邦训练快速测试..."
        python3 train_benchmark.py \
            --participant-counts 2 3 \
            --data-sizes 1000 5000 \
            --model-complexities simple \
            --iterations 1 \
            --max-rounds 20
    # 标准测试
    elif [ "$1" = "standard" ]; then
        log_info "运行联邦训练标准测试..."
        python3 train_benchmark.py \
            --participant-counts 2 5 10 \
            --data-sizes 1000 5000 10000 \
            --model-complexities simple medium \
            --iterations 3 \
            --max-rounds 50
    # 可扩展性测试
    elif [ "$1" = "scalability" ]; then
        log_info "运行联邦训练可扩展性测试..."
        python3 train_benchmark.py \
            --participant-counts 2 5 10 20 50 \
            --data-sizes 10000 \
            --model-complexities medium \
            --iterations 3 \
            --max-rounds 30
    else
        log_info "运行联邦训练完整测试套件..."
        python3 train_benchmark.py
    fi
    
    cd ..
    
    log_success "联邦训练性能测试完成"
}

# 生成综合报告
generate_report() {
    log_info "生成综合性能报告..."
    
    # 创建报告目录
    mkdir -p results/reports
    
    # 生成时间戳
    timestamp=$(date +"%Y%m%d_%H%M%S")
    report_file="results/reports/benchmark_report_${timestamp}.md"
    
    # 生成报告头部
    cat > "$report_file" << EOF
# 联邦风控系统性能基准测试报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**测试环境**: $(uname -s) $(uname -r)
**Python版本**: $(python3 --version)

## 测试概览

本报告包含以下性能测试结果：
- PSI（隐私集合求交）性能测试
- 联邦训练性能测试
- 系统资源使用分析

## PSI性能测试结果

EOF
    
    # 添加PSI测试结果
    if [ -d "results/psi" ]; then
        echo "### PSI测试摘要" >> "$report_file"
        echo "" >> "$report_file"
        
        # 查找最新的PSI报告
        latest_psi_report=$(find results/psi -name "psi_report_*.json" -type f -exec ls -t {} + | head -n1)
        if [ -n "$latest_psi_report" ]; then
            echo "详细结果请参考: \`$latest_psi_report\`" >> "$report_file"
        fi
        echo "" >> "$report_file"
    fi
    
    # 添加联邦训练测试结果
    if [ -d "results/training" ]; then
        echo "## 联邦训练性能测试结果" >> "$report_file"
        echo "" >> "$report_file"
        echo "### 训练测试摘要" >> "$report_file"
        echo "" >> "$report_file"
        
        # 查找最新的训练报告
        latest_training_report=$(find results/training -name "training_report_*.json" -type f -exec ls -t {} + | head -n1)
        if [ -n "$latest_training_report" ]; then
            echo "详细结果请参考: \`$latest_training_report\`" >> "$report_file"
        fi
        echo "" >> "$report_file"
    fi
    
    # 添加系统信息
    cat >> "$report_file" << EOF
## 系统信息

- **操作系统**: $(uname -s) $(uname -r)
- **CPU**: $(python3 -c "import psutil; print(f'{psutil.cpu_count()} cores')")
- **内存**: $(python3 -c "import psutil; print(f'{psutil.virtual_memory().total // (1024**3)} GB')")
- **Python版本**: $(python3 --version)
- **测试时间**: $(date '+%Y-%m-%d %H:%M:%S')

## 建议

基于测试结果，建议：
1. 根据实际业务需求选择合适的PSI算法
2. 根据参与方数量和数据规模调整联邦训练参数
3. 监控系统资源使用情况，确保稳定运行
4. 定期进行性能基准测试，跟踪系统性能变化

EOF
    
    log_success "综合报告已生成: $report_file"
}

# 清理测试数据
cleanup() {
    log_info "清理测试数据..."
    
    # 清理临时文件
    find results -name "*.tmp" -delete 2>/dev/null || true
    find results -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
联邦风控系统性能基准测试工具

用法: $0 [选项] [测试类型]

选项:
  -h, --help              显示此帮助信息
  -i, --install           安装依赖
  -c, --check             检查依赖和服务状态
  -g, --generate-data     生成测试数据
  -r, --report            生成综合报告
  --cleanup               清理测试数据

测试类型:
  all                     运行所有测试（默认）
  psi                     只运行PSI性能测试
  training                只运行联邦训练性能测试
  quick                   运行快速测试
  standard                运行标准测试
  stress                  运行压力测试
  scalability             运行可扩展性测试

示例:
  $0                      # 运行所有测试
  $0 quick                # 运行快速测试
  $0 psi standard         # 运行PSI标准测试
  $0 -i                   # 安装依赖
  $0 -c                   # 检查环境
  $0 -g                   # 生成测试数据
  $0 -r                   # 生成报告

EOF
}

# 主函数
main() {
    local test_type="all"
    local test_level="standard"
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--install)
                install_dependencies
                exit 0
                ;;
            -c|--check)
                check_dependencies
                exit 0
                ;;
            -g|--generate-data)
                generate_test_data
                exit 0
                ;;
            -r|--report)
                generate_report
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            all|psi|training)
                test_type="$1"
                shift
                ;;
            quick|standard|stress|scalability)
                test_level="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 创建结果目录
    mkdir -p results/{data,psi,training,reports}
    
    log_info "开始性能基准测试..."
    log_info "测试类型: $test_type"
    log_info "测试级别: $test_level"
    
    # 检查依赖
    check_dependencies
    
    # 生成测试数据
    if [ "$test_level" != "quick" ]; then
        generate_test_data
    fi
    
    # 运行测试
    case $test_type in
        "all")
            run_psi_benchmark "$test_level"
            run_training_benchmark "$test_level"
            ;;
        "psi")
            run_psi_benchmark "$test_level"
            ;;
        "training")
            run_training_benchmark "$test_level"
            ;;
    esac
    
    # 生成报告
    generate_report
    
    log_success "所有测试完成！"
    log_info "查看结果目录: results/"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi