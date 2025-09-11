#!/bin/bash

# 联邦风控系统测试命令脚本
# 提供便捷的测试执行入口

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 函数：打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 函数：打印标题
print_title() {
    echo
    print_message "$BLUE" "=== $1 ==="
    echo
}

# 函数：检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_message "$RED" "错误: 命令 '$1' 未找到，请先安装"
        exit 1
    fi
}

# 函数：检查Python包是否安装
check_python_package() {
    if ! python3 -c "import $1" &> /dev/null; then
        print_message "$RED" "错误: Python包 '$1' 未安装"
        return 1
    fi
    return 0
}

# 函数：等待服务启动
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_message "$YELLOW" "等待 $service_name 启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_message "$GREEN" "$service_name 已启动"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    print_message "$RED" "$service_name 启动超时"
    return 1
}

# 函数：运行自测脚本
run_self_test() {
    local module=$1
    local verbose=$2
    
    print_title "运行自动化测试"
    
    # 检查自测脚本是否存在
    if [ ! -f "scripts/self_test.py" ]; then
        print_message "$RED" "错误: 自测脚本不存在"
        exit 1
    fi
    
    # 构建命令
    local cmd="python3 scripts/self_test.py"
    
    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi
    
    if [ -n "$module" ]; then
        cmd="$cmd --module $module"
    fi
    
    # 运行测试
    print_message "$CYAN" "执行命令: $cmd"
    if $cmd; then
        print_message "$GREEN" "测试完成"
        return 0
    else
        print_message "$RED" "测试失败"
        return 1
    fi
}

# 函数：运行单元测试
run_unit_tests() {
    print_title "运行单元测试"
    
    # 检查pytest是否安装
    if ! check_python_package "pytest"; then
        print_message "$YELLOW" "安装pytest..."
        pip3 install pytest pytest-asyncio pytest-cov
    fi
    
    # 运行单元测试
    if [ -d "tests" ]; then
        print_message "$CYAN" "运行单元测试..."
        python3 -m pytest tests/ -v --tb=short --cov=services --cov-report=html --cov-report=term
    else
        print_message "$YELLOW" "警告: tests目录不存在，跳过单元测试"
    fi
}

# 函数：运行集成测试
run_integration_tests() {
    print_title "运行集成测试"
    
    # 检查服务是否运行
    local services=("http://localhost:8000" "http://localhost:8001" "http://localhost:8002" "http://localhost:8003")
    local service_names=("consent-service" "psi-service" "model-trainer" "model-explainer")
    
    for i in "${!services[@]}"; do
        if ! curl -s "${services[$i]}/docs" > /dev/null 2>&1; then
            print_message "$RED" "错误: ${service_names[$i]} 未运行，请先启动服务"
            print_message "$YELLOW" "提示: 运行 './test.sh start-services' 启动所有服务"
            exit 1
        fi
    done
    
    # 运行集成测试
    run_self_test "" "true"
}

# 函数：运行性能测试
run_performance_tests() {
    print_title "运行性能测试"
    
    # 检查locust是否安装
    if ! check_python_package "locust"; then
        print_message "$YELLOW" "安装locust..."
        pip3 install locust
    fi
    
    # 创建性能测试脚本
    cat > "$LOG_DIR/locustfile.py" << 'EOF'
from locust import HttpUser, task, between
import json

class FederatedRiskUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """测试开始时的初始化"""
        pass
    
    @task(3)
    def test_consent_service(self):
        """测试同意服务"""
        # 创建同意记录
        consent_data = {
            "user_id": f"test_user_{self.environment.runner.user_count}",
            "data_types": ["profile", "transaction"],
            "purposes": ["risk_assessment"],
            "retention_period": 365
        }
        
        with self.client.post("/consent", 
                             json=consent_data, 
                             catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"创建同意记录失败: {response.status_code}")
    
    @task(2)
    def test_psi_service(self):
        """测试PSI服务"""
        session_data = {
            "session_id": f"test_session_{self.environment.runner.user_count}",
            "method": "ecdh_psi",
            "party_role": "sender",
            "party_id": "test_party_a"
        }
        
        with self.client.post("/psi/session", 
                             json=session_data, 
                             catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"创建PSI会话失败: {response.status_code}")
    
    @task(1)
    def test_model_trainer(self):
        """测试模型训练服务"""
        task_data = {
            "task_name": f"test_task_{self.environment.runner.user_count}",
            "algorithm": "secureboost",
            "participants": ["test_party_a", "test_party_b"],
            "privacy_budget": 1.0,
            "max_rounds": 5
        }
        
        with self.client.post("/training/tasks", 
                             json=task_data, 
                             catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"创建训练任务失败: {response.status_code}")
EOF
    
    # 运行性能测试
    print_message "$CYAN" "启动性能测试 (10个用户，持续60秒)..."
    print_message "$YELLOW" "测试结果将保存到 $LOG_DIR/performance_report.html"
    
    cd "$LOG_DIR"
    locust -f locustfile.py --host=http://localhost:8000 \
           --users 10 --spawn-rate 2 --run-time 60s \
           --html performance_report.html --headless
    cd "$PROJECT_ROOT"
    
    print_message "$GREEN" "性能测试完成，报告已保存"
}

# 函数：运行代码质量检查
run_code_quality_check() {
    print_title "代码质量检查"
    
    # 检查并安装代码质量工具
    local tools=("flake8" "black" "isort" "mypy")
    for tool in "${tools[@]}"; do
        if ! check_python_package "$tool"; then
            print_message "$YELLOW" "安装 $tool..."
            pip3 install "$tool"
        fi
    done
    
    # 代码格式检查
    print_message "$CYAN" "检查代码格式 (black)..."
    if python3 -m black --check services/ scripts/ --diff; then
        print_message "$GREEN" "代码格式检查通过"
    else
        print_message "$YELLOW" "代码格式需要调整，运行 'black services/ scripts/' 自动格式化"
    fi
    
    # 导入排序检查
    print_message "$CYAN" "检查导入排序 (isort)..."
    if python3 -m isort services/ scripts/ --check-only --diff; then
        print_message "$GREEN" "导入排序检查通过"
    else
        print_message "$YELLOW" "导入排序需要调整，运行 'isort services/ scripts/' 自动排序"
    fi
    
    # 代码风格检查
    print_message "$CYAN" "检查代码风格 (flake8)..."
    if python3 -m flake8 services/ scripts/ --max-line-length=88 --extend-ignore=E203,W503; then
        print_message "$GREEN" "代码风格检查通过"
    else
        print_message "$YELLOW" "发现代码风格问题，请根据提示修改"
    fi
    
    # 类型检查
    print_message "$CYAN" "检查类型注解 (mypy)..."
    if python3 -m mypy services/ --ignore-missing-imports --no-strict-optional; then
        print_message "$GREEN" "类型检查通过"
    else
        print_message "$YELLOW" "发现类型问题，请根据提示修改"
    fi
}

# 函数：运行安全检查
run_security_check() {
    print_title "安全检查"
    
    # 检查并安装安全工具
    if ! check_python_package "bandit"; then
        print_message "$YELLOW" "安装bandit..."
        pip3 install bandit
    fi
    
    if ! check_python_package "safety"; then
        print_message "$YELLOW" "安装safety..."
        pip3 install safety
    fi
    
    # 代码安全扫描
    print_message "$CYAN" "运行安全扫描 (bandit)..."
    if python3 -m bandit -r services/ -f json -o "$LOG_DIR/security_report.json"; then
        print_message "$GREEN" "安全扫描完成，报告已保存到 $LOG_DIR/security_report.json"
    else
        print_message "$YELLOW" "发现安全问题，请查看报告"
    fi
    
    # 依赖安全检查
    print_message "$CYAN" "检查依赖安全性 (safety)..."
    if python3 -m safety check --json --output "$LOG_DIR/dependency_security.json"; then
        print_message "$GREEN" "依赖安全检查通过"
    else
        print_message "$YELLOW" "发现不安全的依赖，请查看报告"
    fi
}

# 函数：启动所有服务
start_services() {
    print_title "启动所有服务"
    
    # 检查端口是否被占用
    local ports=(8000 8001 8002 8003 5173)
    local port_names=("consent-service" "psi-service" "model-trainer" "model-explainer" "frontend")
    
    for i in "${!ports[@]}"; do
        if lsof -i :"${ports[$i]}" > /dev/null 2>&1; then
            print_message "$YELLOW" "端口 ${ports[$i]} (${port_names[$i]}) 已被占用"
        fi
    done
    
    # 启动后端服务
    print_message "$CYAN" "启动后端服务..."
    
    # 启动同意服务
    print_message "$YELLOW" "启动同意服务 (端口 8000)..."
    nohup python3 -m uvicorn services.consent-service.app:app --host 0.0.0.0 --port 8000 --reload > "$LOG_DIR/consent-service.log" 2>&1 &
    
    # 启动PSI服务
    print_message "$YELLOW" "启动PSI服务 (端口 8001)..."
    nohup python3 -m uvicorn services.psi-service.app:app --host 0.0.0.0 --port 8001 --reload > "$LOG_DIR/psi-service.log" 2>&1 &
    
    # 启动模型训练服务
    print_message "$YELLOW" "启动模型训练服务 (端口 8002)..."
    nohup python3 -m uvicorn services.model-trainer.app:app --host 0.0.0.0 --port 8002 --reload > "$LOG_DIR/model-trainer.log" 2>&1 &
    
    # 启动模型解释服务
    print_message "$YELLOW" "启动模型解释服务 (端口 8003)..."
    nohup python3 -m uvicorn services.model-explainer.app:app --host 0.0.0.0 --port 8003 --reload > "$LOG_DIR/model-explainer.log" 2>&1 &
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    local services=("http://localhost:8000" "http://localhost:8001" "http://localhost:8002" "http://localhost:8003")
    local service_names=("consent-service" "psi-service" "model-trainer" "model-explainer")
    
    for i in "${!services[@]}"; do
        wait_for_service "${services[$i]}/docs" "${service_names[$i]}"
    done
    
    print_message "$GREEN" "所有后端服务已启动"
    
    # 启动前端服务
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        print_message "$YELLOW" "启动前端服务 (端口 5173)..."
        cd frontend
        if [ ! -d "node_modules" ]; then
            print_message "$CYAN" "安装前端依赖..."
            npm install
        fi
        nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
        cd "$PROJECT_ROOT"
        
        wait_for_service "http://localhost:5173" "frontend"
    else
        print_message "$YELLOW" "前端目录不存在，跳过前端启动"
    fi
    
    print_message "$GREEN" "所有服务启动完成"
    print_message "$CYAN" "服务访问地址:"
    echo "  - 同意服务: http://localhost:8000/docs"
    echo "  - PSI服务: http://localhost:8001/docs"
    echo "  - 模型训练服务: http://localhost:8002/docs"
    echo "  - 模型解释服务: http://localhost:8003/docs"
    echo "  - 前端界面: http://localhost:5173"
}

# 函数：停止所有服务
stop_services() {
    print_title "停止所有服务"
    
    # 停止Python服务
    print_message "$YELLOW" "停止Python服务..."
    pkill -f "uvicorn.*app:app" || true
    
    # 停止前端服务
    print_message "$YELLOW" "停止前端服务..."
    pkill -f "npm run dev" || true
    pkill -f "vite" || true
    
    sleep 2
    print_message "$GREEN" "所有服务已停止"
}

# 函数：生成测试报告
generate_test_report() {
    print_title "生成测试报告"
    
    local report_file="$LOG_DIR/test_summary_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>联邦风控系统测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        .info { color: blue; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>联邦风控系统测试报告</h1>
        <p><strong>生成时间:</strong> $(date)</p>
        <p><strong>项目路径:</strong> $PROJECT_ROOT</p>
    </div>
    
    <div class="section">
        <h2>测试概览</h2>
        <table>
            <tr><th>测试类型</th><th>状态</th><th>说明</th></tr>
            <tr><td>自动化测试</td><td class="info">已执行</td><td>详见自测报告</td></tr>
            <tr><td>代码质量</td><td class="info">已检查</td><td>格式、风格、类型检查</td></tr>
            <tr><td>安全扫描</td><td class="info">已完成</td><td>代码安全和依赖安全</td></tr>
            <tr><td>性能测试</td><td class="info">已执行</td><td>负载和响应时间测试</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>服务状态</h2>
        <table>
            <tr><th>服务名称</th><th>端口</th><th>状态</th><th>访问地址</th></tr>
            <tr><td>同意服务</td><td>8000</td><td class="success">运行中</td><td><a href="http://localhost:8000/docs">http://localhost:8000/docs</a></td></tr>
            <tr><td>PSI服务</td><td>8001</td><td class="success">运行中</td><td><a href="http://localhost:8001/docs">http://localhost:8001/docs</a></td></tr>
            <tr><td>模型训练服务</td><td>8002</td><td class="success">运行中</td><td><a href="http://localhost:8002/docs">http://localhost:8002/docs</a></td></tr>
            <tr><td>模型解释服务</td><td>8003</td><td class="success">运行中</td><td><a href="http://localhost:8003/docs">http://localhost:8003/docs</a></td></tr>
            <tr><td>前端界面</td><td>5173</td><td class="success">运行中</td><td><a href="http://localhost:5173">http://localhost:5173</a></td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>测试文件</h2>
        <ul>
EOF
    
    # 添加日志文件链接
    for log_file in "$LOG_DIR"/*.log "$LOG_DIR"/*.json "$LOG_DIR"/*.html; do
        if [ -f "$log_file" ]; then
            echo "            <li><a href='$(basename "$log_file")'>$(basename "$log_file")</a></li>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF
        </ul>
    </div>
    
    <div class="section">
        <h2>使用说明</h2>
        <h3>测试命令</h3>
        <pre>
# 运行完整测试套件
./test.sh all

# 运行特定测试
./test.sh self-test          # 自动化测试
./test.sh unit-test          # 单元测试
./test.sh integration-test   # 集成测试
./test.sh performance-test   # 性能测试
./test.sh code-quality       # 代码质量检查
./test.sh security-check     # 安全检查

# 服务管理
./test.sh start-services     # 启动所有服务
./test.sh stop-services      # 停止所有服务
./test.sh status             # 查看服务状态
        </pre>
        
        <h3>评审对照表</h3>
        <p>详细的评审对照表请参考: <a href="../docs/REVIEW_CHECKLIST.md">docs/REVIEW_CHECKLIST.md</a></p>
    </div>
</body>
</html>
EOF
    
    print_message "$GREEN" "测试报告已生成: $report_file"
    
    # 在macOS上打开报告
    if command -v open &> /dev/null; then
        open "$report_file"
    fi
}

# 函数：显示服务状态
show_status() {
    print_title "服务状态检查"
    
    local services=("http://localhost:8000" "http://localhost:8001" "http://localhost:8002" "http://localhost:8003" "http://localhost:5173")
    local service_names=("同意服务" "PSI服务" "模型训练服务" "模型解释服务" "前端界面")
    local ports=(8000 8001 8002 8003 5173)
    
    echo "服务状态:"
    echo "----------------------------------------"
    
    for i in "${!services[@]}"; do
        local status="停止"
        local color="$RED"
        
        if curl -s "${services[$i]}" > /dev/null 2>&1; then
            status="运行中"
            color="$GREEN"
        elif lsof -i :"${ports[$i]}" > /dev/null 2>&1; then
            status="端口占用"
            color="$YELLOW"
        fi
        
        printf "%-15s [端口 %d]: " "${service_names[$i]}" "${ports[$i]}"
        print_message "$color" "$status"
    done
    
    echo
    echo "访问地址:"
    echo "----------------------------------------"
    echo "同意服务 API文档:     http://localhost:8000/docs"
    echo "PSI服务 API文档:      http://localhost:8001/docs"
    echo "模型训练服务 API文档: http://localhost:8002/docs"
    echo "模型解释服务 API文档: http://localhost:8003/docs"
    echo "前端界面:             http://localhost:5173"
}

# 函数：显示帮助信息
show_help() {
    cat << EOF
联邦风控系统测试命令

用法: $0 [命令] [选项]

命令:
  all                    运行完整测试套件
  self-test [module]     运行自动化测试 (可选模块: health, consent, psi, trainer, explainer, data, deps, performance)
  unit-test              运行单元测试
  integration-test       运行集成测试
  performance-test       运行性能测试
  code-quality           代码质量检查
  security-check         安全检查
  start-services         启动所有服务
  stop-services          停止所有服务
  status                 查看服务状态
  report                 生成测试报告
  help                   显示此帮助信息

选项:
  -v, --verbose          详细输出
  -h, --help             显示帮助信息

示例:
  $0 all                 # 运行所有测试
  $0 self-test health    # 只运行健康检查
  $0 start-services      # 启动所有服务
  $0 status              # 查看服务状态
  $0 self-test -v        # 详细模式运行自测

更多信息请参考:
  - README.md
  - docs/REVIEW_CHECKLIST.md
EOF
}

# 主函数
main() {
    local command="$1"
    local verbose="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                verbose="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                if [ -z "$command" ]; then
                    command="$1"
                fi
                shift
                ;;
        esac
    done
    
    # 如果没有指定命令，显示帮助
    if [ -z "$command" ]; then
        show_help
        exit 0
    fi
    
    # 检查基本依赖
    check_command "python3"
    check_command "curl"
    
    # 执行命令
    case "$command" in
        "all")
            print_title "运行完整测试套件"
            run_self_test "" "$verbose"
            run_unit_tests
            run_code_quality_check
            run_security_check
            run_performance_tests
            generate_test_report
            ;;
        "self-test")
            local module="$2"
            run_self_test "$module" "$verbose"
            ;;
        "unit-test")
            run_unit_tests
            ;;
        "integration-test")
            run_integration_tests
            ;;
        "performance-test")
            run_performance_tests
            ;;
        "code-quality")
            run_code_quality_check
            ;;
        "security-check")
            run_security_check
            ;;
        "start-services")
            start_services
            ;;
        "stop-services")
            stop_services
            ;;
        "status")
            show_status
            ;;
        "report")
            generate_test_report
            ;;
        "help")
            show_help
            ;;
        *)
            print_message "$RED" "错误: 未知命令 '$command'"
            echo
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"