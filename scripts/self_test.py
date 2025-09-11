#!/usr/bin/env python3
"""
联邦风控系统自动化自测脚本

功能:
1. 服务健康检查
2. API接口测试
3. 核心功能验证
4. 性能基准测试
5. 安全性检查

使用方法:
python3 scripts/self_test.py [--verbose] [--module MODULE]
"""

import asyncio
import json
import time
import argparse
import requests
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/self_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Colors:
    """终端颜色常量"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class TestResult:
    """测试结果类"""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.timestamp = datetime.now()

class FederatedRiskTester:
    """联邦风控系统测试器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.services = {
            'consent-service': 'http://localhost:8000',
            'psi-service': 'http://localhost:8001',
            'model-trainer': 'http://localhost:8002',
            'model-explainer': 'http://localhost:8003',
            'frontend': 'http://localhost:5173'
        }
        
    def log(self, message: str, level: str = 'info'):
        """日志输出"""
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
            
        if self.verbose:
            color = Colors.WHITE
            if level == 'error':
                color = Colors.RED
            elif level == 'warning':
                color = Colors.YELLOW
            print(f"{color}{message}{Colors.END}")
    
    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.results.append(result)
        status_color = Colors.GREEN if result.passed else Colors.RED
        status_text = "PASS" if result.passed else "FAIL"
        
        print(f"[{status_color}{status_text}{Colors.END}] {result.name} ({result.duration:.2f}s)")
        if result.message:
            print(f"    {result.message}")
    
    def test_service_health(self, service_name: str, url: str) -> TestResult:
        """测试服务健康状态"""
        start_time = time.time()
        try:
            # 尝试多个健康检查端点
            health_endpoints = ['/health', '/docs', '/']
            
            for endpoint in health_endpoints:
                try:
                    response = requests.get(f"{url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:  # 404也表示服务在运行
                        duration = time.time() - start_time
                        return TestResult(
                            f"{service_name} 健康检查",
                            True,
                            f"服务正常运行 (状态码: {response.status_code})",
                            duration
                        )
                except requests.exceptions.RequestException:
                    continue
            
            # 所有端点都失败
            duration = time.time() - start_time
            return TestResult(
                f"{service_name} 健康检查",
                False,
                "服务无响应或不可访问",
                duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                f"{service_name} 健康检查",
                False,
                f"检查失败: {str(e)}",
                duration
            )
    
    def test_consent_service_api(self) -> List[TestResult]:
        """测试同意服务API"""
        results = []
        base_url = self.services['consent-service']
        
        # 测试创建同意记录
        start_time = time.time()
        try:
            consent_data = {
                "user_id": "test_user_001",
                "data_types": ["profile", "transaction"],
                "purposes": ["risk_assessment"],
                "retention_period": 365
            }
            
            response = requests.post(
                f"{base_url}/consent",
                json=consent_data,
                timeout=10
            )
            
            duration = time.time() - start_time
            if response.status_code in [200, 201]:
                results.append(TestResult(
                    "同意服务 - 创建同意记录",
                    True,
                    f"成功创建同意记录 (状态码: {response.status_code})",
                    duration
                ))
            else:
                results.append(TestResult(
                    "同意服务 - 创建同意记录",
                    False,
                    f"创建失败 (状态码: {response.status_code})",
                    duration
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                "同意服务 - 创建同意记录",
                False,
                f"请求失败: {str(e)}",
                duration
            ))
        
        # 测试查询同意状态
        start_time = time.time()
        try:
            response = requests.get(
                f"{base_url}/consent/test_user_001/status",
                timeout=10
            )
            
            duration = time.time() - start_time
            if response.status_code == 200:
                results.append(TestResult(
                    "同意服务 - 查询同意状态",
                    True,
                    "成功查询同意状态",
                    duration
                ))
            else:
                results.append(TestResult(
                    "同意服务 - 查询同意状态",
                    False,
                    f"查询失败 (状态码: {response.status_code})",
                    duration
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                "同意服务 - 查询同意状态",
                False,
                f"请求失败: {str(e)}",
                duration
            ))
        
        return results
    
    def test_psi_service_api(self) -> List[TestResult]:
        """测试PSI服务API"""
        results = []
        base_url = self.services['psi-service']
        
        # 测试创建PSI会话
        start_time = time.time()
        try:
            session_data = {
                "session_id": "test_session_001",
                "method": "ecdh_psi",
                "party_role": "sender",
                "party_id": "test_party_a"
            }
            
            response = requests.post(
                f"{base_url}/psi/session",
                json=session_data,
                timeout=10
            )
            
            duration = time.time() - start_time
            if response.status_code in [200, 201]:
                results.append(TestResult(
                    "PSI服务 - 创建会话",
                    True,
                    f"成功创建PSI会话 (状态码: {response.status_code})",
                    duration
                ))
            else:
                results.append(TestResult(
                    "PSI服务 - 创建会话",
                    False,
                    f"创建失败 (状态码: {response.status_code})",
                    duration
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                "PSI服务 - 创建会话",
                False,
                f"请求失败: {str(e)}",
                duration
            ))
        
        return results
    
    def test_model_trainer_api(self) -> List[TestResult]:
        """测试模型训练服务API"""
        results = []
        base_url = self.services['model-trainer']
        
        # 测试创建训练任务
        start_time = time.time()
        try:
            task_data = {
                "task_name": "test_risk_model",
                "algorithm": "secureboost",
                "participants": ["test_party_a", "test_party_b"],
                "privacy_budget": 1.0,
                "max_rounds": 5
            }
            
            response = requests.post(
                f"{base_url}/training/tasks",
                json=task_data,
                timeout=10
            )
            
            duration = time.time() - start_time
            if response.status_code in [200, 201]:
                results.append(TestResult(
                    "模型训练服务 - 创建训练任务",
                    True,
                    f"成功创建训练任务 (状态码: {response.status_code})",
                    duration
                ))
            else:
                results.append(TestResult(
                    "模型训练服务 - 创建训练任务",
                    False,
                    f"创建失败 (状态码: {response.status_code})",
                    duration
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                "模型训练服务 - 创建训练任务",
                False,
                f"请求失败: {str(e)}",
                duration
            ))
        
        return results
    
    def test_model_explainer_api(self) -> List[TestResult]:
        """测试模型解释服务API"""
        results = []
        base_url = self.services['model-explainer']
        
        # 测试SHAP解释
        start_time = time.time()
        try:
            explain_data = {
                "model_id": "test_model_001",
                "method": "shap",
                "sample_data": [[1, 2, 3, 4, 5]],
                "feature_names": ["feature1", "feature2", "feature3", "feature4", "feature5"]
            }
            
            response = requests.post(
                f"{base_url}/explain",
                json=explain_data,
                timeout=15
            )
            
            duration = time.time() - start_time
            if response.status_code in [200, 201]:
                results.append(TestResult(
                    "模型解释服务 - SHAP解释",
                    True,
                    f"成功生成模型解释 (状态码: {response.status_code})",
                    duration
                ))
            else:
                results.append(TestResult(
                    "模型解释服务 - SHAP解释",
                    False,
                    f"解释失败 (状态码: {response.status_code})",
                    duration
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            results.append(TestResult(
                "模型解释服务 - SHAP解释",
                False,
                f"请求失败: {str(e)}",
                duration
            ))
        
        return results
    
    def test_data_files(self) -> List[TestResult]:
        """测试数据文件完整性"""
        results = []
        
        # 检查合成数据文件
        data_files = [
            'data/synth/partyA_bank.csv',
            'data/synth/partyB_ecom.csv',
            'data/synth/metadata.json'
        ]
        
        for file_path in data_files:
            start_time = time.time()
            try:
                path = Path(file_path)
                if path.exists() and path.stat().st_size > 0:
                    duration = time.time() - start_time
                    results.append(TestResult(
                        f"数据文件检查 - {path.name}",
                        True,
                        f"文件存在且非空 (大小: {path.stat().st_size} bytes)",
                        duration
                    ))
                else:
                    duration = time.time() - start_time
                    results.append(TestResult(
                        f"数据文件检查 - {path.name}",
                        False,
                        "文件不存在或为空",
                        duration
                    ))
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    f"数据文件检查 - {file_path}",
                    False,
                    f"检查失败: {str(e)}",
                    duration
                ))
        
        return results
    
    def test_dependencies(self) -> List[TestResult]:
        """测试依赖包"""
        results = []
        
        # Python依赖检查
        python_deps = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 
            'sklearn', 'xgboost', 'shap', 'lime',
            'cryptography', 'redis', 'httpx'
        ]
        
        for dep in python_deps:
            start_time = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, '-c', f'import {dep}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                duration = time.time() - start_time
                if result.returncode == 0:
                    results.append(TestResult(
                        f"依赖检查 - {dep}",
                        True,
                        "依赖包可正常导入",
                        duration
                    ))
                else:
                    results.append(TestResult(
                        f"依赖检查 - {dep}",
                        False,
                        f"导入失败: {result.stderr}",
                        duration
                    ))
                    
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    f"依赖检查 - {dep}",
                    False,
                    f"检查失败: {str(e)}",
                    duration
                ))
        
        return results
    
    def run_performance_test(self) -> List[TestResult]:
        """运行性能测试"""
        results = []
        
        # API响应时间测试
        for service_name, url in self.services.items():
            if service_name == 'frontend':
                continue
                
            start_time = time.time()
            try:
                response = requests.get(f"{url}/docs", timeout=10)
                duration = time.time() - start_time
                
                if duration < 2.0:  # 2秒内响应
                    results.append(TestResult(
                        f"性能测试 - {service_name} 响应时间",
                        True,
                        f"响应时间: {duration:.2f}s (良好)",
                        duration
                    ))
                elif duration < 5.0:  # 5秒内响应
                    results.append(TestResult(
                        f"性能测试 - {service_name} 响应时间",
                        True,
                        f"响应时间: {duration:.2f}s (可接受)",
                        duration
                    ))
                else:
                    results.append(TestResult(
                        f"性能测试 - {service_name} 响应时间",
                        False,
                        f"响应时间: {duration:.2f}s (过慢)",
                        duration
                    ))
                    
            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    f"性能测试 - {service_name} 响应时间",
                    False,
                    f"测试失败: {str(e)}",
                    duration
                ))
        
        return results
    
    def run_all_tests(self, module: Optional[str] = None) -> Dict:
        """运行所有测试"""
        print(f"{Colors.BOLD}{Colors.BLUE}=== 联邦风控系统自动化测试 ==={Colors.END}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        test_modules = {
            'health': self.test_service_health,
            'consent': self.test_consent_service_api,
            'psi': self.test_psi_service_api,
            'trainer': self.test_model_trainer_api,
            'explainer': self.test_model_explainer_api,
            'data': self.test_data_files,
            'deps': self.test_dependencies,
            'performance': self.run_performance_test
        }
        
        if module and module in test_modules:
            # 运行指定模块
            if module == 'health':
                for service_name, url in self.services.items():
                    result = self.test_service_health(service_name, url)
                    self.add_result(result)
            else:
                results = test_modules[module]()
                for result in results:
                    self.add_result(result)
        else:
            # 运行所有测试
            print(f"{Colors.YELLOW}1. 服务健康检查{Colors.END}")
            for service_name, url in self.services.items():
                result = self.test_service_health(service_name, url)
                self.add_result(result)
            
            print(f"\n{Colors.YELLOW}2. API接口测试{Colors.END}")
            for test_func in [self.test_consent_service_api, self.test_psi_service_api, 
                            self.test_model_trainer_api, self.test_model_explainer_api]:
                results = test_func()
                for result in results:
                    self.add_result(result)
            
            print(f"\n{Colors.YELLOW}3. 数据文件检查{Colors.END}")
            results = self.test_data_files()
            for result in results:
                self.add_result(result)
            
            print(f"\n{Colors.YELLOW}4. 依赖包检查{Colors.END}")
            results = self.test_dependencies()
            for result in results:
                self.add_result(result)
            
            print(f"\n{Colors.YELLOW}5. 性能测试{Colors.END}")
            results = self.run_performance_test()
            for result in results:
                self.add_result(result)
        
        # 生成测试报告
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """生成测试报告"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== 测试报告 ==={Colors.END}")
        print(f"总测试数: {total_tests}")
        print(f"{Colors.GREEN}通过: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}失败: {failed_tests}{Colors.END}")
        print(f"成功率: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print(f"\n{Colors.RED}失败的测试:{Colors.END}")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
        
        # 保存详细报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests/total_tests*100
            },
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'duration': r.duration,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        # 保存到文件
        report_file = f"logs/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path('logs').mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_file}")
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='联邦风控系统自动化测试')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--module', '-m', choices=[
        'health', 'consent', 'psi', 'trainer', 'explainer', 
        'data', 'deps', 'performance'
    ], help='运行指定测试模块')
    
    args = parser.parse_args()
    
    # 创建日志目录
    Path('logs').mkdir(exist_ok=True)
    
    # 运行测试
    tester = FederatedRiskTester(verbose=args.verbose)
    report = tester.run_all_tests(args.module)
    
    # 返回适当的退出码
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()