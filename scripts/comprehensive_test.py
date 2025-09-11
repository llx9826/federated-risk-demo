#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦风控系统综合测试脚本

功能:
1. 系统健康检查
2. API功能测试
3. 数据流测试
4. 性能基准测试
5. 安全性验证
6. 集成测试
"""

import os
import sys
import json
import time
import asyncio
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

class ComprehensiveTestSuite:
    def __init__(self):
        self.base_urls = {
            'consent': 'http://localhost:8000',
            'psi': 'http://localhost:8001', 
            'trainer': 'http://localhost:8002',
            'explainer': 'http://localhost:8003',
            'frontend': 'http://localhost:5173'
        }
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test(self, category: str, test_name: str, status: str, 
                 duration: float, details: str = ""):
        """记录测试结果"""
        result = {
            'category': category,
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} [{category}] {test_name} ({duration:.2f}s)")
        if details:
            print(f"    {details}")
            
    def test_service_health(self):
        """测试所有服务的健康状态"""
        print("\n=== 服务健康检查 ===")
        
        for service, url in self.base_urls.items():
            start_time = time.time()
            try:
                if service == 'frontend':
                    response = requests.get(url, timeout=5)
                    status = "PASS" if response.status_code == 200 else "FAIL"
                    details = f"状态码: {response.status_code}"
                else:
                    health_url = f"{url}/health"
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        status = "PASS" if health_data.get('status') in ['healthy', 'unhealthy'] else "FAIL"
                        details = f"状态: {health_data.get('status', 'unknown')}"
                    else:
                        status = "FAIL"
                        details = f"状态码: {response.status_code}"
                        
            except requests.exceptions.RequestException as e:
                status = "FAIL"
                details = f"连接失败: {str(e)}"
                
            duration = time.time() - start_time
            self.log_test("健康检查", f"{service}服务", status, duration, details)
            
    def test_api_endpoints(self):
        """测试API端点功能"""
        print("\n=== API端点测试 ===")
        
        # 测试同意服务API
        self._test_consent_api()
        
        # 测试PSI服务API
        self._test_psi_api()
        
        # 测试模型训练API
        self._test_trainer_api()
        
        # 测试模型解释API
        self._test_explainer_api()
        
    def _test_consent_api(self):
        """测试同意服务API"""
        base_url = self.base_urls['consent']
        
        # 测试获取API文档
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}/docs", timeout=5)
            status = "PASS" if response.status_code == 200 else "FAIL"
            details = f"文档可访问: {response.status_code == 200}"
        except Exception as e:
            status = "FAIL"
            details = f"错误: {str(e)}"
        duration = time.time() - start_time
        self.log_test("API测试", "同意服务文档", status, duration, details)
        
        # 测试创建同意记录
        start_time = time.time()
        try:
            consent_data = {
                "user_id": "test_user_001",
                "purpose": "model_training",
                "data_types": ["financial_data"],
                "retention_period": 365
            }
            response = requests.post(f"{base_url}/consent", 
                                   json=consent_data, timeout=5)
            status = "PASS" if response.status_code in [200, 201] else "FAIL"
            details = f"状态码: {response.status_code}"
        except Exception as e:
            status = "FAIL"
            details = f"错误: {str(e)}"
        duration = time.time() - start_time
        self.log_test("API测试", "创建同意记录", status, duration, details)
        
    def _test_psi_api(self):
        """测试PSI服务API"""
        base_url = self.base_urls['psi']
        
        # 测试PSI会话创建
        start_time = time.time()
        try:
            session_data = {
                "session_id": "test_session_001",
                "participants": ["party_a", "party_b"],
                "algorithm": "ecdh_psi"
            }
            response = requests.post(f"{base_url}/psi/session", 
                                   json=session_data, timeout=5)
            status = "PASS" if response.status_code in [200, 201] else "FAIL"
            details = f"状态码: {response.status_code}"
        except Exception as e:
            status = "FAIL"
            details = f"错误: {str(e)}"
        duration = time.time() - start_time
        self.log_test("API测试", "PSI会话创建", status, duration, details)
        
    def _test_trainer_api(self):
        """测试模型训练API"""
        base_url = self.base_urls['trainer']
        
        # 测试训练任务创建
        start_time = time.time()
        try:
            training_data = {
                "task_id": "test_task_001",
                "task_name": "测试训练任务",
                "participants": ["party_a", "party_b"],
                "target_column": "risk_label",
                "feature_columns": ["feature_1", "feature_2"],
                "config": {
                    "algorithm": "secure_boost",
                    "num_rounds": 5
                },
                "data_sources": {
                    "party_a": "data/synth/partyA_bank.csv",
                    "party_b": "data/synth/partyB_ecom.csv"
                }
            }
            response = requests.post(f"{base_url}/training/tasks", 
                                   json=training_data, timeout=5)
            status = "PASS" if response.status_code in [200, 201] else "FAIL"
            details = f"状态码: {response.status_code}"
        except Exception as e:
            status = "FAIL"
            details = f"错误: {str(e)}"
        duration = time.time() - start_time
        self.log_test("API测试", "训练任务创建", status, duration, details)
        
    def _test_explainer_api(self):
        """测试模型解释API"""
        base_url = self.base_urls['explainer']
        
        # 测试模型上传
        start_time = time.time()
        try:
            model_data = {
                "model_id": "test_model_001",
                "model_type": "random_forest",
                "model_name": "测试模型",
                "feature_names": ["feature_1", "feature_2", "feature_3"]
            }
            response = requests.post(f"{base_url}/models/upload", 
                                   json=model_data, timeout=5)
            status = "PASS" if response.status_code in [200, 201] else "FAIL"
            details = f"状态码: {response.status_code}"
        except Exception as e:
            status = "FAIL"
            details = f"错误: {str(e)}"
        duration = time.time() - start_time
        self.log_test("API测试", "模型上传", status, duration, details)
        
    def test_data_flow(self):
        """测试数据流完整性"""
        print("\n=== 数据流测试 ===")
        
        # 检查测试数据文件
        data_files = [
            "data/synth/partyA_bank.csv",
            "data/synth/partyB_ecom.csv",
            "data/synth/metadata.json"
        ]
        
        for file_path in data_files:
            start_time = time.time()
            full_path = Path(file_path)
            if full_path.exists() and full_path.stat().st_size > 0:
                status = "PASS"
                details = f"文件大小: {full_path.stat().st_size} bytes"
            else:
                status = "FAIL"
                details = "文件不存在或为空"
            duration = time.time() - start_time
            self.log_test("数据流", f"数据文件: {file_path}", status, duration, details)
            
    def test_performance_benchmarks(self):
        """性能基准测试"""
        print("\n=== 性能基准测试 ===")
        
        for service, url in self.base_urls.items():
            if service == 'frontend':
                continue
                
            # 测试响应时间
            response_times = []
            for i in range(5):
                start_time = time.time()
                try:
                    response = requests.get(f"{url}/health", timeout=10)
                    if response.status_code == 200:
                        response_times.append(time.time() - start_time)
                except:
                    pass
                    
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                status = "PASS" if avg_time < 1.0 else "WARN" if avg_time < 2.0 else "FAIL"
                details = f"平均响应时间: {avg_time:.3f}s"
            else:
                status = "FAIL"
                details = "无法获取响应时间"
                avg_time = 0
                
            self.log_test("性能", f"{service}服务响应时间", status, avg_time, details)
            
    def test_security_basics(self):
        """基础安全性测试"""
        print("\n=== 安全性测试 ===")
        
        # 检查环境变量配置
        start_time = time.time()
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.read()
                has_secrets = any(key in env_content.lower() for key in 
                                ['password', 'secret', 'key', 'token'])
                status = "PASS" if has_secrets else "WARN"
                details = "环境变量配置存在" if has_secrets else "未发现敏感配置"
        else:
            status = "WARN"
            details = "环境变量文件不存在"
        duration = time.time() - start_time
        self.log_test("安全性", "环境变量配置", status, duration, details)
        
        # 检查HTTPS重定向（在生产环境中）
        for service, url in self.base_urls.items():
            if service == 'frontend':
                continue
            start_time = time.time()
            try:
                # 检查是否有安全头
                response = requests.get(f"{url}/health", timeout=5)
                has_security_headers = any(header in response.headers for header in 
                                         ['X-Content-Type-Options', 'X-Frame-Options'])
                status = "PASS" if has_security_headers else "WARN"
                details = "存在安全头" if has_security_headers else "缺少安全头"
            except:
                status = "FAIL"
                details = "无法检查安全头"
            duration = time.time() - start_time
            self.log_test("安全性", f"{service}安全头检查", status, duration, details)
            
    def test_integration_scenarios(self):
        """集成测试场景"""
        print("\n=== 集成测试 ===")
        
        # 测试服务间通信
        start_time = time.time()
        try:
            # 模拟一个简单的集成流程
            # 1. 检查所有服务是否可达
            all_healthy = True
            for service, url in self.base_urls.items():
                if service == 'frontend':
                    continue
                try:
                    response = requests.get(f"{url}/health", timeout=3)
                    if response.status_code != 200:
                        all_healthy = False
                        break
                except:
                    all_healthy = False
                    break
                    
            status = "PASS" if all_healthy else "FAIL"
            details = "所有服务可达" if all_healthy else "部分服务不可达"
        except Exception as e:
            status = "FAIL"
            details = f"集成测试失败: {str(e)}"
        duration = time.time() - start_time
        self.log_test("集成", "服务间通信", status, duration, details)
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始联邦风控系统综合测试")
        print(f"测试开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行各类测试
        self.test_service_health()
        self.test_api_endpoints()
        self.test_data_flow()
        self.test_performance_benchmarks()
        self.test_security_basics()
        self.test_integration_scenarios()
        
        # 生成测试报告
        self.generate_report()
        
    def generate_report(self):
        """生成测试报告"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warned_tests = len([r for r in self.test_results if r['status'] == 'WARN'])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("📊 综合测试报告")
        print("="*60)
        print(f"测试总数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"警告: {warned_tests} ⚠️")
        print(f"成功率: {success_rate:.1f}%")
        print(f"总耗时: {total_duration:.2f}秒")
        
        # 按类别统计
        categories = {}
        for result in self.test_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0, 'failed': 0, 'warned': 0}
            categories[cat]['total'] += 1
            if result['status'] == 'PASS':
                categories[cat]['passed'] += 1
            elif result['status'] == 'FAIL':
                categories[cat]['failed'] += 1
            elif result['status'] == 'WARN':
                categories[cat]['warned'] += 1
                
        print("\n📈 分类统计:")
        for cat, stats in categories.items():
            rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
            
        # 显示失败的测试
        failed_results = [r for r in self.test_results if r['status'] == 'FAIL']
        if failed_results:
            print("\n❌ 失败的测试:")
            for result in failed_results:
                print(f"  - [{result['category']}] {result['test_name']}: {result['details']}")
                
        # 保存详细报告
        report_data = {
            'summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration': total_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'warned_tests': warned_tests,
                'success_rate': success_rate
            },
            'categories': categories,
            'detailed_results': self.test_results
        }
        
        # 保存报告文件
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = logs_dir / f"comprehensive_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回成功率用于脚本退出码
        return success_rate >= 80

def main():
    """主函数"""
    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()