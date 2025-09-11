#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦风控系统快速测试脚本

用于开发过程中的快速验证，包括:
1. 服务状态检查
2. 基本API测试
3. 数据文件验证
"""

import requests
import time
import json
from pathlib import Path
from datetime import datetime

class QuickTestSuite:
    def __init__(self):
        self.services = {
            'consent-service': 'http://localhost:8000',
            'psi-service': 'http://localhost:8001',
            'model-trainer': 'http://localhost:8002', 
            'model-explainer': 'http://localhost:8003',
            'frontend': 'http://localhost:5173'
        }
        self.health_endpoints = {
             'consent-service': '/healthz',
             'psi-service': '/health',
             'model-trainer': '/health',
             'model-explainer': '/health'
         }
        self.results = []
        
    def check_service(self, name, url):
        """检查单个服务状态"""
        try:
            start_time = time.time()
            if name == 'frontend':
                response = requests.get(url, timeout=3)
                status = '✅' if response.status_code == 200 else '❌'
                details = f"状态码: {response.status_code}"
            else:
                health_endpoint = self.health_endpoints.get(name, '/health')
                health_url = f"{url}{health_endpoint}"
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    health_data = response.json()
                    service_status = health_data.get('status', 'unknown')
                    status = '✅' if service_status in ['healthy', 'unhealthy'] else '❌'
                    details = f"状态: {service_status}"
                else:
                    status = '❌'
                    details = f"HTTP {response.status_code}"
                    
            duration = time.time() - start_time
            self.results.append({
                'service': name,
                'status': status,
                'duration': duration,
                'details': details
            })
            
            print(f"{status} {name:15} ({duration:.2f}s) - {details}")
            return status == '✅'
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self.results.append({
                'service': name,
                'status': '❌',
                'duration': duration,
                'details': f"连接失败: {str(e)[:50]}"
            })
            print(f"❌ {name:15} ({duration:.2f}s) - 连接失败")
            return False
            
    def check_data_files(self):
        """检查关键数据文件"""
        print("\n📁 数据文件检查:")
        
        data_files = [
            "data/synth/partyA_bank.csv",
            "data/synth/partyB_ecom.csv", 
            "data/synth/metadata.json"
        ]
        
        all_good = True
        for file_path in data_files:
            full_path = Path(file_path)
            if full_path.exists() and full_path.stat().st_size > 0:
                size_kb = full_path.stat().st_size / 1024
                print(f"✅ {file_path:30} ({size_kb:.1f} KB)")
            else:
                print(f"❌ {file_path:30} (不存在或为空)")
                all_good = False
                
        return all_good
        
    def test_basic_apis(self):
        """测试基本API功能"""
        print("\n🔧 基本API测试:")
        
        # 测试同意服务API文档
        try:
            response = requests.get(f"{self.services['consent-service']}/docs", timeout=3)
            status = '✅' if response.status_code == 200 else '❌'
            print(f"{status} 同意服务API文档")
        except:
            print("❌ 同意服务API文档")
            
        # 测试PSI服务API文档
        try:
            response = requests.get(f"{self.services['psi-service']}/docs", timeout=3)
            status = '✅' if response.status_code == 200 else '❌'
            print(f"{status} PSI服务API文档")
        except:
            print("❌ PSI服务API文档")
            
        # 测试模型训练服务API文档
        try:
            response = requests.get(f"{self.services['model-trainer']}/docs", timeout=3)
            status = '✅' if response.status_code == 200 else '❌'
            print(f"{status} 模型训练服务API文档")
        except:
            print("❌ 模型训练服务API文档")
            
        # 测试模型解释服务API文档
        try:
            response = requests.get(f"{self.services['model-explainer']}/docs", timeout=3)
            status = '✅' if response.status_code == 200 else '❌'
            print(f"{status} 模型解释服务API文档")
        except:
            print("❌ 模型解释服务API文档")
            
    def run_quick_test(self):
        """运行快速测试"""
        print("🚀 联邦风控系统快速测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🏥 服务健康检查:")
        
        # 检查所有服务
        healthy_services = 0
        for name, url in self.services.items():
            if self.check_service(name, url):
                healthy_services += 1
                
        # 检查数据文件
        data_ok = self.check_data_files()
        
        # 测试基本API
        self.test_basic_apis()
        
        # 生成简要报告
        total_services = len(self.services)
        service_rate = (healthy_services / total_services * 100) if total_services > 0 else 0
        
        print("\n" + "="*50)
        print("📊 快速测试结果")
        print("="*50)
        print(f"服务状态: {healthy_services}/{total_services} ({service_rate:.0f}%)")
        print(f"数据文件: {'✅ 正常' if data_ok else '❌ 异常'}")
        
        if healthy_services == total_services and data_ok:
            print("\n🎉 系统状态良好，可以开始开发！")
            return True
        else:
            print("\n⚠️  发现问题，请检查上述失败项")
            return False
            
def main():
    """主函数"""
    test_suite = QuickTestSuite()
    success = test_suite.run_quick_test()
    
    # 根据测试结果设置退出码
    exit_code = 0 if success else 1
    return exit_code

if __name__ == "__main__":
    import sys
    sys.exit(main())