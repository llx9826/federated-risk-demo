#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦风控系统测试运行器

统一的测试入口，支持多种测试模式:
1. quick - 快速测试（开发时使用）
2. full - 完整测试（发布前使用）
3. health - 仅健康检查
4. api - 仅API测试
5. perf - 仅性能测试
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def run_script(script_name, description):
    """运行指定的测试脚本"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ 测试脚本不存在: {script_path}")
        return False
        
    print(f"🚀 开始{description}...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description}完成")
            return True
        else:
            print(f"\n❌ {description}失败 (退出码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ 运行{description}时出错: {str(e)}")
        return False

def run_health_check():
    """运行健康检查"""
    print("🏥 系统健康检查")
    print("=" * 30)
    
    import requests
    import time
    
    services = {
        'consent-service': 'http://localhost:8000',
        'psi-service': 'http://localhost:8001',
        'model-trainer': 'http://localhost:8002',
        'model-explainer': 'http://localhost:8003',
        'frontend': 'http://localhost:5173'
    }
    
    healthy_count = 0
    total_count = len(services)
    
    for name, url in services.items():
        try:
            start_time = time.time()
            if name == 'frontend':
                response = requests.get(url, timeout=3)
                status = '✅' if response.status_code == 200 else '❌'
            else:
                health_url = f"{url}/health"
                response = requests.get(health_url, timeout=3)
                status = '✅' if response.status_code == 200 else '❌'
                
            duration = time.time() - start_time
            print(f"{status} {name:15} ({duration:.2f}s)")
            
            if status == '✅':
                healthy_count += 1
                
        except Exception as e:
            print(f"❌ {name:15} (连接失败)")
            
    print(f"\n📊 健康状态: {healthy_count}/{total_count} ({healthy_count/total_count*100:.0f}%)")
    return healthy_count == total_count

def run_api_test():
    """运行API测试"""
    print("🔧 API功能测试")
    print("=" * 30)
    
    import requests
    
    services = {
        'consent-service': 'http://localhost:8000',
        'psi-service': 'http://localhost:8001',
        'model-trainer': 'http://localhost:8002',
        'model-explainer': 'http://localhost:8003'
    }
    
    passed_count = 0
    total_count = len(services)
    
    for name, url in services.items():
        try:
            # 测试API文档
            response = requests.get(f"{url}/docs", timeout=3)
            status = '✅' if response.status_code == 200 else '❌'
            print(f"{status} {name} API文档")
            
            if status == '✅':
                passed_count += 1
                
        except Exception as e:
            print(f"❌ {name} API文档 (连接失败)")
            
    print(f"\n📊 API测试: {passed_count}/{total_count} ({passed_count/total_count*100:.0f}%)")
    return passed_count == total_count

def run_performance_test():
    """运行性能测试"""
    print("⚡ 性能基准测试")
    print("=" * 30)
    
    import requests
    import time
    
    services = {
        'consent-service': 'http://localhost:8000',
        'psi-service': 'http://localhost:8001',
        'model-trainer': 'http://localhost:8002',
        'model-explainer': 'http://localhost:8003'
    }
    
    good_performance = 0
    total_services = len(services)
    
    for name, url in services.items():
        response_times = []
        
        for i in range(3):
            try:
                start_time = time.time()
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    response_times.append(time.time() - start_time)
            except:
                pass
                
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            status = '✅' if avg_time < 1.0 else '⚠️' if avg_time < 2.0 else '❌'
            print(f"{status} {name:15} 平均响应: {avg_time:.3f}s")
            
            if avg_time < 2.0:
                good_performance += 1
        else:
            print(f"❌ {name:15} 无响应")
            
    print(f"\n📊 性能测试: {good_performance}/{total_services} ({good_performance/total_services*100:.0f}%)")
    return good_performance == total_services

def main():
    parser = argparse.ArgumentParser(
        description='联邦风控系统测试运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式说明:
  quick    - 快速测试，适合开发时使用
  full     - 完整测试，适合发布前使用
  health   - 仅检查服务健康状态
  api      - 仅测试API功能
  perf     - 仅测试性能基准
  
示例:
  python scripts/test_runner.py quick
  python scripts/test_runner.py full
  python scripts/test_runner.py health
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['quick', 'full', 'health', 'api', 'perf'],
        help='测试模式'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    print(f"🎯 联邦风控系统测试运行器")
    print(f"模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    success = True
    
    if args.mode == 'quick':
        success = run_script('quick_test.py', '快速测试')
        
    elif args.mode == 'full':
        success = run_script('comprehensive_test.py', '完整测试')
        
    elif args.mode == 'health':
        success = run_health_check()
        
    elif args.mode == 'api':
        success = run_api_test()
        
    elif args.mode == 'perf':
        success = run_performance_test()
        
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试完成，所有检查通过！")
        sys.exit(0)
    else:
        print("⚠️ 测试完成，发现问题需要处理")
        sys.exit(1)

if __name__ == '__main__':
    main()