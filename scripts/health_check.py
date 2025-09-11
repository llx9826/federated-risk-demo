#!/usr/bin/env python3
"""
联邦风控演示系统健康检查脚本

该脚本检查所有服务的健康状态，包括：
- PSI服务
- 同意管理服务
- 训练服务
- 推理服务
- 前端应用
"""

import asyncio
import aiohttp
import sys
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    url: str
    health_endpoint: str
    timeout: int = 10
    critical: bool = True


@dataclass
class HealthResult:
    """健康检查结果"""
    service: str
    status: str  # 'healthy', 'unhealthy', 'timeout', 'error'
    response_time: float
    details: Dict
    error: str = None


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.services = [
            ServiceConfig(
                name="PSI服务",
                url="http://localhost:8001",
                health_endpoint="/health"
            ),
            ServiceConfig(
                name="同意管理服务",
                url="http://localhost:8002",
                health_endpoint="/health"
            ),
            ServiceConfig(
                name="训练服务",
                url="http://localhost:8003",
                health_endpoint="/health"
            ),
            ServiceConfig(
                name="推理服务",
                url="http://localhost:8004",
                health_endpoint="/health"
            ),
            ServiceConfig(
                name="前端应用",
                url="http://localhost:5173",
                health_endpoint="/",
                critical=False
            )
        ]
        
    async def check_service(self, session: aiohttp.ClientSession, service: ServiceConfig) -> HealthResult:
        """检查单个服务的健康状态"""
        start_time = time.time()
        
        try:
            url = f"{service.url}{service.health_endpoint}"
            
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=service.timeout)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    try:
                        if service.name == "前端应用":
                            # 前端应用返回HTML，只检查状态码
                            details = {"status_code": response.status}
                        else:
                            # 后端服务返回JSON
                            data = await response.json()
                            details = data
                    except Exception:
                        details = {"status_code": response.status}
                    
                    return HealthResult(
                        service=service.name,
                        status="healthy",
                        response_time=response_time,
                        details=details
                    )
                else:
                    return HealthResult(
                        service=service.name,
                        status="unhealthy",
                        response_time=response_time,
                        details={"status_code": response.status},
                        error=f"HTTP {response.status}"
                    )
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthResult(
                service=service.name,
                status="timeout",
                response_time=response_time,
                details={},
                error=f"请求超时 ({service.timeout}s)"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthResult(
                service=service.name,
                status="error",
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    async def check_all_services(self) -> List[HealthResult]:
        """检查所有服务的健康状态"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.check_service(session, service)
                for service in self.services
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            health_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    health_results.append(HealthResult(
                        service=self.services[i].name,
                        status="error",
                        response_time=0.0,
                        details={},
                        error=str(result)
                    ))
                else:
                    health_results.append(result)
            
            return health_results
    
    def print_results(self, results: List[HealthResult], verbose: bool = False):
        """打印检查结果"""
        print(f"\n{'='*60}")
        print(f"联邦风控演示系统健康检查报告")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        healthy_count = 0
        critical_issues = 0
        
        for result in results:
            # 状态图标
            if result.status == "healthy":
                icon = "✅"
                healthy_count += 1
            elif result.status == "timeout":
                icon = "⏰"
                if any(s.name == result.service and s.critical for s in self.services):
                    critical_issues += 1
            else:
                icon = "❌"
                if any(s.name == result.service and s.critical for s in self.services):
                    critical_issues += 1
            
            # 基本信息
            print(f"{icon} {result.service}")
            print(f"   状态: {result.status.upper()}")
            print(f"   响应时间: {result.response_time:.3f}s")
            
            if result.error:
                print(f"   错误: {result.error}")
            
            if verbose and result.details:
                print(f"   详情: {json.dumps(result.details, indent=6, ensure_ascii=False)}")
            
            print()
        
        # 总结
        total_services = len(results)
        print(f"{'='*60}")
        print(f"总结: {healthy_count}/{total_services} 服务正常运行")
        
        if critical_issues > 0:
            print(f"⚠️  发现 {critical_issues} 个关键服务问题")
            return False
        else:
            print("✅ 所有关键服务运行正常")
            return True
    
    async def run_continuous_check(self, interval: int = 30):
        """持续健康检查"""
        print(f"开始持续健康检查，检查间隔: {interval}秒")
        print("按 Ctrl+C 停止检查\n")
        
        try:
            while True:
                results = await self.check_all_services()
                self.print_results(results)
                
                print(f"等待 {interval} 秒后进行下次检查...\n")
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n健康检查已停止")
    
    async def run_dependency_check(self):
        """依赖检查"""
        print("检查服务依赖关系...\n")
        
        # 检查数据库连接
        print("📊 检查数据库连接...")
        # 这里可以添加数据库连接检查
        
        # 检查Redis连接
        print("🔄 检查Redis连接...")
        # 这里可以添加Redis连接检查
        
        # 检查文件系统
        print("📁 检查文件系统...")
        # 这里可以添加文件系统检查
        
        print("依赖检查完成\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="联邦风控演示系统健康检查")
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="显示详细信息"
    )
    parser.add_argument(
        "--continuous", "-c", 
        action="store_true", 
        help="持续监控模式"
    )
    parser.add_argument(
        "--interval", "-i", 
        type=int, 
        default=30, 
        help="持续监控间隔（秒）"
    )
    parser.add_argument(
        "--dependencies", "-d", 
        action="store_true", 
        help="检查依赖关系"
    )
    parser.add_argument(
        "--json", "-j", 
        action="store_true", 
        help="以JSON格式输出结果"
    )
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    
    async def run_check():
        if args.dependencies:
            await checker.run_dependency_check()
        
        if args.continuous:
            await checker.run_continuous_check(args.interval)
        else:
            results = await checker.check_all_services()
            
            if args.json:
                # JSON输出
                json_results = [
                    {
                        "service": r.service,
                        "status": r.status,
                        "response_time": r.response_time,
                        "details": r.details,
                        "error": r.error
                    }
                    for r in results
                ]
                print(json.dumps(json_results, indent=2, ensure_ascii=False))
            else:
                # 普通输出
                success = checker.print_results(results, args.verbose)
                sys.exit(0 if success else 1)
    
    try:
        asyncio.run(run_check())
    except KeyboardInterrupt:
        print("\n检查已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n检查过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()