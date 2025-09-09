#!/usr/bin/env python3
"""
联邦风控演示系统启动脚本

该脚本用于启动整个系统，包括：
- 检查依赖
- 启动后端服务
- 启动前端应用
- 健康检查
"""

import os
import sys
import subprocess
import time
import asyncio
import signal
from pathlib import Path
from typing import List, Dict
import psutil


class SystemManager:
    """系统管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.processes: List[subprocess.Popen] = []
        self.services = {
            "psi_service": {
                "name": "PSI服务",
                "port": 8001,
                "path": "backend/psi_service",
                "command": ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
            },
            "consent_service": {
                "name": "同意管理服务",
                "port": 8002,
                "path": "backend/consent_service",
                "command": ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
            },
            "training_service": {
                "name": "训练服务",
                "port": 8003,
                "path": "backend/training_service",
                "command": ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
            },
            "inference_service": {
                "name": "推理服务",
                "port": 8004,
                "path": "backend/inference_service",
                "command": ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]
            },
            "frontend": {
                "name": "前端应用",
                "port": 5173,
                "path": "frontend",
                "command": ["npm", "run", "dev"]
            }
        }
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在关闭系统...")
        self.stop_all_services()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        print("🔍 检查系统依赖...")
        
        # 检查Python
        try:
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                print("❌ Python版本需要3.8或更高")
                return False
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        except Exception as e:
            print(f"❌ Python检查失败: {e}")
            return False
        
        # 检查Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js {result.stdout.strip()}")
            else:
                print("❌ Node.js未安装")
                return False
        except FileNotFoundError:
            print("❌ Node.js未安装")
            return False
        
        # 检查npm
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ npm {result.stdout.strip()}")
            else:
                print("❌ npm未安装")
                return False
        except FileNotFoundError:
            print("❌ npm未安装")
            return False
        
        # 检查端口占用
        print("\n🔍 检查端口占用...")
        for service_id, config in self.services.items():
            if self.is_port_in_use(config["port"]):
                print(f"⚠️  端口 {config['port']} 已被占用 ({config['name']})")
            else:
                print(f"✅ 端口 {config['port']} 可用 ({config['name']})")
        
        print("\n依赖检查完成\n")
        return True
    
    def is_port_in_use(self, port: int) -> bool:
        """检查端口是否被占用"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def install_dependencies(self):
        """安装依赖"""
        print("📦 安装依赖包...")
        
        # 安装Python依赖
        print("\n安装Python依赖...")
        for service_id, config in self.services.items():
            if service_id == "frontend":
                continue
                
            service_path = self.project_root / config["path"]
            requirements_file = service_path / "requirements.txt"
            
            if requirements_file.exists():
                print(f"安装 {config['name']} 依赖...")
                try:
                    subprocess.run(
                        ["pip", "install", "-r", str(requirements_file)],
                        cwd=service_path,
                        check=True
                    )
                    print(f"✅ {config['name']} 依赖安装完成")
                except subprocess.CalledProcessError as e:
                    print(f"❌ {config['name']} 依赖安装失败: {e}")
                    return False
        
        # 安装前端依赖
        print("\n安装前端依赖...")
        frontend_path = self.project_root / "frontend"
        if (frontend_path / "package.json").exists():
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_path,
                    check=True
                )
                print("✅ 前端依赖安装完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ 前端依赖安装失败: {e}")
                return False
        
        print("\n📦 依赖安装完成\n")
        return True
    
    def start_service(self, service_id: str, config: Dict) -> subprocess.Popen:
        """启动单个服务"""
        service_path = self.project_root / config["path"]
        
        print(f"🚀 启动 {config['name']} (端口 {config['port']})...")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONPATH"] = str(service_path)
            
            process = subprocess.Popen(
                config["command"],
                cwd=service_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(process)
            print(f"✅ {config['name']} 启动成功 (PID: {process.pid})")
            
            return process
            
        except Exception as e:
            print(f"❌ {config['name']} 启动失败: {e}")
            return None
    
    def start_all_services(self, services_to_start: List[str] = None):
        """启动所有服务"""
        if services_to_start is None:
            services_to_start = list(self.services.keys())
        
        print("🚀 启动系统服务...\n")
        
        # 先启动后端服务
        backend_services = [s for s in services_to_start if s != "frontend"]
        for service_id in backend_services:
            config = self.services[service_id]
            process = self.start_service(service_id, config)
            if process:
                # 等待服务启动
                time.sleep(2)
        
        # 等待后端服务完全启动
        if backend_services:
            print("\n⏳ 等待后端服务启动...")
            time.sleep(5)
        
        # 启动前端服务
        if "frontend" in services_to_start:
            config = self.services["frontend"]
            self.start_service("frontend", config)
            
            print("\n⏳ 等待前端服务启动...")
            time.sleep(10)
        
        print("\n🎉 所有服务启动完成！\n")
    
    def stop_all_services(self):
        """停止所有服务"""
        print("\n🛑 停止所有服务...")
        
        for process in self.processes:
            try:
                if process.poll() is None:  # 进程仍在运行
                    print(f"停止进程 {process.pid}...")
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=5)
                        print(f"✅ 进程 {process.pid} 已停止")
                    except subprocess.TimeoutExpired:
                        print(f"⚠️  进程 {process.pid} 未响应，强制终止...")
                        process.kill()
                        process.wait()
                        print(f"✅ 进程 {process.pid} 已强制终止")
            except Exception as e:
                print(f"❌ 停止进程 {process.pid} 时出错: {e}")
        
        self.processes.clear()
        print("🛑 所有服务已停止")
    
    async def run_health_check(self):
        """运行健康检查"""
        print("🏥 运行健康检查...\n")
        
        # 导入健康检查模块
        sys.path.append(str(self.project_root / "scripts"))
        from health_check import HealthChecker
        
        checker = HealthChecker()
        results = await checker.check_all_services()
        success = checker.print_results(results)
        
        return success
    
    def show_service_urls(self):
        """显示服务URL"""
        print("🌐 服务访问地址:")
        print("="*50)
        
        for service_id, config in self.services.items():
            url = f"http://localhost:{config['port']}"
            if service_id == "frontend":
                print(f"📱 {config['name']}: {url}")
            else:
                print(f"🔧 {config['name']}: {url}")
                print(f"   健康检查: {url}/health")
                print(f"   API文档: {url}/docs")
        
        print("="*50)
    
    def monitor_services(self):
        """监控服务状态"""
        print("\n👀 监控服务状态 (按 Ctrl+C 停止)...\n")
        
        try:
            while True:
                print(f"\n[{time.strftime('%H:%M:%S')}] 服务状态:")
                
                for i, process in enumerate(self.processes):
                    if process.poll() is None:
                        print(f"  ✅ 进程 {process.pid} 运行中")
                    else:
                        print(f"  ❌ 进程 {process.pid} 已停止 (退出码: {process.returncode})")
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n监控已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="联邦风控演示系统启动脚本")
    parser.add_argument(
        "--services", "-s",
        nargs="+",
        choices=["psi_service", "consent_service", "training_service", "inference_service", "frontend"],
        help="指定要启动的服务"
    )
    parser.add_argument(
        "--no-deps", 
        action="store_true", 
        help="跳过依赖检查"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true", 
        help="安装依赖包"
    )
    parser.add_argument(
        "--health-check", 
        action="store_true", 
        help="启动后运行健康检查"
    )
    parser.add_argument(
        "--monitor", 
        action="store_true", 
        help="监控服务状态"
    )
    parser.add_argument(
        "--stop", 
        action="store_true", 
        help="停止所有服务"
    )
    
    args = parser.parse_args()
    
    manager = SystemManager()
    
    try:
        if args.stop:
            manager.stop_all_services()
            return
        
        if args.install_deps:
            if not manager.install_dependencies():
                sys.exit(1)
        
        if not args.no_deps:
            if not manager.check_dependencies():
                print("\n❌ 依赖检查失败，请解决上述问题后重试")
                sys.exit(1)
        
        # 启动服务
        manager.start_all_services(args.services)
        
        # 显示服务URL
        manager.show_service_urls()
        
        # 健康检查
        if args.health_check:
            async def run_check():
                await asyncio.sleep(5)  # 等待服务完全启动
                await manager.run_health_check()
            
            asyncio.run(run_check())
        
        # 监控模式
        if args.monitor:
            manager.monitor_services()
        else:
            print("\n✨ 系统启动完成！")
            print("按 Ctrl+C 停止所有服务")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    except Exception as e:
        print(f"\n❌ 启动过程中发生错误: {e}")
        sys.exit(1)
    
    finally:
        manager.stop_all_services()


if __name__ == "__main__":
    main()