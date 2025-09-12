#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大数据PSI隐私求交会话创建脚本
用于创建和管理大规模数据的隐私求交计算会话
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

# PSI服务配置
PSI_SERVICE_URL = "http://localhost:8001"
DATA_DIR = "data/synth"
RESULT_DIR = "data/psi_results"

def check_psi_service_health() -> bool:
    """检查PSI服务健康状态"""
    try:
        response = requests.get(f"{PSI_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ PSI服务连接失败: {e}")
        return False

def create_psi_session(session_id: str, party_id: str, other_parties: List[str], method: str = "ecdh_psi") -> bool:
    """创建PSI会话"""
    try:
        session_request = {
            "session_id": session_id,
            "method": method,
            "party_role": "coordinator",
            "party_id": party_id,
            "other_parties": other_parties,
            "timeout_seconds": 3600,
            "metadata": {
                "test_type": "performance",
                "data_size": "large",
                "created_by": "automated_script"
            }
        }
        
        response = requests.post(
            f"{PSI_SERVICE_URL}/psi/session",
            json=session_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ PSI会话创建成功: {session_id}")
            return True
        else:
            print(f"❌ PSI会话创建失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 创建PSI会话时发生错误: {e}")
        return False

def upload_dataset(session_id: str, party_id: str, file_path: str) -> bool:
    """上传数据集到PSI会话"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ 数据文件不存在: {file_path}")
            return False
            
        # 准备文件上传
        with open(file_path, 'rb') as f:
            files = {'file': (f'{party_id}.json', f, 'application/json')}
            data = {
                'session_id': session_id,
                'party_id': party_id
            }
            
            response = requests.post(
                f"{PSI_SERVICE_URL}/psi/upload",
                files=files,
                data=data,
                timeout=120  # 大数据上传需要更长时间
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 数据上传成功 - {party_id}: {result.get('element_count', 0)} 条记录")
            return True
        else:
            print(f"❌ 数据上传失败 - {party_id}: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 上传数据时发生错误 - {party_id}: {e}")
        return False

def start_psi_computation(session_id: str, party_id: str) -> bool:
    """启动PSI计算"""
    try:
        compute_request = {
            "session_id": session_id,
            "party_id": party_id,
            "return_intersection": False
        }
        
        response = requests.post(
            f"{PSI_SERVICE_URL}/psi/compute",
            json=compute_request,
            timeout=300  # 大数据计算需要更长时间
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ PSI计算启动成功")
            return True
        else:
            print(f"❌ PSI计算启动失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 启动PSI计算时发生错误: {e}")
        return False

def check_psi_status(session_id: str) -> Dict[str, Any]:
    """检查PSI计算状态"""
    try:
        response = requests.get(
            f"{PSI_SERVICE_URL}/psi/sessions/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 获取PSI状态失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 检查PSI状态时发生错误: {e}")
        return None

def get_psi_results(session_id: str) -> Dict[str, Any]:
    """获取PSI计算结果"""
    try:
        response = requests.get(
            f"{PSI_SERVICE_URL}/psi/results/{session_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 获取PSI结果失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 获取PSI结果时发生错误: {e}")
        return None

def save_results(session_id: str, results: Dict[str, Any]):
    """保存PSI计算结果"""
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{RESULT_DIR}/psi_results_{session_id}_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"💾 PSI结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"❌ 保存结果时发生错误: {e}")

def main():
    """主函数 - 执行完整的PSI流程"""
    print("🚀 开始大数据PSI隐私求交流程...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 检查PSI服务状态
    print("🔍 检查PSI服务状态...")
    if not check_psi_service_health():
        print("❌ PSI服务不可用，请确保服务正在运行")
        return
    print("✅ PSI服务运行正常")
    
    # 2. 生成会话ID和配置
    session_id = f"perf_test_{int(time.time())}"
    party_id = "bank"
    other_parties = ["ecommerce"]
    
    # 3. 创建PSI会话
    print("\n📝 创建PSI会话...")
    if not create_psi_session(session_id, party_id, other_parties, "ecdh_psi"):
        print("❌ 无法创建PSI会话，流程终止")
        return
    
    # 4. 上传数据集
    print("\n📤 上传数据集...")
    
    # 上传银行数据
    bank_file = f"{DATA_DIR}/large_bank_data.json"
    bank_upload_success = upload_dataset(session_id, "bank", bank_file)
    
    # 上传电商数据
    ecommerce_file = f"{DATA_DIR}/large_ecom_data.json"
    ecommerce_upload_success = upload_dataset(session_id, "ecommerce", ecommerce_file)
    
    if not (bank_upload_success and ecommerce_upload_success):
        print("❌ 数据上传失败，流程终止")
        return
    
    # 5. 启动PSI计算
    print("\n🔄 启动PSI计算...")
    if not start_psi_computation(session_id, party_id):
        print("❌ PSI计算启动失败，流程终止")
        return
    
    # 6. 监控计算进度
    print("\n⏳ 监控PSI计算进度...")
    max_wait_time = 600  # 最大等待10分钟
    check_interval = 10  # 每10秒检查一次
    waited_time = 0
    
    while waited_time < max_wait_time:
        status = check_psi_status(session_id)
        if status:
            state = status.get('status', 'unknown')
            progress = status.get('progress_percentage', 0)
            
            print(f"📊 计算状态: {state}, 进度: {progress}%")
            
            if state == 'completed':
                print("✅ PSI计算完成！")
                break
            elif state == 'failed':
                print(f"❌ PSI计算失败: {status.get('error_message', '未知错误')}")
                return
        
        time.sleep(check_interval)
        waited_time += check_interval
    
    if waited_time >= max_wait_time:
        print("⏰ PSI计算超时，请检查服务状态")
        return
    
    # 7. 获取计算结果
    print("\n📊 获取PSI计算结果...")
    results = get_psi_results(session_id)
    if results:
        intersection_size = results.get('intersection_size', 0)
        computation_time = results.get('computation_time_ms', 0)
        
        print(f"🎯 交集大小: {intersection_size:,} 条记录")
        print(f"⏱️  计算耗时: {computation_time:,} 毫秒")
        print(f"📈 处理速度: {(200000 / (computation_time / 1000)):.0f} 记录/秒")
        
        # 保存结果
        save_results(session_id, results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"✅ 大数据PSI流程完成！")
    print(f"🆔 会话ID: {session_id}")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print(f"🎉 大数据隐私求交测试成功！")
    print(f"🔗 可通过前端页面查看详细结果: http://localhost:3000")

if __name__ == "__main__":
    main()