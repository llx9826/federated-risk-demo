#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习模型训练脚本
使用PSI隐私求交结果训练联邦风险评估模型
"""

import requests
import json
import time
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# 服务配置
MODEL_TRAINER_URL = "http://localhost:8002"
PSI_SERVICE_URL = "http://localhost:8001"
DATA_DIR = "data/synth"
RESULT_DIR = "data/federated_results"

def check_model_trainer_health() -> bool:
    """检查模型训练服务健康状态"""
    try:
        response = requests.get(f"{MODEL_TRAINER_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 模型训练服务连接失败: {e}")
        return False

def get_latest_psi_result() -> Optional[Dict[str, Any]]:
    """获取最新的PSI计算结果"""
    try:
        # 查找最新的PSI结果文件
        result_files = []
        if os.path.exists("data/psi_results"):
            for file in os.listdir("data/psi_results"):
                if file.startswith("psi_results_") and file.endswith(".json"):
                    result_files.append(file)
        
        if not result_files:
            print("❌ 未找到PSI结果文件")
            return None
        
        # 按时间排序，获取最新的
        result_files.sort(reverse=True)
        latest_file = result_files[0]
        
        with open(f"data/psi_results/{latest_file}", 'r', encoding='utf-8') as f:
            psi_result = json.load(f)
        
        print(f"📊 加载PSI结果: {latest_file}")
        print(f"🎯 交集大小: {psi_result.get('intersection_size', 0):,} 条记录")
        
        return psi_result
        
    except Exception as e:
        print(f"❌ 获取PSI结果失败: {e}")
        return None

def prepare_training_data(psi_result: Dict[str, Any]) -> Dict[str, Any]:
    """准备联邦学习训练数据"""
    try:
        # 加载原始数据
        bank_data = []
        ecom_data = []
        
        # 加载银行数据
        bank_file = os.path.join(DATA_DIR, "large_bank_data.json")
        if os.path.exists(bank_file):
            with open(bank_file, 'r', encoding='utf-8') as f:
                bank_data = json.load(f)
        
        # 加载电商数据
        ecom_file = os.path.join(DATA_DIR, "large_ecom_data.json")
        if os.path.exists(ecom_file):
            with open(ecom_file, 'r', encoding='utf-8') as f:
                ecom_data = json.load(f)
        
        print(f"📊 银行数据: {len(bank_data):,} 条记录")
        print(f"📊 电商数据: {len(ecom_data):,} 条记录")
        
        # 准备特征数据
        training_data = {
            "bank_features": [],
            "ecom_features": [],
            "labels": [],
            "metadata": {
                "total_samples": len(bank_data) + len(ecom_data),
                "bank_samples": len(bank_data),
                "ecom_samples": len(ecom_data),
                "intersection_size": psi_result.get('intersection_size', 0),
                "psi_session_id": psi_result.get('session_id', 'unknown')
            }
        }
        
        # 提取银行特征
        for record in bank_data[:10000]:  # 限制训练数据量
            features = [
                float(record.get('credit_score', 0)) / 850.0,  # 归一化信用分数
                float(record.get('annual_income', 0)) / 200000.0,  # 归一化年收入
                float(record.get('debt_ratio', 0)),
                1.0 if record.get('has_mortgage', False) else 0.0,
                float(record.get('account_age_months', 0)) / 360.0  # 归一化账户年龄
            ]
            training_data["bank_features"].append(features)
            # 基于信用分数和债务比例生成风险标签
            risk_score = (record.get('credit_score', 600) / 850.0) * (1 - record.get('debt_ratio', 0.5))
            training_data["labels"].append(1 if risk_score > 0.6 else 0)
        
        # 提取电商特征
        for record in ecom_data[:10000]:  # 限制训练数据量
            features = [
                float(record.get('total_spent', 0)) / 10000.0,  # 归一化总消费
                float(record.get('order_count', 0)) / 100.0,  # 归一化订单数
                float(record.get('avg_order_value', 0)) / 500.0,  # 归一化平均订单价值
                float(record.get('return_rate', 0)),
                float(record.get('days_since_last_order', 0)) / 365.0  # 归一化天数
            ]
            training_data["ecom_features"].append(features)
            # 基于消费行为生成风险标签
            activity_score = (record.get('total_spent', 0) / 10000.0) * (1 - record.get('return_rate', 0.1))
            training_data["labels"].append(1 if activity_score > 0.5 else 0)
        
        print(f"✅ 训练数据准备完成")
        print(f"📊 银行特征: {len(training_data['bank_features']):,} 样本")
        print(f"📊 电商特征: {len(training_data['ecom_features']):,} 样本")
        print(f"📊 标签分布: {sum(training_data['labels']):,} 正样本 / {len(training_data['labels']) - sum(training_data['labels']):,} 负样本")
        
        return training_data
        
    except Exception as e:
        print(f"❌ 准备训练数据失败: {e}")
        return None

def create_federated_training_job(training_data: Dict[str, Any]) -> Optional[str]:
    """创建联邦学习训练任务"""
    try:
        task_id = f"federated_risk_model_{int(time.time())}"
        job_request = {
            "task_id": task_id,
            "task_name": "联邦风险评估模型训练",
            "participants": ["bank", "ecommerce"],
            "target_column": "is_fraud",
            "feature_columns": ["credit_score", "annual_income", "debt_ratio", "has_mortgage", "account_age", "total_spent", "order_count", "avg_order_value", "return_rate", "days_since_last_order"],
            "config": {
                "algorithm": "secure_boost",
                "privacy_level": "medium",
                "enable_secure_agg": True,
                "num_rounds": 10,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 10
            },
            "data_sources": {
                "bank": "psi_intersection",
                "ecommerce": "psi_intersection"
            },
            "metadata": training_data["metadata"]
        }
        
        response = requests.post(
            f"{MODEL_TRAINER_URL}/train",
            json=job_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            print(f"✅ 联邦学习任务创建成功: {task_id}")
            print(f"响应详情: {result}")
            return task_id
        else:
            print(f"❌ 创建联邦学习任务失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 创建联邦学习任务时发生错误: {e}")
        return None

def upload_training_data(job_id: str, training_data: Dict[str, Any], party: str) -> bool:
    """上传训练数据"""
    try:
        if party == "bank":
            data_payload = {
                "job_id": job_id,
                "party_id": "bank",
                "features": training_data["bank_features"],
                "labels": training_data["labels"][:len(training_data["bank_features"])],
                "feature_names": ["credit_score", "annual_income", "debt_ratio", "has_mortgage", "account_age"]
            }
        else:
            data_payload = {
                "job_id": job_id,
                "party_id": "ecommerce",
                "features": training_data["ecom_features"],
                "labels": training_data["labels"][len(training_data["bank_features"]):],
                "feature_names": ["total_spent", "order_count", "avg_order_value", "return_rate", "days_since_last_order"]
            }
        
        response = requests.post(
            f"{MODEL_TRAINER_URL}/federated/data/upload",
            json=data_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"✅ {party} 数据上传成功")
            return True
        else:
            print(f"❌ {party} 数据上传失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 上传 {party} 数据时发生错误: {e}")
        return False

def start_federated_training(job_id: str) -> bool:
    """启动联邦学习训练"""
    try:
        response = requests.post(
            f"{MODEL_TRAINER_URL}/federated/jobs/{job_id}/start",
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ 联邦学习训练启动成功")
            return True
        else:
            print(f"❌ 启动联邦学习训练失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 启动联邦学习训练时发生错误: {e}")
        return False

def monitor_training_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """监控训练进度"""
    try:
        max_wait_time = 300  # 最大等待5分钟
        check_interval = 10  # 每10秒检查一次
        waited_time = 0
        
        while waited_time < max_wait_time:
            response = requests.get(
                f"{MODEL_TRAINER_URL}/tasks/{job_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                status = response.json()
                state = status.get('status', 'unknown')
                progress = status.get('progress_percentage', 0)
                
                print(f"📊 训练状态: {state}, 进度: {progress}%")
                
                if state == 'completed':
                    print("✅ 联邦学习训练完成！")
                    return status
                elif state == 'failed':
                    print(f"❌ 联邦学习训练失败: {status.get('error_message', '未知错误')}")
                    return None
            
            time.sleep(check_interval)
            waited_time += check_interval
        
        print("⏰ 联邦学习训练超时")
        return None
        
    except Exception as e:
        print(f"❌ 监控训练进度时发生错误: {e}")
        return None

def get_training_results(job_id: str) -> Optional[Dict[str, Any]]:
    """获取训练结果"""
    try:
        response = requests.get(
            f"{MODEL_TRAINER_URL}/tasks/{job_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 获取训练结果失败: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ 获取训练结果时发生错误: {e}")
        return None

def save_results(job_id: str, results: Dict[str, Any]):
    """保存训练结果"""
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{RESULT_DIR}/federated_results_{job_id}_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 训练结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    print("🚀 开始联邦学习模型训练流程...")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 检查模型训练服务状态
    print("🔍 检查模型训练服务状态...")
    if not check_model_trainer_health():
        print("❌ 模型训练服务不可用，请检查服务状态")
        return
    print("✅ 模型训练服务运行正常")
    
    # 2. 获取PSI结果
    print("\n📊 获取PSI计算结果...")
    psi_result = get_latest_psi_result()
    if not psi_result:
        print("❌ 无法获取PSI结果，请先运行PSI计算")
        return
    
    # 3. 准备训练数据
    print("\n📝 准备联邦学习训练数据...")
    training_data = prepare_training_data(psi_result)
    if not training_data:
        print("❌ 训练数据准备失败")
        return
    
    # 4. 创建联邦学习任务
    print("\n🎯 创建联邦学习训练任务...")
    job_id = create_federated_training_job(training_data)
    if not job_id:
        print("❌ 联邦学习任务创建失败")
        return
    
    print(f"🎯 任务ID: {job_id}")
    print("\n📝 注意: 模型训练服务使用模拟数据，训练任务已自动启动")
    
    # 7. 监控训练进度
    print("\n⏳ 监控训练进度...")
    training_status = monitor_training_progress(job_id)
    if not training_status:
        print("❌ 训练未能成功完成")
        return
    
    # 8. 获取训练结果
    print("\n📊 获取训练结果...")
    results = get_training_results(job_id)
    if results:
        accuracy = results.get('accuracy', 0)
        loss = results.get('final_loss', 0)
        training_time = results.get('training_time_ms', 0)
        
        print(f"🎯 模型准确率: {accuracy:.4f}")
        print(f"📉 最终损失: {loss:.6f}")
        print(f"⏱️  训练耗时: {training_time:,} 毫秒")
        
        # 保存结果
        save_results(job_id, results)
    
    # 9. 获取训练报告
    print("\n📋 获取训练报告...")
    try:
        response = requests.get(
            f"{MODEL_TRAINER_URL}/tasks/{job_id}/report",
            timeout=30
        )
        
        if response.status_code == 200:
            report_data = response.json()
            report_path = f"./reports/federated_training_report_{job_id}.json"
            
            # 确保reports目录存在
            os.makedirs("./reports", exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 训练报告已保存: {report_path}")
            print("✅ 训练报告获取完成")
            
        else:
            print(f"⚠️ 获取训练报告失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"⚠️ 获取训练报告异常: {e}")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ 联邦学习训练流程完成！")
    print(f"🆔 任务ID: {job_id}")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print("🎉 联邦风险评估模型训练成功！")
    print("🔗 可通过前端页面查看详细结果: http://localhost:3000")

if __name__ == "__main__":
    main()