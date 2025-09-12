#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量创建数据使用同意记录
为大数据测试创建多个同意记录
"""

import requests
import json
import time
from datetime import datetime, timedelta
import random

# 同意管理服务配置
CONSENT_SERVICE_URL = "http://localhost:8000"

def create_consent_record(title, description, data_types, purpose, requester, duration_days=365):
    """创建单个同意记录"""
    # 生成用户ID
    subject = f"user_{random.randint(100000, 999999)}"
    
    consent_data = {
        "subject": subject,
        "purpose": purpose,
        "scope_features": data_types,
        "ttl_hours": duration_days * 24,
        "issuer": requester,
        "metadata": {
            "title": title,
            "description": description,
            "created_by": "batch_script",
            "original_requester": requester
        }
    }
    
    try:
        response = requests.post(
            f"{CONSENT_SERVICE_URL}/consent/issue",
            json=consent_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 同意记录创建成功: {title} (ID: {result.get('consent_id', 'N/A')})")
            return result
        else:
            print(f"❌ 创建失败: {title} - {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 请求异常: {title} - {str(e)}")
        return None

def create_batch_consent_records():
    """批量创建同意记录"""
    print("=" * 60)
    print("批量创建数据使用同意记录")
    print("=" * 60)
    
    # 定义多个同意记录模板（使用允许的purpose值）
    consent_templates = [
        {
            "title": "银行信用评分数据使用同意",
            "description": "同意银行使用个人金融数据进行信用评分，包括账户余额、交易记录、信用历史等信息。",
            "data_types": ["financial_data", "transaction_history", "credit_score", "account_info"],
            "purpose": "credit_scoring",
            "requester": "银行风控部门",
            "duration_days": 730
        },
        {
            "title": "电商风险评估数据使用同意",
            "description": "同意电商平台使用用户购买行为、浏览记录等数据进行风险评估和欺诈检测。",
            "data_types": ["purchase_history", "browsing_behavior", "user_reviews", "preference_data"],
            "purpose": "risk_assessment",
            "requester": "电商风控团队",
            "duration_days": 365
        },
        {
            "title": "个性化营销数据使用同意",
            "description": "同意使用用户偏好和行为数据进行个性化营销推荐，提升用户体验。",
            "data_types": ["preference_data", "demographic_info", "interaction_history"],
            "purpose": "marketing",
            "requester": "营销团队",
            "duration_days": 365
        },
        {
            "title": "联邦学习研究数据使用同意",
            "description": "同意参与联邦学习研究，使用去标识化的用户数据改进机器学习模型。",
            "data_types": ["anonymized_features", "risk_labels", "demographic_data"],
            "purpose": "research",
            "requester": "AI研究团队",
            "duration_days": 1095
        },
        {
            "title": "跨机构风险评估数据共享同意",
            "description": "同意在银行和电商平台间进行隐私保护的数据共享，用于风险评估。",
            "data_types": ["user_identifiers", "risk_indicators", "fraud_signals"],
            "purpose": "risk_assessment",
            "requester": "跨机构风控联盟",
            "duration_days": 365
        },
        {
            "title": "信用评分模型研究同意",
            "description": "同意使用PSI技术进行隐私保护的数据研究，改进信用评分模型。",
            "data_types": ["hashed_identifiers", "encrypted_features"],
            "purpose": "credit_scoring",
            "requester": "PSI研究服务",
            "duration_days": 180
        },
        {
            "title": "大数据营销研究同意",
            "description": "同意使用模拟数据进行大规模营销效果研究，验证营销策略有效性。",
            "data_types": ["synthetic_data", "performance_metrics", "system_logs"],
            "purpose": "marketing",
            "requester": "营销研究团队",
            "duration_days": 90
        },
        {
            "title": "数据质量研究同意",
            "description": "同意对数据质量进行研究和分析，包括数据完整性检查、异常值检测等。",
            "data_types": ["raw_data", "data_quality_metrics", "cleansing_logs"],
            "purpose": "research",
            "requester": "数据研究团队",
            "duration_days": 365
        },
        {
            "title": "实时风险评估同意",
            "description": "同意进行实时风险评估，使用机器学习算法检测异常交易和潜在风险。",
            "data_types": ["real_time_transactions", "risk_scores", "alert_data"],
            "purpose": "risk_assessment",
            "requester": "风险评估系统",
            "duration_days": 365
        },
        {
            "title": "信用评分优化研究同意",
            "description": "同意参与信用评分算法优化研究，提升评分准确性和公平性。",
            "data_types": ["credit_history", "payment_behavior", "financial_metrics"],
            "purpose": "credit_scoring",
            "requester": "信用评分研发团队",
            "duration_days": 730
        }
    ]
    
    created_records = []
    
    # 检查服务可用性
    try:
        health_response = requests.get(f"{CONSENT_SERVICE_URL}/healthz")
        if health_response.status_code != 200:
            print(f"❌ 同意管理服务不可用: {health_response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 无法连接到同意管理服务: {str(e)}")
        return []
    
    print(f"🚀 开始创建 {len(consent_templates)} 个同意记录...\n")
    
    for i, template in enumerate(consent_templates, 1):
        print(f"[{i}/{len(consent_templates)}] 创建: {template['title']}")
        
        result = create_consent_record(
            title=template['title'],
            description=template['description'],
            data_types=template['data_types'],
            purpose=template['purpose'],
            requester=template['requester'],
            duration_days=template['duration_days']
        )
        
        if result:
            created_records.append(result)
        
        # 添加小延迟避免过快请求
        time.sleep(0.5)
        print()
    
    print("=" * 60)
    print(f"✅ 同意记录创建完成: {len(created_records)}/{len(consent_templates)} 成功")
    
    if created_records:
        print("\n📋 创建的同意记录:")
        for record in created_records:
            print(f"   - {record.get('title', 'N/A')} (ID: {record.get('id', 'N/A')})")
        
        # 保存创建结果
        with open('data/synth/created_consent_records.json', 'w', encoding='utf-8') as f:
            json.dump(created_records, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: data/synth/created_consent_records.json")
    
    return created_records

def verify_consent_records():
    """验证创建的同意记录"""
    print("\n🔍 验证同意记录...")
    
    try:
        response = requests.get(f"{CONSENT_SERVICE_URL}/consent")
        if response.status_code == 200:
            result = response.json()
            consents = result.get('consents', [])
            print(f"✅ 当前系统中共有 {len(consents)} 个同意记录")
            
            # 显示最近创建的记录
            if consents:
                print("\n📝 最新的同意记录:")
                for consent in consents[-5:]:  # 显示最后5个
                    print(f"   - {consent.get('request_id', 'N/A')} ({consent.get('decision', 'N/A')})")
            
            return consents
        else:
            print(f"❌ 获取同意记录失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 验证异常: {str(e)}")
        return []

if __name__ == "__main__":
    start_time = time.time()
    
    # 创建同意记录
    created_records = create_batch_consent_records()
    
    # 验证创建结果
    all_consents = verify_consent_records()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed:.2f} 秒")
    
    if created_records:
        print(f"🎉 同意记录创建流程完成！")
        print(f"📊 成功创建 {len(created_records)} 个同意记录")
        print(f"🔗 可通过前端页面查看: http://localhost:3000")
    else:
        print(f"⚠️  未能创建任何同意记录，请检查服务状态")