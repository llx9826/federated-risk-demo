#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模测试数据生成器
生成银行和电商数据各10万条记录用于性能测试
"""

import pandas as pd
import numpy as np
import json
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
import time

# 设置随机种子确保可重现性
np.random.seed(42)
random.seed(42)

def generate_user_id():
    """生成用户ID"""
    return f"user_{random.randint(100000, 999999)}"

def generate_phone():
    """生成手机号"""
    return f"1{random.choice([3,4,5,7,8,9])}{random.randint(10000000, 99999999)}"

def generate_email():
    """生成邮箱"""
    domains = ['gmail.com', '163.com', 'qq.com', 'sina.com', 'hotmail.com']
    username = f"user{random.randint(1000, 9999)}"
    return f"{username}@{random.choice(domains)}"

def generate_id_card():
    """生成身份证号（模拟）"""
    prefix = random.choice(['110101', '310101', '440101', '500101', '320101'])
    birth_year = random.randint(1970, 2000)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    suffix = f"{random.randint(100, 999)}{random.randint(0, 9)}"
    return f"{prefix}{birth_year:04d}{birth_month:02d}{birth_day:02d}{suffix}"

def generate_bank_data(num_records=100000):
    """生成银行数据"""
    print(f"正在生成 {num_records:,} 条银行数据...")
    start_time = time.time()
    
    # 生成共同用户池（用于PSI交集）
    common_users = [generate_user_id() for _ in range(num_records // 3)]  # 约1/3用户重叠
    
    data = []
    for i in range(num_records):
        if i % 10000 == 0:
            print(f"  已生成 {i:,} 条银行记录...")
        
        # 2/3概率使用共同用户，1/3概率使用独有用户
        if i < len(common_users) * 2:
            user_id = common_users[i % len(common_users)]
        else:
            user_id = generate_user_id()
        
        record = {
            'user_id': user_id,
            'phone': generate_phone(),
            'email': generate_email(),
            'id_card': generate_id_card(),
            'account_balance': round(random.uniform(1000, 1000000), 2),
            'credit_score': random.randint(300, 850),
            'loan_amount': round(random.uniform(0, 500000), 2),
            'deposit_amount': round(random.uniform(10000, 2000000), 2),
            'transaction_count': random.randint(1, 1000),
            'account_age_months': random.randint(1, 120),
            'risk_level': random.choice(['low', 'medium', 'high']),
            'account_type': random.choice(['savings', 'checking', 'credit', 'loan']),
            'branch_code': f"B{random.randint(1000, 9999)}",
            'last_transaction_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            'kyc_status': random.choice(['verified', 'pending', 'rejected']),
            'monthly_income': round(random.uniform(3000, 50000), 2),
            'employment_status': random.choice(['employed', 'self_employed', 'unemployed', 'retired']),
            'education_level': random.choice(['high_school', 'bachelor', 'master', 'phd']),
            'marital_status': random.choice(['single', 'married', 'divorced', 'widowed']),
            'age': random.randint(18, 80)
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # 保存为CSV和JSON格式
    csv_path = 'data/synth/large_bank_data.csv'
    json_path = 'data/synth/large_bank_data.json'
    
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)
    
    # 生成用于PSI的标识符文件
    identifiers = []
    for _, row in df.iterrows():
        # 使用多个字段组合生成唯一标识符
        identifier_str = f"{row['user_id']}|{row['phone']}|{row['email']}"
        identifier_hash = hashlib.sha256(identifier_str.encode()).hexdigest()[:16]
        identifiers.append(identifier_hash)
    
    with open('data/synth/large_bank_identifiers.txt', 'w') as f:
        for identifier in identifiers:
            f.write(f"{identifier}\n")
    
    elapsed = time.time() - start_time
    print(f"✅ 银行数据生成完成: {num_records:,} 条记录，耗时 {elapsed:.2f} 秒")
    print(f"   CSV文件: {csv_path}")
    print(f"   JSON文件: {json_path}")
    print(f"   标识符文件: data/synth/large_bank_identifiers.txt")
    
    return common_users

def generate_ecom_data(num_records=100000, common_users=None):
    """生成电商数据"""
    print(f"\n正在生成 {num_records:,} 条电商数据...")
    start_time = time.time()
    
    data = []
    for i in range(num_records):
        if i % 10000 == 0:
            print(f"  已生成 {i:,} 条电商记录...")
        
        # 使用共同用户池创建重叠用户
        if common_users and i < len(common_users) and random.random() < 0.4:  # 40%概率使用共同用户
            user_id = common_users[i % len(common_users)]
        else:
            user_id = generate_user_id()
        
        record = {
            'user_id': user_id,
            'phone': generate_phone(),
            'email': generate_email(),
            'total_orders': random.randint(1, 500),
            'total_amount': round(random.uniform(100, 50000), 2),
            'avg_order_value': round(random.uniform(50, 1000), 2),
            'favorite_category': random.choice(['electronics', 'clothing', 'books', 'home', 'sports', 'beauty']),
            'membership_level': random.choice(['bronze', 'silver', 'gold', 'platinum']),
            'registration_date': (datetime.now() - timedelta(days=random.randint(30, 1095))).isoformat(),
            'last_order_date': (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
            'return_rate': round(random.uniform(0, 0.3), 3),
            'review_count': random.randint(0, 200),
            'avg_rating': round(random.uniform(3.0, 5.0), 1),
            'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'alipay', 'wechat_pay']),
            'shipping_address_count': random.randint(1, 5),
            'cart_abandonment_rate': round(random.uniform(0, 0.8), 3),
            'mobile_usage_rate': round(random.uniform(0.3, 1.0), 3),
            'social_media_connected': random.choice([True, False]),
            'newsletter_subscribed': random.choice([True, False]),
            'customer_service_contacts': random.randint(0, 20),
            'loyalty_points': random.randint(0, 10000),
            'referral_count': random.randint(0, 50),
            'seasonal_buyer': random.choice([True, False]),
            'price_sensitivity': random.choice(['low', 'medium', 'high']),
            'brand_loyalty': random.choice(['low', 'medium', 'high'])
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # 保存为CSV和JSON格式
    csv_path = 'data/synth/large_ecom_data.csv'
    json_path = 'data/synth/large_ecom_data.json'
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)
    
    # 生成用于PSI的标识符文件
    identifiers = []
    for _, row in df.iterrows():
        # 使用多个字段组合生成唯一标识符
        identifier_str = f"{row['user_id']}|{row['phone']}|{row['email']}"
        identifier_hash = hashlib.sha256(identifier_str.encode()).hexdigest()[:16]
        identifiers.append(identifier_hash)
    
    with open('data/synth/large_ecom_identifiers.txt', 'w') as f:
        for identifier in identifiers:
            f.write(f"{identifier}\n")
    
    elapsed = time.time() - start_time
    print(f"✅ 电商数据生成完成: {num_records:,} 条记录，耗时 {elapsed:.2f} 秒")
    print(f"   CSV文件: {csv_path}")
    print(f"   JSON文件: {json_path}")
    print(f"   标识符文件: data/synth/large_ecom_identifiers.txt")

def generate_metadata():
    """生成元数据文件"""
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'bank_data': {
            'record_count': 100000,
            'file_path': 'data/synth/large_bank_data.csv',
            'identifier_path': 'data/synth/large_bank_identifiers.txt',
            'schema': {
                'user_id': 'string',
                'phone': 'string',
                'email': 'string',
                'id_card': 'string',
                'account_balance': 'float',
                'credit_score': 'int',
                'loan_amount': 'float',
                'deposit_amount': 'float',
                'transaction_count': 'int',
                'account_age_months': 'int',
                'risk_level': 'categorical',
                'account_type': 'categorical',
                'branch_code': 'string',
                'last_transaction_date': 'datetime',
                'kyc_status': 'categorical',
                'monthly_income': 'float',
                'employment_status': 'categorical',
                'education_level': 'categorical',
                'marital_status': 'categorical',
                'age': 'int'
            }
        },
        'ecom_data': {
            'record_count': 100000,
            'file_path': 'data/synth/large_ecom_data.csv',
            'identifier_path': 'data/synth/large_ecom_identifiers.txt',
            'schema': {
                'user_id': 'string',
                'phone': 'string',
                'email': 'string',
                'total_orders': 'int',
                'total_amount': 'float',
                'avg_order_value': 'float',
                'favorite_category': 'categorical',
                'membership_level': 'categorical',
                'registration_date': 'datetime',
                'last_order_date': 'datetime',
                'return_rate': 'float',
                'review_count': 'int',
                'avg_rating': 'float',
                'payment_method': 'categorical',
                'shipping_address_count': 'int',
                'cart_abandonment_rate': 'float',
                'mobile_usage_rate': 'float',
                'social_media_connected': 'boolean',
                'newsletter_subscribed': 'boolean',
                'customer_service_contacts': 'int',
                'loyalty_points': 'int',
                'referral_count': 'int',
                'seasonal_buyer': 'boolean',
                'price_sensitivity': 'categorical',
                'brand_loyalty': 'categorical'
            }
        },
        'psi_config': {
            'expected_intersection_size': 'approximately 30,000-40,000 records',
            'hash_algorithm': 'SHA256',
            'identifier_format': '16-character hex string'
        }
    }
    
    with open('data/synth/large_data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 元数据文件生成完成: data/synth/large_data_metadata.json")

if __name__ == "__main__":
    print("=" * 60)
    print("大规模测试数据生成器")
    print("=" * 60)
    
    total_start = time.time()
    
    # 生成银行数据
    common_users = generate_bank_data(100000)
    
    # 生成电商数据（使用部分共同用户）
    generate_ecom_data(100000, common_users)
    
    # 生成元数据
    generate_metadata()
    
    total_elapsed = time.time() - total_start
    print(f"\n🎉 所有数据生成完成！")
    print(f"📊 总计: 200,000 条记录")
    print(f"⏱️  总耗时: {total_elapsed:.2f} 秒")
    print(f"💾 数据大小估计: ~50-80 MB")
    print(f"🔗 预期PSI交集: ~30,000-40,000 条记录")
    
    # 显示文件信息
    files = [
        'data/synth/large_bank_data.csv',
        'data/synth/large_bank_data.json',
        'data/synth/large_bank_identifiers.txt',
        'data/synth/large_ecom_data.csv',
        'data/synth/large_ecom_data.json',
        'data/synth/large_ecom_identifiers.txt',
        'data/synth/large_data_metadata.json'
    ]
    
    print("\n📁 生成的文件:")
    for file_path in files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"   {file_path} ({size:,} bytes)")