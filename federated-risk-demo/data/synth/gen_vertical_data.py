#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纵向联邦学习合成数据生成脚本
生成银行方(A)和电商方(B)的垂直分割数据
"""

import pandas as pd
import numpy as np
import hashlib
import uuid
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_psi_token(identifier, salt="federated_demo_2024"):
    """生成PSI token (SHA256哈希)"""
    combined = f"{salt}||{identifier}"
    return hashlib.sha256(combined.encode()).hexdigest()

def generate_phone_email_pairs(n):
    """生成手机号和邮箱对"""
    phones = []
    emails = []
    
    for i in range(n):
        # 生成手机号 (13x-xxxx-xxxx格式)
        phone = f"13{np.random.randint(0, 10)}{np.random.randint(1000, 9999)}{np.random.randint(1000, 9999)}"
        phones.append(phone)
        
        # 生成邮箱
        domains = ['qq.com', '163.com', 'gmail.com', 'sina.com', 'outlook.com']
        username = f"user{i}_{np.random.randint(100, 999)}"
        email = f"{username}@{np.random.choice(domains)}"
        emails.append(email)
    
    return phones, emails

def generate_bank_data(n_samples=4200):
    """生成银行方数据 (包含标签)"""
    print(f"生成银行方数据: {n_samples} 条记录")
    
    # 生成基础标识
    phones, emails = generate_phone_email_pairs(n_samples)
    
    # 生成PSI token
    psi_tokens = [generate_psi_token(phone) for phone in phones]
    
    # 生成银行特征
    data = {
        'psi_token': psi_tokens,
        'phone': phones,
        'email': emails,
        
        # 银行核心特征
        'credit_score': np.random.normal(650, 120, n_samples).astype(int),
        'annual_income': np.random.lognormal(10.5, 0.8, n_samples).astype(int),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples),
        'credit_history_months': np.random.exponential(36, n_samples).astype(int),
        'num_credit_cards': np.random.poisson(2.5, n_samples),
        'mortgage_balance': np.random.exponential(150000, n_samples),
        'savings_balance': np.random.exponential(25000, n_samples),
        'num_late_payments_12m': np.random.poisson(0.8, n_samples),
        'credit_utilization': np.random.beta(2, 3, n_samples),
        'age': np.random.normal(38, 12, n_samples).astype(int),
        
        # 银行行为特征
        'avg_monthly_balance': np.random.lognormal(8.5, 1.2, n_samples),
        'num_transactions_3m': np.random.poisson(45, n_samples),
        'overdraft_count_12m': np.random.poisson(0.3, n_samples),
        'loan_approval_rate': np.random.beta(3, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 数据清洗
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['annual_income'] = np.clip(df['annual_income'], 20000, 500000)
    df['debt_to_income_ratio'] = np.clip(df['debt_to_income_ratio'], 0, 1)
    df['credit_history_months'] = np.clip(df['credit_history_months'], 0, 360)
    df['age'] = np.clip(df['age'], 18, 80)
    df['credit_utilization'] = np.clip(df['credit_utilization'], 0, 1)
    
    # 生成违约标签 (基于特征的逻辑回归模型)
    # 构造违约概率
    logit = (
        -2.5 +
        -0.008 * (df['credit_score'] - 600) +
        -0.0001 * (df['annual_income'] - 50000) +
        2.0 * df['debt_to_income_ratio'] +
        -0.01 * df['credit_history_months'] +
        0.15 * df['num_late_payments_12m'] +
        1.5 * df['credit_utilization'] +
        0.1 * df['overdraft_count_12m'] +
        -0.02 * (df['age'] - 35)
    )
    
    default_prob = 1 / (1 + np.exp(-logit))
    df['default_label'] = np.random.binomial(1, default_prob, n_samples)
    
    # 添加噪声确保真实性
    noise_mask = np.random.random(n_samples) < 0.05
    df.loc[noise_mask, 'default_label'] = 1 - df.loc[noise_mask, 'default_label']
    
    print(f"银行方违约率: {df['default_label'].mean():.3f}")
    
    return df

def generate_ecommerce_data(n_samples=3800, overlap_ratio=0.68):
    """生成电商方数据 (仅特征，无标签)"""
    print(f"生成电商方数据: {n_samples} 条记录")
    
    # 生成基础标识
    phones, emails = generate_phone_email_pairs(n_samples)
    
    # 生成PSI token
    psi_tokens = [generate_psi_token(phone) for phone in phones]
    
    # 生成电商特征
    data = {
        'psi_token': psi_tokens,
        'phone': phones,
        'email': emails,
        
        # 电商核心特征
        'total_orders_12m': np.random.poisson(12, n_samples),
        'avg_order_value': np.random.lognormal(4.5, 0.8, n_samples),
        'total_spent_12m': np.random.lognormal(7.5, 1.2, n_samples),
        'days_since_last_order': np.random.exponential(15, n_samples).astype(int),
        'favorite_category': np.random.choice(['electronics', 'clothing', 'books', 'home', 'sports'], n_samples),
        'return_rate': np.random.beta(1, 9, n_samples),
        'review_score_avg': np.random.normal(4.2, 0.8, n_samples),
        'num_reviews_given': np.random.poisson(3, n_samples),
        
        # 电商行为特征
        'cart_abandonment_rate': np.random.beta(3, 2, n_samples),
        'mobile_usage_ratio': np.random.beta(6, 2, n_samples),
        'coupon_usage_rate': np.random.beta(2, 3, n_samples),
        'peak_shopping_hour': np.random.choice(range(24), n_samples),
        'num_wishlist_items': np.random.poisson(8, n_samples),
        'social_shares': np.random.poisson(1.5, n_samples),
        'customer_service_contacts': np.random.poisson(0.5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 数据清洗
    df['avg_order_value'] = np.clip(df['avg_order_value'], 10, 2000)
    df['total_spent_12m'] = np.clip(df['total_spent_12m'], 0, 50000)
    df['days_since_last_order'] = np.clip(df['days_since_last_order'], 0, 365)
    df['return_rate'] = np.clip(df['return_rate'], 0, 0.5)
    df['review_score_avg'] = np.clip(df['review_score_avg'], 1, 5)
    df['cart_abandonment_rate'] = np.clip(df['cart_abandonment_rate'], 0, 1)
    df['mobile_usage_ratio'] = np.clip(df['mobile_usage_ratio'], 0, 1)
    df['coupon_usage_rate'] = np.clip(df['coupon_usage_rate'], 0, 1)
    
    return df

def create_overlap(bank_df, ecom_df, target_overlap=2600):
    """创建两个数据集的交集"""
    print(f"创建目标交集: {target_overlap} 条记录")
    
    # 从银行数据中随机选择一部分作为交集基础
    overlap_indices = np.random.choice(len(bank_df), target_overlap, replace=False)
    overlap_bank_tokens = bank_df.iloc[overlap_indices]['psi_token'].tolist()
    
    # 替换电商数据中的前target_overlap条记录的psi_token
    ecom_df_copy = ecom_df.copy()
    ecom_df_copy.iloc[:target_overlap, ecom_df_copy.columns.get_loc('psi_token')] = overlap_bank_tokens
    
    # 更新对应的phone和email
    for i, token in enumerate(overlap_bank_tokens):
        bank_row = bank_df[bank_df['psi_token'] == token].iloc[0]
        ecom_df_copy.iloc[i, ecom_df_copy.columns.get_loc('phone')] = bank_row['phone']
        ecom_df_copy.iloc[i, ecom_df_copy.columns.get_loc('email')] = bank_row['email']
    
    # 验证交集
    actual_overlap = len(set(bank_df['psi_token']) & set(ecom_df_copy['psi_token']))
    print(f"实际交集大小: {actual_overlap}")
    print(f"银行方对齐率: {actual_overlap / len(bank_df):.3f}")
    print(f"电商方对齐率: {actual_overlap / len(ecom_df_copy):.3f}")
    
    return ecom_df_copy

def main():
    """主函数"""
    print("=== 联邦学习合成数据生成 ===")
    print(f"生成时间: {datetime.now()}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # 生成银行方数据
    bank_data = generate_bank_data(n_samples=4200)
    
    # 生成电商方数据
    ecom_data = generate_ecommerce_data(n_samples=3800)
    
    # 创建交集
    ecom_data_aligned = create_overlap(bank_data, ecom_data, target_overlap=2600)
    
    # 保存数据
    bank_file = os.path.join(os.path.dirname(__file__), 'partyA_bank.csv')
    ecom_file = os.path.join(os.path.dirname(__file__), 'partyB_ecom.csv')
    
    # 银行方保存所有字段
    bank_data.to_csv(bank_file, index=False, encoding='utf-8')
    print(f"银行方数据已保存: {bank_file}")
    
    # 电商方不保存标签
    ecom_columns = [col for col in ecom_data_aligned.columns if col != 'default_label']
    ecom_data_aligned[ecom_columns].to_csv(ecom_file, index=False, encoding='utf-8')
    print(f"电商方数据已保存: {ecom_file}")
    
    # 统计信息
    print("\n=== 数据统计 ===")
    print(f"银行方记录数: {len(bank_data)}")
    print(f"电商方记录数: {len(ecom_data_aligned)}")
    print(f"交集大小: {len(set(bank_data['psi_token']) & set(ecom_data_aligned['psi_token']))}")
    print(f"银行方违约率: {bank_data['default_label'].mean():.3f}")
    
    # 保存元数据
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'bank_records': len(bank_data),
        'ecom_records': len(ecom_data_aligned),
        'intersection_size': len(set(bank_data['psi_token']) & set(ecom_data_aligned['psi_token'])),
        'default_rate': float(bank_data['default_label'].mean()),
        'bank_features': [col for col in bank_data.columns if col not in ['psi_token', 'phone', 'email', 'default_label']],
        'ecom_features': [col for col in ecom_data_aligned.columns if col not in ['psi_token', 'phone', 'email']]
    }
    
    import json
    metadata_file = os.path.join(os.path.dirname(__file__), 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"元数据已保存: {metadata_file}")
    print("\n数据生成完成！")

if __name__ == '__main__':
    main()