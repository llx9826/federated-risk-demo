#!/usr/bin/env python3
"""
联邦风控系统数据生成工具

生成用于性能测试的模拟数据，包括：
- 用户特征数据
- 交易记录
- 风险标签
- PSI测试数据
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from faker import Faker
from loguru import logger
from tqdm import tqdm

# 配置日志
logger.add("data_generation.log", rotation="10 MB")

class DataGenerator:
    """数据生成器"""
    
    def __init__(self, config_path: str):
        """初始化数据生成器"""
        self.config = self._load_config(config_path)
        self.fake = Faker(['zh_CN', 'en_US'])
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        Faker.seed(self.config.get('random_seed', 42))
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def generate_user_features(self, num_users: int) -> pd.DataFrame:
        """生成用户特征数据"""
        logger.info(f"生成 {num_users} 个用户的特征数据...")
        
        users = []
        for i in tqdm(range(num_users), desc="生成用户特征"):
            user = {
                'user_id': f"user_{i:08d}",
                'age': random.randint(18, 80),
                'gender': random.choice(['M', 'F']),
                'income': np.random.lognormal(10, 1),  # 收入分布
                'education': random.choice(['高中', '本科', '硕士', '博士']),
                'city_tier': random.choice([1, 2, 3, 4]),
                'credit_score': random.randint(300, 850),
                'account_age_days': random.randint(30, 3650),
                'total_assets': np.random.lognormal(12, 1.5),
                'debt_ratio': random.uniform(0, 0.8),
                'employment_status': random.choice(['employed', 'unemployed', 'student', 'retired']),
                'marital_status': random.choice(['single', 'married', 'divorced']),
                'num_dependents': random.randint(0, 5),
                'home_ownership': random.choice(['own', 'rent', 'mortgage']),
                'phone_verified': random.choice([True, False]),
                'email_verified': random.choice([True, False]),
                'created_at': self.fake.date_time_between(start_date='-2y', end_date='now')
            }
            users.append(user)
        
        df = pd.DataFrame(users)
        
        # 保存数据
        output_path = self.output_dir / 'user_features.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"用户特征数据已保存到: {output_path}")
        
        return df
    
    def generate_transactions(self, user_df: pd.DataFrame, transactions_per_user: int = 50) -> pd.DataFrame:
        """生成交易记录"""
        num_transactions = len(user_df) * transactions_per_user
        logger.info(f"生成 {num_transactions} 条交易记录...")
        
        transactions = []
        
        for _, user in tqdm(user_df.iterrows(), total=len(user_df), desc="生成交易记录"):
            user_transactions = random.randint(1, transactions_per_user * 2)
            
            for j in range(user_transactions):
                # 基于用户特征生成交易
                base_amount = user['income'] / 12 / 30  # 日均可支配收入
                
                transaction = {
                    'transaction_id': f"txn_{user['user_id']}_{j:04d}",
                    'user_id': user['user_id'],
                    'amount': max(1, np.random.exponential(base_amount)),
                    'transaction_type': random.choice(['transfer', 'payment', 'withdrawal', 'deposit']),
                    'merchant_category': random.choice([
                        'grocery', 'restaurant', 'gas_station', 'retail', 'online',
                        'entertainment', 'healthcare', 'education', 'travel', 'other'
                    ]),
                    'channel': random.choice(['mobile', 'web', 'atm', 'branch']),
                    'device_id': f"device_{random.randint(1, 10000):06d}",
                    'ip_address': self.fake.ipv4(),
                    'location_city': self.fake.city(),
                    'location_country': random.choice(['CN', 'US', 'UK', 'JP']),
                    'is_weekend': random.choice([True, False]),
                    'hour_of_day': random.randint(0, 23),
                    'days_since_last_transaction': random.randint(0, 30),
                    'velocity_1h': random.randint(0, 10),
                    'velocity_24h': random.randint(0, 50),
                    'velocity_7d': random.randint(0, 200),
                    'timestamp': self.fake.date_time_between(start_date='-1y', end_date='now')
                }
                transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        # 保存数据
        output_path = self.output_dir / 'transactions.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"交易记录已保存到: {output_path}")
        
        return df
    
    def generate_risk_labels(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """生成风险标签"""
        logger.info("生成风险标签...")
        
        # 基于交易特征生成风险概率
        def calculate_risk_probability(row):
            risk_score = 0
            
            # 金额异常
            if row['amount'] > 10000:
                risk_score += 0.3
            
            # 时间异常
            if row['hour_of_day'] < 6 or row['hour_of_day'] > 22:
                risk_score += 0.2
            
            # 频率异常
            if row['velocity_1h'] > 5:
                risk_score += 0.25
            
            # 地理位置异常
            if row['location_country'] != 'CN':
                risk_score += 0.15
            
            # 设备异常
            if random.random() < 0.05:  # 5%概率设备异常
                risk_score += 0.1
            
            return min(risk_score, 0.95)
        
        # 计算风险概率
        transaction_df['risk_probability'] = transaction_df.apply(calculate_risk_probability, axis=1)
        
        # 生成标签（基于概率）
        transaction_df['is_fraud'] = transaction_df['risk_probability'].apply(
            lambda p: random.random() < p
        )
        
        # 添加风险等级
        def get_risk_level(prob):
            if prob < 0.1:
                return 'low'
            elif prob < 0.3:
                return 'medium'
            elif prob < 0.6:
                return 'high'
            else:
                return 'critical'
        
        transaction_df['risk_level'] = transaction_df['risk_probability'].apply(get_risk_level)
        
        # 保存标签数据
        label_df = transaction_df[['transaction_id', 'user_id', 'is_fraud', 'risk_level', 'risk_probability']]
        output_path = self.output_dir / 'risk_labels.csv'
        label_df.to_csv(output_path, index=False)
        logger.info(f"风险标签已保存到: {output_path}")
        
        return label_df
    
    def generate_psi_test_data(self, num_records: int, num_parties: int = 2) -> List[pd.DataFrame]:
        """生成PSI测试数据"""
        logger.info(f"生成PSI测试数据: {num_records} 条记录, {num_parties} 方")
        
        # 生成基础ID池
        all_ids = [f"id_{i:08d}" for i in range(num_records * 2)]
        
        party_data = []
        
        for party_idx in range(num_parties):
            # 每方随机选择一部分ID，确保有交集
            if party_idx == 0:
                # 第一方选择前70%的ID
                party_ids = random.sample(all_ids[:int(num_records * 1.4)], num_records)
            else:
                # 其他方选择后70%的ID，确保有30-40%的交集
                party_ids = random.sample(all_ids[int(num_records * 0.6):], num_records)
            
            # 为每个ID生成特征数据
            party_records = []
            for record_id in party_ids:
                record = {
                    'id': record_id,
                    'feature_1': random.uniform(0, 100),
                    'feature_2': random.randint(1, 1000),
                    'feature_3': random.choice(['A', 'B', 'C', 'D']),
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
                }
                party_records.append(record)
            
            df = pd.DataFrame(party_records)
            party_data.append(df)
            
            # 保存各方数据
            output_path = self.output_dir / f'psi_party_{party_idx + 1}.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"第{party_idx + 1}方PSI数据已保存到: {output_path}")
        
        return party_data
    
    def generate_model_training_data(self, user_df: pd.DataFrame, transaction_df: pd.DataFrame, 
                                   label_df: pd.DataFrame) -> pd.DataFrame:
        """生成模型训练数据"""
        logger.info("生成模型训练数据...")
        
        # 合并数据
        training_data = transaction_df.merge(user_df, on='user_id', how='left')
        training_data = training_data.merge(label_df, on=['transaction_id', 'user_id'], how='left')
        
        # 特征工程
        training_data['amount_log'] = np.log1p(training_data['amount'])
        training_data['income_log'] = np.log1p(training_data['income'])
        training_data['assets_log'] = np.log1p(training_data['total_assets'])
        
        # 类别特征编码
        categorical_features = ['gender', 'education', 'employment_status', 'marital_status', 
                              'home_ownership', 'transaction_type', 'merchant_category', 'channel']
        
        for feature in categorical_features:
            training_data[f'{feature}_encoded'] = pd.Categorical(training_data[feature]).codes
        
        # 选择训练特征
        feature_columns = [
            'age', 'income_log', 'city_tier', 'credit_score', 'account_age_days',
            'assets_log', 'debt_ratio', 'num_dependents', 'amount_log',
            'hour_of_day', 'days_since_last_transaction', 'velocity_1h', 'velocity_24h', 'velocity_7d'
        ] + [f'{f}_encoded' for f in categorical_features] + [
            'phone_verified', 'email_verified', 'is_weekend'
        ]
        
        # 转换布尔值为数值
        bool_columns = ['phone_verified', 'email_verified', 'is_weekend']
        for col in bool_columns:
            training_data[col] = training_data[col].astype(int)
        
        # 选择最终的训练数据
        final_features = feature_columns + ['is_fraud']
        model_data = training_data[final_features].copy()
        
        # 处理缺失值
        model_data = model_data.fillna(0)
        
        # 保存训练数据
        output_path = self.output_dir / 'model_training_data.csv'
        model_data.to_csv(output_path, index=False)
        logger.info(f"模型训练数据已保存到: {output_path}")
        
        # 生成数据统计报告
        self._generate_data_report(model_data)
        
        return model_data
    
    def _generate_data_report(self, data: pd.DataFrame):
        """生成数据统计报告"""
        report = {
            'data_summary': {
                'total_records': len(data),
                'total_features': len(data.columns) - 1,  # 排除标签列
                'fraud_rate': data['is_fraud'].mean(),
                'missing_values': data.isnull().sum().to_dict()
            },
            'feature_statistics': data.describe().to_dict(),
            'generation_time': datetime.now().isoformat()
        }
        
        # 保存报告
        report_path = self.output_dir / 'data_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据报告已保存到: {report_path}")
        
        # 打印摘要
        logger.info(f"数据生成完成:")
        logger.info(f"  - 总记录数: {report['data_summary']['total_records']:,}")
        logger.info(f"  - 特征数量: {report['data_summary']['total_features']}")
        logger.info(f"  - 欺诈率: {report['data_summary']['fraud_rate']:.2%}")
    
    def run(self, num_users: int, transactions_per_user: int = 50, 
            psi_records: int = 10000, psi_parties: int = 2):
        """运行完整的数据生成流程"""
        start_time = time.time()
        logger.info("开始数据生成流程...")
        
        try:
            # 1. 生成用户特征
            user_df = self.generate_user_features(num_users)
            
            # 2. 生成交易记录
            transaction_df = self.generate_transactions(user_df, transactions_per_user)
            
            # 3. 生成风险标签
            label_df = self.generate_risk_labels(transaction_df)
            
            # 4. 生成PSI测试数据
            psi_data = self.generate_psi_test_data(psi_records, psi_parties)
            
            # 5. 生成模型训练数据
            training_data = self.generate_model_training_data(user_df, transaction_df, label_df)
            
            elapsed_time = time.time() - start_time
            logger.info(f"数据生成完成，耗时: {elapsed_time:.2f} 秒")
            
            return {
                'user_data': user_df,
                'transaction_data': transaction_df,
                'label_data': label_df,
                'psi_data': psi_data,
                'training_data': training_data
            }
            
        except Exception as e:
            logger.error(f"数据生成失败: {e}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='联邦风控系统数据生成工具')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--users', type=int, default=10000, help='用户数量')
    parser.add_argument('--transactions-per-user', type=int, default=50, help='每用户交易数')
    parser.add_argument('--psi-records', type=int, default=10000, help='PSI测试记录数')
    parser.add_argument('--psi-parties', type=int, default=2, help='PSI参与方数量')
    parser.add_argument('--output-dir', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    try:
        # 创建数据生成器
        generator = DataGenerator(args.config)
        
        # 如果指定了输出目录，覆盖配置
        if args.output_dir:
            generator.output_dir = Path(args.output_dir)
            generator.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行数据生成
        generator.run(
            num_users=args.users,
            transactions_per_user=args.transactions_per_user,
            psi_records=args.psi_records,
            psi_parties=args.psi_parties
        )
        
        logger.info("数据生成任务完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())