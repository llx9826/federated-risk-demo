#!/usr/bin/env python3
"""
纵向联邦学习合成数据生成器
支持多方数据生成，确保有效信号和可学习性
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class VerticalFLDataGenerator:
    """纵向联邦学习数据生成器"""
    
    def __init__(self, n_samples: int = 50000, overlap: float = 0.6, 
                 parties: List[str] = None, seed: int = 42, 
                 bad_rate: float = 0.12, noise: float = 0.15):
        """
        初始化数据生成器
        
        Args:
            n_samples: 每方样本数量
            overlap: 交集比例
            parties: 参与方列表 ['A', 'B', 'C']
            seed: 随机种子
            bad_rate: 坏账率
            noise: 噪声水平
        """
        self.n_samples = n_samples
        self.overlap = overlap
        self.parties = parties or ['A', 'B']
        self.seed = seed
        self.bad_rate = bad_rate
        self.noise = noise
        self.public_salt = "federated_risk_demo_2025"
        
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        # 数据质量统计
        self.data_profile = {
            'generation_time': datetime.now().isoformat(),
            'parameters': {
                'n_samples': n_samples,
                'overlap': overlap,
                'parties': parties,
                'seed': seed,
                'bad_rate': bad_rate,
                'noise': noise
            },
            'quality_metrics': {},
            'feature_correlations': {},
            'baseline_performance': {}
        }
    
    def generate_phone_numbers(self, n: int) -> List[str]:
        """生成手机号码作为PSI标识符"""
        phones = []
        for i in range(n):
            # 生成中国手机号格式
            prefix = random.choice(['138', '139', '150', '151', '188', '189'])
            suffix = f"{random.randint(10000000, 99999999):08d}"
            phones.append(f"{prefix}{suffix}")
        return phones
    
    def generate_psi_tokens(self, phones: List[str]) -> List[str]:
        """生成PSI令牌"""
        tokens = []
        for phone in phones:
            token_input = f"{self.public_salt}{phone}"
            token = hashlib.sha256(token_input.encode()).hexdigest()
            tokens.append(token)
        return tokens
    
    def create_overlapping_identifiers(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """创建有重叠的标识符"""
        # 生成共同用户池
        overlap_size = int(self.n_samples * self.overlap)
        common_phones = self.generate_phone_numbers(overlap_size)
        
        party_phones = {}
        party_tokens = {}
        
        for party in self.parties:
            # 每方都包含共同用户
            party_specific_size = self.n_samples - overlap_size
            party_specific_phones = self.generate_phone_numbers(party_specific_size)
            
            # 合并共同用户和专有用户
            all_phones = common_phones + party_specific_phones
            random.shuffle(all_phones)
            
            party_phones[party] = all_phones
            party_tokens[party] = self.generate_psi_tokens(all_phones)
        
        return common_phones, party_tokens
    
    def generate_bank_features(self, n: int) -> pd.DataFrame:
        """生成银行方特征（A方，含标签）"""
        data = {
            # 收入相关
            'annual_income': np.random.lognormal(10.5, 0.8, n),  # 年收入
            'monthly_income': np.random.lognormal(8.5, 0.6, n),  # 月收入
            
            # 债务相关
            'debt_to_income': np.random.beta(2, 5, n) * 2,  # 债务收入比
            'cc_utilization': np.random.beta(2, 3, n),  # 信用卡使用率
            'total_debt': np.random.lognormal(9.0, 1.2, n),  # 总债务
            
            # 信用历史
            'credit_len_yrs': np.random.gamma(2, 5, n),  # 信用历史长度
            'credit_score': np.random.normal(650, 120, n),  # 信用评分
            
            # 违约历史
            'late_3m': np.random.binomial(1, 0.15, n),  # 近3月逾期
            'delinq_12m': np.random.poisson(0.3, n),  # 近12月违约次数
            'bankruptcy_flag': np.random.binomial(1, 0.02, n),  # 破产标记
            
            # 账户信息
            'num_accounts': np.random.poisson(3, n),  # 账户数量
            'account_age_months': np.random.gamma(3, 12, n),  # 账户年龄
            
            # 其他
            'employment_years': np.random.gamma(2, 3, n),  # 工作年限
            'home_ownership': np.random.choice(['own', 'rent', 'mortgage'], n, p=[0.3, 0.4, 0.3])
        }
        
        df = pd.DataFrame(data)
        
        # 数据清理和约束
        df['annual_income'] = np.clip(df['annual_income'], 20000, 2000000)
        df['monthly_income'] = np.clip(df['monthly_income'], 2000, 200000)
        df['debt_to_income'] = np.clip(df['debt_to_income'], 0, 2)
        df['cc_utilization'] = np.clip(df['cc_utilization'], 0, 1)
        df['credit_score'] = np.clip(df['credit_score'], 300, 850)
        df['credit_len_yrs'] = np.clip(df['credit_len_yrs'], 0, 50)
        df['employment_years'] = np.clip(df['employment_years'], 0, 40)
        
        return df
    
    def generate_ecom_features(self, n: int) -> pd.DataFrame:
        """生成电商方特征（B方）"""
        data = {
            # 购买行为
            'order_cnt_6m': np.random.poisson(8, n),  # 6月订单数
            'monetary_6m': np.random.lognormal(6.5, 1.5, n),  # 6月消费金额
            'avg_order_value': np.random.lognormal(4.5, 0.8, n),  # 平均订单价值
            
            # 退货行为
            'return_rate': np.random.beta(1, 9, n),  # 退货率
            'return_cnt_6m': np.random.poisson(1.2, n),  # 6月退货次数
            
            # 活跃度
            'recency_days': np.random.gamma(2, 15, n),  # 最近购买天数
            'frequency_score': np.random.gamma(2, 2, n),  # 购买频率评分
            'session_duration_avg': np.random.lognormal(3.5, 0.8, n),  # 平均会话时长
            
            # 行为模式
            'midnight_orders_ratio': np.random.beta(1, 19, n),  # 深夜下单比例
            'weekend_orders_ratio': np.random.beta(3, 7, n),  # 周末下单比例
            'mobile_orders_ratio': np.random.beta(7, 3, n),  # 移动端下单比例
            
            # 偏好
            'category_diversity': np.random.gamma(2, 1.5, n),  # 品类多样性
            'brand_loyalty_score': np.random.beta(3, 2, n),  # 品牌忠诚度
            
            # 风险指标
            'payment_failures': np.random.poisson(0.5, n),  # 支付失败次数
            'address_changes': np.random.poisson(0.3, n),  # 地址变更次数
        }
        
        df = pd.DataFrame(data)
        
        # 数据清理和约束
        df['monetary_6m'] = np.clip(df['monetary_6m'], 0, 100000)
        df['avg_order_value'] = np.clip(df['avg_order_value'], 10, 5000)
        df['recency_days'] = np.clip(df['recency_days'], 0, 365)
        df['session_duration_avg'] = np.clip(df['session_duration_avg'], 30, 7200)
        df['category_diversity'] = np.clip(df['category_diversity'], 1, 20)
        
        return df
    
    def generate_telco_features(self, n: int) -> pd.DataFrame:
        """生成运营商方特征（C方，可选）"""
        data = {
            # 账单信息
            'monthly_bill': np.random.lognormal(4.0, 0.6, n),  # 月账单
            'bill_overdue_days': np.random.gamma(1, 3, n),  # 账单逾期天数
            'payment_method': np.random.choice(['auto', 'manual', 'prepaid'], n, p=[0.6, 0.3, 0.1]),
            
            # 使用行为
            'data_usage_gb': np.random.lognormal(2.5, 1.0, n),  # 数据使用量
            'call_minutes': np.random.lognormal(5.0, 0.8, n),  # 通话时长
            'sms_count': np.random.poisson(50, n),  # 短信数量
            
            # 服务信息
            'tenure_months': np.random.gamma(3, 8, n),  # 在网时长
            'plan_changes': np.random.poisson(0.5, n),  # 套餐变更次数
            'service_calls': np.random.poisson(1, n),  # 客服通话次数
            
            # 网络行为
            'roaming_usage': np.random.exponential(0.1, n),  # 漫游使用
            'night_usage_ratio': np.random.beta(2, 8, n),  # 夜间使用比例
            'weekend_usage_ratio': np.random.beta(3, 7, n),  # 周末使用比例
        }
        
        df = pd.DataFrame(data)
        
        # 数据清理和约束
        df['monthly_bill'] = np.clip(df['monthly_bill'], 20, 500)
        df['bill_overdue_days'] = np.clip(df['bill_overdue_days'], 0, 90)
        df['data_usage_gb'] = np.clip(df['data_usage_gb'], 0, 100)
        df['call_minutes'] = np.clip(df['call_minutes'], 0, 5000)
        df['tenure_months'] = np.clip(df['tenure_months'], 1, 120)
        
        return df
    
    def generate_target_with_signal(self, bank_df: pd.DataFrame, 
                                   ecom_df: pd.DataFrame, 
                                   telco_df: pd.DataFrame = None) -> np.ndarray:
        """基于特征生成有信号的目标变量"""
        n = len(bank_df)
        
        # 银行特征权重（正相关表示增加违约概率）
        bank_weights = {
            'debt_to_income': 2.5,
            'cc_utilization': 1.8,
            'late_3m': 1.5,
            'delinq_12m': 1.2,
            'bankruptcy_flag': 3.0,
            'annual_income': -0.8,  # 负相关
            'credit_len_yrs': -0.6,
            'credit_score': -1.5,
            'employment_years': -0.4
        }
        
        # 电商特征权重
        ecom_weights = {
            'return_rate': 1.5,
            'recency_days': 0.8,
            'midnight_orders_ratio': 1.2,
            'payment_failures': 2.0,
            'order_cnt_6m': -0.6,  # 负相关
            'monetary_6m': -0.9,
            'brand_loyalty_score': -0.5
        }
        
        # 运营商特征权重（如果存在）
        telco_weights = {
            'bill_overdue_days': 1.8,
            'plan_changes': 0.8,
            'service_calls': 0.6,
            'tenure_months': -0.7  # 负相关
        } if telco_df is not None else {}
        
        # 标准化特征
        scaler = StandardScaler()
        
        # 计算银行线性项
        bank_linear = np.zeros(n)
        for feature, weight in bank_weights.items():
            if feature in bank_df.columns:
                feature_std = scaler.fit_transform(bank_df[[feature]]).flatten()
                bank_linear += weight * feature_std
        
        # 计算电商线性项
        ecom_linear = np.zeros(n)
        for feature, weight in ecom_weights.items():
            if feature in ecom_df.columns:
                feature_std = scaler.fit_transform(ecom_df[[feature]]).flatten()
                ecom_linear += weight * feature_std
        
        # 计算运营商线性项
        telco_linear = np.zeros(n)
        if telco_df is not None:
            for feature, weight in telco_weights.items():
                if feature in telco_df.columns:
                    feature_std = scaler.fit_transform(telco_df[[feature]]).flatten()
                    telco_linear += weight * feature_std
        
        # 交互项（A×B）
        interaction_terms = np.zeros(n)
        
        # 债务收入比 × 退货率
        if 'debt_to_income' in bank_df.columns and 'return_rate' in ecom_df.columns:
            dti_std = scaler.fit_transform(bank_df[['debt_to_income']]).flatten()
            return_std = scaler.fit_transform(ecom_df[['return_rate']]).flatten()
            interaction_terms += 1.5 * dti_std * return_std
        
        # 信用卡使用率 × 最近购买天数
        if 'cc_utilization' in bank_df.columns and 'recency_days' in ecom_df.columns:
            cc_std = scaler.fit_transform(bank_df[['cc_utilization']]).flatten()
            recency_std = scaler.fit_transform(ecom_df[['recency_days']]).flatten()
            interaction_terms += 1.2 * cc_std * recency_std
        
        # 逾期记录 × 支付失败
        if 'late_3m' in bank_df.columns and 'payment_failures' in ecom_df.columns:
            late_std = scaler.fit_transform(bank_df[['late_3m']]).flatten()
            payment_std = scaler.fit_transform(ecom_df[['payment_failures']]).flatten()
            interaction_terms += 2.0 * late_std * payment_std
        
        # 合并所有项
        logits = bank_linear + ecom_linear + telco_linear + interaction_terms
        
        # 添加噪声
        noise = np.random.normal(0, self.noise, n)
        logits += noise
        
        # 转换为概率
        probabilities = 1 / (1 + np.exp(-logits))
        
        # 校准到目标坏账率
        threshold = np.percentile(probabilities, (1 - self.bad_rate) * 100)
        labels = (probabilities > threshold).astype(int)
        
        # 确保坏账率在合理范围内
        actual_bad_rate = labels.mean()
        if not (0.08 <= actual_bad_rate <= 0.20):
            # 重新校准
            sorted_probs = np.sort(probabilities)
            target_threshold_idx = int((1 - self.bad_rate) * len(sorted_probs))
            threshold = sorted_probs[target_threshold_idx]
            labels = (probabilities > threshold).astype(int)
        
        return labels
    
    def validate_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证数据质量"""
        print("\n🔍 验证数据质量...")
        
        all_passed = True
        
        for party, df in datasets.items():
            print(f"\n📊 验证 {party} 方数据:")
            
            # 检查样本数量
            if len(df) < self.n_samples * 0.95:
                print(f"❌ 样本数量不足: {len(df)} < {self.n_samples * 0.95}")
                all_passed = False
            else:
                print(f"✅ 样本数量: {len(df)}")
            
            # 检查缺失率
            missing_rates = df.isnull().mean()
            high_missing = missing_rates[missing_rates > 0.4]
            if len(high_missing) > 0:
                print(f"❌ 高缺失率特征: {high_missing.to_dict()}")
                all_passed = False
            else:
                print(f"✅ 缺失率检查通过")
            
            # 检查常量列
            constant_cols = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                print(f"❌ 常量列: {constant_cols}")
                all_passed = False
            else:
                print(f"✅ 无常量列")
            
            # 检查重复样本
            duplicate_rate = df.duplicated().mean()
            if duplicate_rate > 0.001:
                print(f"❌ 重复样本率过高: {duplicate_rate:.4f}")
                all_passed = False
            else:
                print(f"✅ 重复样本率: {duplicate_rate:.4f}")
        
        return all_passed
    
    def calculate_feature_correlations(self, bank_df: pd.DataFrame, 
                                     ecom_df: pd.DataFrame, 
                                     labels: np.ndarray) -> Dict[str, float]:
        """计算特征与标签的相关性"""
        correlations = {}
        
        # 银行特征相关性
        for col in bank_df.select_dtypes(include=[np.number]).columns:
            corr = np.corrcoef(bank_df[col], labels)[0, 1]
            if not np.isnan(corr):
                correlations[f'bank_{col}'] = abs(corr)
        
        # 电商特征相关性
        for col in ecom_df.select_dtypes(include=[np.number]).columns:
            corr = np.corrcoef(ecom_df[col], labels)[0, 1]
            if not np.isnan(corr):
                correlations[f'ecom_{col}'] = abs(corr)
        
        # 检查有效信号数量
        strong_signals = {k: v for k, v in correlations.items() if v >= 0.1}
        
        self.data_profile['feature_correlations'] = correlations
        self.data_profile['strong_signals_count'] = len(strong_signals)
        
        print(f"\n📈 强信号特征数量: {len(strong_signals)}/6 (要求≥6)")
        for feature, corr in sorted(strong_signals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {corr:.3f}")
        
        return correlations
    
    def test_baseline_performance(self, bank_df: pd.DataFrame, 
                                ecom_df: pd.DataFrame, 
                                labels: np.ndarray) -> Dict[str, float]:
        """测试基线模型性能"""
        print("\n🎯 测试基线模型性能...")
        
        # 合并特征
        features = pd.concat([bank_df.select_dtypes(include=[np.number]), 
                            ecom_df.select_dtypes(include=[np.number])], axis=1)
        
        # 处理缺失值
        features = features.fillna(features.median())
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=self.seed, stratify=labels
        )
        
        results = {}
        
        # 逻辑回归
        lr = LogisticRegression(random_state=self.seed, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred)
        
        # 梯度提升
        gb = GradientBoostingClassifier(random_state=self.seed, n_estimators=100)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict_proba(X_test)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_pred)
        
        # 计算KS值
        def calculate_ks(y_true, y_prob):
            from scipy.stats import ks_2samp
            pos_scores = y_prob[y_true == 1]
            neg_scores = y_prob[y_true == 0]
            return ks_2samp(pos_scores, neg_scores).statistic
        
        lr_ks = calculate_ks(y_test, lr_pred)
        gb_ks = calculate_ks(y_test, gb_pred)
        
        results = {
            'logistic_auc': lr_auc,
            'logistic_ks': lr_ks,
            'gradient_boosting_auc': gb_auc,
            'gradient_boosting_ks': gb_ks,
            'best_auc': max(lr_auc, gb_auc),
            'best_ks': max(lr_ks, gb_ks)
        }
        
        self.data_profile['baseline_performance'] = results
        
        print(f"📊 逻辑回归 - AUC: {lr_auc:.3f}, KS: {lr_ks:.3f}")
        print(f"📊 梯度提升 - AUC: {gb_auc:.3f}, KS: {gb_ks:.3f}")
        
        # 检查性能要求
        if results['best_auc'] < 0.70 or results['best_ks'] < 0.25:
            print(f"❌ 基线性能不达标 (AUC: {results['best_auc']:.3f} < 0.70 或 KS: {results['best_ks']:.3f} < 0.25)")
            return None
        else:
            print(f"✅ 基线性能达标")
            return results
    
    def generate_dataset_readme(self, datasets: Dict[str, pd.DataFrame], 
                              labels: np.ndarray) -> str:
        """生成数据集说明文档"""
        readme_content = f"""# 联邦学习合成数据集说明

## 生成参数
- 样本数量: {self.n_samples:,}
- 交集比例: {self.overlap:.1%}
- 参与方: {', '.join(self.parties)}
- 随机种子: {self.seed}
- 坏账率: {self.bad_rate:.1%}
- 噪声水平: {self.noise}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览

### 标签分布
- 正样本 (违约): {labels.sum():,} ({labels.mean():.1%})
- 负样本 (正常): {(1-labels).sum():,} ({(1-labels).mean():.1%})

### 各方特征数量
"""
        
        for party, df in datasets.items():
            party_name = {'A': '银行', 'B': '电商', 'C': '运营商'}.get(party, party)
            readme_content += f"- {party_name}方: {len(df.columns)} 个特征\n"
        
        readme_content += "\n## 特征说明\n\n"
        
        # 银行特征说明
        if 'A' in datasets:
            readme_content += """### 银行方特征 (A)
- `annual_income`: 年收入
- `monthly_income`: 月收入  
- `debt_to_income`: 债务收入比 (↑违约风险)
- `cc_utilization`: 信用卡使用率 (↑违约风险)
- `total_debt`: 总债务
- `credit_len_yrs`: 信用历史长度 (↓违约风险)
- `credit_score`: 信用评分 (↓违约风险)
- `late_3m`: 近3月逾期标记 (↑违约风险)
- `delinq_12m`: 近12月违约次数 (↑违约风险)
- `bankruptcy_flag`: 破产标记
- `num_accounts`: 账户数量
- `account_age_months`: 账户年龄
- `employment_years`: 工作年限 (↓违约风险)
- `home_ownership`: 房屋所有权

"""
        
        # 电商特征说明
        if 'B' in datasets:
            readme_content += """### 电商方特征 (B)
- `order_cnt_6m`: 6月订单数 (↓违约风险)
- `monetary_6m`: 6月消费金额 (↓违约风险)
- `avg_order_value`: 平均订单价值
- `return_rate`: 退货率 (↑违约风险)
- `return_cnt_6m`: 6月退货次数
- `recency_days`: 最近购买天数 (↑违约风险)
- `frequency_score`: 购买频率评分
- `session_duration_avg`: 平均会话时长
- `midnight_orders_ratio`: 深夜下单比例 (↑违约风险)
- `weekend_orders_ratio`: 周末下单比例
- `mobile_orders_ratio`: 移动端下单比例
- `category_diversity`: 品类多样性
- `brand_loyalty_score`: 品牌忠诚度 (↓违约风险)
- `payment_failures`: 支付失败次数 (↑违约风险)
- `address_changes`: 地址变更次数

"""
        
        # 运营商特征说明
        if 'C' in datasets:
            readme_content += """### 运营商方特征 (C)
- `monthly_bill`: 月账单
- `bill_overdue_days`: 账单逾期天数 (↑违约风险)
- `payment_method`: 支付方式
- `data_usage_gb`: 数据使用量
- `call_minutes`: 通话时长
- `sms_count`: 短信数量
- `tenure_months`: 在网时长 (↓违约风险)
- `plan_changes`: 套餐变更次数 (↑违约风险)
- `service_calls`: 客服通话次数
- `roaming_usage`: 漫游使用
- `night_usage_ratio`: 夜间使用比例
- `weekend_usage_ratio`: 周末使用比例

"""
        
        readme_content += """## PSI标识符
- `psi_token`: SHA256(public_salt || phone) 用于隐私求交
- 仅用于数据对齐，不参与特征工程

## 数据质量保证
- 所有特征缺失率 < 40%
- 无全常量列
- 重复样本 < 0.1%
- 无 NaN/Inf 值
- 至少6个特征与标签相关性 |ρ| ≥ 0.1

## 基线性能
- 要求明文基线 AUC ≥ 0.70, KS ≥ 0.25
- 支持逻辑回归和梯度提升模型

## 使用说明
```bash
# 生成数据
python tools/seed/synth_vertical_v2.py --n 50000 --overlap 0.6 --parties A,B --bad_rate 0.12 --noise 0.15

# 验证数据合约
python tools/contract/data_contract.py --files partyA_bank.csv partyB_ecom.csv
```
"""
        
        return readme_content
    
    def generate(self, output_dir: str = "data/synth") -> bool:
        """生成数据集"""
        print(f"🚀 开始生成纵向联邦学习数据集...")
        print(f"📊 参数: n={self.n_samples}, overlap={self.overlap}, parties={self.parties}")
        print(f"📊 坏账率={self.bad_rate}, 噪声={self.noise}, 种子={self.seed}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\n🔄 尝试 {attempt + 1}/{max_retries}")
                
                # 1. 生成重叠标识符
                common_phones, party_tokens = self.create_overlapping_identifiers()
                
                # 2. 生成各方特征
                datasets = {}
                
                if 'A' in self.parties:
                    bank_df = self.generate_bank_features(self.n_samples)
                    bank_df['psi_token'] = party_tokens['A']
                    datasets['A'] = bank_df
                
                if 'B' in self.parties:
                    ecom_df = self.generate_ecom_features(self.n_samples)
                    ecom_df['psi_token'] = party_tokens['B']
                    datasets['B'] = ecom_df
                
                if 'C' in self.parties:
                    telco_df = self.generate_telco_features(self.n_samples)
                    telco_df['psi_token'] = party_tokens['C']
                    datasets['C'] = telco_df
                
                # 3. 生成标签（基于A方）
                if 'A' in datasets:
                    labels = self.generate_target_with_signal(
                        datasets['A'], 
                        datasets.get('B', pd.DataFrame()), 
                        datasets.get('C')
                    )
                    datasets['A']['default_flag'] = labels
                
                # 4. 验证数据质量
                if not self.validate_data_quality(datasets):
                    print(f"❌ 数据质量验证失败，重新生成...")
                    continue
                
                # 5. 计算特征相关性
                if 'A' in datasets and 'B' in datasets:
                    correlations = self.calculate_feature_correlations(
                        datasets['A'], datasets['B'], labels
                    )
                    
                    if self.data_profile['strong_signals_count'] < 6:
                        print(f"❌ 强信号特征不足 ({self.data_profile['strong_signals_count']}/6)，重新生成...")
                        continue
                
                # 6. 测试基线性能
                if 'A' in datasets and 'B' in datasets:
                    baseline_results = self.test_baseline_performance(
                        datasets['A'], datasets['B'], labels
                    )
                    
                    if baseline_results is None:
                        print(f"❌ 基线性能不达标，重新生成...")
                        continue
                
                # 7. 保存数据集
                file_mapping = {
                    'A': 'partyA_bank.csv',
                    'B': 'partyB_ecom.csv', 
                    'C': 'partyC_telco.csv'
                }
                
                for party, df in datasets.items():
                    filename = file_mapping[party]
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"💾 已保存: {filepath} ({len(df)} 行, {len(df.columns)} 列)")
                
                # 8. 保存数据概况
                profile_path = os.path.join(output_dir, 'data_profile.json')
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data_profile, f, indent=2, ensure_ascii=False)
                print(f"📊 数据概况已保存: {profile_path}")
                
                # 9. 生成README
                readme_content = self.generate_dataset_readme(datasets, labels)
                readme_path = os.path.join(output_dir, 'DATASET_README.md')
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                print(f"📖 数据说明已保存: {readme_path}")
                
                print(f"\n✅ 数据生成成功!")
                print(f"📊 交集大小: {len(common_phones):,} ({len(common_phones)/self.n_samples:.1%})")
                print(f"📊 坏账率: {labels.mean():.1%}")
                print(f"📊 基线AUC: {baseline_results['best_auc']:.3f}")
                print(f"📊 基线KS: {baseline_results['best_ks']:.3f}")
                
                return True
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                if attempt == max_retries - 1:
                    print(f"\n💡 修复建议:")
                    print(f"1. 检查随机种子设置")
                    print(f"2. 调整坏账率参数 (0.08-0.20)")
                    print(f"3. 增加噪声水平")
                    print(f"4. 检查特征权重设置")
                    return False
                continue
        
        return False


def main():
    parser = argparse.ArgumentParser(description='纵向联邦学习合成数据生成器')
    parser.add_argument('--n', type=int, default=50000, help='每方样本数量')
    parser.add_argument('--overlap', type=float, default=0.6, help='交集比例')
    parser.add_argument('--parties', type=str, default='A,B', help='参与方列表，逗号分隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--bad_rate', type=float, default=0.12, help='坏账率')
    parser.add_argument('--noise', type=float, default=0.15, help='噪声水平')
    parser.add_argument('--output', type=str, default='data/synth', help='输出目录')
    
    args = parser.parse_args()
    
    # 解析参与方
    parties = [p.strip().upper() for p in args.parties.split(',')]
    
    # 验证参数
    if not (0.08 <= args.bad_rate <= 0.20):
        print(f"❌ 坏账率必须在 0.08-0.20 之间，当前: {args.bad_rate}")
        sys.exit(1)
    
    if not (0.4 <= args.overlap <= 0.8):
        print(f"❌ 交集比例必须在 0.4-0.8 之间，当前: {args.overlap}")
        sys.exit(1)
    
    if args.n < 10000:
        print(f"❌ 样本数量不能少于 10000，当前: {args.n}")
        sys.exit(1)
    
    # 生成数据
    generator = VerticalFLDataGenerator(
        n_samples=args.n,
        overlap=args.overlap,
        parties=parties,
        seed=args.seed,
        bad_rate=args.bad_rate,
        noise=args.noise
    )
    
    success = generator.generate(args.output)
    
    if success:
        print(f"\n🎉 数据生成完成! 输出目录: {args.output}")
        sys.exit(0)
    else:
        print(f"\n❌ 数据生成失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()