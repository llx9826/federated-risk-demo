#!/usr/bin/env python3
"""
数据合约校验器
实现严格的数据质量检查，失败即停
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


class DataContractValidator:
    """数据合约校验器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化校验器
        
        Args:
            config: 校验配置
        """
        self.config = config or self._get_default_config()
        self.violations = []
        self.data_profile = {
            'validation_time': datetime.now().isoformat(),
            'config': self.config,
            'datasets': {},
            'violations': [],
            'quality_metrics': {},
            'recommendations': []
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据层护栏
            'min_samples': 1000,  # 交集≥1000
            'max_missing_rate': 0.4,
            'max_duplicate_rate': 0.001,
            'min_overlap_ratio': 0.58,  # 0.6 ± 0.02
            'max_overlap_ratio': 0.62,
            'min_bad_rate': 0.08,
            'max_bad_rate': 0.20,
            'min_class_ratio': 0.03,
            'min_correlation_threshold': 0.1,  # ≥6个特征|ρ|≥0.1
            'min_strong_signals': 6,
            'min_baseline_auc': 0.70,
            'max_constant_rate': 0.95,  # 常量/NaN/Inf剔除
            'max_inf_rate': 0.01,
            'min_baseline_ks': 0.25,
            'required_bank_features': [
                'debt_to_income', 'cc_utilization', 'late_3m', 
                'delinq_12m', 'annual_income', 'credit_len_yrs'
            ],
            'required_ecom_features': [
                'return_rate', 'recency_days', 'midnight_orders_ratio',
                'order_cnt_6m', 'monetary_6m'
            ],
            'forbidden_patterns': [
                '未来', 'future', '是否逾期', '是否违约', '是否拒绝',
                'overdue_flag', 'default_flag', 'reject_flag'
            ]
        }
    
    def add_violation(self, level: str, category: str, message: str, 
                     suggestion: str = None, data: Dict = None):
        """添加违规记录"""
        violation = {
            'level': level,  # 'error', 'warning', 'info'
            'category': category,
            'message': message,
            'suggestion': suggestion,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }
        self.violations.append(violation)
        self.data_profile['violations'].append(violation)
        
        # 打印违规信息
        icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(level, '•')
        print(f"{icon} [{category}] {message}")
        if suggestion:
            print(f"   💡 建议: {suggestion}")
    
    def validate_structure(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证数据结构"""
        print("\n🔍 验证数据结构...")
        
        passed = True
        
        # 检查参与方数量
        if len(datasets) < 2:
            self.add_violation(
                'error', 'structure', 
                f"参与方数量不足: {len(datasets)} < 2",
                "确保至少有两个参与方"
            )
            passed = False
        
        # 检查交集大小（数据层护栏）
        min_samples = min(len(df) for df in datasets.values())
        if min_samples < self.config['min_samples']:
            self.add_violation(
                'error', 'structure',
                f"交集样本数不足: {min_samples} < {self.config['min_samples']}",
                "增加数据量或检查PSI对齐结果"
            )
            passed = False
        
        # 检查数据一致性
        sample_counts = {party: len(df) for party, df in datasets.items()}
        if len(set(sample_counts.values())) > 1:
            self.add_violation(
                'error', 'structure',
                f"各方样本数不一致: {sample_counts}",
                "检查PSI对齐过程"
            )
            passed = False
        
        # 检查必需的参与方
        if 'A' not in datasets:
            self.add_violation(
                'error', 'structure',
                "缺少银行方(A)数据",
                "银行方数据必须包含标签列"
            )
            passed = False
        
        if 'B' not in datasets:
            self.add_violation(
                'error', 'structure',
                "缺少电商方(B)数据",
                "电商方数据用于特征补充"
            )
            passed = False
        
        # 检查样本数量
        for party, df in datasets.items():
            if len(df) < self.config['min_samples']:
                self.add_violation(
                    'error', 'structure',
                    f"{party}方样本数量不足: {len(df)} < {self.config['min_samples']}",
                    f"增加样本数量到至少 {self.config['min_samples']}"
                )
                passed = False
            
            # 记录数据集信息
            self.data_profile['datasets'][party] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        return passed
    
    def validate_psi_tokens(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证PSI标识符"""
        print("\n🔍 验证PSI标识符...")
        
        passed = True
        
        # 检查PSI列存在性
        for party, df in datasets.items():
            if 'psi_token' not in df.columns:
                self.add_violation(
                    'error', 'psi',
                    f"{party}方缺少psi_token列",
                    "添加psi_token列用于隐私求交"
                )
                passed = False
                continue
            
            # 检查唯一性
            unique_rate = df['psi_token'].nunique() / len(df)
            if unique_rate < 0.999:
                self.add_violation(
                    'error', 'psi',
                    f"{party}方psi_token重复率过高: {1-unique_rate:.4f} > 0.001",
                    "检查PSI令牌生成逻辑，确保唯一性"
                )
                passed = False
            
            # 检查格式
            if not all(isinstance(token, str) and len(token) == 64 for token in df['psi_token'].head(100)):
                self.add_violation(
                    'warning', 'psi',
                    f"{party}方psi_token格式异常",
                    "PSI令牌应为64位十六进制字符串(SHA256)"
                )
        
        # 检查交集比例
        if 'A' in datasets and 'B' in datasets and passed:
            tokens_a = set(datasets['A']['psi_token'])
            tokens_b = set(datasets['B']['psi_token'])
            intersection = tokens_a & tokens_b
            
            overlap_ratio = len(intersection) / min(len(tokens_a), len(tokens_b))
            
            if not (self.config['min_overlap_ratio'] <= overlap_ratio <= self.config['max_overlap_ratio']):
                self.add_violation(
                    'error', 'psi',
                    f"交集比例异常: {overlap_ratio:.3f} 不在 [{self.config['min_overlap_ratio']:.2f}, {self.config['max_overlap_ratio']:.2f}] 范围内",
                    "调整数据生成参数，确保合理的交集比例"
                )
                passed = False
            
            self.data_profile['quality_metrics']['overlap_ratio'] = overlap_ratio
            self.data_profile['quality_metrics']['intersection_size'] = len(intersection)
        
        return passed
    
    def validate_features(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证特征质量"""
        print("\n🔍 验证特征质量...")
        
        passed = True
        
        for party, df in datasets.items():
            print(f"\n📊 检查 {party} 方特征:")
            
            # 排除PSI列和标签列
            feature_cols = [col for col in df.columns 
                          if col not in ['psi_token', 'default_flag']]
            
            if len(feature_cols) == 0:
                self.add_violation(
                    'error', 'features',
                    f"{party}方无有效特征列",
                    "添加业务特征列"
                )
                passed = False
                continue
            
            # 检查缺失率
            missing_rates = df[feature_cols].isnull().mean()
            high_missing = missing_rates[missing_rates > self.config['max_missing_rate']]
            
            if len(high_missing) > 0:
                self.add_violation(
                    'error', 'features',
                    f"{party}方高缺失率特征: {dict(high_missing)}",
                    f"处理缺失值或移除缺失率>{self.config['max_missing_rate']*100}%的特征"
                )
                passed = False
            
            # 检查常量列（数据层护栏）
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
            constant_cols = []
            near_constant_cols = []
            
            for col in numeric_cols:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
                elif df[col].std() == 0:
                    constant_cols.append(col)
                else:
                    # 检查近似常量列（95%以上相同值）
                    value_counts = df[col].value_counts(normalize=True)
                    if value_counts.iloc[0] > self.config['max_constant_rate']:
                        near_constant_cols.append((col, value_counts.iloc[0]))
            
            if constant_cols:
                self.add_violation(
                    'error', 'features',
                    f"{party}方常量列: {constant_cols}",
                    "移除无变化的特征列"
                )
                passed = False
            
            if near_constant_cols:
                for col, ratio in near_constant_cols:
                    self.add_violation(
                        'warning', 'features',
                        f"{party}方近似常量列 {col}: {ratio:.1%}相同值",
                        "考虑移除或重新设计该特征"
                    )
            
            # 检查异常值
            for col in numeric_cols:
                if col not in constant_cols:
                    # 检查无穷值（数据层护栏）
                    inf_count = np.isinf(df[col]).sum()
                    inf_rate = inf_count / len(df)
                    if inf_rate > self.config['max_inf_rate']:
                        self.add_violation(
                            'error', 'features',
                            f"{party}方特征{col}无穷值比例过高: {inf_rate:.1%}",
                            "处理无穷值或移除该特征"
                        )
                        passed = False
                    elif inf_count > 0:
                        self.add_violation(
                            'warning', 'features',
                            f"{party}方特征{col}包含{inf_count}个无穷值",
                            "建议处理无穷值"
                        )
                    
                    # 检查NaN值
                    nan_count = df[col].isna().sum()
                    if nan_count > len(df) * self.config['max_missing_rate']:
                        self.add_violation(
                            'warning', 'features',
                            f"{party}方特征{col}缺失率过高: {nan_count/len(df):.1%}",
                            "考虑特征工程或移除该特征"
                        )
            
            # 检查重复样本
            duplicate_rate = df.duplicated().mean()
            if duplicate_rate > self.config['max_duplicate_rate']:
                self.add_violation(
                    'error', 'features',
                    f"{party}方重复样本率过高: {duplicate_rate:.4f} > {self.config['max_duplicate_rate']}",
                    "去除重复样本"
                )
                passed = False
            
            # 检查必需特征
            required_features = {
                'A': self.config['required_bank_features'],
                'B': self.config['required_ecom_features']
            }.get(party, [])
            
            missing_required = [f for f in required_features if f not in df.columns]
            if missing_required:
                self.add_violation(
                    'error', 'features',
                    f"{party}方缺少必需特征: {missing_required}",
                    "添加缺少的核心业务特征"
                )
                passed = False
        
        return passed
    
    def validate_labels(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证标签质量"""
        print("\n🔍 验证标签质量...")
        
        passed = True
        
        if 'A' not in datasets:
            return passed
        
        df = datasets['A']
        
        # 检查标签列存在性
        if 'default_flag' not in df.columns:
            self.add_violation(
                'error', 'labels',
                "银行方缺少default_flag标签列",
                "添加二分类标签列"
            )
            return False
        
        labels = df['default_flag']
        
        # 检查标签类型
        unique_labels = labels.unique()
        if not set(unique_labels).issubset({0, 1}):
            self.add_violation(
                'error', 'labels',
                f"标签值异常: {unique_labels}，应为0/1",
                "确保标签为二分类(0/1)"
            )
            passed = False
        
        # 检查标签分布（数据层护栏：标签两类存在）
        if len(unique_labels) < 2:
            self.add_violation(
                'error', 'labels',
                f"标签类别不足: {len(unique_labels)} < 2",
                "确保正负样本都存在"
            )
            passed = False
        
        # 检查标签缺失
        missing_labels = labels.isnull().sum()
        if missing_labels > 0:
            self.add_violation(
                'error', 'labels',
                f"标签存在{missing_labels}个缺失值",
                "处理标签缺失值"
            )
            passed = False
        
        # 检查类别平衡（数据层护栏：bad_rate∈[0.08,0.20]）
        bad_rate = labels.mean()
        if not (self.config['min_bad_rate'] <= bad_rate <= self.config['max_bad_rate']):
            self.add_violation(
                'error', 'labels',
                f"坏账率异常: {bad_rate:.3f} 不在 [{self.config['min_bad_rate']:.2f}, {self.config['max_bad_rate']:.2f}] 范围内",
                "调整标签生成逻辑或重新采样"
            )
            passed = False
            
        # 记录详细标签统计
        positive_count = int(labels.sum())
        negative_count = int((1 - labels).sum())
        print(f"📊 标签分布: 正样本 {positive_count} ({bad_rate:.1%}), 负样本 {negative_count} ({1-bad_rate:.1%})")
        
        # 检查最小类别比例
        min_class_ratio = min(bad_rate, 1 - bad_rate)
        if min_class_ratio < self.config['min_class_ratio']:
            self.add_violation(
                'error', 'labels',
                f"最小类别比例过低: {min_class_ratio:.3f} < {self.config['min_class_ratio']}",
                "增加少数类样本或启用重加权"
            )
            passed = False
        
        self.data_profile['quality_metrics']['bad_rate'] = bad_rate
        self.data_profile['quality_metrics']['label_distribution'] = {
            'positive': int(labels.sum()),
            'negative': int((1 - labels).sum())
        }
        
        return passed
    
    def validate_data_leakage(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证数据泄漏"""
        print("\n🔍 验证数据泄漏...")
        
        passed = True
        
        for party, df in datasets.items():
            # 检查列名中的泄漏模式
            for col in df.columns:
                col_lower = col.lower()
                for pattern in self.config['forbidden_patterns']:
                    if pattern.lower() in col_lower:
                        self.add_violation(
                            'error', 'leakage',
                            f"{party}方特征'{col}'可能包含未来信息",
                            f"移除或重命名包含'{pattern}'的特征"
                        )
                        passed = False
            
            # 检查时间相关特征
            time_related_cols = [col for col in df.columns 
                               if any(keyword in col.lower() 
                                    for keyword in ['date', 'time', 'day', 'month', 'year'])]
            
            if time_related_cols:
                self.add_violation(
                    'warning', 'leakage',
                    f"{party}方包含时间相关特征: {time_related_cols}",
                    "确认时间特征不包含未来信息"
                )
        
        return passed
    
    def validate_signal_strength(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证信号强度"""
        print("\n🔍 验证信号强度...")
        
        passed = True
        
        if 'A' not in datasets or 'default_flag' not in datasets['A'].columns:
            return passed
        
        labels = datasets['A']['default_flag']
        correlations = {}
        
        # 计算各方特征与标签的相关性
        for party, df in datasets.items():
            feature_cols = [col for col in df.columns 
                          if col not in ['psi_token', 'default_flag']]
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    # 计算点双列相关系数
                    corr = np.corrcoef(df[col].fillna(df[col].median()), labels)[0, 1]
                    if not np.isnan(corr):
                        correlations[f'{party}_{col}'] = abs(corr)
                except:
                    continue
        
        # 检查强信号数量（数据层护栏：≥6个特征|ρ|≥0.1）
        strong_signals = {k: v for k, v in correlations.items() 
                         if v >= self.config['min_correlation_threshold']}
        
        if len(strong_signals) < self.config['min_strong_signals']:
            self.add_violation(
                'error', 'signal',
                f"强信号特征不足: {len(strong_signals)}/{self.config['min_strong_signals']}",
                "增加与标签相关的特征或调整特征工程"
            )
            passed = False
        
        # 检查是否存在过强信号（可能数据泄露）
        very_strong_signals = {k: v for k, v in correlations.items() if v > 0.8}
        if very_strong_signals:
            self.add_violation(
                'warning', 'signal',
                f"发现过强信号特征: {very_strong_signals}",
                "检查是否存在数据泄露"
            )
        
        # 记录相关性信息
        self.data_profile['quality_metrics']['feature_correlations'] = correlations
        self.data_profile['quality_metrics']['strong_signals'] = strong_signals
        
        print(f"📈 强信号特征: {len(strong_signals)}/{self.config['min_strong_signals']}")
        for feature, corr in sorted(strong_signals.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {corr:.3f}")
        
        return passed
    
    def validate_baseline_performance(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """验证基线性能"""
        print("\n🔍 验证基线性能...")
        
        passed = True
        
        if 'A' not in datasets or 'B' not in datasets:
            return passed
        
        if 'default_flag' not in datasets['A'].columns:
            return passed
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            
            # 准备数据
            bank_features = datasets['A'].select_dtypes(include=[np.number]).drop(columns=['default_flag'], errors='ignore')
            ecom_features = datasets['B'].select_dtypes(include=[np.number])
            
            # 合并特征
            features = pd.concat([bank_features, ecom_features], axis=1)
            features = features.fillna(features.median())
            
            labels = datasets['A']['default_flag']
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练逻辑回归
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            
            # 计算AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # 计算KS
            def calculate_ks(y_true, y_prob):
                from scipy.stats import ks_2samp
                pos_scores = y_prob[y_true == 1]
                neg_scores = y_prob[y_true == 0]
                return ks_2samp(pos_scores, neg_scores).statistic
            
            ks = calculate_ks(y_test, y_pred_proba)
            
            # 检查性能要求
            if auc < self.config['min_baseline_auc']:
                self.add_violation(
                    'error', 'performance',
                    f"基线AUC不达标: {auc:.3f} < {self.config['min_baseline_auc']}",
                    "增强特征工程或调整数据生成逻辑"
                )
                passed = False
            
            if ks < self.config['min_baseline_ks']:
                self.add_violation(
                    'error', 'performance',
                    f"基线KS不达标: {ks:.3f} < {self.config['min_baseline_ks']}",
                    "增强特征区分度或调整标签生成"
                )
                passed = False
            
            self.data_profile['quality_metrics']['baseline_performance'] = {
                'auc': auc,
                'ks': ks,
                'feature_count': len(features.columns)
            }
            
            print(f"📊 基线性能 - AUC: {auc:.3f}, KS: {ks:.3f}")
            
        except Exception as e:
            self.add_violation(
                'warning', 'performance',
                f"基线性能测试失败: {e}",
                "检查数据格式和依赖库"
            )
        
        return passed
    
    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 按违规类别分组
        error_violations = [v for v in self.violations if v['level'] == 'error']
        
        if error_violations:
            recommendations.append("🚨 严重问题需要立即修复:")
            
            # 结构问题
            structure_errors = [v for v in error_violations if v['category'] == 'structure']
            if structure_errors:
                recommendations.append("  📁 数据结构问题:")
                for v in structure_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
            
            # PSI问题
            psi_errors = [v for v in error_violations if v['category'] == 'psi']
            if psi_errors:
                recommendations.append("  🔗 PSI标识符问题:")
                for v in psi_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
            
            # 特征问题
            feature_errors = [v for v in error_violations if v['category'] == 'features']
            if feature_errors:
                recommendations.append("  📊 特征质量问题:")
                for v in feature_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
            
            # 标签问题
            label_errors = [v for v in error_violations if v['category'] == 'labels']
            if label_errors:
                recommendations.append("  🎯 标签质量问题:")
                for v in label_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
            
            # 信号强度问题
            signal_errors = [v for v in error_violations if v['category'] == 'signal']
            if signal_errors:
                recommendations.append("  📈 信号强度问题:")
                for v in signal_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
            
            # 性能问题
            perf_errors = [v for v in error_violations if v['category'] == 'performance']
            if perf_errors:
                recommendations.append("  🎯 性能问题:")
                for v in perf_errors:
                    recommendations.append(f"    - {v['message']}")
                    if v['suggestion']:
                        recommendations.append(f"      💡 {v['suggestion']}")
        
        # 通用建议
        if error_violations:
            recommendations.extend([
                "",
                "🔧 通用修复步骤:",
                "  1. 重新生成数据: python tools/seed/synth_vertical_v2.py --n 50000 --bad_rate 0.12",
                "  2. 调整生成参数: 修改 --overlap, --noise, --seed 参数",
                "  3. 检查特征工程: 确保特征与业务逻辑一致",
                "  4. 验证数据质量: python tools/contract/data_contract.py --files *.csv"
            ])
        
        self.data_profile['recommendations'] = recommendations
        return recommendations
    
    def apply_data_guards(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """应用数据层护栏，自动修复可修复的问题"""
        print("\n🛡️ 应用数据层护栏...")
        
        cleaned_datasets = {}
        
        for party, df in datasets.items():
            cleaned_df = df.copy()
            
            # 移除常量列
            feature_cols = [col for col in cleaned_df.columns 
                          if col not in ['psi_token', 'default_flag']]
            numeric_cols = cleaned_df[feature_cols].select_dtypes(include=[np.number]).columns
            
            constant_cols = []
            for col in numeric_cols:
                if cleaned_df[col].nunique() <= 1 or cleaned_df[col].std() == 0:
                    constant_cols.append(col)
            
            if constant_cols:
                cleaned_df = cleaned_df.drop(columns=constant_cols)
                print(f"🧹 {party}方移除常量列: {constant_cols}")
            
            # 处理无穷值
            for col in numeric_cols:
                if col in cleaned_df.columns:
                    inf_mask = np.isinf(cleaned_df[col])
                    if inf_mask.any():
                        # 用中位数替换无穷值
                        median_val = cleaned_df[col][~inf_mask].median()
                        cleaned_df.loc[inf_mask, col] = median_val
                        print(f"🔧 {party}方特征{col}的{inf_mask.sum()}个无穷值已替换为中位数")
            
            # 处理NaN值
            for col in numeric_cols:
                if col in cleaned_df.columns:
                    nan_count = cleaned_df[col].isnull().sum()
                    if nan_count > 0:
                        median_val = cleaned_df[col].median()
                        cleaned_df[col].fillna(median_val, inplace=True)
                        print(f"🔧 {party}方特征{col}的{nan_count}个缺失值已填充")
            
            cleaned_datasets[party] = cleaned_df
        
        return cleaned_datasets
    
    def validate(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """执行完整验证"""
        print(f"🚀 开始数据合约验证...")
        print(f"📊 数据集: {list(datasets.keys())}")
        
        # 应用数据层护栏
        cleaned_datasets = self.apply_data_guards(datasets)
        
        all_passed = True
        
        # 1. 结构验证
        if not self.validate_structure(cleaned_datasets):
            all_passed = False
        
        # 2. PSI标识符验证
        if not self.validate_psi_tokens(cleaned_datasets):
            all_passed = False
        
        # 3. 特征质量验证
        if not self.validate_features(cleaned_datasets):
            all_passed = False
        
        # 4. 标签质量验证
        if not self.validate_labels(cleaned_datasets):
            all_passed = False
        
        # 5. 数据泄漏验证
        if not self.validate_data_leakage(cleaned_datasets):
            all_passed = False
        
        # 6. 信号强度验证
        if not self.validate_signal_strength(cleaned_datasets):
            all_passed = False
        
        # 7. 基线性能验证
        if not self.validate_baseline_performance(cleaned_datasets):
            all_passed = False
        
        # 生成建议
        recommendations = self.generate_recommendations()
        
        # 输出结果
        print(f"\n{'='*60}")
        if all_passed:
            print("✅ 数据合约验证通过!")
        else:
            print("❌ 数据合约验证失败!")
            print(f"\n📋 修复建议:")
            for rec in recommendations:
                print(rec)
            
            # 如果有错误，直接中止并给出修复建议
            error_count = len([v for v in self.violations if v['level'] == 'error'])
            if error_count > 0:
                print("\n🛑 发现严重错误，训练中止")
                print("\n📋 修复建议:")
                for violation in self.violations:
                    if violation['level'] == 'error' and violation['suggestion']:
                        print(f"  • {violation['suggestion']}")
        
        print(f"\n📊 验证统计:")
        error_count = len([v for v in self.violations if v['level'] == 'error'])
        warning_count = len([v for v in self.violations if v['level'] == 'warning'])
        print(f"  错误: {error_count}")
        print(f"  警告: {warning_count}")
        print(f"  总计: {len(self.violations)}")
        
        return all_passed
    
    def save_profile(self, output_path: str):
        """保存数据概况"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_profile, f, indent=2, ensure_ascii=False)
        print(f"📊 数据概况已保存: {output_path}")


def load_datasets(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """加载数据集"""
    datasets = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
        
        try:
            # 根据文件名推断参与方
            filename = os.path.basename(file_path).lower()
            if 'bank' in filename or 'partya' in filename:
                party = 'A'
            elif 'ecom' in filename or 'partyb' in filename:
                party = 'B'
            elif 'telco' in filename or 'partyc' in filename:
                party = 'C'
            else:
                party = filename.split('.')[0].upper()
            
            # 加载数据
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                print(f"⚠️ 不支持的文件格式: {file_path}")
                continue
            
            datasets[party] = df
            print(f"📁 已加载 {party} 方数据: {file_path} ({len(df)} 行, {len(df.columns)} 列)")
            
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(description='数据合约校验器')
    parser.add_argument('--files', nargs='+', required=True, help='数据文件路径列表')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output', type=str, default='data_profile.json', help='输出文件路径')
    parser.add_argument('--strict', action='store_true', help='严格模式，任何错误都返回失败')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 加载数据集
    datasets = load_datasets(args.files)
    
    if not datasets:
        print("❌ 没有成功加载任何数据集")
        sys.exit(1)
    
    # 执行验证
    validator = DataContractValidator(config)
    success = validator.validate(datasets)
    
    # 保存结果
    validator.save_profile(args.output)
    
    # 返回结果
    if success:
        print(f"\n🎉 数据合约验证成功!")
        sys.exit(0)
    else:
        print(f"\n💥 数据合约验证失败!")
        if args.strict:
            sys.exit(1)
        else:
            print("⚠️ 非严格模式，继续执行")
            sys.exit(0)


if __name__ == '__main__':
    main()