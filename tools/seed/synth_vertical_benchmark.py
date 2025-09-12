#!/usr/bin/env python3
"""
纵向联邦学习基准数据生成器
支持多方数据生成，A方含标签，B/C方含行为特征
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_random_seed(seed: int):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)


def generate_base_features(n_samples: int, n_features: int, n_informative: int, 
                          random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成基础特征和标签"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=random_state
    )
    return X, y


def add_realistic_features(X: np.ndarray, y: np.ndarray, noise_level: float) -> pd.DataFrame:
    """添加现实的金融特征"""
    n_samples = X.shape[0]
    
    # 基础特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 构建现实特征
    df = pd.DataFrame()
    
    # 信用评分 (300-850)
    df['credit_score'] = 300 + (X_scaled[:, 0] + 3) * 91.67 + np.random.normal(0, 20, n_samples)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    
    # 年收入 (20k-200k)
    df['annual_income'] = 20000 + (X_scaled[:, 1] + 3) * 30000 + np.random.normal(0, 5000, n_samples)
    df['annual_income'] = np.clip(df['annual_income'], 20000, 200000)
    
    # 债务比率 (0-1)
    df['debt_ratio'] = 0.1 + (X_scaled[:, 2] + 3) * 0.15 + np.random.normal(0, 0.05, n_samples)
    df['debt_ratio'] = np.clip(df['debt_ratio'], 0, 1)
    
    # 就业年限
    df['employment_years'] = np.maximum(0, X_scaled[:, 3] * 5 + 10 + np.random.normal(0, 2, n_samples))
    
    # 房贷状态
    df['has_mortgage'] = (X_scaled[:, 0] + np.random.normal(0, 0.5, n_samples)) > 0
    
    # 信用卡数量
    df['num_credit_cards'] = np.maximum(0, np.round(X_scaled[:, 4] * 2 + 3 + np.random.normal(0, 1, n_samples)))
    
    # 银行关系年限
    df['bank_relationship_years'] = np.maximum(0, X_scaled[:, 1] * 3 + 5 + np.random.normal(0, 1, n_samples))
    
    # 月均消费
    df['monthly_spending'] = df['annual_income'] * 0.6 / 12 + np.random.normal(0, 500, n_samples)
    df['monthly_spending'] = np.maximum(0, df['monthly_spending'])
    
    # 储蓄账户余额
    df['savings_balance'] = df['annual_income'] * 0.2 + np.random.normal(0, 2000, n_samples)
    df['savings_balance'] = np.maximum(0, df['savings_balance'])
    
    # 逾期次数
    df['delinquency_count'] = np.maximum(0, np.round(-X_scaled[:, 0] + np.random.normal(0, 1, n_samples)))
    
    # 添加噪声
    if noise_level > 0:
        for col in df.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, df[col].std() * noise_level, n_samples)
            df[col] += noise
    
    # 添加标签
    df['default_risk'] = y
    
    # 生成唯一ID
    df['customer_id'] = [f'cust_{i:06d}' for i in range(n_samples)]
    
    return df


def split_vertical_data(df: pd.DataFrame, parties: int, overlap_rate: float) -> Dict[str, pd.DataFrame]:
    """按纵向联邦学习方式分割数据"""
    n_samples = len(df)
    overlap_size = int(n_samples * overlap_rate)
    
    # 生成重叠的客户ID
    all_ids = df['customer_id'].tolist()
    overlap_ids = random.sample(all_ids, overlap_size)
    
    party_data = {}
    
    # Party A: 标签方 (含标签和部分特征)
    party_a_features = ['customer_id', 'credit_score', 'annual_income', 'default_risk']
    party_a_df = df[df['customer_id'].isin(overlap_ids)][party_a_features].copy()
    party_data['party_a'] = party_a_df
    
    # 剩余特征分配给其他方
    remaining_features = [col for col in df.columns if col not in party_a_features]
    features_per_party = len(remaining_features) // (parties - 1)
    
    for i in range(1, parties):
        party_name = f'party_{chr(ord("b") + i - 1)}'
        start_idx = (i - 1) * features_per_party
        end_idx = start_idx + features_per_party if i < parties - 1 else len(remaining_features)
        
        party_features = ['customer_id'] + remaining_features[start_idx:end_idx]
        party_df = df[df['customer_id'].isin(overlap_ids)][party_features].copy()
        
        # 为非标签方添加一些独有客户
        unique_size = int(overlap_size * 0.3)  # 30%独有客户
        non_overlap_ids = [id for id in all_ids if id not in overlap_ids]
        if non_overlap_ids and unique_size > 0:
            unique_ids = random.sample(non_overlap_ids, min(unique_size, len(non_overlap_ids)))
            unique_df = df[df['customer_id'].isin(unique_ids)][party_features].copy()
            party_df = pd.concat([party_df, unique_df], ignore_index=True)
        
        party_data[party_name] = party_df
    
    return party_data


def generate_metadata(party_data: Dict[str, pd.DataFrame], config: Dict) -> Dict:
    """生成数据集元数据"""
    metadata = {
        'generation_config': config,
        'parties': {},
        'overlap_analysis': {},
        'feature_distribution': {},
        'quality_metrics': {}
    }
    
    # 分析每方数据
    all_ids = set()
    for party_name, df in party_data.items():
        party_ids = set(df['customer_id'])
        all_ids.update(party_ids)
        
        metadata['parties'][party_name] = {
            'sample_count': len(df),
            'feature_count': len(df.columns) - 1,  # 排除customer_id
            'features': [col for col in df.columns if col != 'customer_id'],
            'missing_rate': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'unique_customers': len(party_ids)
        }
    
    # 重叠分析
    party_ids = {name: set(df['customer_id']) for name, df in party_data.items()}
    intersection = set.intersection(*party_ids.values())
    union = set.union(*party_ids.values())
    
    metadata['overlap_analysis'] = {
        'intersection_size': len(intersection),
        'union_size': len(union),
        'overlap_rate': len(intersection) / len(union) if union else 0,
        'jaccard_similarity': len(intersection) / len(union) if union else 0
    }
    
    # 特征分布分析
    if 'party_a' in party_data and 'default_risk' in party_data['party_a'].columns:
        y = party_data['party_a']['default_risk']
        metadata['quality_metrics'] = {
            'label_distribution': {
                'positive_rate': float(y.mean()),
                'negative_rate': float(1 - y.mean()),
                'class_balance': float(min(y.mean(), 1 - y.mean()) / max(y.mean(), 1 - y.mean()))
            }
        }
    
    return metadata


def save_party_data(party_data: Dict[str, pd.DataFrame], output_dir: str, metadata: Dict):
    """保存各方数据和元数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存各方数据
    for party_name, df in party_data.items():
        csv_path = output_path / f'{party_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"保存 {party_name} 数据: {csv_path} ({len(df)} 样本, {len(df.columns)-1} 特征)")
    
    # 保存元数据
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"保存元数据: {metadata_path}")
    
    # 生成数据摘要
    summary_path = output_path / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("纵向联邦学习数据集摘要\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"总样本数: {metadata['overlap_analysis']['union_size']}\n")
        f.write(f"重叠样本数: {metadata['overlap_analysis']['intersection_size']}\n")
        f.write(f"重叠率: {metadata['overlap_analysis']['overlap_rate']:.3f}\n\n")
        
        f.write("各方数据统计:\n")
        for party_name, party_info in metadata['parties'].items():
            f.write(f"  {party_name}: {party_info['sample_count']} 样本, {party_info['feature_count']} 特征\n")
            f.write(f"    特征: {', '.join(party_info['features'])}\n")
        
        if 'label_distribution' in metadata['quality_metrics']:
            label_dist = metadata['quality_metrics']['label_distribution']
            f.write(f"\n标签分布:\n")
            f.write(f"  正样本率: {label_dist['positive_rate']:.3f}\n")
            f.write(f"  类别平衡度: {label_dist['class_balance']:.3f}\n")
    
    print(f"保存数据摘要: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='纵向联邦学习基准数据生成器')
    parser.add_argument('--n', type=int, default=10000, help='总样本数')
    parser.add_argument('--overlap', type=float, default=0.3, help='重叠率 (0-1)')
    parser.add_argument('--parties', type=int, default=2, help='参与方数量')
    parser.add_argument('--bad_rate', type=float, default=0.15, help='坏样本率')
    parser.add_argument('--noise', type=float, default=0.05, help='噪声水平')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.overlap < 0 or args.overlap > 1:
        raise ValueError("重叠率必须在0-1之间")
    if args.parties < 2:
        raise ValueError("参与方数量必须至少为2")
    if args.bad_rate < 0 or args.bad_rate > 1:
        raise ValueError("坏样本率必须在0-1之间")
    
    print(f"开始生成纵向联邦学习数据集...")
    print(f"参数: n={args.n}, overlap={args.overlap}, parties={args.parties}, bad_rate={args.bad_rate}")
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 生成基础特征
    n_features = 10
    n_informative = 6
    X, y_binary = generate_base_features(args.n, n_features, n_informative, args.seed)
    
    # 调整标签分布
    n_positive = int(args.n * args.bad_rate)
    y = np.zeros(args.n)
    positive_indices = np.random.choice(args.n, n_positive, replace=False)
    y[positive_indices] = 1
    
    # 生成现实特征
    df = add_realistic_features(X, y, args.noise)
    
    # 纵向分割数据
    party_data = split_vertical_data(df, args.parties, args.overlap)
    
    # 生成元数据
    config = {
        'n_samples': args.n,
        'overlap_rate': args.overlap,
        'parties': args.parties,
        'bad_rate': args.bad_rate,
        'noise_level': args.noise,
        'random_seed': args.seed,
        'generation_timestamp': pd.Timestamp.now().isoformat()
    }
    metadata = generate_metadata(party_data, config)
    
    # 保存数据
    save_party_data(party_data, args.output, metadata)
    
    print(f"\n数据生成完成！")
    print(f"输出目录: {args.output}")
    print(f"重叠客户数: {metadata['overlap_analysis']['intersection_size']}")
    print(f"实际重叠率: {metadata['overlap_analysis']['overlap_rate']:.3f}")


if __name__ == '__main__':
    main()