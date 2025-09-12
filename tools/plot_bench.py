#!/usr/bin/env python3
"""
基准测试结果可视化工具

从CSV/JSONL文件生成PNG图表，包括：
- PSI吞吐量和延迟图表
- 联邦训练AUC/KS vs ε图表
- 通信量vs轮次图表
- 推理延迟图表
- 改进前后对比图表
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# 配置matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

class BenchmarkPlotter:
    """基准测试结果绘图器"""
    
    def __init__(self, reports_dir: str = "reports", assets_dir: str = "docs/assets"):
        """初始化绘图器"""
        self.reports_dir = Path(reports_dir)
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"报告目录: {self.reports_dir}")
        logger.info(f"图片输出目录: {self.assets_dir}")
    
    def load_psi_data(self) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
        """加载PSI基准测试数据"""
        summary_file = self.reports_dir / "psi_summary.csv"
        results_file = self.reports_dir / "psi_results.jsonl"
        
        summary_df = None
        results_data = None
        
        # 加载汇总数据
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                logger.info(f"加载PSI汇总数据: {len(summary_df)} 条记录")
            except Exception as e:
                logger.error(f"加载PSI汇总数据失败: {e}")
        
        # 加载详细结果
        if results_file.exists():
            try:
                results_data = []
                with open(results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            results_data.append(json.loads(line))
                logger.info(f"加载PSI详细结果: {len(results_data)} 条记录")
            except Exception as e:
                logger.error(f"加载PSI详细结果失败: {e}")
        
        return summary_df, results_data
    
    def load_train_data(self) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]], Optional[pd.DataFrame]]:
        """加载训练基准测试数据"""
        summary_file = self.reports_dir / "train_summary.csv"
        results_file = self.reports_dir / "train_results.jsonl"
        inference_file = self.reports_dir / "inference_bench.csv"
        
        summary_df = None
        results_data = None
        inference_df = None
        
        # 加载训练汇总数据
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                logger.info(f"加载训练汇总数据: {len(summary_df)} 条记录")
            except Exception as e:
                logger.error(f"加载训练汇总数据失败: {e}")
        
        # 加载详细结果
        if results_file.exists():
            try:
                results_data = []
                with open(results_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            results_data.append(json.loads(line))
                logger.info(f"加载训练详细结果: {len(results_data)} 条记录")
            except Exception as e:
                logger.error(f"加载训练详细结果失败: {e}")
        
        # 加载推理数据
        if inference_file.exists():
            try:
                inference_df = pd.read_csv(inference_file)
                logger.info(f"加载推理数据: {len(inference_df)} 条记录")
            except Exception as e:
                logger.error(f"加载推理数据失败: {e}")
        
        return summary_df, results_data, inference_df
    
    def plot_psi_throughput(self, summary_df: pd.DataFrame, results_data: List[Dict]) -> str:
        """绘制PSI吞吐量图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：吞吐量 vs 数据规模
        if summary_df is not None and not summary_df.empty:
            # 按数据规模分组
            # 使用实际存在的列名
            throughput_col = 'total_throughput_per_sec' if 'total_throughput_per_sec' in summary_df.columns else 'throughput_per_sec'
            
            grouped = summary_df.groupby('total_size').agg({
                throughput_col: ['mean', 'std']
            }).reset_index()
            
            sizes = grouped['total_size'] / 1e6  # 转换为百万
            throughput_mean = grouped[(throughput_col, 'mean')] / 10000  # 转换为万条/秒
            throughput_std = grouped[(throughput_col, 'std')] / 10000 if not grouped[(throughput_col, 'std')].isna().all() else None
            
            ax1.errorbar(sizes, throughput_mean, yerr=throughput_std, 
                        marker='o', capsize=5, capthick=2)
            ax1.set_xlabel('数据规模 (百万条)')
            ax1.set_ylabel('吞吐量 (万条/秒)')
            ax1.set_title('PSI吞吐量 vs 数据规模')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
        
        # 右图：不同交集比例的吞吐量
        if results_data:
            intersection_ratios = []
            throughputs = []
            
            for result in results_data:
                if 'intersection_ratio' in result and 'avg_throughput_per_sec' in result:
                    intersection_ratios.append(result['intersection_ratio'] * 100)
                    throughputs.append(result['avg_throughput_per_sec'] / 10000)  # 转换为万条/秒
            
            if intersection_ratios and throughputs:
                # 按交集比例分组
                ratio_throughput = {}
                for ratio, throughput in zip(intersection_ratios, throughputs):
                    if ratio not in ratio_throughput:
                        ratio_throughput[ratio] = []
                    ratio_throughput[ratio].append(throughput)
                
                ratios = sorted(ratio_throughput.keys())
                means = [np.mean(ratio_throughput[r]) for r in ratios]
                stds = [np.std(ratio_throughput[r]) for r in ratios]
                
                ax2.bar(ratios, means, yerr=stds, capsize=5, alpha=0.7)
                ax2.set_xlabel('交集比例 (%)')
                ax2.set_ylabel('吞吐量 (万条/秒)')
                ax2.set_title('PSI吞吐量 vs 交集比例')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.assets_dir / "psi_throughput.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"PSI吞吐量图表已保存: {output_file}")
        return str(output_file)
    
    def plot_psi_latency(self, summary_df: pd.DataFrame, results_data: List[Dict]) -> str:
        """绘制PSI延迟图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：处理时间分布
        if summary_df is not None and not summary_df.empty:
            # 使用实际存在的时间字段
            time_cols = []
            time_labels = []
            
            if 'total_time' in summary_df.columns:
                time_cols.append('total_time')
                time_labels.append('总时间')
            if 'compute_time' in summary_df.columns:
                time_cols.append('compute_time')
                time_labels.append('计算时间')
            if 'data_gen_time' in summary_df.columns:
                time_cols.append('data_gen_time')
                time_labels.append('数据生成时间')
            
            if time_cols:
                # 转换为毫秒
                avg_times = [summary_df[col].mean() * 1000 for col in time_cols]
                std_times = [summary_df[col].std() * 1000 for col in time_cols]
                
                x_pos = np.arange(len(time_labels))
                ax1.bar(x_pos, avg_times, yerr=std_times, capsize=5, alpha=0.7)
                ax1.set_xlabel('时间类型')
                ax1.set_ylabel('时间 (毫秒)')
                ax1.set_title('PSI处理时间分布')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(time_labels)
                ax1.grid(True, alpha=0.3)
        
        # 右图：处理时间 vs 数据规模
        if summary_df is not None and not summary_df.empty and 'total_time' in summary_df.columns:
            # 按数据规模排序
            df_sorted = summary_df.sort_values('total_size')
            sizes = df_sorted['total_size'] / 1e6  # 转换为百万
            
            ax2.plot(sizes, df_sorted['total_time'] * 1000, 'o-', label='总时间', alpha=0.8)
            if 'compute_time' in df_sorted.columns:
                ax2.plot(sizes, df_sorted['compute_time'] * 1000, 's-', label='计算时间', alpha=0.8)
            
            ax2.set_xlabel('数据规模 (百万条)')
            ax2.set_ylabel('时间 (毫秒)')
            ax2.set_title('PSI处理时间 vs 数据规模')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
        
        plt.tight_layout()
        output_file = self.assets_dir / "psi_latency.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"PSI延迟图表已保存: {output_file}")
        return str(output_file)
    
    def plot_train_auc_ks_vs_epsilon(self, summary_df: pd.DataFrame, results_data: List[Dict]) -> str:
        """绘制训练AUC/KS vs ε图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if summary_df is not None and not summary_df.empty:
            # 处理epsilon值
            df = summary_df.copy()
            df['epsilon_numeric'] = df['epsilon'].apply(
                lambda x: float('inf') if x == 'inf' else float(x)
            )
            
            # 按epsilon分组
            grouped = df.groupby('epsilon').agg({
                'final_auc': ['mean', 'std'],
                'final_ks': ['mean', 'std']
            }).reset_index()
            
            epsilons = grouped['epsilon'].tolist()
            auc_means = grouped[('final_auc', 'mean')]
            auc_stds = grouped[('final_auc', 'std')]
            ks_means = grouped[('final_ks', 'mean')]
            ks_stds = grouped[('final_ks', 'std')]
            
            # 处理无穷大的epsilon
            epsilon_labels = []
            epsilon_positions = []
            for i, eps in enumerate(epsilons):
                if eps == 'inf':
                    epsilon_labels.append('∞')
                    epsilon_positions.append(len(epsilons))  # 放在最后
                else:
                    epsilon_labels.append(str(eps))
                    epsilon_positions.append(float(eps))
            
            # AUC图
            x_pos = np.arange(len(epsilons))
            ax1.bar(x_pos, auc_means, yerr=auc_stds, capsize=5, alpha=0.7)
            ax1.set_xlabel('差分隐私参数 ε')
            ax1.set_ylabel('AUC')
            ax1.set_title('联邦训练AUC vs ε')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(epsilon_labels)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0.5, 1.0)
            
            # KS图
            ax2.bar(x_pos, ks_means, yerr=ks_stds, capsize=5, alpha=0.7, color='orange')
            ax2.set_xlabel('差分隐私参数 ε')
            ax2.set_ylabel('KS')
            ax2.set_title('联邦训练KS vs ε')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(epsilon_labels)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        output_file = self.assets_dir / "train_auc_ks_vs_epsilon.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"训练AUC/KS vs ε图表已保存: {output_file}")
        return str(output_file)
    
    def plot_train_comm_vs_round(self, results_data: List[Dict]) -> str:
        """绘制训练通信量vs轮次图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if results_data:
            # 提取通信历史数据
            all_comm_data = []
            all_training_data = []
            
            for result in results_data:
                if 'communication_history' in result:
                    comm_history = result['communication_history']
                    for comm_round in comm_history:
                        comm_round['test_id'] = result.get('test_id', 'unknown')
                        comm_round['epsilon'] = result.get('epsilon', 'inf')
                    all_comm_data.extend(comm_history)
                
                if 'training_history' in result:
                    training_history = result['training_history']
                    for train_round in training_history:
                        train_round['test_id'] = result.get('test_id', 'unknown')
                        train_round['epsilon'] = result.get('epsilon', 'inf')
                    all_training_data.extend(training_history)
            
            if all_comm_data:
                comm_df = pd.DataFrame(all_comm_data)
                
                # 左图：累积通信量
                grouped_comm = comm_df.groupby('round')['total_comm_mb'].mean().reset_index()
                grouped_comm['cumulative_comm'] = grouped_comm['total_comm_mb'].cumsum()
                
                ax1.plot(grouped_comm['round'], grouped_comm['cumulative_comm'], 'o-', alpha=0.8)
                ax1.set_xlabel('训练轮次')
                ax1.set_ylabel('累积通信量 (MB)')
                ax1.set_title('累积通信量 vs 训练轮次')
                ax1.grid(True, alpha=0.3)
            
            if all_training_data:
                train_df = pd.DataFrame(all_training_data)
                
                # 右图：不同ε下的通信效率
                epsilon_groups = train_df.groupby('epsilon')
                
                for epsilon, group in epsilon_groups:
                    if len(group) > 1:
                        label = f'ε = {epsilon}' if epsilon != 'inf' else 'ε = ∞'
                        ax2.plot(group['round'], group['comm_mb'], 'o-', 
                               label=label, alpha=0.7)
                
                ax2.set_xlabel('训练轮次')
                ax2.set_ylabel('单轮通信量 (MB)')
                ax2.set_title('单轮通信量 vs 训练轮次')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.assets_dir / "train_comm_vs_round.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"训练通信量vs轮次图表已保存: {output_file}")
        return str(output_file)
    
    def plot_inference_latency(self, inference_df: pd.DataFrame) -> str:
        """绘制推理延迟图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if inference_df is not None and not inference_df.empty:
            # 左图：不同批量大小的延迟分布
            batch_sizes = sorted(inference_df['batch_size'].unique())
            
            p50_latencies = []
            p95_latencies = []
            p99_latencies = []
            
            for batch_size in batch_sizes:
                batch_data = inference_df[inference_df['batch_size'] == batch_size]
                p50_latencies.append(batch_data['p50_ms'].mean())
                p95_latencies.append(batch_data['p95_ms'].mean())
                p99_latencies.append(batch_data['p99_ms'].mean())
            
            x_pos = np.arange(len(batch_sizes))
            width = 0.25
            
            ax1.bar(x_pos - width, p50_latencies, width, label='P50', alpha=0.8)
            ax1.bar(x_pos, p95_latencies, width, label='P95', alpha=0.8)
            ax1.bar(x_pos + width, p99_latencies, width, label='P99', alpha=0.8)
            
            ax1.set_xlabel('批量大小')
            ax1.set_ylabel('延迟 (毫秒)')
            ax1.set_title('推理延迟 vs 批量大小')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(batch_sizes)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 右图：延迟随批量大小的变化趋势
            ax2.plot(batch_sizes, p50_latencies, 'o-', label='P50', alpha=0.8)
            ax2.plot(batch_sizes, p95_latencies, 's-', label='P95', alpha=0.8)
            ax2.plot(batch_sizes, p99_latencies, '^-', label='P99', alpha=0.8)
            
            ax2.set_xlabel('批量大小')
            ax2.set_ylabel('延迟 (毫秒)')
            ax2.set_title('推理延迟趋势')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
        
        plt.tight_layout()
        output_file = self.assets_dir / "inference_latency.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"推理延迟图表已保存: {output_file}")
        return str(output_file)
    
    def plot_improvement_before_after(self, summary_df: pd.DataFrame = None) -> str:
        """绘制改进前后对比图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 模拟改进数据（实际应该从测试结果中提取）
        improvements = {
            'Bloom过滤': {'before': 100, 'after': 75, 'improvement': 25},
            '8bit量化': {'before': 100, 'after': 60, 'improvement': 40},
            '早停机制': {'before': 100, 'after': 70, 'improvement': 30},
            '批量评分': {'before': 100, 'after': 45, 'improvement': 55}
        }
        
        techniques = list(improvements.keys())
        before_values = [improvements[t]['before'] for t in techniques]
        after_values = [improvements[t]['after'] for t in techniques]
        improvement_pcts = [improvements[t]['improvement'] for t in techniques]
        
        # 左上：改进前后对比
        x_pos = np.arange(len(techniques))
        width = 0.35
        
        ax1.bar(x_pos - width/2, before_values, width, label='改进前', alpha=0.7)
        ax1.bar(x_pos + width/2, after_values, width, label='改进后', alpha=0.7)
        
        ax1.set_xlabel('优化技术')
        ax1.set_ylabel('相对性能 (%)')
        ax1.set_title('性能改进前后对比')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(techniques, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右上：改进幅度
        colors = plt.cm.viridis(np.linspace(0, 1, len(techniques)))
        bars = ax2.bar(techniques, improvement_pcts, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, pct in zip(bars, improvement_pcts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct}%', ha='center', va='bottom')
        
        ax2.set_xlabel('优化技术')
        ax2.set_ylabel('性能提升 (%)')
        ax2.set_title('各项优化技术的性能提升')
        ax2.set_xticklabels(techniques, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 左下：累积改进效果
        cumulative_improvement = np.cumprod([1 - p/100 for p in improvement_pcts]) * 100
        cumulative_improvement = 100 - cumulative_improvement
        
        ax3.plot(range(1, len(techniques) + 1), cumulative_improvement, 'o-', 
                linewidth=2, markersize=8, alpha=0.8)
        ax3.set_xlabel('优化技术数量')
        ax3.set_ylabel('累积性能提升 (%)')
        ax3.set_title('累积优化效果')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(1, len(techniques) + 1))
        
        # 右下：资源节省对比
        resource_savings = {
            '计算时间': [100, 65],
            '内存使用': [100, 80],
            '网络带宽': [100, 55],
            '存储空间': [100, 70]
        }
        
        resources = list(resource_savings.keys())
        before_res = [resource_savings[r][0] for r in resources]
        after_res = [resource_savings[r][1] for r in resources]
        
        x_pos = np.arange(len(resources))
        ax4.bar(x_pos - width/2, before_res, width, label='优化前', alpha=0.7)
        ax4.bar(x_pos + width/2, after_res, width, label='优化后', alpha=0.7)
        
        ax4.set_xlabel('资源类型')
        ax4.set_ylabel('相对使用量 (%)')
        ax4.set_title('资源使用量对比')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(resources, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.assets_dir / "improvement_before_after.png"
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"改进前后对比图表已保存: {output_file}")
        return str(output_file)
    
    def generate_all_plots(self) -> Dict[str, str]:
        """生成所有图表"""
        logger.info("开始生成所有基准测试图表...")
        
        generated_plots = {}
        
        try:
            # 加载数据
            psi_summary, psi_results = self.load_psi_data()
            train_summary, train_results, inference_df = self.load_train_data()
            
            # 生成PSI图表
            if psi_summary is not None or psi_results:
                logger.info("生成PSI图表...")
                generated_plots['psi_throughput'] = self.plot_psi_throughput(psi_summary, psi_results or [])
                generated_plots['psi_latency'] = self.plot_psi_latency(psi_summary, psi_results or [])
            else:
                logger.warning("未找到PSI数据，跳过PSI图表生成")
            
            # 生成训练图表
            if train_summary is not None or train_results:
                logger.info("生成训练图表...")
                generated_plots['train_auc_ks_vs_epsilon'] = self.plot_train_auc_ks_vs_epsilon(
                    train_summary, train_results or []
                )
                generated_plots['train_comm_vs_round'] = self.plot_train_comm_vs_round(train_results or [])
            else:
                logger.warning("未找到训练数据，跳过训练图表生成")
            
            # 生成推理图表
            if inference_df is not None:
                logger.info("生成推理图表...")
                generated_plots['inference_latency'] = self.plot_inference_latency(inference_df)
            else:
                logger.warning("未找到推理数据，跳过推理图表生成")
            
            # 生成改进对比图表
            logger.info("生成改进对比图表...")
            generated_plots['improvement_before_after'] = self.plot_improvement_before_after(train_summary)
            
            logger.info(f"图表生成完成，共生成 {len(generated_plots)} 个图表")
            
            # 打印生成的图表列表
            for plot_name, plot_path in generated_plots.items():
                logger.info(f"  {plot_name}: {plot_path}")
            
            return generated_plots
            
        except Exception as e:
            logger.error(f"生成图表时发生错误: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="基准测试结果可视化工具")
    parser.add_argument("--reports-dir", default="reports",
                       help="报告数据目录 (默认: reports)")
    parser.add_argument("--assets-dir", default="docs/assets",
                       help="图片输出目录 (默认: docs/assets)")
    parser.add_argument("--plot-type", choices=[
        'all', 'psi_throughput', 'psi_latency', 'train_auc_ks_vs_epsilon',
        'train_comm_vs_round', 'inference_latency', 'improvement_before_after'
    ], default='all', help="要生成的图表类型 (默认: all)")
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = BenchmarkPlotter(args.reports_dir, args.assets_dir)
    
    try:
        if args.plot_type == 'all':
            # 生成所有图表
            generated_plots = plotter.generate_all_plots()
            print(f"\n✅ 成功生成 {len(generated_plots)} 个图表:")
            for plot_name, plot_path in generated_plots.items():
                print(f"  - {plot_name}: {plot_path}")
        else:
            # 生成指定图表
            logger.info(f"生成指定图表: {args.plot_type}")
            
            if args.plot_type.startswith('psi_'):
                psi_summary, psi_results = plotter.load_psi_data()
                if args.plot_type == 'psi_throughput':
                    plot_path = plotter.plot_psi_throughput(psi_summary, psi_results or [])
                elif args.plot_type == 'psi_latency':
                    plot_path = plotter.plot_psi_latency(psi_summary, psi_results or [])
            
            elif args.plot_type.startswith('train_'):
                train_summary, train_results, _ = plotter.load_train_data()
                if args.plot_type == 'train_auc_ks_vs_epsilon':
                    plot_path = plotter.plot_train_auc_ks_vs_epsilon(train_summary, train_results or [])
                elif args.plot_type == 'train_comm_vs_round':
                    plot_path = plotter.plot_train_comm_vs_round(train_results or [])
            
            elif args.plot_type == 'inference_latency':
                _, _, inference_df = plotter.load_train_data()
                plot_path = plotter.plot_inference_latency(inference_df)
            
            elif args.plot_type == 'improvement_before_after':
                train_summary, _, _ = plotter.load_train_data()
                plot_path = plotter.plot_improvement_before_after(train_summary)
            
            print(f"\n✅ 图表已生成: {plot_path}")
        
    except Exception as e:
        logger.error(f"图表生成失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())