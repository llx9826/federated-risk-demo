#!/usr/bin/env python3
"""
基准测试图表生成器
生成PSI吞吐、训练收敛、评分延迟等可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ChartGenerator:
    def __init__(self, reports_dir: str = "reports", output_dir: str = "reports/bench/plots"):
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_psi_throughput_chart(self, psi_data: Dict[str, Any]):
        """生成PSI吞吐量图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Workers vs Throughput
        workers_data = psi_data.get('scalability_analysis', {}).get('workers_vs_throughput', [])
        if workers_data:
            workers = [d['workers'] for d in workers_data]
            throughput = [d['throughput'] for d in workers_data]
            
            ax1.plot(workers, throughput, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Workers')
            ax1.set_ylabel('Throughput (records/sec)')
            ax1.set_title('PSI Throughput vs Workers')
            ax1.grid(True, alpha=0.3)
        
        # Shards vs Throughput
        shards_data = psi_data.get('scalability_analysis', {}).get('shards_vs_throughput', [])
        if shards_data:
            shards = [d['shards'] for d in shards_data]
            throughput = [d['throughput'] for d in shards_data]
            
            ax2.plot(shards, throughput, 's-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Shards')
            ax2.set_ylabel('Throughput (records/sec)')
            ax2.set_title('PSI Throughput vs Shards')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'psi_throughput.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PSI throughput chart saved to {self.output_dir / 'psi_throughput.png'}")
    
    def generate_training_convergence_chart(self, train_data: Dict[str, Any]):
        """生成训练收敛图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convergence curve
        convergence_data = train_data.get('convergence_curve', [])
        if convergence_data:
            rounds = [d['round'] for d in convergence_data]
            auc = [d['auc'] for d in convergence_data]
            ks = [d['ks'] for d in convergence_data]
            
            ax1.plot(rounds, auc, 'o-', label='AUC', linewidth=2, markersize=6)
            ax1.plot(rounds, ks, 's-', label='KS', linewidth=2, markersize=6)
            ax1.set_xlabel('Training Round')
            ax1.set_ylabel('Metric Value')
            ax1.set_title('Model Performance Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Communication analysis
        if convergence_data:
            communication = [d['communication_mb'] for d in convergence_data]
            
            ax2.bar(rounds, communication, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Training Round')
            ax2.set_ylabel('Communication (MB)')
            ax2.set_title('Communication Cost per Round')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'train_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training convergence chart saved to {self.output_dir / 'train_convergence.png'}")
    
    def generate_scoring_latency_chart(self, score_data: Dict[str, Any]):
        """生成评分延迟图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency distribution
        latency_dist = score_data.get('latency_distribution', {})
        if latency_dist:
            percentiles = ['p50_ms', 'p90_ms', 'p95_ms', 'p99_ms']
            values = [latency_dist.get(p, 0) for p in percentiles]
            labels = ['P50', 'P90', 'P95', 'P99']
            
            bars = ax1.bar(labels, values, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Response Latency Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value}ms', ha='center', va='bottom')
        
        # Stage analysis
        stage_data = score_data.get('stage_analysis', [])
        if stage_data:
            stages = [d['stage'] for d in stage_data]
            actual_rps = [d['actual_rps'] for d in stage_data]
            p95_latency = [d['p95_ms'] for d in stage_data]
            
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(stages, actual_rps, alpha=0.7, color='lightblue', label='RPS')
            line1 = ax2_twin.plot(stages, p95_latency, 'ro-', linewidth=2, markersize=6, label='P95 Latency')
            
            ax2.set_ylabel('Requests per Second', color='blue')
            ax2_twin.set_ylabel('P95 Latency (ms)', color='red')
            ax2.set_title('Performance vs Load Stages')
            ax2.tick_params(axis='x', rotation=45)
            
            # 合并图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_latency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Scoring latency chart saved to {self.output_dir / 'score_latency.png'}")
    
    def generate_extrapolation_chart(self, extrap_data: Dict[str, Any]):
        """生成外推分析图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSI scaling extrapolation
        psi_component = extrap_data.get('components', {}).get('psi', {})
        calibration_data = psi_component.get('calibration_data', [])
        
        if calibration_data:
            scales = [d['scale'] for d in calibration_data]
            measured = [d['measured_throughput'] for d in calibration_data]
            predicted = [d['predicted_throughput'] for d in calibration_data]
            
            # 添加外推点
            extrapolations = psi_component.get('extrapolations', {})
            if extrapolations:
                for scale_name, extrap in extrapolations.items():
                    if '1e8' in scale_name:
                        scales.append(1e8)
                        measured.append(None)
                        predicted.append(extrap['predicted_throughput'])
                    elif '1e9' in scale_name:
                        scales.append(1e9)
                        measured.append(None)
                        predicted.append(extrap['predicted_throughput'])
            
            ax1.loglog(scales[:len(calibration_data)], measured[:len(calibration_data)], 
                      'o', markersize=8, label='Measured', color='blue')
            ax1.loglog(scales, predicted, 's-', linewidth=2, markersize=6, 
                      label='Model Prediction', color='red', alpha=0.7)
            ax1.set_xlabel('Data Scale')
            ax1.set_ylabel('Throughput (records/sec)')
            ax1.set_title('PSI Scaling Extrapolation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Training scaling extrapolation
        train_component = extrap_data.get('components', {}).get('federated_training', {})
        train_calibration = train_component.get('calibration_data', [])
        
        if train_calibration:
            scales = [d['scale'] for d in train_calibration]
            measured_time = [d['measured_time'] for d in train_calibration]
            predicted_time = [d['predicted_time'] for d in train_calibration]
            
            # 添加外推点
            train_extrapolations = train_component.get('extrapolations', {})
            if train_extrapolations:
                for extrap_name, extrap in train_extrapolations.items():
                    if '1e6' in extrap_name:
                        scales.append(1e6)
                        measured_time.append(None)
                        predicted_time.append(extrap['predicted_time_minutes'])
            
            ax2.loglog(scales[:len(train_calibration)], measured_time[:len(train_calibration)], 
                      'o', markersize=8, label='Measured', color='green')
            ax2.loglog(scales, predicted_time, 's-', linewidth=2, markersize=6, 
                      label='Model Prediction', color='orange', alpha=0.7)
            ax2.set_xlabel('Data Scale')
            ax2.set_ylabel('Training Time (minutes)')
            ax2.set_title('Training Scaling Extrapolation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'extrapolation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Extrapolation analysis chart saved to {self.output_dir / 'extrapolation_analysis.png'}")
    
    def generate_all_charts(self):
        """生成所有图表"""
        print("🎨 Generating benchmark visualization charts...")
        
        # 加载数据文件
        try:
            # PSI基准测试图表
            psi_file = self.reports_dir / "bench" / "psi_benchmark_sample.json"
            if psi_file.exists():
                with open(psi_file) as f:
                    psi_data = json.load(f)
                self.generate_psi_throughput_chart(psi_data)
            
            # 训练基准测试图表
            train_file = self.reports_dir / "bench" / "train_benchmark_sample.json"
            if train_file.exists():
                with open(train_file) as f:
                    train_data = json.load(f)
                self.generate_training_convergence_chart(train_data)
            
            # 评分基准测试图表
            score_file = self.reports_dir / "bench" / "score_benchmark_sample.json"
            if score_file.exists():
                with open(score_file) as f:
                    score_data = json.load(f)
                self.generate_scoring_latency_chart(score_data)
            
            # 外推分析图表
            extrap_file = self.reports_dir / "extrapolation_analysis.json"
            if extrap_file.exists():
                with open(extrap_file) as f:
                    extrap_data = json.load(f)
                self.generate_extrapolation_chart(extrap_data)
            
            print("\n✅ All charts generated successfully!")
            print(f"📁 Charts saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualization charts")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory")
    parser.add_argument("--output-dir", default="reports/bench/plots", help="Output directory for charts")
    
    args = parser.parse_args()
    
    generator = ChartGenerator(args.reports_dir, args.output_dir)
    generator.generate_all_charts()

if __name__ == "__main__":
    main()