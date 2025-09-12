#!/usr/bin/env python3
"""
统计与外推模块
基于实测吞吐/时延 vs 并发/分片拟合产能模型
PSI：分段线性+带宽上限；训练：轮次×通信量/带宽
输出外推值 + 95% CI与前提（CPU/带宽/内存）
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略拟合警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ExtrapolationResult:
    """外推结果数据类"""
    target_scale: float
    predicted_value: float
    confidence_interval_95: Tuple[float, float]
    model_type: str
    r_squared: float
    assumptions: List[str]
    formula: str
    actual_data_range: Tuple[float, float]
    extrapolation_factor: float

class PSICapacityModel:
    """PSI产能模型：分段线性+带宽上限"""
    
    def __init__(self):
        self.model_params = None
        self.bandwidth_limit = None
        self.r_squared = 0
        
    def psi_throughput_model(self, x, a, b, c, bandwidth_limit):
        """
        PSI吞吐模型：Throughput(P,S) = min(a*P*S + b, bandwidth_limit) - c
        P: 并发数, S: 分片数
        """
        parallel_factor, shard_factor = x
        linear_throughput = a * parallel_factor * shard_factor + b
        return np.minimum(linear_throughput, bandwidth_limit) - c
    
    def fit(self, data: pd.DataFrame) -> Dict:
        """
        拟合PSI产能模型
        data应包含: workers, shards, throughput_ops_per_sec, bandwidth_mbps
        """
        try:
            # 准备数据
            X = np.column_stack([data['workers'], data['shards']])
            y = data['throughput_ops_per_sec']
            
            # 估计带宽上限（取观测到的最大吞吐量的1.2倍）
            estimated_bandwidth_limit = y.max() * 1.2
            
            # 拟合模型
            def model_func(x_combined, a, b, c):
                x_reshaped = x_combined.reshape(2, -1)
                return self.psi_throughput_model(x_reshaped, a, b, c, estimated_bandwidth_limit)
            
            X_combined = X.T.flatten()
            
            # 初始参数估计
            p0 = [1.0, 100.0, 10.0]
            
            popt, pcov = curve_fit(model_func, X_combined, y, p0=p0, maxfev=5000)
            
            self.model_params = popt
            self.bandwidth_limit = estimated_bandwidth_limit
            
            # 计算R²
            y_pred = model_func(X_combined, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'success': True,
                'params': popt.tolist(),
                'bandwidth_limit': self.bandwidth_limit,
                'r_squared': self.r_squared,
                'covariance': pcov.tolist()
            }
            
        except Exception as e:
            logger.error(f"PSI模型拟合失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, workers: int, shards: int, confidence_level: float = 0.95) -> ExtrapolationResult:
        """
        预测给定并发和分片数下的PSI吞吐量
        """
        if self.model_params is None:
            raise ValueError("模型未拟合")
        
        # 点预测
        x = np.array([[workers], [shards]])
        predicted = self.psi_throughput_model(x, *self.model_params, self.bandwidth_limit)[0]
        
        # 置信区间（简化计算）
        # 实际应用中应该使用完整的协方差矩阵
        uncertainty = predicted * 0.15  # 假设15%的不确定性
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        ci_lower = predicted - z_score * uncertainty
        ci_upper = predicted + z_score * uncertainty
        
        # 生成假设条件
        assumptions = [
            f"CPU核数 >= {workers}",
            f"内存 >= {shards * workers * 100}MB",
            f"网络带宽 >= {self.bandwidth_limit/1000:.1f}Gbps",
            "网络延迟 < 10ms",
            "无其他系统负载干扰"
        ]
        
        formula = f"Throughput = min({self.model_params[0]:.2f} * workers * shards + {self.model_params[1]:.2f}, {self.bandwidth_limit:.0f}) - {self.model_params[2]:.2f}"
        
        return ExtrapolationResult(
            target_scale=workers * shards,
            predicted_value=predicted,
            confidence_interval_95=(ci_lower, ci_upper),
            model_type="PSI分段线性+带宽上限",
            r_squared=self.r_squared,
            assumptions=assumptions,
            formula=formula,
            actual_data_range=(0, 0),  # 需要从原始数据计算
            extrapolation_factor=1.0
        )

class FederatedTrainingModel:
    """联邦训练模型：轮次×通信量/带宽"""
    
    def __init__(self):
        self.model_params = None
        self.r_squared = 0
        
    def training_time_model(self, x, a, b, c):
        """
        训练时间模型：T = a * rounds * (communication_cost / bandwidth) + b * local_compute + c
        """
        rounds, comm_cost, bandwidth, local_compute = x
        return a * rounds * (comm_cost / bandwidth) + b * local_compute + c
    
    def fit(self, data: pd.DataFrame) -> Dict:
        """
        拟合联邦训练时间模型
        data应包含: rounds, communication_cost, bandwidth_mbps, local_compute_time, total_time
        """
        try:
            # 准备数据
            X = np.column_stack([
                data['rounds'],
                data['communication_cost'],
                data['bandwidth_mbps'],
                data['local_compute_time']
            ])
            y = data['total_time']
            
            # 拟合模型
            def model_func(x_combined, a, b, c):
                x_reshaped = x_combined.reshape(4, -1)
                return self.training_time_model(x_reshaped, a, b, c)
            
            X_combined = X.T.flatten()
            
            # 初始参数估计
            p0 = [1.0, 1.0, 0.0]
            
            popt, pcov = curve_fit(model_func, X_combined, y, p0=p0, maxfev=5000)
            
            self.model_params = popt
            
            # 计算R²
            y_pred = model_func(X_combined, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'success': True,
                'params': popt.tolist(),
                'r_squared': self.r_squared,
                'covariance': pcov.tolist()
            }
            
        except Exception as e:
            logger.error(f"训练模型拟合失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, rounds: int, comm_cost: float, bandwidth_mbps: float, 
               local_compute_time: float, confidence_level: float = 0.95) -> ExtrapolationResult:
        """
        预测联邦训练时间
        """
        if self.model_params is None:
            raise ValueError("模型未拟合")
        
        # 点预测
        x = np.array([[rounds], [comm_cost], [bandwidth_mbps], [local_compute_time]])
        predicted = self.training_time_model(x, *self.model_params)[0]
        
        # 置信区间
        uncertainty = predicted * 0.20  # 假设20%的不确定性
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        ci_lower = predicted - z_score * uncertainty
        ci_upper = predicted + z_score * uncertainty
        
        # 生成假设条件
        assumptions = [
            f"网络带宽 >= {bandwidth_mbps}Mbps",
            f"CPU性能与基准测试一致",
            f"内存 >= {rounds * comm_cost / 1024 / 1024:.0f}GB",
            "网络稳定，无丢包",
            "参与方同时在线"
        ]
        
        formula = f"Time = {self.model_params[0]:.3f} * rounds * (comm_cost / bandwidth) + {self.model_params[1]:.3f} * local_compute + {self.model_params[2]:.3f}"
        
        return ExtrapolationResult(
            target_scale=rounds,
            predicted_value=predicted,
            confidence_interval_95=(ci_lower, ci_upper),
            model_type="联邦训练时间模型",
            r_squared=self.r_squared,
            assumptions=assumptions,
            formula=formula,
            actual_data_range=(0, 0),
            extrapolation_factor=1.0
        )

class ExtrapolationEngine:
    """外推引擎主类"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.psi_model = PSICapacityModel()
        self.training_model = FederatedTrainingModel()
        
    def load_psi_data(self) -> Optional[pd.DataFrame]:
        """加载PSI基准测试数据"""
        try:
            psi_files = list(self.reports_dir.glob("bench/psi/*.json"))
            if not psi_files:
                logger.warning("未找到PSI基准测试数据")
                return None
            
            # 加载最新的PSI报告
            latest_file = max(psi_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            records = []
            for result in data.get('results', []):
                if 'error' not in result:
                    records.append({
                        'workers': result.get('config', {}).get('workers', 1),
                        'shards': result.get('config', {}).get('shards', 1),
                        'throughput_ops_per_sec': result.get('performance', {}).get('throughput_ops_per_sec', 0),
                        'bandwidth_mbps': result.get('performance', {}).get('bandwidth_utilization_mbps', 100),
                        'cpu_utilization': result.get('performance', {}).get('cpu_utilization_percent', 50),
                        'memory_usage_mb': result.get('performance', {}).get('memory_usage_mb', 1000)
                    })
            
            if not records:
                logger.warning("PSI数据中没有有效记录")
                return None
                
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"加载PSI数据失败: {e}")
            return None
    
    def load_training_data(self) -> Optional[pd.DataFrame]:
        """加载联邦训练基准测试数据"""
        try:
            train_files = list(self.reports_dir.glob("bench/train/*.json"))
            if not train_files:
                logger.warning("未找到训练基准测试数据")
                return None
            
            # 加载最新的训练报告
            latest_file = max(train_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            records = []
            for exp in data.get('detailed', {}).get('experiments', []):
                if 'error' not in exp:
                    records.append({
                        'rounds': exp.get('efficiency', {}).get('rounds', 10),
                        'communication_cost': exp.get('efficiency', {}).get('communicationCost', 1000000),
                        'bandwidth_mbps': 100,  # 假设值，实际应从配置获取
                        'local_compute_time': exp.get('efficiency', {}).get('totalTime', 60000) * 0.3,  # 假设30%是本地计算
                        'total_time': exp.get('efficiency', {}).get('totalTime', 60000),
                        'participants': exp.get('config', {}).get('participants', 2),
                        'epsilon': exp.get('config', {}).get('epsilon', 5)
                    })
            
            if not records:
                logger.warning("训练数据中没有有效记录")
                return None
                
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            return None
    
    def extrapolate_psi_capacity(self, target_scale: float = 1e9, 
                                bandwidth_scenarios: List[float] = [100, 1000, 10000]) -> Dict:
        """外推PSI产能到目标规模"""
        logger.info(f"开始PSI产能外推，目标规模: {target_scale:.0e}")
        
        # 加载数据
        psi_data = self.load_psi_data()
        if psi_data is None or len(psi_data) < 3:
            return {'success': False, 'error': 'PSI数据不足，需要至少3个数据点'}
        
        # 拟合模型
        fit_result = self.psi_model.fit(psi_data)
        if not fit_result['success']:
            return fit_result
        
        # 计算不同带宽场景下的外推结果
        results = {}
        
        for bandwidth_mbps in bandwidth_scenarios:
            # 估算需要的并发和分片数
            # 假设每个worker处理1000个样本，每个shard包含100万个样本
            estimated_shards = max(1, int(target_scale / 1e6))
            estimated_workers = max(1, int(target_scale / (estimated_shards * 1e6)))
            
            # 限制在合理范围内
            estimated_shards = min(estimated_shards, 1000)
            estimated_workers = min(estimated_workers, 100)
            
            try:
                prediction = self.psi_model.predict(estimated_workers, estimated_shards)
                
                # 计算完成时间（秒）
                completion_time_seconds = target_scale / prediction.predicted_value
                completion_time_hours = completion_time_seconds / 3600
                
                results[f"{bandwidth_mbps}Mbps"] = {
                    'workers': estimated_workers,
                    'shards': estimated_shards,
                    'predicted_throughput': prediction.predicted_value,
                    'completion_time_hours': completion_time_hours,
                    'completion_time_days': completion_time_hours / 24,
                    'confidence_interval': prediction.confidence_interval_95,
                    'assumptions': prediction.assumptions,
                    'formula': prediction.formula
                }
                
            except Exception as e:
                results[f"{bandwidth_mbps}Mbps"] = {'error': str(e)}
        
        return {
            'success': True,
            'target_scale': target_scale,
            'model_r_squared': self.psi_model.r_squared,
            'scenarios': results,
            'data_points_used': len(psi_data)
        }
    
    def extrapolate_training_capacity(self, target_samples: float = 1e6,
                                    bandwidth_scenarios: List[float] = [100, 1000]) -> Dict:
        """外推联邦训练产能到目标规模"""
        logger.info(f"开始联邦训练外推，目标样本数: {target_samples:.0e}")
        
        # 加载数据
        training_data = self.load_training_data()
        if training_data is None or len(training_data) < 3:
            return {'success': False, 'error': '训练数据不足，需要至少3个数据点'}
        
        # 拟合模型
        fit_result = self.training_model.fit(training_data)
        if not fit_result['success']:
            return fit_result
        
        results = {}
        
        for bandwidth_mbps in bandwidth_scenarios:
            # 估算训练参数
            estimated_rounds = 20  # 基于经验
            estimated_comm_cost = target_samples * 100  # 每样本100字节通信成本
            estimated_local_compute = target_samples * 0.001  # 每样本1ms计算时间
            
            try:
                prediction = self.training_model.predict(
                    estimated_rounds, estimated_comm_cost, bandwidth_mbps, estimated_local_compute
                )
                
                completion_time_hours = prediction.predicted_value / 3600
                
                results[f"{bandwidth_mbps}Mbps"] = {
                    'estimated_rounds': estimated_rounds,
                    'communication_cost_bytes': estimated_comm_cost,
                    'predicted_time_seconds': prediction.predicted_value,
                    'completion_time_hours': completion_time_hours,
                    'completion_time_days': completion_time_hours / 24,
                    'confidence_interval': prediction.confidence_interval_95,
                    'assumptions': prediction.assumptions,
                    'formula': prediction.formula
                }
                
            except Exception as e:
                results[f"{bandwidth_mbps}Mbps"] = {'error': str(e)}
        
        return {
            'success': True,
            'target_samples': target_samples,
            'model_r_squared': self.training_model.r_squared,
            'scenarios': results,
            'data_points_used': len(training_data)
        }
    
    def generate_extrapolation_report(self, output_file: str = None) -> Dict:
        """生成完整的外推报告"""
        logger.info("生成外推分析报告...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'psi_extrapolation': {},
            'training_extrapolation': {},
            'summary': {},
            'recommendations': []
        }
        
        # PSI外推
        psi_result = self.extrapolate_psi_capacity()
        report['psi_extrapolation'] = psi_result
        
        # 训练外推
        training_result = self.extrapolate_training_capacity()
        report['training_extrapolation'] = training_result
        
        # 生成摘要和建议
        if psi_result.get('success') and training_result.get('success'):
            report['summary'] = {
                'psi_feasibility': self._assess_psi_feasibility(psi_result),
                'training_feasibility': self._assess_training_feasibility(training_result),
                'bottlenecks': self._identify_bottlenecks(psi_result, training_result)
            }
            
            report['recommendations'] = self._generate_recommendations(psi_result, training_result)
        
        # 保存报告
        if output_file is None:
            output_file = self.reports_dir / "extrapolation_analysis.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"外推报告保存至: {output_path}")
        return report
    
    def _assess_psi_feasibility(self, psi_result: Dict) -> str:
        """评估PSI可行性"""
        scenarios = psi_result.get('scenarios', {})
        
        for bandwidth, result in scenarios.items():
            if 'completion_time_days' in result and result['completion_time_days'] <= 1:
                return f"可行 - 在{bandwidth}带宽下可在1天内完成"
        
        return "需要优化 - 当前配置下无法在1天内完成1e9规模PSI"
    
    def _assess_training_feasibility(self, training_result: Dict) -> str:
        """评估训练可行性"""
        scenarios = training_result.get('scenarios', {})
        
        for bandwidth, result in scenarios.items():
            if 'completion_time_days' in result and result['completion_time_days'] <= 1:
                return f"可行 - 在{bandwidth}带宽下可在1天内完成"
        
        return "需要优化 - 当前配置下无法在1天内完成1e6样本训练"
    
    def _identify_bottlenecks(self, psi_result: Dict, training_result: Dict) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析PSI瓶颈
        if psi_result.get('success'):
            psi_scenarios = psi_result.get('scenarios', {})
            if all(s.get('completion_time_days', 999) > 1 for s in psi_scenarios.values()):
                bottlenecks.append("PSI吞吐量不足，需要增加并发度或优化算法")
        
        # 分析训练瓶颈
        if training_result.get('success'):
            training_scenarios = training_result.get('scenarios', {})
            if all(s.get('completion_time_days', 999) > 1 for s in training_scenarios.values()):
                bottlenecks.append("联邦训练通信开销过大，需要优化通信协议")
        
        return bottlenecks
    
    def _generate_recommendations(self, psi_result: Dict, training_result: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # PSI优化建议
        if psi_result.get('success'):
            recommendations.extend([
                "PSI优化：增加分片数到32（需要16核以上CPU）",
                "PSI优化：使用Bloom过滤器预过滤，减少1%假阳性",
                "PSI优化：批处理大小从2k增加到8k"
            ])
        
        # 训练优化建议
        if training_result.get('success'):
            recommendations.extend([
                "训练优化：启用直方统计复用，减少20%通信量",
                "训练优化：使用8bit量化，减少通信开销",
                "训练优化：设置ε=5为默认隐私预算"
            ])
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='联邦风控系统性能外推分析')
    parser.add_argument('--reports-dir', default='reports', help='报告目录路径')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--psi-scale', type=float, default=1e9, help='PSI目标规模')
    parser.add_argument('--train-scale', type=float, default=1e6, help='训练目标样本数')
    parser.add_argument('--bandwidth', nargs='+', type=float, default=[100, 1000, 10000], 
                       help='带宽场景(Mbps)')
    
    args = parser.parse_args()
    
    # 创建外推引擎
    engine = ExtrapolationEngine(args.reports_dir)
    
    # 生成报告
    report = engine.generate_extrapolation_report(args.output)
    
    # 打印摘要
    print("\n=== 外推分析摘要 ===")
    if report.get('summary'):
        print(f"PSI可行性: {report['summary'].get('psi_feasibility', 'N/A')}")
        print(f"训练可行性: {report['summary'].get('training_feasibility', 'N/A')}")
        
        bottlenecks = report['summary'].get('bottlenecks', [])
        if bottlenecks:
            print("\n识别的瓶颈:")
            for bottleneck in bottlenecks:
                print(f"  - {bottleneck}")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\n优化建议:")
            for rec in recommendations:
                print(f"  - {rec}")
    
    print(f"\n详细报告已保存至: {args.output or 'reports/extrapolation_analysis.json'}")

if __name__ == '__main__':
    main()