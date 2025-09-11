#!/usr/bin/env python3
"""
联邦训练性能基准测试工具

测试不同场景下的联邦训练性能，包括：
- 不同参与方数量的训练性能
- 不同数据规模的训练性能
- 不同模型复杂度的训练性能
- 通信开销分析
- 收敛性分析
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from tqdm import tqdm

# 配置日志
logger.add("train_benchmark.log", rotation="10 MB")

class FederatedTrainingBenchmark:
    """联邦训练性能基准测试"""
    
    def __init__(self, orchestrator_url: str = "http://localhost:8002"):
        """初始化联邦训练基准测试"""
        self.orchestrator_url = orchestrator_url
        self.results = []
        self.process = psutil.Process()
        
        # 创建结果目录
        self.results_dir = Path("../results/training")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def _monitor_resources(self, duration: float) -> Dict:
        """监控资源使用情况"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        network_sent = []
        network_recv = []
        
        initial_net = psutil.net_io_counters()
        
        while time.time() - start_time < duration:
            cpu_samples.append(self.process.cpu_percent())
            memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB
            
            net_io = psutil.net_io_counters()
            network_sent.append(net_io.bytes_sent - initial_net.bytes_sent)
            network_recv.append(net_io.bytes_recv - initial_net.bytes_recv)
            
            await asyncio.sleep(0.5)
        
        return {
            'avg_cpu_percent': np.mean(cpu_samples),
            'max_cpu_percent': np.max(cpu_samples),
            'avg_memory_mb': np.mean(memory_samples),
            'max_memory_mb': np.max(memory_samples),
            'total_network_sent_mb': max(network_sent) / 1024 / 1024 if network_sent else 0,
            'total_network_recv_mb': max(network_recv) / 1024 / 1024 if network_recv else 0
        }
    
    async def _create_training_session(self, session_config: Dict) -> Tuple[bool, Optional[str]]:
        """创建联邦训练会话"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.orchestrator_url}/training/sessions",
                    json=session_config
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        session_id = result.get('session_id')
                        logger.debug(f"训练会话创建成功: {session_id}")
                        return True, session_id
                    else:
                        logger.error(f"训练会话创建失败: {response.status}")
                        return False, None
                        
            except Exception as e:
                logger.error(f"创建训练会话异常: {e}")
                return False, None
    
    async def _register_participants(self, session_id: str, participants: List[Dict]) -> bool:
        """注册参与方"""
        async with aiohttp.ClientSession() as session:
            try:
                for participant in participants:
                    async with session.post(
                        f"{self.orchestrator_url}/training/sessions/{session_id}/participants",
                        json=participant
                    ) as response:
                        if response.status != 200:
                            logger.error(f"参与方注册失败: {participant['party_id']}")
                            return False
                
                logger.debug(f"所有参与方注册成功: {len(participants)} 个")
                return True
                        
            except Exception as e:
                logger.error(f"注册参与方异常: {e}")
                return False
    
    async def _start_training(self, session_id: str) -> bool:
        """开始训练"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.orchestrator_url}/training/sessions/{session_id}/start"
                ) as response:
                    if response.status == 200:
                        logger.debug(f"训练开始成功: {session_id}")
                        return True
                    else:
                        logger.error(f"训练开始失败: {response.status}")
                        return False
                        
            except Exception as e:
                logger.error(f"开始训练异常: {e}")
                return False
    
    async def _monitor_training_progress(self, session_id: str, timeout: int = 1800) -> Dict:
        """监控训练进度"""
        start_time = time.time()
        rounds_completed = 0
        convergence_history = []
        communication_rounds = []
        
        while time.time() - start_time < timeout:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.orchestrator_url}/training/sessions/{session_id}/status"
                    ) as response:
                        if response.status == 200:
                            status = await response.json()
                            
                            current_round = status.get('current_round', 0)
                            if current_round > rounds_completed:
                                rounds_completed = current_round
                                
                                # 记录收敛指标
                                metrics = status.get('metrics', {})
                                if 'loss' in metrics:
                                    convergence_history.append({
                                        'round': current_round,
                                        'loss': metrics['loss'],
                                        'accuracy': metrics.get('accuracy', 0),
                                        'timestamp': time.time() - start_time
                                    })
                                
                                # 记录通信轮次
                                comm_stats = status.get('communication_stats', {})
                                if comm_stats:
                                    communication_rounds.append({
                                        'round': current_round,
                                        'bytes_sent': comm_stats.get('bytes_sent', 0),
                                        'bytes_received': comm_stats.get('bytes_received', 0),
                                        'round_time': comm_stats.get('round_time', 0)
                                    })
                            
                            if status.get('status') == 'completed':
                                logger.debug(f"训练完成: {session_id}, 轮次: {rounds_completed}")
                                return {
                                    'completed': True,
                                    'rounds_completed': rounds_completed,
                                    'total_time': time.time() - start_time,
                                    'convergence_history': convergence_history,
                                    'communication_rounds': communication_rounds,
                                    'final_metrics': status.get('final_metrics', {})
                                }
                            elif status.get('status') == 'failed':
                                logger.error(f"训练失败: {status.get('error')}")
                                return {
                                    'completed': False,
                                    'error': status.get('error'),
                                    'rounds_completed': rounds_completed,
                                    'total_time': time.time() - start_time
                                }
                        
                except Exception as e:
                    logger.error(f"监控训练进度异常: {e}")
            
            await asyncio.sleep(2)
        
        logger.error(f"训练监控超时: {timeout} 秒")
        return {
            'completed': False,
            'error': 'timeout',
            'rounds_completed': rounds_completed,
            'total_time': timeout
        }
    
    def _generate_participant_config(self, num_participants: int, data_size_per_participant: int) -> List[Dict]:
        """生成参与方配置"""
        participants = []
        
        for i in range(num_participants):
            participant = {
                'party_id': f'party_{i+1}',
                'endpoint': f'http://localhost:{8010 + i}',
                'data_config': {
                    'data_size': data_size_per_participant,
                    'feature_dim': 20,
                    'label_distribution': 'balanced' if i % 2 == 0 else 'imbalanced'
                },
                'model_config': {
                    'model_type': 'logistic_regression',
                    'learning_rate': 0.01,
                    'batch_size': min(32, data_size_per_participant // 10)
                }
            }
            participants.append(participant)
        
        return participants
    
    async def benchmark_single_training(self, num_participants: int, data_size_per_participant: int,
                                      model_complexity: str = 'simple', max_rounds: int = 50) -> Dict:
        """单次联邦训练性能测试"""
        session_id = f"bench_{num_participants}p_{data_size_per_participant}d_{int(time.time())}"
        
        logger.info(f"开始联邦训练测试: 参与方={num_participants}, 数据量={data_size_per_participant}, "
                   f"模型复杂度={model_complexity}, 最大轮次={max_rounds}")
        
        start_time = time.time()
        
        try:
            # 1. 创建训练会话
            session_config = {
                'session_id': session_id,
                'algorithm': 'fedavg',
                'model_type': model_complexity,
                'max_rounds': max_rounds,
                'min_participants': num_participants,
                'convergence_threshold': 0.001,
                'privacy_config': {
                    'differential_privacy': True,
                    'noise_multiplier': 0.1,
                    'l2_norm_clip': 1.0
                },
                'aggregation_config': {
                    'strategy': 'weighted_average',
                    'min_fit_clients': num_participants,
                    'min_eval_clients': max(1, num_participants // 2)
                }
            }
            
            session_created, session_id = await self._create_training_session(session_config)
            if not session_created:
                return {'error': '训练会话创建失败'}
            
            session_creation_time = time.time() - start_time
            
            # 2. 注册参与方
            participants = self._generate_participant_config(num_participants, data_size_per_participant)
            
            registration_start = time.time()
            participants_registered = await self._register_participants(session_id, participants)
            if not participants_registered:
                return {'error': '参与方注册失败'}
            
            registration_time = time.time() - registration_start
            
            # 3. 开始训练并监控
            training_start = time.time()
            
            # 启动资源监控
            monitor_task = asyncio.create_task(
                self._monitor_resources(duration=1800)  # 监控30分钟
            )
            
            # 开始训练
            training_started = await self._start_training(session_id)
            if not training_started:
                return {'error': '训练启动失败'}
            
            # 监控训练进度
            training_result = await self._monitor_training_progress(session_id, timeout=1800)
            
            training_time = time.time() - training_start
            total_time = time.time() - start_time
            
            # 停止资源监控
            monitor_task.cancel()
            try:
                resource_stats = await monitor_task
            except asyncio.CancelledError:
                resource_stats = {'avg_cpu_percent': 0, 'max_cpu_percent': 0, 
                                'avg_memory_mb': 0, 'max_memory_mb': 0,
                                'total_network_sent_mb': 0, 'total_network_recv_mb': 0}
            
            # 计算性能指标
            rounds_completed = training_result.get('rounds_completed', 0)
            convergence_history = training_result.get('convergence_history', [])
            communication_rounds = training_result.get('communication_rounds', [])
            
            # 计算收敛性指标
            convergence_metrics = self._analyze_convergence(convergence_history)
            
            # 计算通信开销
            communication_metrics = self._analyze_communication(communication_rounds)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'num_participants': num_participants,
                'data_size_per_participant': data_size_per_participant,
                'model_complexity': model_complexity,
                'max_rounds': max_rounds,
                'rounds_completed': rounds_completed,
                'session_creation_time': session_creation_time,
                'registration_time': registration_time,
                'training_time': training_time,
                'total_time': total_time,
                'training_completed': training_result.get('completed', False),
                'final_metrics': training_result.get('final_metrics', {}),
                'convergence_metrics': convergence_metrics,
                'communication_metrics': communication_metrics,
                'resource_usage': resource_stats,
                'throughput_rounds_per_minute': rounds_completed / (training_time / 60) if training_time > 0 else 0,
                'success': training_result.get('completed', False)
            }
            
            if not result['success']:
                result['error'] = training_result.get('error', 'unknown')
            
            logger.info(f"联邦训练测试完成: {num_participants} 参与方, {rounds_completed} 轮次, "
                       f"耗时 {total_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            logger.error(f"联邦训练测试异常: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'num_participants': num_participants,
                'data_size_per_participant': data_size_per_participant,
                'model_complexity': model_complexity,
                'error': str(e),
                'success': False
            }
    
    def _analyze_convergence(self, convergence_history: List[Dict]) -> Dict:
        """分析收敛性"""
        if not convergence_history:
            return {}
        
        losses = [h['loss'] for h in convergence_history]
        accuracies = [h['accuracy'] for h in convergence_history]
        
        # 计算收敛速度
        convergence_speed = 0
        if len(losses) > 1:
            initial_loss = losses[0]
            final_loss = losses[-1]
            convergence_speed = (initial_loss - final_loss) / len(losses)
        
        # 检测是否收敛
        converged = False
        convergence_round = None
        if len(losses) >= 5:
            # 检查最后5轮的损失变化
            recent_losses = losses[-5:]
            loss_variance = np.var(recent_losses)
            if loss_variance < 0.001:  # 损失变化很小
                converged = True
                convergence_round = len(losses) - 5
        
        return {
            'initial_loss': losses[0] if losses else 0,
            'final_loss': losses[-1] if losses else 0,
            'best_loss': min(losses) if losses else 0,
            'initial_accuracy': accuracies[0] if accuracies else 0,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'convergence_speed': convergence_speed,
            'converged': converged,
            'convergence_round': convergence_round,
            'loss_variance': np.var(losses) if losses else 0
        }
    
    def _analyze_communication(self, communication_rounds: List[Dict]) -> Dict:
        """分析通信开销"""
        if not communication_rounds:
            return {}
        
        total_bytes_sent = sum(r['bytes_sent'] for r in communication_rounds)
        total_bytes_received = sum(r['bytes_received'] for r in communication_rounds)
        round_times = [r['round_time'] for r in communication_rounds]
        
        return {
            'total_bytes_sent': total_bytes_sent,
            'total_bytes_received': total_bytes_received,
            'total_communication_mb': (total_bytes_sent + total_bytes_received) / 1024 / 1024,
            'avg_round_time': np.mean(round_times) if round_times else 0,
            'max_round_time': np.max(round_times) if round_times else 0,
            'min_round_time': np.min(round_times) if round_times else 0,
            'communication_efficiency': total_bytes_sent / (total_bytes_sent + total_bytes_received) if (total_bytes_sent + total_bytes_received) > 0 else 0
        }
    
    async def run_benchmark_suite(self, participant_counts: List[int], data_sizes: List[int],
                                 model_complexities: List[str], iterations: int = 3):
        """运行完整的基准测试套件"""
        logger.info(f"开始联邦训练基准测试套件: {len(participant_counts)} 个参与方配置, "
                   f"{len(data_sizes)} 个数据规模, {len(model_complexities)} 个模型复杂度, {iterations} 次迭代")
        
        total_tests = len(participant_counts) * len(data_sizes) * len(model_complexities) * iterations
        progress_bar = tqdm(total=total_tests, desc="联邦训练基准测试")
        
        for model_complexity in model_complexities:
            for num_participants in participant_counts:
                for data_size in data_sizes:
                    for iteration in range(iterations):
                        result = await self.benchmark_single_training(
                            num_participants=num_participants,
                            data_size_per_participant=data_size,
                            model_complexity=model_complexity,
                            max_rounds=50
                        )
                        
                        result['iteration'] = iteration + 1
                        self.results.append(result)
                        
                        progress_bar.update(1)
                        
                        # 测试间隔，避免服务过载
                        await asyncio.sleep(5)
        
        progress_bar.close()
        
        # 保存结果
        await self._save_results()
        
        # 生成报告
        await self._generate_report()
    
    async def _save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = self.results_dir / f"training_benchmark_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {results_file}")
        
        # 保存CSV格式
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.results_dir / f"training_benchmark_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV结果已保存到: {csv_file}")
    
    async def _generate_report(self):
        """生成性能报告"""
        if not self.results:
            logger.warning("没有测试结果，跳过报告生成")
            return
        
        df = pd.DataFrame([r for r in self.results if r.get('success', False)])
        
        if df.empty:
            logger.warning("没有成功的测试结果，跳过报告生成")
            return
        
        # 按配置分组统计
        summary = df.groupby(['num_participants', 'data_size_per_participant', 'model_complexity']).agg({
            'total_time': ['mean', 'std', 'min', 'max'],
            'rounds_completed': ['mean', 'std', 'min', 'max'],
            'throughput_rounds_per_minute': ['mean', 'std', 'min', 'max'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        # 生成报告
        report = {
            'test_summary': {
                'total_tests': len(self.results),
                'successful_tests': len(df),
                'participant_counts_tested': sorted(df['num_participants'].unique().tolist()),
                'data_sizes_tested': sorted(df['data_size_per_participant'].unique().tolist()),
                'model_complexities_tested': df['model_complexity'].unique().tolist(),
                'test_date': datetime.now().isoformat()
            },
            'performance_summary': summary.to_dict(),
            'best_performance': {
                'highest_throughput': {
                    'value': df['throughput_rounds_per_minute'].max(),
                    'config': df.loc[df['throughput_rounds_per_minute'].idxmax(), 
                                   ['num_participants', 'data_size_per_participant', 'model_complexity']].to_dict()
                },
                'fastest_training': {
                    'value': df['total_time'].min(),
                    'config': df.loc[df['total_time'].idxmin(), 
                                   ['num_participants', 'data_size_per_participant', 'model_complexity']].to_dict()
                }
            }
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"training_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存到: {report_file}")
        
        # 打印摘要
        logger.info("=== 联邦训练性能测试摘要 ===")
        logger.info(f"总测试数: {report['test_summary']['total_tests']}")
        logger.info(f"成功测试数: {report['test_summary']['successful_tests']}")
        logger.info(f"最高吞吐量: {report['best_performance']['highest_throughput']['value']:.2f} 轮次/分钟")
        logger.info(f"最快训练时间: {report['best_performance']['fastest_training']['value']:.2f} 秒")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='联邦训练性能基准测试工具')
    parser.add_argument('--orchestrator-url', default='http://localhost:8002', 
                       help='联邦编排服务URL')
    parser.add_argument('--participant-counts', nargs='+', type=int, 
                       default=[2, 3, 5], 
                       help='参与方数量')
    parser.add_argument('--data-sizes', nargs='+', type=int, 
                       default=[1000, 5000, 10000], 
                       help='每个参与方的数据规模')
    parser.add_argument('--model-complexities', nargs='+', 
                       default=['simple', 'medium'], 
                       help='模型复杂度')
    parser.add_argument('--iterations', type=int, default=3, 
                       help='每个测试的迭代次数')
    parser.add_argument('--max-rounds', type=int, default=50, 
                       help='最大训练轮次')
    parser.add_argument('--single-test', action='store_true', 
                       help='只运行单次测试')
    parser.add_argument('--num-participants', type=int, default=3, 
                       help='单次测试的参与方数量')
    parser.add_argument('--data-size', type=int, default=5000, 
                       help='单次测试的数据规模')
    parser.add_argument('--model-complexity', default='simple', 
                       help='单次测试的模型复杂度')
    
    args = parser.parse_args()
    
    async def run_tests():
        benchmark = FederatedTrainingBenchmark(args.orchestrator_url)
        
        if args.single_test:
            # 单次测试
            result = await benchmark.benchmark_single_training(
                num_participants=args.num_participants,
                data_size_per_participant=args.data_size,
                model_complexity=args.model_complexity,
                max_rounds=args.max_rounds
            )
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # 完整基准测试套件
            await benchmark.run_benchmark_suite(
                participant_counts=args.participant_counts,
                data_sizes=args.data_sizes,
                model_complexities=args.model_complexities,
                iterations=args.iterations
            )
    
    try:
        asyncio.run(run_tests())
        logger.info("联邦训练基准测试完成")
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())