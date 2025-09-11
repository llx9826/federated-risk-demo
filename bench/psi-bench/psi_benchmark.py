#!/usr/bin/env python3
"""
PSI性能基准测试工具

测试不同数据规模下的PSI计算性能，包括：
- ECDH-PSI性能测试
- Token-join性能测试
- 内存使用分析
- 网络传输分析
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
logger.add("psi_benchmark.log", rotation="10 MB")

class PSIBenchmark:
    """PSI性能基准测试"""
    
    def __init__(self, psi_service_url: str = "http://localhost:8003"):
        """初始化PSI基准测试"""
        self.psi_service_url = psi_service_url
        self.results = []
        self.process = psutil.Process()
        
        # 创建结果目录
        self.results_dir = Path("../results/psi")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def _monitor_resources(self, duration: float) -> Dict:
        """监控资源使用情况"""
        start_time = time.time()
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            cpu_samples.append(self.process.cpu_percent())
            memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB
            await asyncio.sleep(0.1)
        
        return {
            'avg_cpu_percent': np.mean(cpu_samples),
            'max_cpu_percent': np.max(cpu_samples),
            'avg_memory_mb': np.mean(memory_samples),
            'max_memory_mb': np.max(memory_samples)
        }
    
    async def _create_psi_session(self, session_id: str, algorithm: str = "ecdh") -> bool:
        """创建PSI会话"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "session_id": session_id,
                    "algorithm": algorithm,
                    "participants": ["party_a", "party_b"],
                    "config": {
                        "hash_function": "sha256",
                        "curve": "secp256r1" if algorithm == "ecdh" else None
                    }
                }
                
                async with session.post(
                    f"{self.psi_service_url}/sessions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.debug(f"PSI会话创建成功: {session_id}")
                        return True
                    else:
                        logger.error(f"PSI会话创建失败: {response.status}")
                        return False
                        
            except Exception as e:
                logger.error(f"创建PSI会话异常: {e}")
                return False
    
    async def _upload_data(self, session_id: str, party_id: str, data: List[str]) -> bool:
        """上传数据到PSI会话"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "party_id": party_id,
                    "data": data
                }
                
                async with session.post(
                    f"{self.psi_service_url}/sessions/{session_id}/data",
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.debug(f"数据上传成功: {party_id}, {len(data)} 条记录")
                        return True
                    else:
                        logger.error(f"数据上传失败: {response.status}")
                        return False
                        
            except Exception as e:
                logger.error(f"上传数据异常: {e}")
                return False
    
    async def _compute_intersection(self, session_id: str) -> Tuple[bool, Optional[List[str]]]:
        """计算交集"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.psi_service_url}/sessions/{session_id}/compute"
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"交集计算成功: {len(result.get('intersection', []))} 条记录")
                        return True, result.get('intersection', [])
                    else:
                        logger.error(f"交集计算失败: {response.status}")
                        return False, None
                        
            except Exception as e:
                logger.error(f"计算交集异常: {e}")
                return False, None
    
    async def _wait_for_completion(self, session_id: str, timeout: int = 300) -> bool:
        """等待PSI计算完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.psi_service_url}/sessions/{session_id}/status"
                    ) as response:
                        if response.status == 200:
                            status = await response.json()
                            if status.get('status') == 'completed':
                                return True
                            elif status.get('status') == 'failed':
                                logger.error(f"PSI计算失败: {status.get('error')}")
                                return False
                        
                except Exception as e:
                    logger.error(f"检查状态异常: {e}")
            
            await asyncio.sleep(1)
        
        logger.error(f"PSI计算超时: {timeout} 秒")
        return False
    
    def _generate_test_data(self, size: int, overlap_ratio: float = 0.3) -> Tuple[List[str], List[str]]:
        """生成测试数据"""
        # 生成基础数据池
        total_unique = int(size * 2 - size * overlap_ratio)
        base_data = [f"id_{i:08d}" for i in range(total_unique)]
        
        # 第一方数据：前size个
        party_a_data = base_data[:size]
        
        # 第二方数据：后size个，确保有overlap_ratio的重叠
        overlap_size = int(size * overlap_ratio)
        party_b_unique = base_data[size:size + (size - overlap_size)]
        party_b_overlap = base_data[:overlap_size]
        party_b_data = party_b_unique + party_b_overlap
        
        # 随机打乱
        np.random.shuffle(party_a_data)
        np.random.shuffle(party_b_data)
        
        return party_a_data, party_b_data
    
    async def benchmark_single_test(self, data_size: int, algorithm: str = "ecdh", 
                                   overlap_ratio: float = 0.3) -> Dict:
        """单次PSI性能测试"""
        session_id = f"bench_{algorithm}_{data_size}_{int(time.time())}"
        
        logger.info(f"开始PSI测试: 算法={algorithm}, 数据量={data_size}, 重叠率={overlap_ratio:.1%}")
        
        # 生成测试数据
        party_a_data, party_b_data = self._generate_test_data(data_size, overlap_ratio)
        expected_intersection = len(set(party_a_data) & set(party_b_data))
        
        start_time = time.time()
        
        try:
            # 1. 创建会话
            session_created = await self._create_psi_session(session_id, algorithm)
            if not session_created:
                return {'error': '会话创建失败'}
            
            session_creation_time = time.time() - start_time
            
            # 2. 上传数据
            upload_start = time.time()
            
            # 启动资源监控
            monitor_task = asyncio.create_task(
                self._monitor_resources(duration=60)  # 监控60秒
            )
            
            # 并行上传数据
            upload_tasks = [
                self._upload_data(session_id, "party_a", party_a_data),
                self._upload_data(session_id, "party_b", party_b_data)
            ]
            
            upload_results = await asyncio.gather(*upload_tasks)
            upload_time = time.time() - upload_start
            
            if not all(upload_results):
                return {'error': '数据上传失败'}
            
            # 3. 计算交集
            compute_start = time.time()
            success, intersection = await self._compute_intersection(session_id)
            
            if not success:
                return {'error': '交集计算失败'}
            
            # 等待计算完成
            completed = await self._wait_for_completion(session_id)
            compute_time = time.time() - compute_start
            
            if not completed:
                return {'error': '计算超时'}
            
            total_time = time.time() - start_time
            
            # 停止资源监控
            monitor_task.cancel()
            try:
                resource_stats = await monitor_task
            except asyncio.CancelledError:
                resource_stats = {'avg_cpu_percent': 0, 'max_cpu_percent': 0, 
                                'avg_memory_mb': 0, 'max_memory_mb': 0}
            
            # 验证结果
            actual_intersection = len(intersection) if intersection else 0
            accuracy = actual_intersection / expected_intersection if expected_intersection > 0 else 1.0
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'algorithm': algorithm,
                'data_size': data_size,
                'overlap_ratio': overlap_ratio,
                'expected_intersection': expected_intersection,
                'actual_intersection': actual_intersection,
                'accuracy': accuracy,
                'session_creation_time': session_creation_time,
                'upload_time': upload_time,
                'compute_time': compute_time,
                'total_time': total_time,
                'throughput_records_per_sec': data_size * 2 / total_time,
                'resource_usage': resource_stats,
                'success': True
            }
            
            logger.info(f"PSI测试完成: {data_size} 条记录, 耗时 {total_time:.2f} 秒, "
                       f"吞吐量 {result['throughput_records_per_sec']:.0f} 记录/秒")
            
            return result
            
        except Exception as e:
            logger.error(f"PSI测试异常: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'algorithm': algorithm,
                'data_size': data_size,
                'error': str(e),
                'success': False
            }
    
    async def run_benchmark_suite(self, data_sizes: List[int], algorithms: List[str], 
                                 iterations: int = 3, overlap_ratio: float = 0.3):
        """运行完整的基准测试套件"""
        logger.info(f"开始PSI基准测试套件: {len(data_sizes)} 个数据规模, "
                   f"{len(algorithms)} 个算法, {iterations} 次迭代")
        
        total_tests = len(data_sizes) * len(algorithms) * iterations
        progress_bar = tqdm(total=total_tests, desc="PSI基准测试")
        
        for algorithm in algorithms:
            for data_size in data_sizes:
                for iteration in range(iterations):
                    result = await self.benchmark_single_test(
                        data_size=data_size,
                        algorithm=algorithm,
                        overlap_ratio=overlap_ratio
                    )
                    
                    result['iteration'] = iteration + 1
                    self.results.append(result)
                    
                    progress_bar.update(1)
                    
                    # 测试间隔，避免服务过载
                    await asyncio.sleep(1)
        
        progress_bar.close()
        
        # 保存结果
        await self._save_results()
        
        # 生成报告
        await self._generate_report()
    
    async def _save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = self.results_dir / f"psi_benchmark_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {results_file}")
        
        # 保存CSV格式
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = self.results_dir / f"psi_benchmark_{timestamp}.csv"
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
        
        # 按算法和数据规模分组统计
        summary = df.groupby(['algorithm', 'data_size']).agg({
            'total_time': ['mean', 'std', 'min', 'max'],
            'throughput_records_per_sec': ['mean', 'std', 'min', 'max'],
            'accuracy': ['mean', 'std'],
            'compute_time': ['mean', 'std'],
            'upload_time': ['mean', 'std']
        }).round(4)
        
        # 生成报告
        report = {
            'test_summary': {
                'total_tests': len(self.results),
                'successful_tests': len(df),
                'algorithms_tested': df['algorithm'].unique().tolist(),
                'data_sizes_tested': sorted(df['data_size'].unique().tolist()),
                'test_date': datetime.now().isoformat()
            },
            'performance_summary': summary.to_dict(),
            'best_performance': {
                'highest_throughput': {
                    'value': df['throughput_records_per_sec'].max(),
                    'algorithm': df.loc[df['throughput_records_per_sec'].idxmax(), 'algorithm'],
                    'data_size': df.loc[df['throughput_records_per_sec'].idxmax(), 'data_size']
                },
                'lowest_latency': {
                    'value': df['total_time'].min(),
                    'algorithm': df.loc[df['total_time'].idxmin(), 'algorithm'],
                    'data_size': df.loc[df['total_time'].idxmin(), 'data_size']
                }
            }
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"psi_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存到: {report_file}")
        
        # 打印摘要
        logger.info("=== PSI性能测试摘要 ===")
        logger.info(f"总测试数: {report['test_summary']['total_tests']}")
        logger.info(f"成功测试数: {report['test_summary']['successful_tests']}")
        logger.info(f"最高吞吐量: {report['best_performance']['highest_throughput']['value']:.0f} 记录/秒 "
                   f"({report['best_performance']['highest_throughput']['algorithm']})")
        logger.info(f"最低延迟: {report['best_performance']['lowest_latency']['value']:.2f} 秒 "
                   f"({report['best_performance']['lowest_latency']['algorithm']})")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PSI性能基准测试工具')
    parser.add_argument('--service-url', default='http://localhost:8003', 
                       help='PSI服务URL')
    parser.add_argument('--data-sizes', nargs='+', type=int, 
                       default=[1000, 5000, 10000, 50000], 
                       help='测试数据规模')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['ecdh', 'token_join'], 
                       help='测试算法')
    parser.add_argument('--iterations', type=int, default=3, 
                       help='每个测试的迭代次数')
    parser.add_argument('--overlap-ratio', type=float, default=0.3, 
                       help='数据重叠比例')
    parser.add_argument('--single-test', action='store_true', 
                       help='只运行单次测试')
    parser.add_argument('--data-size', type=int, default=10000, 
                       help='单次测试的数据规模')
    parser.add_argument('--algorithm', default='ecdh', 
                       help='单次测试的算法')
    
    args = parser.parse_args()
    
    async def run_tests():
        benchmark = PSIBenchmark(args.service_url)
        
        if args.single_test:
            # 单次测试
            result = await benchmark.benchmark_single_test(
                data_size=args.data_size,
                algorithm=args.algorithm,
                overlap_ratio=args.overlap_ratio
            )
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # 完整基准测试套件
            await benchmark.run_benchmark_suite(
                data_sizes=args.data_sizes,
                algorithms=args.algorithms,
                iterations=args.iterations,
                overlap_ratio=args.overlap_ratio
            )
    
    try:
        asyncio.run(run_tests())
        logger.info("PSI基准测试完成")
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())