#!/usr/bin/env python3
"""
联邦学习基准测试完整运行脚本

执行端到端的基准测试流程：
1. 启动Ray集群
2. 运行十亿级PSI基准测试
3. 运行百万级联邦训练基准测试
4. 生成可视化图表
5. 生成展示文档
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class FullBenchmarkRunner:
    """完整基准测试运行器"""
    
    def __init__(self, project_root: str):
        """初始化运行器"""
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.assets_dir = self.project_root / "docs" / "assets"
        self.scripts_dir = self.project_root / "scripts"
        self.bench_dir = self.project_root / "bench"
        
        # 创建必要目录
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"项目根目录: {self.project_root}")
        logger.info(f"报告目录: {self.reports_dir}")
        logger.info(f"图片目录: {self.assets_dir}")
    
    def check_dependencies(self) -> bool:
        """检查依赖项"""
        logger.info("检查依赖项...")
        
        required_packages = [
            'ray', 'numpy', 'pandas', 'matplotlib', 'sklearn',
            'xgboost', 'pybloom_live', 'cryptography', 'fastapi', 'uvicorn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.debug(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package}")
        
        if missing_packages:
            logger.error(f"缺少依赖包: {missing_packages}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✓ 所有依赖项检查通过")
        return True
    
    def start_ray_cluster(self, workers: int = 16) -> bool:
        """启动Ray集群"""
        logger.info(f"启动Ray集群 (workers={workers})...")
        
        cluster_script = self.scripts_dir / "cluster_up.sh"
        if not cluster_script.exists():
            logger.error(f"集群启动脚本不存在: {cluster_script}")
            return False
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['WORKERS'] = str(workers)
            
            # 运行集群启动脚本
            result = subprocess.run(
                ["bash", str(cluster_script)],
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                logger.info("✓ Ray集群启动成功")
                logger.debug(f"启动输出: {result.stdout}")
                return True
            else:
                logger.error(f"Ray集群启动失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Ray集群启动超时")
            return False
        except Exception as e:
            logger.error(f"启动Ray集群时发生错误: {e}")
            return False
    
    def run_psi_benchmark(self, target_size: int = 1_000_000_000) -> bool:
        """运行PSI基准测试"""
        logger.info(f"运行PSI基准测试 (目标规模: {target_size:,})...")
        
        psi_script = self.bench_dir / "psi-bench" / "psi_benchmark.py"
        if not psi_script.exists():
            logger.error(f"PSI基准测试脚本不存在: {psi_script}")
            return False
        
        try:
            # 运行PSI基准测试
            cmd = [
                sys.executable, str(psi_script),
                "--target-size", str(target_size),
                "--intersection-ratios", "0.01,0.05,0.10",
                "--output-dir", str(self.reports_dir),
                "--use-ray", "true",
                "--max-time-hours", "24"
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=24 * 3600  # 24小时超时
            )
            
            if result.returncode == 0:
                logger.info("✓ PSI基准测试完成")
                logger.debug(f"测试输出: {result.stdout}")
                return True
            else:
                logger.error(f"PSI基准测试失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("PSI基准测试超时 (24小时)")
            return False
        except Exception as e:
            logger.error(f"运行PSI基准测试时发生错误: {e}")
            return False
    
    def run_train_benchmark(self, target_size: int = 1_000_000) -> bool:
        """运行联邦训练基准测试"""
        logger.info(f"运行联邦训练基准测试 (目标规模: {target_size:,})...")
        
        train_script = self.bench_dir / "train-bench" / "train_benchmark.py"
        if not train_script.exists():
            logger.error(f"训练基准测试脚本不存在: {train_script}")
            return False
        
        try:
            # 运行训练基准测试
            cmd = [
                sys.executable, str(train_script),
                "--target-size", str(target_size),
                "--epsilon-values", "inf,8,5,3",
                "--output-dir", str(self.reports_dir),
                "--use-ray", "true",
                "--max-time-hours", "24"
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=24 * 3600  # 24小时超时
            )
            
            if result.returncode == 0:
                logger.info("✓ 联邦训练基准测试完成")
                logger.debug(f"测试输出: {result.stdout}")
                return True
            else:
                logger.error(f"联邦训练基准测试失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("联邦训练基准测试超时 (24小时)")
            return False
        except Exception as e:
            logger.error(f"运行联邦训练基准测试时发生错误: {e}")
            return False
    
    def generate_plots(self) -> bool:
        """生成可视化图表"""
        logger.info("生成可视化图表...")
        
        plot_script = self.project_root / "tools" / "plot_bench.py"
        if not plot_script.exists():
            logger.error(f"绘图脚本不存在: {plot_script}")
            return False
        
        try:
            # 运行绘图脚本
            cmd = [
                sys.executable, str(plot_script),
                "--reports-dir", str(self.reports_dir),
                "--assets-dir", str(self.assets_dir),
                "--plot-type", "all"
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                logger.info("✓ 可视化图表生成完成")
                logger.debug(f"绘图输出: {result.stdout}")
                return True
            else:
                logger.error(f"图表生成失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("图表生成超时")
            return False
        except Exception as e:
            logger.error(f"生成图表时发生错误: {e}")
            return False
    
    def validate_results(self) -> Dict[str, bool]:
        """验证测试结果"""
        logger.info("验证测试结果...")
        
        validation_results = {
            'psi_summary_exists': False,
            'psi_target_met': False,
            'train_summary_exists': False,
            'train_target_met': False,
            'all_plots_exist': False,
            'markdown_ready': False
        }
        
        # 检查PSI结果
        psi_summary_file = self.reports_dir / "psi_summary.csv"
        if psi_summary_file.exists():
            validation_results['psi_summary_exists'] = True
            try:
                import pandas as pd
                psi_df = pd.read_csv(psi_summary_file)
                total_processed = psi_df['total_processed'].sum()
                max_time = psi_df['wall_clock_hours'].max()
                
                if total_processed >= 1_000_000_000 and max_time <= 24:
                    validation_results['psi_target_met'] = True
                    logger.info(f"✓ PSI目标达成: {total_processed:,} 条记录，{max_time:.2f} 小时")
                else:
                    logger.warning(f"✗ PSI目标未达成: {total_processed:,} 条记录，{max_time:.2f} 小时")
            except Exception as e:
                logger.error(f"验证PSI结果时发生错误: {e}")
        
        # 检查训练结果
        train_summary_file = self.reports_dir / "train_summary.csv"
        if train_summary_file.exists():
            validation_results['train_summary_exists'] = True
            try:
                import pandas as pd
                train_df = pd.read_csv(train_summary_file)
                max_size = train_df['n_samples'].max()
                max_time = train_df['wall_clock_hours'].max()
                
                if max_size >= 1_000_000 and max_time <= 24:
                    validation_results['train_target_met'] = True
                    logger.info(f"✓ 训练目标达成: {max_size:,} 样本，{max_time:.2f} 小时")
                else:
                    logger.warning(f"✗ 训练目标未达成: {max_size:,} 样本，{max_time:.2f} 小时")
            except Exception as e:
                logger.error(f"验证训练结果时发生错误: {e}")
        
        # 检查图表文件
        required_plots = [
            "psi_throughput.png",
            "psi_latency.png",
            "train_auc_ks_vs_epsilon.png",
            "train_comm_vs_round.png",
            "inference_latency.png",
            "improvement_before_after.png"
        ]
        
        all_plots_exist = True
        for plot_name in required_plots:
            plot_file = self.assets_dir / plot_name
            if not plot_file.exists():
                logger.warning(f"✗ 缺少图表: {plot_name}")
                all_plots_exist = False
            else:
                logger.debug(f"✓ 图表存在: {plot_name}")
        
        validation_results['all_plots_exist'] = all_plots_exist
        if all_plots_exist:
            logger.info("✓ 所有必需图表已生成")
        
        # 检查是否准备好生成Markdown
        validation_results['markdown_ready'] = (
            validation_results['psi_target_met'] and
            validation_results['train_target_met'] and
            validation_results['all_plots_exist']
        )
        
        return validation_results
    
    def stop_ray_cluster(self) -> bool:
        """停止Ray集群"""
        logger.info("停止Ray集群...")
        
        cluster_script = self.scripts_dir / "cluster_down.sh"
        if not cluster_script.exists():
            logger.warning(f"集群停止脚本不存在: {cluster_script}")
            return False
        
        try:
            # 运行集群停止脚本
            result = subprocess.run(
                ["bash", str(cluster_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60  # 1分钟超时
            )
            
            if result.returncode == 0:
                logger.info("✓ Ray集群已停止")
                return True
            else:
                logger.warning(f"停止Ray集群时出现警告: {result.stderr}")
                return True  # 即使有警告也认为成功
                
        except subprocess.TimeoutExpired:
            logger.warning("停止Ray集群超时")
            return False
        except Exception as e:
            logger.warning(f"停止Ray集群时发生错误: {e}")
            return False
    
    def run_full_benchmark(self, psi_target: int = 1_000_000_000, 
                          train_target: int = 1_000_000, 
                          workers: int = 16) -> bool:
        """运行完整基准测试"""
        logger.info("开始完整基准测试流程...")
        
        start_time = time.time()
        
        try:
            # 1. 检查依赖
            if not self.check_dependencies():
                return False
            
            # 2. 启动Ray集群
            if not self.start_ray_cluster(workers):
                return False
            
            # 3. 运行PSI基准测试
            if not self.run_psi_benchmark(psi_target):
                return False
            
            # 4. 运行训练基准测试
            if not self.run_train_benchmark(train_target):
                return False
            
            # 5. 生成图表
            if not self.generate_plots():
                return False
            
            # 6. 验证结果
            validation_results = self.validate_results()
            
            # 7. 输出结果摘要
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"\n{'='*60}")
            logger.info("基准测试完成摘要")
            logger.info(f"{'='*60}")
            logger.info(f"总耗时: {total_time/3600:.2f} 小时")
            logger.info(f"PSI目标达成: {'✓' if validation_results['psi_target_met'] else '✗'}")
            logger.info(f"训练目标达成: {'✓' if validation_results['train_target_met'] else '✗'}")
            logger.info(f"图表生成完成: {'✓' if validation_results['all_plots_exist'] else '✗'}")
            logger.info(f"准备生成文档: {'✓' if validation_results['markdown_ready'] else '✗'}")
            
            if validation_results['markdown_ready']:
                logger.info("\n🎉 所有基准测试目标已达成，可以生成展示文档！")
                return True
            else:
                logger.warning("\n⚠️  部分目标未达成，请检查测试结果")
                return False
            
        except KeyboardInterrupt:
            logger.warning("\n用户中断测试")
            return False
        except Exception as e:
            logger.error(f"\n基准测试过程中发生错误: {e}")
            return False
        finally:
            # 清理：停止Ray集群
            self.stop_ray_cluster()

def main():
    parser = argparse.ArgumentParser(description="联邦学习完整基准测试")
    parser.add_argument("--psi-target", type=int, default=1_000_000_000,
                       help="PSI目标数据规模 (默认: 1,000,000,000)")
    parser.add_argument("--train-target", type=int, default=1_000_000,
                       help="训练目标数据规模 (默认: 1,000,000)")
    parser.add_argument("--workers", type=int, default=16,
                       help="Ray集群worker数量 (默认: 16)")
    parser.add_argument("--project-root", default=".",
                       help="项目根目录 (默认: 当前目录)")
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = FullBenchmarkRunner(args.project_root)
    
    # 运行完整基准测试
    success = runner.run_full_benchmark(
        psi_target=args.psi_target,
        train_target=args.train_target,
        workers=args.workers
    )
    
    if success:
        print("\n✅ 基准测试成功完成！")
        print("\n下一步：")
        print("1. 检查 reports/ 目录中的测试结果")
        print("2. 查看 docs/assets/ 目录中的图表")
        print("3. 运行文档生成脚本创建展示文档")
        return 0
    else:
        print("\n❌ 基准测试未能完全成功")
        print("请检查日志输出并解决问题后重试")
        return 1

if __name__ == '__main__':
    exit(main())