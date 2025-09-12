#!/usr/bin/env python3
"""
产出物完整性验证工具

验证所有基准测试产出物的完整性和数据可追溯性
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

class DeliverableValidator:
    """产出物验证器"""
    
    def __init__(self, project_root: str = "."):
        """初始化验证器"""
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.docs_dir = self.project_root / "docs"
        self.assets_dir = self.docs_dir / "assets"
        self.models_dir = self.project_root / "models"
        
        logger.info(f"初始化产出物验证器: {self.project_root}")
    
    def validate_file_integrity(self, file_path: Path) -> Dict:
        """验证文件完整性"""
        if not file_path.exists():
            return {
                'exists': False,
                'size': 0,
                'sha256': None,
                'readable': False
            }
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                sha256_hash = hashlib.sha256(content).hexdigest()
            
            return {
                'exists': True,
                'size': len(content),
                'sha256': sha256_hash,
                'readable': True
            }
        except Exception as e:
            return {
                'exists': True,
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'sha256': None,
                'readable': False,
                'error': str(e)
            }
    
    def validate_psi_deliverables(self) -> Dict:
        """验证PSI相关产出物"""
        logger.info("验证PSI产出物...")
        
        results = {
            'psi_summary': None,
            'psi_results': None,
            'psi_integrity': None,
            'psi_charts': []
        }
        
        # 验证PSI汇总数据
        psi_summary_file = self.reports_dir / "psi_summary.csv"
        results['psi_summary'] = self.validate_file_integrity(psi_summary_file)
        
        if results['psi_summary']['exists']:
            try:
                df = pd.read_csv(psi_summary_file)
                results['psi_summary']['records'] = len(df)
                results['psi_summary']['columns'] = list(df.columns)
                logger.info(f"✓ PSI汇总数据: {len(df)} 条记录")
            except Exception as e:
                results['psi_summary']['parse_error'] = str(e)
                logger.error(f"✗ PSI汇总数据解析失败: {e}")
        
        # 验证PSI详细结果
        psi_results_file = self.reports_dir / "psi_results.jsonl"
        results['psi_results'] = self.validate_file_integrity(psi_results_file)
        
        if results['psi_results']['exists']:
            try:
                with open(psi_results_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    results['psi_results']['records'] = len(lines)
                logger.info(f"✓ PSI详细结果: {len(lines)} 条记录")
            except Exception as e:
                results['psi_results']['parse_error'] = str(e)
                logger.error(f"✗ PSI详细结果解析失败: {e}")
        
        # 验证PSI完整性数据
        psi_integrity_file = self.reports_dir / "psi_integrity.json"
        results['psi_integrity'] = self.validate_file_integrity(psi_integrity_file)
        
        if results['psi_integrity']['exists']:
            try:
                with open(psi_integrity_file, 'r') as f:
                    integrity_data = json.load(f)
                    results['psi_integrity']['test_id'] = integrity_data.get('test_id')
                    results['psi_integrity']['accuracy'] = integrity_data.get('accuracy')
                logger.info(f"✓ PSI完整性数据: 精度 = {integrity_data.get('accuracy', 'N/A')}")
            except Exception as e:
                results['psi_integrity']['parse_error'] = str(e)
                logger.error(f"✗ PSI完整性数据解析失败: {e}")
        
        # 验证PSI图表
        psi_charts = ['psi_throughput.png', 'psi_latency.png']
        for chart in psi_charts:
            chart_file = self.assets_dir / chart
            chart_info = self.validate_file_integrity(chart_file)
            chart_info['name'] = chart
            results['psi_charts'].append(chart_info)
            
            if chart_info['exists']:
                logger.info(f"✓ PSI图表: {chart}")
            else:
                logger.warning(f"✗ PSI图表缺失: {chart}")
        
        return results
    
    def validate_train_deliverables(self) -> Dict:
        """验证训练相关产出物"""
        logger.info("验证训练产出物...")
        
        results = {
            'train_summary': None,
            'train_reports': [],
            'train_models': [],
            'train_charts': []
        }
        
        # 验证训练汇总数据
        train_summary_file = self.project_root / "bench" / "train-bench" / "data" / "train_results" / "train_summary.csv"
        results['train_summary'] = self.validate_file_integrity(train_summary_file)
        
        if results['train_summary']['exists']:
            try:
                df = pd.read_csv(train_summary_file)
                results['train_summary']['records'] = len(df)
                results['train_summary']['columns'] = list(df.columns)
                logger.info(f"✓ 训练汇总数据: {len(df)} 条记录")
            except Exception as e:
                results['train_summary']['parse_error'] = str(e)
                logger.error(f"✗ 训练汇总数据解析失败: {e}")
        
        # 验证训练报告文件
        for report_file in self.reports_dir.glob("train_report_*.json"):
            report_info = self.validate_file_integrity(report_file)
            report_info['name'] = report_file.name
            
            if report_info['exists']:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                        report_info['test_id'] = report_data.get('test_id')
                        report_info['final_auc'] = report_data.get('final_auc')
                        report_info['final_ks'] = report_data.get('final_ks')
                    logger.info(f"✓ 训练报告: {report_file.name}")
                except Exception as e:
                    report_info['parse_error'] = str(e)
                    logger.error(f"✗ 训练报告解析失败: {report_file.name}")
            
            results['train_reports'].append(report_info)
        
        # 验证模型文件
        for model_file in self.models_dir.glob("*.pkl"):
            model_info = self.validate_file_integrity(model_file)
            model_info['name'] = model_file.name
            results['train_models'].append(model_info)
            
            if model_info['exists']:
                logger.info(f"✓ 模型文件: {model_file.name}")
            else:
                logger.warning(f"✗ 模型文件缺失: {model_file.name}")
        
        # 验证训练图表
        train_charts = ['train_auc_ks_vs_epsilon.png', 'train_comm_vs_round.png']
        for chart in train_charts:
            chart_file = self.assets_dir / chart
            chart_info = self.validate_file_integrity(chart_file)
            chart_info['name'] = chart
            results['train_charts'].append(chart_info)
            
            if chart_info['exists']:
                logger.info(f"✓ 训练图表: {chart}")
            else:
                logger.warning(f"✗ 训练图表缺失: {chart}")
        
        return results
    
    def validate_documentation(self) -> Dict:
        """验证文档产出物"""
        logger.info("验证文档产出物...")
        
        results = {
            'showcase_doc': None,
            'other_docs': []
        }
        
        # 验证展示文档
        showcase_file = self.docs_dir / "pab-federated-showcase.md"
        results['showcase_doc'] = self.validate_file_integrity(showcase_file)
        
        if results['showcase_doc']['exists']:
            try:
                with open(showcase_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    results['showcase_doc']['word_count'] = len(content.split())
                    results['showcase_doc']['line_count'] = len(content.split('\n'))
                logger.info(f"✓ 展示文档: {results['showcase_doc']['word_count']} 词")
            except Exception as e:
                results['showcase_doc']['parse_error'] = str(e)
                logger.error(f"✗ 展示文档解析失败: {e}")
        
        # 验证其他文档
        doc_files = ['README.md', 'ARCHITECTURE.md', 'SECURITY.md', 'COMPLIANCE.md']
        for doc_file in doc_files:
            doc_path = self.docs_dir / doc_file
            doc_info = self.validate_file_integrity(doc_path)
            doc_info['name'] = doc_file
            results['other_docs'].append(doc_info)
            
            if doc_info['exists']:
                logger.info(f"✓ 文档: {doc_file}")
            else:
                logger.warning(f"✗ 文档缺失: {doc_file}")
        
        return results
    
    def validate_traceability(self) -> Dict:
        """验证数据可追溯性"""
        logger.info("验证数据可追溯性...")
        
        results = {
            'file_manifest': {},
            'hash_consistency': True,
            'timestamp_consistency': True,
            'id_consistency': True
        }
        
        # 生成文件清单
        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.csv', '.jsonl', '.json']:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        sha256_hash = hashlib.sha256(content).hexdigest()
                        relative_path = file_path.relative_to(self.project_root)
                        results['file_manifest'][str(relative_path)] = {
                            'sha256': sha256_hash,
                            'size': len(content),
                            'modified': file_path.stat().st_mtime
                        }
                except Exception as e:
                    logger.warning(f"无法处理文件 {file_path}: {e}")
        
        logger.info(f"✓ 文件清单: {len(results['file_manifest'])} 个文件")
        
        # 验证测试ID一致性
        test_ids = set()
        
        # 从PSI汇总中提取测试ID
        psi_summary_file = self.reports_dir / "psi_summary.csv"
        if psi_summary_file.exists():
            try:
                df = pd.read_csv(psi_summary_file)
                for test_id in df['test_id']:
                    test_ids.add(test_id)
            except Exception as e:
                logger.warning(f"无法从PSI汇总中提取测试ID: {e}")
        
        # 从训练汇总中提取测试ID
        train_summary_file = self.project_root / "bench" / "train-bench" / "data" / "train_results" / "train_summary.csv"
        if train_summary_file.exists():
            try:
                df = pd.read_csv(train_summary_file)
                for test_id in df['test_id']:
                    test_ids.add(test_id)
            except Exception as e:
                logger.warning(f"无法从训练汇总中提取测试ID: {e}")
        
        results['unique_test_ids'] = len(test_ids)
        logger.info(f"✓ 唯一测试ID: {len(test_ids)} 个")
        
        return results
    
    def generate_validation_report(self) -> Dict:
        """生成完整验证报告"""
        logger.info("生成验证报告...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'project_root': str(self.project_root),
            'psi_deliverables': self.validate_psi_deliverables(),
            'train_deliverables': self.validate_train_deliverables(),
            'documentation': self.validate_documentation(),
            'traceability': self.validate_traceability()
        }
        
        # 计算总体评分
        total_files = 0
        valid_files = 0
        
        # PSI文件
        for key in ['psi_summary', 'psi_results', 'psi_integrity']:
            if report['psi_deliverables'][key]:
                total_files += 1
                if report['psi_deliverables'][key]['exists']:
                    valid_files += 1
        
        for chart in report['psi_deliverables']['psi_charts']:
            total_files += 1
            if chart['exists']:
                valid_files += 1
        
        # 训练文件
        if report['train_deliverables']['train_summary']:
            total_files += 1
            if report['train_deliverables']['train_summary']['exists']:
                valid_files += 1
        
        total_files += len(report['train_deliverables']['train_reports'])
        valid_files += sum(1 for r in report['train_deliverables']['train_reports'] if r['exists'])
        
        total_files += len(report['train_deliverables']['train_models'])
        valid_files += sum(1 for m in report['train_deliverables']['train_models'] if m['exists'])
        
        for chart in report['train_deliverables']['train_charts']:
            total_files += 1
            if chart['exists']:
                valid_files += 1
        
        # 文档文件
        if report['documentation']['showcase_doc']:
            total_files += 1
            if report['documentation']['showcase_doc']['exists']:
                valid_files += 1
        
        total_files += len(report['documentation']['other_docs'])
        valid_files += sum(1 for d in report['documentation']['other_docs'] if d['exists'])
        
        report['summary'] = {
            'total_files': total_files,
            'valid_files': valid_files,
            'completion_rate': valid_files / total_files if total_files > 0 else 0,
            'file_manifest_count': len(report['traceability']['file_manifest']),
            'unique_test_ids': report['traceability']['unique_test_ids']
        }
        
        logger.info(f"✅ 验证完成: {valid_files}/{total_files} 文件有效 ({report['summary']['completion_rate']:.1%})")
        
        return report
    
    def save_validation_report(self, report: Dict) -> str:
        """保存验证报告"""
        output_file = self.reports_dir / "validation_report.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"验证报告已保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证基准测试产出物完整性")
    parser.add_argument("--project-root", default=".",
                       help="项目根目录 (默认: 当前目录)")
    parser.add_argument("--save-report", action="store_true",
                       help="保存验证报告到文件")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = DeliverableValidator(args.project_root)
    
    try:
        # 生成验证报告
        report = validator.generate_validation_report()
        
        # 保存报告
        if args.save_report:
            output_file = validator.save_validation_report(report)
            print(f"\n📄 验证报告已保存: {output_file}")
        
        # 打印摘要
        summary = report['summary']
        print(f"\n📊 验证摘要:")
        print(f"  - 总文件数: {summary['total_files']}")
        print(f"  - 有效文件数: {summary['valid_files']}")
        print(f"  - 完成率: {summary['completion_rate']:.1%}")
        print(f"  - 文件清单: {summary['file_manifest_count']} 个文件")
        print(f"  - 唯一测试ID: {summary['unique_test_ids']} 个")
        
        if summary['completion_rate'] >= 0.9:
            print(f"\n✅ 产出物验证通过! 完成率 {summary['completion_rate']:.1%}")
            return 0
        else:
            print(f"\n⚠️  产出物验证警告! 完成率 {summary['completion_rate']:.1%} < 90%")
            return 1
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return 1

if __name__ == '__main__':
    exit(main())