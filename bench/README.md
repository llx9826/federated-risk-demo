# 性能测试基准

本目录包含联邦风控系统的性能测试脚本和基准测试工具。

## 目录结构

```
bench/
├── README.md              # 本文件
├── data-gen/              # 数据生成工具
│   ├── generate_data.py   # 主数据生成脚本
│   ├── config.yaml        # 数据生成配置
│   └── requirements.txt   # 依赖包
├── psi-bench/             # PSI性能测试
│   ├── psi_benchmark.py   # PSI基准测试
│   ├── load_test.py       # 负载测试
│   └── requirements.txt   # 依赖包
├── train-bench/           # 训练性能测试
│   ├── train_benchmark.py # 训练基准测试
│   ├── memory_test.py     # 内存使用测试
│   └── requirements.txt   # 依赖包
└── results/               # 测试结果输出
    ├── psi/               # PSI测试结果
    ├── training/          # 训练测试结果
    └── reports/           # 综合报告
```

## 快速开始

### 1. 数据生成

```bash
cd bench/data-gen
pip install -r requirements.txt
python generate_data.py --config config.yaml
```

### 2. PSI性能测试

```bash
cd bench/psi-bench
pip install -r requirements.txt
python psi_benchmark.py --data-size 10000
```

### 3. 训练性能测试

```bash
cd bench/train-bench
pip install -r requirements.txt
python train_benchmark.py --epochs 10
```

## 测试场景

### PSI测试场景
- **小规模**: 1K-10K记录
- **中规模**: 10K-100K记录
- **大规模**: 100K-1M记录
- **超大规模**: 1M+记录

### 训练测试场景
- **单机训练**: 本地模型训练性能
- **联邦训练**: 多方协作训练性能
- **内存使用**: 训练过程内存占用
- **收敛速度**: 模型收敛时间分析

## 性能指标

### PSI指标
- **吞吐量**: 每秒处理记录数
- **延迟**: 单次PSI计算时间
- **内存使用**: 峰值内存占用
- **网络带宽**: 数据传输量

### 训练指标
- **训练时间**: 每轮训练耗时
- **收敛轮数**: 达到目标精度的轮数
- **内存使用**: 训练过程内存占用
- **模型精度**: 最终模型性能

## 报告生成

测试完成后，可以生成综合性能报告：

```bash
python generate_report.py --output results/reports/
```

## 注意事项

1. **环境要求**: 确保测试环境与生产环境配置相似
2. **数据隐私**: 使用模拟数据，避免真实敏感数据
3. **资源监控**: 测试过程中监控CPU、内存、网络使用情况
4. **多次测试**: 进行多次测试取平均值，确保结果可靠性