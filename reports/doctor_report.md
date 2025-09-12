# 联邦风控系统医生诊断报告

**生成时间**: {timestamp}  
**诊断会话ID**: {session_id}  
**系统版本**: {version}  
**执行模式**: {mode}  

---

## 📊 数据层诊断

### 数据规模统计
- **总样本数**: {total_samples:,}
- **交集大小**: {intersection_size:,} ({intersection_rate:.2%})
- **坏账率**: {bad_rate:.3f} (目标范围: 0.08-0.20)
- **特征数量**: {feature_count}

### 数据质量检查
- **缺失值比例**: {missing_rate:.3f}
- **常量列数量**: {constant_columns_count}
- **无穷值比例**: {inf_rate:.3f}
- **重复样本**: {duplicate_samples:,}

### 剔除列清单
{removed_columns}

### 信号强度分析
- **强信号特征数**: {strong_signal_count} (|ρ|≥0.1)
- **最强相关性**: {max_correlation:.3f}
- **平均相关性**: {avg_correlation:.3f}

**状态**: {data_layer_status} ✅❌

---

## 🔗 对齐层诊断

### PSI对齐统计
- **盐一致性**: {salt_consistency} ✅❌
- **Token唯一率**: {token_uniqueness:.4f} (要求>99.9%)
- **对齐率**: {alignment_rate:.3f} (配置±2%范围)
- **对齐耗时**: {alignment_duration_ms}ms

### 样例哈希对比
```
甲方样例哈希: {party_a_sample_hashes}
乙方样例哈希: {party_b_sample_hashes}
盐值: {salt_value}
```

**状态**: {alignment_layer_status} ✅❌

---

## 🚀 训练层诊断

### 训练配置
- **算法**: {algorithm}
- **差分隐私**: ε={dp_epsilon} (∞表示关闭)
- **迭代轮数**: {iterations}
- **学习率**: {learning_rate}
- **最大深度**: {max_depth}
- **类别权重**: {scale_pos_weight}

### 训练过程监控
- **总耗时**: {training_duration_ms}ms
- **平均单轮耗时**: {avg_iteration_time_ms}ms
- **Loss收敛**: {loss_convergence} ✅❌
- **早停触发**: {early_stopping_triggered}

### 联邦vs明文对比
- **联邦AUC**: {federated_auc:.4f}
- **明文AUC**: {plaintext_auc:.4f}
- **AUC差异**: {auc_difference:.4f} (阈值<0.03)
- **性能退化**: {performance_degradation} ✅❌

**状态**: {training_layer_status} ✅❌

---

## 📈 评估层诊断

### 模型性能指标
- **AUC**: {auc:.4f} (要求≥0.65)
- **KS**: {ks:.4f} (要求≥0.20)
- **准确率**: {accuracy:.4f}
- **精确率**: {precision:.4f}
- **召回率**: {recall:.4f}

### 预测分布检查
- **预测标准差**: {prediction_std:.4f} (要求>0.01)
- **预测范围**: [{prediction_min:.4f}, {prediction_max:.4f}]
- **0分占比**: {zero_score_ratio:.3f}
- **1分占比**: {one_score_ratio:.3f}

### 模型文件检查
- **模型文件大小**: {model_file_size_kb}KB (要求>10KB)
- **报告文件**: {report_file_exists} ✅❌
- **模型哈希**: {model_hash}

**状态**: {evaluation_layer_status} ✅❌

---

## 🎯 服务层诊断

### 在线评分测试 (20样本)
- **评分分布非退化**: {scoring_distribution_ok} ✅❌
- **评分标准差**: {scoring_std:.4f} (要求>0.01)
- **0/1占比检查**: 0分{zero_ratio:.1%}, 1分{one_ratio:.1%} (各<95%)
- **平均响应时间**: {avg_response_time_ms}ms

### 审计字段完整性
{audit_fields_check}

**状态**: {serving_layer_status} ✅❌

---

## 🔧 自愈策略执行记录

### 自愈轮次: {self_healing_rounds}/3

{self_healing_attempts}

### 参数调优历史
```
轮次1: eta={eta_1}, max_depth={max_depth_1}, ε={epsilon_1} → AUC={auc_1:.4f}
轮次2: eta={eta_2}, max_depth={max_depth_2}, ε={epsilon_2} → AUC={auc_2:.4f}
轮次3: eta={eta_3}, max_depth={max_depth_3}, ε={epsilon_3} → AUC={auc_3:.4f}
```

### 差分隐私影响分析
- **ε=∞**: AUC={auc_no_dp:.4f}
- **ε=8**: AUC={auc_dp_8:.4f}
- **ε=5**: AUC={auc_dp_5:.4f}
- **ε=3**: AUC={auc_dp_3:.4f}

**自愈结果**: {self_healing_result} ✅❌

---

## 🎯 最终结论

### 系统状态: {overall_status}

{final_conclusion}

### 最优配置
- **推荐阈值**: {recommended_threshold:.3f}
- **是否启用校准**: {calibration_enabled}
- **差分隐私设置**: ε={final_dp_epsilon}
- **模型参数**: eta={final_eta}, max_depth={final_max_depth}

### 性能达标情况
- ✅ AUC≥0.70: {auc_target_met}
- ✅ KS≥0.25: {ks_target_met}
- ✅ 预测分布非退化: {distribution_target_met}
- ✅ 模型文件真实落盘: {model_persisted}

---

## 🚨 人工干预清单

{manual_intervention_list}

---

## 📋 诊断树输出

```
{diagnostic_tree_output}
```

---

## 📁 相关文件

- **事故包**: `incidents/{incident_package_id}.zip`
- **模型文件**: `models/{model_hash}.pkl`
- **训练日志**: `logs/train_{session_id}.json`
- **评估报告**: `reports/eval_{session_id}.json`
- **追踪文件**: `traces/*_traces.jsonl`

---

**报告生成完成时间**: {completion_timestamp}  
**下次建议检查**: {next_check_recommendation}