#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练服务单元测试
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, calculate_auc, calculate_ks, add_dp_noise

client = TestClient(app)

class TestTrainService:
    """训练服务测试类"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "secretflow_status" in data
        assert "data_status" in data
        assert "artifacts_status" in data
    
    def test_calculate_auc(self):
        """测试AUC计算"""
        # 完美分类
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        auc = calculate_auc(y_true, y_pred)
        assert auc == 1.0
        
        # 随机分类
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        auc = calculate_auc(y_true, y_pred)
        assert auc == 0.5
    
    def test_calculate_ks(self):
        """测试KS计算"""
        # 完美分离
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        ks = calculate_ks(y_true, y_pred)
        assert ks > 0.5  # 应该有较高的KS值
        
        # 无分离能力
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        ks = calculate_ks(y_true, y_pred)
        assert ks == 0.0
    
    def test_add_dp_noise_gaussian(self):
        """测试高斯差分隐私噪声"""
        values = np.array([1.0, 2.0, 3.0])
        epsilon = 1.0
        
        noisy_values = add_dp_noise(values, epsilon, noise_type="gaussian")
        
        # 检查形状不变
        assert noisy_values.shape == values.shape
        # 检查值有变化（概率很高）
        assert not np.array_equal(values, noisy_values)
    
    def test_add_dp_noise_laplace(self):
        """测试拉普拉斯差分隐私噪声"""
        values = np.array([1.0, 2.0, 3.0])
        epsilon = 1.0
        
        noisy_values = add_dp_noise(values, epsilon, noise_type="laplace")
        
        # 检查形状不变
        assert noisy_values.shape == values.shape
        # 检查值有变化（概率很高）
        assert not np.array_equal(values, noisy_values)
    
    def test_add_dp_noise_invalid_type(self):
        """测试无效噪声类型"""
        values = np.array([1.0, 2.0, 3.0])
        epsilon = 1.0
        
        with pytest.raises(ValueError):
            add_dp_noise(values, epsilon, noise_type="invalid")
    
    @patch('app.load_psi_mapping')
    @patch('app.os.path.exists')
    @patch('app.pd.read_csv')
    def test_secureboost_training_success(self, mock_read_csv, mock_exists, mock_load_psi):
        """测试SecureBoost训练成功"""
        # Mock PSI映射
        mock_load_psi.return_value = {
            'aligned_tokens': ['token1', 'token2', 'token3']
        }
        
        # Mock文件存在
        mock_exists.return_value = True
        
        # Mock数据
        df_a = pd.DataFrame({
            'psi_token': ['token1', 'token2', 'token3'],
            'age': [25, 35, 45],
            'income': [50000, 70000, 90000],
            'default_label': [0, 1, 0]
        })
        
        df_b = pd.DataFrame({
            'psi_token': ['token1', 'token2', 'token3'],
            'purchase_amount': [1000, 2000, 3000],
            'purchase_frequency': [5, 10, 15]
        })
        
        mock_read_csv.side_effect = [df_a, df_b]
        
        # 训练请求
        request_data = {
            "dp": {
                "enable": False,
                "epsilon": 1.0
            },
            "psi_mapping_key": "test_key",
            "features": {
                "A": ["age", "income"],
                "B": ["purchase_amount", "purchase_frequency"]
            },
            "max_depth": 3,
            "num_boost_round": 5
        }
        
        response = client.post("/train/secureboost", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "run_id" in data
        assert "model_hash" in data
        assert data["algorithm"] == "secureboost"
        assert "auc" in data
        assert "ks" in data
        assert data["status"] == "completed"
    
    @patch('app.load_psi_mapping')
    @patch('app.os.path.exists')
    @patch('app.pd.read_csv')
    def test_hetero_lr_training_success(self, mock_read_csv, mock_exists, mock_load_psi):
        """测试Hetero-LR训练成功"""
        # Mock PSI映射
        mock_load_psi.return_value = {
            'aligned_tokens': ['token1', 'token2', 'token3', 'token4']
        }
        
        # Mock文件存在
        mock_exists.return_value = True
        
        # Mock数据
        df_a = pd.DataFrame({
            'psi_token': ['token1', 'token2', 'token3', 'token4'],
            'age': [25, 35, 45, 55],
            'income': [50000, 70000, 90000, 110000],
            'default_label': [0, 1, 0, 1]
        })
        
        df_b = pd.DataFrame({
            'psi_token': ['token1', 'token2', 'token3', 'token4'],
            'purchase_amount': [1000, 2000, 3000, 4000],
            'purchase_frequency': [5, 10, 15, 20]
        })
        
        mock_read_csv.side_effect = [df_a, df_b]
        
        # 训练请求
        request_data = {
            "dp": {
                "enable": True,
                "epsilon": 2.0,
                "noise_type": "gaussian"
            },
            "psi_mapping_key": "test_key",
            "features": {
                "A": ["age", "income"],
                "B": ["purchase_amount", "purchase_frequency"]
            },
            "max_iter": 50,
            "learning_rate": 0.01
        }
        
        response = client.post("/train/hetero_lr", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "run_id" in data
        assert "model_hash" in data
        assert data["algorithm"] == "hetero_lr"
        assert "auc" in data
        assert "ks" in data
        assert data["dp_epsilon"] == 2.0
        assert data["status"] == "completed"
    
    def test_training_invalid_psi_key(self):
        """测试无效PSI键"""
        request_data = {
            "psi_mapping_key": "invalid_key",
            "features": {
                "A": ["age"],
                "B": ["purchase_amount"]
            }
        }
        
        response = client.post("/train/secureboost", json=request_data)
        assert response.status_code in [400, 500]  # 应该返回错误
    
    def test_training_invalid_features(self):
        """测试无效特征配置"""
        request_data = {
            "psi_mapping_key": "test_key",
            "features": {
                "A": [],  # 空特征列表
                "B": ["purchase_amount"]
            }
        }
        
        response = client.post("/train/secureboost", json=request_data)
        assert response.status_code == 422  # 验证错误
    
    def test_dp_config_validation(self):
        """测试差分隐私配置验证"""
        # 无效的epsilon
        request_data = {
            "dp": {
                "enable": True,
                "epsilon": -1.0  # 无效值
            },
            "psi_mapping_key": "test_key",
            "features": {
                "A": ["age"],
                "B": ["purchase_amount"]
            }
        }
        
        response = client.post("/train/secureboost", json=request_data)
        assert response.status_code == 422  # 验证错误
    
    def test_list_models_empty(self):
        """测试列出模型（空）"""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)
        assert data["total"] >= 0
    
    def test_list_sessions_empty(self):
        """测试列出训练会话（空）"""
        response = client.get("/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)
        assert data["total"] >= 0
    
    def test_get_metrics_not_found(self):
        """测试获取不存在的训练指标"""
        response = client.get("/metrics/nonexistent_run_id")
        assert response.status_code == 404
    
    def test_explain_global_not_found(self):
        """测试全局解释（模型不存在）"""
        request_data = {
            "model_hash": "nonexistent_hash",
            "top_n": 5
        }
        
        response = client.post("/explain/global", json=request_data)
        assert response.status_code == 404
    
    def test_explain_local_not_found(self):
        """测试本地解释（模型不存在）"""
        request_data = {
            "model_hash": "nonexistent_hash",
            "sample_id": "sample123",
            "party": "A"
        }
        
        response = client.post("/explain/local", json=request_data)
        assert response.status_code == 404
    
    @patch('app.model_registry')
    def test_explain_global_success(self, mock_registry):
        """测试全局解释成功"""
        # Mock模型注册表
        mock_registry.__contains__ = lambda self, key: key == "test_hash"
        mock_registry.__getitem__ = lambda self, key: {
            'algorithm': 'secureboost',
            'feature_importance': {
                'age': 0.3,
                'income': 0.5,
                'purchase_amount': 0.2
            }
        }
        
        request_data = {
            "model_hash": "test_hash",
            "top_n": 3,
            "dp_noise": False
        }
        
        response = client.post("/explain/global", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_hash"] == "test_hash"
        assert "global_importance" in data
        assert len(data["global_importance"]) <= 3
        assert data["dp_noise_applied"] == False
    
    @patch('app.model_registry')
    def test_explain_local_success(self, mock_registry):
        """测试本地解释成功"""
        # Mock模型注册表
        mock_registry.__contains__ = lambda self, key: key == "test_hash"
        mock_registry.__getitem__ = lambda self, key: {
            'algorithm': 'hetero_lr'
        }
        
        request_data = {
            "model_hash": "test_hash",
            "sample_id": "sample123",
            "party": "A"
        }
        
        response = client.post("/explain/local", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_hash"] == "test_hash"
        assert data["sample_id"] == "sample123"
        assert data["party"] == "A"
        assert "shap_values" in data
        assert "base_value" in data
        assert "prediction" in data
    
    def test_explain_local_invalid_party(self):
        """测试本地解释无效方"""
        request_data = {
            "model_hash": "test_hash",
            "sample_id": "sample123",
            "party": "C"  # 只支持A和B
        }
        
        response = client.post("/explain/local", json=request_data)
        # 应该仍然成功，但返回空的SHAP值或默认值
        assert response.status_code in [200, 404]

if __name__ == "__main__":
    pytest.main(["-v", __file__])