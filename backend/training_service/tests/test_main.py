import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime
import numpy as np
import tempfile
import os

from main import app
from models import TrainingJob, TrainingConfig, ModelMetrics, TrainingStatus
from privacy import DifferentialPrivacy
from federated import FederatedTraining

client = TestClient(app)

class TestTrainingService:
    """训练服务测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "secretflow_status" in data
        assert "gpu_available" in data
    
    def test_metrics_endpoint(self):
        """测试指标接口"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
        assert "running_jobs" in data
        assert "completed_jobs" in data
        assert "failed_jobs" in data
        assert "avg_training_time" in data
        assert "gpu_utilization" in data
    
    def test_create_training_job(self):
        """测试创建训练任务"""
        job_data = {
            "name": "测试训练任务",
            "description": "用于测试的联邦学习训练任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b", "party_c"],
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.01,
                "privacy_budget": 1.0,
                "noise_multiplier": 1.1,
                "max_grad_norm": 1.0
            },
            "data_schema": {
                "features": ["age", "income", "credit_score"],
                "target": "default_risk",
                "feature_types": {"age": "numeric", "income": "numeric", "credit_score": "numeric"}
            }
        }
        
        response = client.post("/training/jobs", json=job_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == job_data["name"]
        assert data["algorithm"] == job_data["algorithm"]
        assert data["status"] == "created"
        assert "id" in data
        assert "created_at" in data
        assert len(data["participants"]) == 3
    
    def test_create_training_job_validation(self):
        """测试训练任务参数验证"""
        # 测试缺少必需字段
        invalid_data = {
            "name": "测试任务"
            # 缺少其他必需字段
        }
        response = client.post("/training/jobs", json=invalid_data)
        assert response.status_code == 422
        
        # 测试无效的算法
        invalid_data = {
            "name": "测试任务",
            "algorithm": "invalid_algorithm",
            "participants": ["party_a"],
            "config": {"epochs": 10},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        response = client.post("/training/jobs", json=invalid_data)
        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]
        
        # 测试参与方数量不足
        invalid_data = {
            "name": "测试任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a"],  # 只有一个参与方
            "config": {"epochs": 10},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        response = client.post("/training/jobs", json=invalid_data)
        assert response.status_code == 400
        assert "At least 2 participants" in response.json()["detail"]
    
    def test_get_training_jobs(self):
        """测试获取训练任务列表"""
        # 创建几个训练任务
        for i in range(3):
            job_data = {
                "name": f"训练任务 {i}",
                "description": f"描述 {i}",
                "algorithm": "logistic_regression",
                "participants": ["party_a", "party_b"],
                "config": {"epochs": 10},
                "data_schema": {"features": ["f1"], "target": "t1"}
            }
            client.post("/training/jobs", json=job_data)
        
        # 获取任务列表
        response = client.get("/training/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert all("id" in item for item in data)
        assert all("status" in item for item in data)
        assert all("name" in item for item in data)
    
    def test_get_training_job_by_id(self):
        """测试根据ID获取训练任务"""
        # 创建训练任务
        job_data = {
            "name": "特定训练任务",
            "description": "用于ID查询测试",
            "algorithm": "neural_network",
            "participants": ["party_a", "party_b", "party_c"],
            "config": {"epochs": 20, "batch_size": 64},
            "data_schema": {"features": ["f1", "f2"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 获取特定任务
        response = client.get(f"/training/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["name"] == job_data["name"]
        assert data["algorithm"] == job_data["algorithm"]
    
    def test_get_nonexistent_training_job(self):
        """测试获取不存在的训练任务"""
        response = client.get("/training/jobs/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('federated.FederatedTraining.start_training')
    def test_start_training_job(self, mock_start_training):
        """测试启动训练任务"""
        mock_start_training.return_value = AsyncMock()
        
        # 创建训练任务
        job_data = {
            "name": "启动测试任务",
            "description": "测试启动功能",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 5},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 启动训练
        response = client.post(f"/training/jobs/{job_id}/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "started_at" in data
        
        # 验证启动方法被调用
        mock_start_training.assert_called_once()
    
    def test_start_already_running_job(self):
        """测试启动已在运行的任务"""
        # 创建并启动任务
        job_data = {
            "name": "运行中任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 5},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        with patch('federated.FederatedTraining.start_training'):
            client.post(f"/training/jobs/{job_id}/start")
        
        # 再次尝试启动
        response = client.post(f"/training/jobs/{job_id}/start")
        assert response.status_code == 400
        assert "already running" in response.json()["detail"]
    
    @patch('federated.FederatedTraining.stop_training')
    def test_stop_training_job(self, mock_stop_training):
        """测试停止训练任务"""
        mock_stop_training.return_value = True
        
        # 创建并启动任务
        job_data = {
            "name": "停止测试任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 10},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        with patch('federated.FederatedTraining.start_training'):
            client.post(f"/training/jobs/{job_id}/start")
        
        # 停止训练
        response = client.post(f"/training/jobs/{job_id}/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert "stopped_at" in data
        
        # 验证停止方法被调用
        mock_stop_training.assert_called_once()
    
    def test_stop_not_running_job(self):
        """测试停止未运行的任务"""
        # 创建任务但不启动
        job_data = {
            "name": "未运行任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 5},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 尝试停止
        response = client.post(f"/training/jobs/{job_id}/stop")
        assert response.status_code == 400
        assert "not running" in response.json()["detail"]
    
    def test_delete_training_job(self):
        """测试删除训练任务"""
        # 创建任务
        job_data = {
            "name": "待删除任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 5},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 删除任务
        response = client.delete(f"/training/jobs/{job_id}")
        assert response.status_code == 200
        
        # 验证任务已删除
        get_response = client.get(f"/training/jobs/{job_id}")
        assert get_response.status_code == 404
    
    def test_delete_running_job(self):
        """测试删除运行中的任务"""
        # 创建并启动任务
        job_data = {
            "name": "运行中待删除任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 10},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        with patch('federated.FederatedTraining.start_training'):
            client.post(f"/training/jobs/{job_id}/start")
        
        # 尝试删除运行中的任务
        response = client.delete(f"/training/jobs/{job_id}")
        assert response.status_code == 400
        assert "running" in response.json()["detail"]
    
    def test_get_training_logs(self):
        """测试获取训练日志"""
        # 创建任务
        job_data = {
            "name": "日志测试任务",
            "algorithm": "logistic_regression",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 5},
            "data_schema": {"features": ["f1"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 获取日志
        response = client.get(f"/training/jobs/{job_id}/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)
    
    def test_get_training_metrics(self):
        """测试获取训练指标"""
        # 创建任务
        job_data = {
            "name": "指标测试任务",
            "algorithm": "neural_network",
            "participants": ["party_a", "party_b"],
            "config": {"epochs": 10},
            "data_schema": {"features": ["f1", "f2"], "target": "t1"}
        }
        create_response = client.post("/training/jobs", json=job_data)
        job_id = create_response.json()["id"]
        
        # 获取指标
        response = client.get(f"/training/jobs/{job_id}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
        
        # 检查基本指标字段
        metrics = data["metrics"]
        expected_fields = ["accuracy", "loss", "precision", "recall", "f1_score"]
        for field in expected_fields:
            assert field in metrics


class TestDifferentialPrivacy:
    """差分隐私测试类"""
    
    def test_privacy_budget_calculation(self):
        """测试隐私预算计算"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 测试基本预算计算
        budget = dp.calculate_privacy_budget(num_queries=10, sensitivity=1.0)
        assert budget > 0
        assert budget <= 1.0
    
    def test_noise_generation(self):
        """测试噪声生成"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 生成拉普拉斯噪声
        noise = dp.generate_laplace_noise(sensitivity=1.0, size=100)
        assert len(noise) == 100
        assert isinstance(noise, np.ndarray)
        
        # 生成高斯噪声
        noise = dp.generate_gaussian_noise(sensitivity=1.0, size=50)
        assert len(noise) == 50
        assert isinstance(noise, np.ndarray)
    
    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 创建测试梯度
        gradients = [np.random.randn(10, 5) for _ in range(3)]
        max_norm = 1.0
        
        clipped_gradients = dp.clip_gradients(gradients, max_norm)
        
        # 验证梯度被正确裁剪
        for grad in clipped_gradients:
            norm = np.linalg.norm(grad)
            assert norm <= max_norm + 1e-6  # 允许小的数值误差
    
    def test_privacy_accounting(self):
        """测试隐私会计"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 模拟多次查询
        for i in range(5):
            dp.add_privacy_cost(epsilon=0.1, delta=1e-6)
        
        total_epsilon, total_delta = dp.get_total_privacy_cost()
        assert total_epsilon <= 1.0  # 不应超过预算
        assert total_delta <= 1e-5
    
    def test_privacy_budget_exceeded(self):
        """测试隐私预算超限"""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # 尝试添加超出预算的成本
        with pytest.raises(ValueError, match="Privacy budget exceeded"):
            dp.add_privacy_cost(epsilon=1.5, delta=0)


class TestFederatedTraining:
    """联邦训练测试类"""
    
    @patch('secretflow.init')
    def test_federated_training_initialization(self, mock_sf_init):
        """测试联邦训练初始化"""
        participants = ["party_a", "party_b", "party_c"]
        config = {
            "algorithm": "logistic_regression",
            "epochs": 10,
            "batch_size": 32
        }
        
        ft = FederatedTraining(participants, config)
        assert ft.participants == participants
        assert ft.config == config
        assert ft.status == "initialized"
    
    @patch('secretflow.ml.nn.FLModel')
    async def test_start_training(self, mock_fl_model):
        """测试启动训练"""
        mock_model = MagicMock()
        mock_fl_model.return_value = mock_model
        
        participants = ["party_a", "party_b"]
        config = {
            "algorithm": "neural_network",
            "epochs": 5,
            "learning_rate": 0.01
        }
        
        ft = FederatedTraining(participants, config)
        
        # 模拟数据
        mock_data = {
            "party_a": np.random.randn(100, 10),
            "party_b": np.random.randn(100, 10)
        }
        
        await ft.start_training(mock_data)
        
        assert ft.status == "running"
        mock_model.fit.assert_called_once()
    
    def test_participant_validation(self):
        """测试参与方验证"""
        # 测试参与方数量不足
        with pytest.raises(ValueError, match="At least 2 participants"):
            FederatedTraining(["party_a"], {})
        
        # 测试重复参与方
        with pytest.raises(ValueError, match="Duplicate participants"):
            FederatedTraining(["party_a", "party_a"], {})
    
    def test_algorithm_validation(self):
        """测试算法验证"""
        participants = ["party_a", "party_b"]
        
        # 测试不支持的算法
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            config = {"algorithm": "unsupported_algo"}
            FederatedTraining(participants, config)
    
    @patch('secretflow.ml.nn.FLModel')
    async def test_training_with_privacy(self, mock_fl_model):
        """测试带隐私保护的训练"""
        mock_model = MagicMock()
        mock_fl_model.return_value = mock_model
        
        participants = ["party_a", "party_b"]
        config = {
            "algorithm": "logistic_regression",
            "epochs": 5,
            "privacy_budget": 1.0,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0
        }
        
        ft = FederatedTraining(participants, config)
        
        # 验证隐私配置
        assert ft.privacy_config["epsilon"] == 1.0
        assert ft.privacy_config["noise_multiplier"] == 1.1
        assert ft.privacy_config["max_grad_norm"] == 1.0
    
    def test_model_aggregation(self):
        """测试模型聚合"""
        participants = ["party_a", "party_b", "party_c"]
        config = {"algorithm": "logistic_regression"}
        
        ft = FederatedTraining(participants, config)
        
        # 模拟本地模型参数
        local_models = {
            "party_a": {"weights": np.array([1.0, 2.0, 3.0]), "bias": 0.5},
            "party_b": {"weights": np.array([2.0, 3.0, 4.0]), "bias": 1.0},
            "party_c": {"weights": np.array([3.0, 4.0, 5.0]), "bias": 1.5}
        }
        
        # 执行聚合
        aggregated_model = ft.aggregate_models(local_models)
        
        # 验证聚合结果（简单平均）
        expected_weights = np.array([2.0, 3.0, 4.0])
        expected_bias = 1.0
        
        np.testing.assert_array_almost_equal(
            aggregated_model["weights"], expected_weights
        )
        assert abs(aggregated_model["bias"] - expected_bias) < 1e-6
    
    def test_secure_aggregation(self):
        """测试安全聚合"""
        participants = ["party_a", "party_b"]
        config = {
            "algorithm": "neural_network",
            "secure_aggregation": True
        }
        
        ft = FederatedTraining(participants, config)
        
        # 模拟加密的模型参数
        encrypted_models = {
            "party_a": {"encrypted_weights": "encrypted_data_a"},
            "party_b": {"encrypted_weights": "encrypted_data_b"}
        }
        
        # 验证安全聚合配置
        assert ft.config["secure_aggregation"] is True
        
        # 这里只是验证配置，实际的安全聚合需要更复杂的密码学实现
        result = ft.secure_aggregate(encrypted_models)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])