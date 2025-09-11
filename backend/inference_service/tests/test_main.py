import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime
import numpy as np
import tempfile
import os
import pickle

from main import app
from models import InferenceJob, ModelInfo, PredictionRequest, PredictionResponse
from model_registry import ModelRegistry
from audit import AuditLogger

client = TestClient(app)

class TestInferenceService:
    """推理服务测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "model_registry_status" in data
        assert "gpu_available" in data
    
    def test_metrics_endpoint(self):
        """测试指标接口"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "active_models" in data
        assert "avg_inference_time" in data
        assert "error_rate" in data
        assert "throughput" in data
        assert "gpu_utilization" in data
    
    def test_register_model(self):
        """测试注册模型"""
        model_data = {
            "name": "风险评估模型",
            "version": "1.0.0",
            "description": "用于信用风险评估的联邦学习模型",
            "algorithm": "logistic_regression",
            "input_schema": {
                "features": ["age", "income", "credit_score"],
                "feature_types": {
                    "age": "numeric",
                    "income": "numeric", 
                    "credit_score": "numeric"
                }
            },
            "output_schema": {
                "prediction": "numeric",
                "probability": "numeric",
                "risk_level": "categorical"
            },
            "model_path": "/models/risk_model_v1.pkl",
            "metadata": {
                "training_data_size": 10000,
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88
            }
        }
        
        response = client.post("/models/register", json=model_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == model_data["name"]
        assert data["version"] == model_data["version"]
        assert data["status"] == "registered"
        assert "id" in data
        assert "registered_at" in data
    
    def test_register_model_validation(self):
        """测试模型注册参数验证"""
        # 测试缺少必需字段
        invalid_data = {
            "name": "测试模型"
            # 缺少其他必需字段
        }
        response = client.post("/models/register", json=invalid_data)
        assert response.status_code == 422
        
        # 测试无效的算法
        invalid_data = {
            "name": "测试模型",
            "version": "1.0.0",
            "algorithm": "invalid_algorithm",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/path/to/model"
        }
        response = client.post("/models/register", json=invalid_data)
        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]
    
    def test_get_models(self):
        """测试获取模型列表"""
        # 注册几个模型
        for i in range(3):
            model_data = {
                "name": f"模型 {i}",
                "version": f"1.{i}.0",
                "description": f"描述 {i}",
                "algorithm": "logistic_regression",
                "input_schema": {"features": ["f1"]},
                "output_schema": {"prediction": "numeric"},
                "model_path": f"/models/model_{i}.pkl"
            }
            client.post("/models/register", json=model_data)
        
        # 获取模型列表
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert all("id" in item for item in data)
        assert all("name" in item for item in data)
        assert all("status" in item for item in data)
    
    def test_get_model_by_id(self):
        """测试根据ID获取模型"""
        # 注册模型
        model_data = {
            "name": "特定模型",
            "version": "2.0.0",
            "description": "用于ID查询测试",
            "algorithm": "neural_network",
            "input_schema": {"features": ["f1", "f2"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/specific_model.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        # 获取特定模型
        response = client.get(f"/models/{model_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model_id
        assert data["name"] == model_data["name"]
        assert data["version"] == model_data["version"]
    
    def test_get_nonexistent_model(self):
        """测试获取不存在的模型"""
        response = client.get("/models/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('model_registry.ModelRegistry.load_model')
    def test_activate_model(self, mock_load_model):
        """测试激活模型"""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # 注册模型
        model_data = {
            "name": "激活测试模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/activate_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        # 激活模型
        response = client.post(f"/models/{model_id}/activate")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "activated_at" in data
        
        # 验证加载方法被调用
        mock_load_model.assert_called_once()
    
    def test_activate_nonexistent_model(self):
        """测试激活不存在的模型"""
        response = client.post("/models/999999/activate")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_deactivate_model(self):
        """测试停用模型"""
        # 注册并激活模型
        model_data = {
            "name": "停用测试模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/deactivate_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        with patch('model_registry.ModelRegistry.load_model'):
            client.post(f"/models/{model_id}/activate")
        
        # 停用模型
        response = client.post(f"/models/{model_id}/deactivate")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "inactive"
        assert "deactivated_at" in data
    
    def test_delete_model(self):
        """测试删除模型"""
        # 注册模型
        model_data = {
            "name": "待删除模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/delete_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        # 删除模型
        response = client.delete(f"/models/{model_id}")
        assert response.status_code == 200
        
        # 验证模型已删除
        get_response = client.get(f"/models/{model_id}")
        assert get_response.status_code == 404
    
    def test_delete_active_model(self):
        """测试删除活跃模型"""
        # 注册并激活模型
        model_data = {
            "name": "活跃待删除模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/active_delete_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        with patch('model_registry.ModelRegistry.load_model'):
            client.post(f"/models/{model_id}/activate")
        
        # 尝试删除活跃模型
        response = client.delete(f"/models/{model_id}")
        assert response.status_code == 400
        assert "active" in response.json()["detail"]
    
    @patch('model_registry.ModelRegistry.predict')
    def test_make_prediction(self, mock_predict):
        """测试进行预测"""
        # 模拟预测结果
        mock_predict.return_value = {
            "prediction": 0.75,
            "probability": 0.85,
            "risk_level": "medium"
        }
        
        # 注册并激活模型
        model_data = {
            "name": "预测测试模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {
                "features": ["age", "income", "credit_score"],
                "feature_types": {
                    "age": "numeric",
                    "income": "numeric",
                    "credit_score": "numeric"
                }
            },
            "output_schema": {
                "prediction": "numeric",
                "probability": "numeric",
                "risk_level": "categorical"
            },
            "model_path": "/models/prediction_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        with patch('model_registry.ModelRegistry.load_model'):
            client.post(f"/models/{model_id}/activate")
        
        # 进行预测
        prediction_data = {
            "model_id": model_id,
            "features": {
                "age": 35,
                "income": 50000,
                "credit_score": 720
            },
            "request_id": "test_request_001"
        }
        
        response = client.post("/predict", json=prediction_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.75
        assert data["probability"] == 0.85
        assert data["risk_level"] == "medium"
        assert data["request_id"] == "test_request_001"
        assert "inference_time" in data
        
        # 验证预测方法被调用
        mock_predict.assert_called_once()
    
    def test_prediction_with_inactive_model(self):
        """测试使用未激活模型进行预测"""
        # 注册模型但不激活
        model_data = {
            "name": "未激活模型",
            "version": "1.0.0",
            "algorithm": "logistic_regression",
            "input_schema": {"features": ["f1"]},
            "output_schema": {"prediction": "numeric"},
            "model_path": "/models/inactive_test.pkl"
        }
        register_response = client.post("/models/register", json=model_data)
        model_id = register_response.json()["id"]
        
        # 尝试预测
        prediction_data = {
            "model_id": model_id,
            "features": {"f1": 1.0},
            "request_id": "test_request_002"
        }
        
        response = client.post("/predict", json=prediction_data)
        assert response.status_code == 400
        assert "not active" in response.json()["detail"]
    
    def test_prediction_validation(self):
        """测试预测请求参数验证"""
        # 测试缺少必需字段
        invalid_data = {
            "features": {"f1": 1.0}
            # 缺少 model_id
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
        
        # 测试无效的模型ID
        invalid_data = {
            "model_id": 999999,
            "features": {"f1": 1.0},
            "request_id": "test_request_003"
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_batch_prediction(self):
        """测试批量预测"""
        with patch('model_registry.ModelRegistry.predict') as mock_predict:
            # 模拟批量预测结果
            mock_predict.side_effect = [
                {"prediction": 0.8, "probability": 0.9, "risk_level": "high"},
                {"prediction": 0.3, "probability": 0.4, "risk_level": "low"},
                {"prediction": 0.6, "probability": 0.7, "risk_level": "medium"}
            ]
            
            # 注册并激活模型
            model_data = {
                "name": "批量预测模型",
                "version": "1.0.0",
                "algorithm": "logistic_regression",
                "input_schema": {"features": ["f1", "f2"]},
                "output_schema": {"prediction": "numeric"},
                "model_path": "/models/batch_test.pkl"
            }
            register_response = client.post("/models/register", json=model_data)
            model_id = register_response.json()["id"]
            
            with patch('model_registry.ModelRegistry.load_model'):
                client.post(f"/models/{model_id}/activate")
            
            # 批量预测
            batch_data = {
                "model_id": model_id,
                "batch_features": [
                    {"f1": 1.0, "f2": 2.0},
                    {"f1": 3.0, "f2": 4.0},
                    {"f1": 5.0, "f2": 6.0}
                ],
                "request_id": "batch_request_001"
            }
            
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 3
            assert data["request_id"] == "batch_request_001"
            assert "total_inference_time" in data
            
            # 验证预测方法被调用3次
            assert mock_predict.call_count == 3
    
    def test_get_prediction_history(self):
        """测试获取预测历史"""
        response = client.get("/predictions/history")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "total_count" in data
    
    def test_get_prediction_by_request_id(self):
        """测试根据请求ID获取预测结果"""
        # 先进行一次预测
        with patch('model_registry.ModelRegistry.predict') as mock_predict:
            mock_predict.return_value = {"prediction": 0.5}
            
            # 注册并激活模型
            model_data = {
                "name": "历史查询模型",
                "version": "1.0.0",
                "algorithm": "logistic_regression",
                "input_schema": {"features": ["f1"]},
                "output_schema": {"prediction": "numeric"},
                "model_path": "/models/history_test.pkl"
            }
            register_response = client.post("/models/register", json=model_data)
            model_id = register_response.json()["id"]
            
            with patch('model_registry.ModelRegistry.load_model'):
                client.post(f"/models/{model_id}/activate")
            
            # 进行预测
            prediction_data = {
                "model_id": model_id,
                "features": {"f1": 1.0},
                "request_id": "history_request_001"
            }
            client.post("/predict", json=prediction_data)
        
        # 查询预测历史
        response = client.get("/predictions/history/history_request_001")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "history_request_001"
        assert "prediction" in data
        assert "timestamp" in data


class TestModelRegistry:
    """模型注册表测试类"""
    
    def test_model_registry_initialization(self):
        """测试模型注册表初始化"""
        registry = ModelRegistry()
        assert registry.models == {}
        assert registry.active_models == {}
    
    def test_register_model(self):
        """测试注册模型"""
        registry = ModelRegistry()
        
        model_info = ModelInfo(
            name="测试模型",
            version="1.0.0",
            algorithm="logistic_regression",
            input_schema={"features": ["f1"]},
            output_schema={"prediction": "numeric"},
            model_path="/models/test.pkl"
        )
        
        model_id = registry.register_model(model_info)
        assert model_id in registry.models
        assert registry.models[model_id].name == "测试模型"
        assert registry.models[model_id].status == "registered"
    
    def test_load_model(self):
        """测试加载模型"""
        registry = ModelRegistry()
        
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # 创建一个简单的模型对象
            model = {"weights": np.array([1, 2, 3]), "bias": 0.5}
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            model_info = ModelInfo(
                name="加载测试模型",
                version="1.0.0",
                algorithm="logistic_regression",
                input_schema={"features": ["f1"]},
                output_schema={"prediction": "numeric"},
                model_path=temp_path
            )
            
            model_id = registry.register_model(model_info)
            loaded_model = registry.load_model(model_id)
            
            assert loaded_model is not None
            assert "weights" in loaded_model
            assert "bias" in loaded_model
        finally:
            # 清理临时文件
            os.unlink(temp_path)
    
    def test_activate_model(self):
        """测试激活模型"""
        registry = ModelRegistry()
        
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model = {"weights": np.array([1, 2, 3])}
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            model_info = ModelInfo(
                name="激活测试模型",
                version="1.0.0",
                algorithm="logistic_regression",
                input_schema={"features": ["f1"]},
                output_schema={"prediction": "numeric"},
                model_path=temp_path
            )
            
            model_id = registry.register_model(model_info)
            registry.activate_model(model_id)
            
            assert model_id in registry.active_models
            assert registry.models[model_id].status == "active"
        finally:
            os.unlink(temp_path)
    
    def test_deactivate_model(self):
        """测试停用模型"""
        registry = ModelRegistry()
        
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model = {"weights": np.array([1, 2, 3])}
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            model_info = ModelInfo(
                name="停用测试模型",
                version="1.0.0",
                algorithm="logistic_regression",
                input_schema={"features": ["f1"]},
                output_schema={"prediction": "numeric"},
                model_path=temp_path
            )
            
            model_id = registry.register_model(model_info)
            registry.activate_model(model_id)
            registry.deactivate_model(model_id)
            
            assert model_id not in registry.active_models
            assert registry.models[model_id].status == "inactive"
        finally:
            os.unlink(temp_path)
    
    def test_predict_with_model(self):
        """测试使用模型进行预测"""
        registry = ModelRegistry()
        
        # 创建模拟模型
        class MockModel:
            def predict(self, features):
                return {"prediction": sum(features.values()) * 0.1}
        
        mock_model = MockModel()
        
        # 直接设置活跃模型（跳过文件加载）
        model_id = "test_model_001"
        registry.active_models[model_id] = mock_model
        
        # 进行预测
        features = {"f1": 10, "f2": 20}
        result = registry.predict(model_id, features)
        
        assert result["prediction"] == 3.0  # (10 + 20) * 0.1
    
    def test_predict_with_inactive_model(self):
        """测试使用未激活模型进行预测"""
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="not active"):
            registry.predict("nonexistent_model", {"f1": 1.0})


class TestAuditLogger:
    """审计日志测试类"""
    
    def test_audit_logger_initialization(self):
        """测试审计日志初始化"""
        logger = AuditLogger()
        assert logger.logs == []
    
    def test_log_prediction(self):
        """测试记录预测日志"""
        logger = AuditLogger()
        
        prediction_data = {
            "model_id": "model_001",
            "request_id": "req_001",
            "features": {"f1": 1.0, "f2": 2.0},
            "prediction": 0.75,
            "inference_time": 0.05
        }
        
        logger.log_prediction(prediction_data)
        
        assert len(logger.logs) == 1
        log_entry = logger.logs[0]
        assert log_entry["event_type"] == "prediction"
        assert log_entry["model_id"] == "model_001"
        assert log_entry["request_id"] == "req_001"
        assert "timestamp" in log_entry
    
    def test_log_model_operation(self):
        """测试记录模型操作日志"""
        logger = AuditLogger()
        
        operation_data = {
            "model_id": "model_002",
            "operation": "activate",
            "user_id": "user_001",
            "details": "模型激活成功"
        }
        
        logger.log_model_operation(operation_data)
        
        assert len(logger.logs) == 1
        log_entry = logger.logs[0]
        assert log_entry["event_type"] == "model_operation"
        assert log_entry["operation"] == "activate"
        assert log_entry["user_id"] == "user_001"
    
    def test_get_logs_by_model(self):
        """测试根据模型ID获取日志"""
        logger = AuditLogger()
        
        # 添加多个日志
        logger.log_prediction({
            "model_id": "model_001",
            "request_id": "req_001",
            "prediction": 0.5
        })
        
        logger.log_prediction({
            "model_id": "model_002",
            "request_id": "req_002",
            "prediction": 0.8
        })
        
        logger.log_model_operation({
            "model_id": "model_001",
            "operation": "activate"
        })
        
        # 获取特定模型的日志
        model_logs = logger.get_logs_by_model("model_001")
        assert len(model_logs) == 2
        assert all(log["model_id"] == "model_001" for log in model_logs)
    
    def test_get_logs_by_time_range(self):
        """测试根据时间范围获取日志"""
        logger = AuditLogger()
        
        # 添加日志
        logger.log_prediction({
            "model_id": "model_001",
            "request_id": "req_001",
            "prediction": 0.5
        })
        
        # 获取最近1小时的日志
        from datetime import datetime, timedelta
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        recent_logs = logger.get_logs_by_time_range(start_time, end_time)
        assert len(recent_logs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])