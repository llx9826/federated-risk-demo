import pytest
import json
import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# 导入应用
from app import app
from model_registry import ModelRegistry
from feature_view import FeatureView
from consent_client import ConsentClient, ConsentCache
from audit_log import AuditLogger

client = TestClient(app)

class TestServingService:
    """推理服务测试"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint(self):
        """测试指标接口"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "total_scores" in data
        assert "avg_processing_time" in data
        assert "model_registry_size" in data
    
    @patch('consent_client.ConsentClient.verify_consent')
    @patch('feature_view.FeatureView.get_features')
    def test_score_endpoint_success(self, mock_get_features, mock_verify_consent):
        """测试成功的评分请求"""
        # Mock同意验证
        mock_verify_consent.return_value = {
            "valid": True,
            "consent_fingerprint": "test_fingerprint",
            "policy_version": "v1.0"
        }
        
        # Mock特征获取
        mock_get_features.return_value = {
            "age": 30,
            "income": 50000,
            "credit_score": 750,
            "transaction_count": 100
        }
        
        # 测试评分请求
        score_request = {
            "subject": "user123",
            "consent_token": "valid_token",
            "model_name": "risk_model_v1",
            "threshold": 0.5,
            "features": ["age", "income", "credit_score"]
        }
        
        response = client.post("/score", json=score_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert "score" in data
        assert "decision" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
        assert data["decision"] in ["approve", "reject"]
        assert 0 <= data["score"] <= 1
    
    @patch('consent_client.ConsentClient.verify_consent')
    def test_score_endpoint_invalid_consent(self, mock_verify_consent):
        """测试无效同意票据"""
        # Mock同意验证失败
        mock_verify_consent.return_value = {
            "valid": False,
            "error": "Invalid consent token"
        }
        
        score_request = {
            "subject": "user123",
            "consent_token": "invalid_token",
            "model_name": "risk_model_v1",
            "threshold": 0.5,
            "features": ["age", "income"]
        }
        
        response = client.post("/score", json=score_request)
        assert response.status_code == 403
        
        data = response.json()
        assert "error" in data
        assert "consent" in data["error"].lower()
    
    def test_score_endpoint_missing_fields(self):
        """测试缺少必需字段"""
        incomplete_request = {
            "subject": "user123",
            # 缺少consent_token和其他字段
        }
        
        response = client.post("/score", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    def test_promote_model_success(self):
        """测试模型提升成功"""
        promote_request = {
            "model_name": "risk_model_v2",
            "version": "2.0",
            "promoted_by": "admin",
            "reason": "Better performance metrics"
        }
        
        response = client.post("/models/promote", json=promote_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "message" in data
    
    def test_promote_model_not_found(self):
        """测试提升不存在的模型"""
        promote_request = {
            "model_name": "nonexistent_model",
            "version": "1.0",
            "promoted_by": "admin",
            "reason": "Test"
        }
        
        response = client.post("/models/promote", json=promote_request)
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
    
    def test_list_models(self):
        """测试列出模型"""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_get_production_model(self):
        """测试获取生产模型"""
        response = client.get("/models/production")
        assert response.status_code == 200
        
        data = response.json()
        if data.get("model"):  # 如果有生产模型
            assert "name" in data["model"]
            assert "version" in data["model"]
            assert "status" in data["model"]
    
    def test_register_model_success(self):
        """测试注册模型成功"""
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "algorithm": "SecureBoost",
            "features": ["age", "income", "credit_score"],
            "metrics": {
                "auc": 0.85,
                "ks": 0.45
            },
            "created_by": "test_user",
            "description": "Test model for unit testing"
        }
        
        response = client.post("/models/register", json=model_info)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "model_id" in data
    
    def test_get_features_success(self):
        """测试获取特征成功"""
        response = client.get("/features/user123")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], dict)
    
    def test_update_features_success(self):
        """测试更新特征成功"""
        feature_update = {
            "subject": "user123",
            "features": {
                "age": 35,
                "income": 60000,
                "credit_score": 780
            },
            "source": "manual_update"
        }
        
        response = client.post("/features/update", json=feature_update)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "updated_features" in data

class TestModelRegistry:
    """模型注册表测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def test_register_model(self):
        """测试注册模型"""
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "algorithm": "SecureBoost",
            "features": ["age", "income"],
            "metrics": {"auc": 0.85}
        }
        
        model_id = self.registry.register_model(model_info)
        assert model_id is not None
        assert len(model_id) > 0
        
        # 验证模型已注册
        registered_model = self.registry.get_model_info(model_id)
        assert registered_model is not None
        assert registered_model["name"] == "test_model"
    
    def test_promote_model(self):
        """测试提升模型"""
        # 先注册模型
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "algorithm": "SecureBoost",
            "features": ["age", "income"],
            "metrics": {"auc": 0.85}
        }
        
        model_id = self.registry.register_model(model_info)
        
        # 提升为生产模型
        success = self.registry.promote_to_production(
            model_id, "admin", "Better performance"
        )
        assert success is True
        
        # 验证生产模型
        production_model = self.registry.get_production_model()
        assert production_model is not None
        assert production_model["id"] == model_id
    
    def test_list_models(self):
        """测试列出模型"""
        # 注册几个模型
        for i in range(3):
            model_info = {
                "name": f"test_model_{i}",
                "version": "1.0",
                "algorithm": "SecureBoost",
                "features": ["age", "income"],
                "metrics": {"auc": 0.8 + i * 0.01}
            }
            self.registry.register_model(model_info)
        
        models = self.registry.list_models()
        assert len(models) == 3
        assert all("name" in model for model in models)

class TestFeatureView:
    """特征视图测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.feature_view = FeatureView(self.temp_dir)
    
    def test_get_features_with_mock_data(self):
        """测试获取特征（使用模拟数据）"""
        features = self.feature_view.get_features("user123")
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # 验证基本特征存在
        expected_features = ["age", "income", "credit_score", "transaction_count"]
        for feature in expected_features:
            assert feature in features
    
    def test_update_features(self):
        """测试更新特征"""
        new_features = {
            "age": 35,
            "income": 60000,
            "credit_score": 780
        }
        
        success = self.feature_view.update_features("user123", new_features)
        assert success is True
        
        # 验证更新后的特征
        updated_features = self.feature_view.get_features("user123")
        for key, value in new_features.items():
            assert updated_features.get(key) == value
    
    def test_feature_statistics(self):
        """测试特征统计"""
        stats = self.feature_view.get_feature_statistics()
        assert isinstance(stats, dict)
        assert "total_subjects" in stats
        assert "feature_counts" in stats
    
    def test_feature_schemas(self):
        """测试特征模式"""
        schemas = self.feature_view.get_feature_schemas()
        assert isinstance(schemas, dict)
        assert len(schemas) > 0
        
        # 验证模式结构
        for feature_name, schema in schemas.items():
            assert "type" in schema
            assert "description" in schema

class TestConsentClient:
    """同意客户端测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.consent_client = ConsentClient("http://localhost:7002")
    
    @patch('requests.post')
    def test_verify_consent_success(self, mock_post):
        """测试同意验证成功"""
        # Mock成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "consent_fingerprint": "test_fingerprint",
            "policy_version": "v1.0"
        }
        mock_post.return_value = mock_response
        
        result = self.consent_client.verify_consent("valid_token")
        assert result["valid"] is True
        assert "consent_fingerprint" in result
    
    @patch('requests.post')
    def test_verify_consent_failure(self, mock_post):
        """测试同意验证失败"""
        # Mock失败响应
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "valid": False,
            "error": "Invalid token"
        }
        mock_post.return_value = mock_response
        
        result = self.consent_client.verify_consent("invalid_token")
        assert result["valid"] is False
        assert "error" in result
    
    @patch('requests.post')
    def test_record_audit_success(self, mock_post):
        """测试审计记录成功"""
        # Mock成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        audit_data = {
            "subject": "user123",
            "action": "score_request",
            "resource": "risk_model",
            "result": "success"
        }
        
        success = self.consent_client.record_audit(audit_data)
        assert success is True

class TestConsentCache:
    """同意缓存测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.cache = ConsentCache(max_size=100, ttl_seconds=300)
    
    def test_cache_operations(self):
        """测试缓存操作"""
        # 测试设置和获取
        consent_result = {
            "valid": True,
            "consent_fingerprint": "test_fingerprint"
        }
        
        self.cache.set("test_token", consent_result)
        cached_result = self.cache.get("test_token")
        
        assert cached_result is not None
        assert cached_result["valid"] is True
        assert cached_result["consent_fingerprint"] == "test_fingerprint"
    
    def test_cache_expiry(self):
        """测试缓存过期"""
        # 使用很短的TTL
        short_cache = ConsentCache(max_size=100, ttl_seconds=0.1)
        
        consent_result = {"valid": True}
        short_cache.set("test_token", consent_result)
        
        # 立即获取应该成功
        assert short_cache.get("test_token") is not None
        
        # 等待过期后应该返回None
        import time
        time.sleep(0.2)
        assert short_cache.get("test_token") is None
    
    def test_cache_statistics(self):
        """测试缓存统计"""
        # 添加一些缓存项
        for i in range(5):
            self.cache.set(f"token_{i}", {"valid": True})
        
        # 获取一些项（命中）
        for i in range(3):
            self.cache.get(f"token_{i}")
        
        # 尝试获取不存在的项（未命中）
        self.cache.get("nonexistent_token")
        
        stats = self.cache.get_statistics()
        assert stats["size"] == 5
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75

class TestAuditLogger:
    """审计日志测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(self.temp_dir)
    
    def test_record_audit(self):
        """测试记录审计"""
        audit_record = {
            "request_id": "test_request_123",
            "subject": "user123",
            "consent_fingerprint": "test_fingerprint",
            "model_hash": "model_hash_123",
            "decision": "approve",
            "score": 0.75,
            "features_used": ["age", "income", "credit_score"],
            "processing_time_ms": 150
        }
        
        success = self.audit_logger.record_audit(audit_record)
        assert success is True
    
    def test_query_audit_records(self):
        """测试查询审计记录"""
        # 先记录一些审计数据
        for i in range(3):
            audit_record = {
                "request_id": f"test_request_{i}",
                "subject": f"user{i}",
                "consent_fingerprint": f"fingerprint_{i}",
                "model_hash": "model_hash_123",
                "decision": "approve" if i % 2 == 0 else "reject",
                "score": 0.5 + i * 0.1,
                "features_used": ["age", "income"],
                "processing_time_ms": 100 + i * 10
            }
            self.audit_logger.record_audit(audit_record)
        
        # 查询所有记录
        records = self.audit_logger.query_audit_records(limit=10)
        assert len(records) == 3
        
        # 按subject查询
        user_records = self.audit_logger.query_audit_records(subject="user1")
        assert len(user_records) == 1
        assert user_records[0]["subject"] == "user1"
    
    def test_audit_statistics(self):
        """测试审计统计"""
        # 记录一些测试数据
        decisions = ["approve", "reject", "approve"]
        for i, decision in enumerate(decisions):
            audit_record = {
                "request_id": f"test_request_{i}",
                "subject": f"user{i}",
                "consent_fingerprint": f"fingerprint_{i}",
                "model_hash": "model_hash_123",
                "decision": decision,
                "score": 0.3 + i * 0.2,
                "features_used": ["age", "income"],
                "processing_time_ms": 100 + i * 50
            }
            self.audit_logger.record_audit(audit_record)
        
        stats = self.audit_logger.get_audit_statistics()
        assert stats["total_records"] == 3
        assert "decision_distribution" in stats
        assert stats["decision_distribution"]["approve"] == 2
        assert stats["decision_distribution"]["reject"] == 1
    
    def test_database_info(self):
        """测试数据库信息"""
        info = self.audit_logger.get_database_info()
        assert "database_path" in info
        assert "json_log_path" in info
        assert "total_records" in info
        assert info["database_exists"] is True

if __name__ == "__main__":
    pytest.main(["-v", __file__])