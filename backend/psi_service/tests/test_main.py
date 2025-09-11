import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from main import app
from models import PSIRequest, PSIResponse
from psi_protocol import PSIProtocol

client = TestClient(app)

class TestPSIService:
    """PSI服务测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
    
    def test_metrics_endpoint(self):
        """测试指标接口"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "active_sessions" in data
        assert "avg_response_time" in data
        assert "memory_usage" in data
        assert "cpu_usage" in data
    
    def test_psi_request_validation(self):
        """测试PSI请求参数验证"""
        # 测试空数据
        response = client.post("/psi/compute", json={
            "party_a_data": [],
            "party_b_data": ["item1", "item2"]
        })
        assert response.status_code == 400
        assert "Party A data cannot be empty" in response.json()["detail"]
        
        # 测试数据过大
        large_data = [f"item_{i}" for i in range(10001)]
        response = client.post("/psi/compute", json={
            "party_a_data": large_data,
            "party_b_data": ["item1"]
        })
        assert response.status_code == 400
        assert "exceeds maximum size" in response.json()["detail"]
        
        # 测试无效数据类型
        response = client.post("/psi/compute", json={
            "party_a_data": "invalid",
            "party_b_data": ["item1"]
        })
        assert response.status_code == 422
    
    @patch('psi_protocol.PSIProtocol.compute_intersection')
    def test_psi_compute_success(self, mock_compute):
        """测试PSI计算成功场景"""
        # 模拟PSI计算结果
        mock_compute.return_value = {
            "intersection": ["common1", "common2"],
            "intersection_size": 2,
            "computation_time": 1.5,
            "privacy_budget_used": 0.1
        }
        
        request_data = {
            "party_a_data": ["item1", "common1", "common2"],
            "party_b_data": ["item2", "common1", "common2"],
            "privacy_budget": 1.0,
            "hash_function": "sha256"
        }
        
        response = client.post("/psi/compute", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["intersection"] == ["common1", "common2"]
        assert data["intersection_size"] == 2
        assert data["computation_time"] == 1.5
        assert data["privacy_budget_used"] == 0.1
        assert "session_id" in data
        
        # 验证mock被正确调用
        mock_compute.assert_called_once()
    
    @patch('psi_protocol.PSIProtocol.compute_intersection')
    def test_psi_compute_error(self, mock_compute):
        """测试PSI计算错误场景"""
        # 模拟计算错误
        mock_compute.side_effect = Exception("Computation failed")
        
        request_data = {
            "party_a_data": ["item1", "item2"],
            "party_b_data": ["item3", "item4"]
        }
        
        response = client.post("/psi/compute", json=request_data)
        assert response.status_code == 500
        assert "PSI computation failed" in response.json()["detail"]
    
    def test_psi_status_endpoint(self):
        """测试PSI状态查询接口"""
        # 测试不存在的会话
        response = client.get("/psi/status/nonexistent")
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
    
    @patch('psi_protocol.PSIProtocol.compute_intersection')
    def test_psi_with_different_hash_functions(self, mock_compute):
        """测试不同哈希函数的PSI计算"""
        mock_compute.return_value = {
            "intersection": ["common1"],
            "intersection_size": 1,
            "computation_time": 1.0,
            "privacy_budget_used": 0.05
        }
        
        hash_functions = ["sha256", "sha512", "blake2b"]
        
        for hash_func in hash_functions:
            request_data = {
                "party_a_data": ["item1", "common1"],
                "party_b_data": ["item2", "common1"],
                "hash_function": hash_func
            }
            
            response = client.post("/psi/compute", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["intersection"] == ["common1"]
    
    def test_concurrent_psi_requests(self):
        """测试并发PSI请求"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                with patch('psi_protocol.PSIProtocol.compute_intersection') as mock_compute:
                    mock_compute.return_value = {
                        "intersection": ["common1"],
                        "intersection_size": 1,
                        "computation_time": 0.5,
                        "privacy_budget_used": 0.1
                    }
                    
                    request_data = {
                        "party_a_data": ["item1", "common1"],
                        "party_b_data": ["item2", "common1"]
                    }
                    
                    response = client.post("/psi/compute", json=request_data)
                    results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # 创建多个并发请求
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results)
    
    def test_privacy_budget_tracking(self):
        """测试隐私预算跟踪"""
        with patch('psi_protocol.PSIProtocol.compute_intersection') as mock_compute:
            mock_compute.return_value = {
                "intersection": ["common1"],
                "intersection_size": 1,
                "computation_time": 1.0,
                "privacy_budget_used": 0.5
            }
            
            # 第一次请求
            request_data = {
                "party_a_data": ["item1", "common1"],
                "party_b_data": ["item2", "common1"],
                "privacy_budget": 1.0
            }
            
            response = client.post("/psi/compute", json=request_data)
            assert response.status_code == 200
            
            # 检查隐私预算使用情况
            response = client.get("/metrics")
            data = response.json()
            assert "privacy_budget_used" in data


class TestPSIProtocol:
    """PSI协议测试类"""
    
    def test_hash_function_consistency(self):
        """测试哈希函数一致性"""
        protocol = PSIProtocol()
        
        data = "test_item"
        hash1 = protocol._hash_item(data, "sha256")
        hash2 = protocol._hash_item(data, "sha256")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-character hex string
    
    def test_different_hash_functions(self):
        """测试不同哈希函数"""
        protocol = PSIProtocol()
        
        data = "test_item"
        sha256_hash = protocol._hash_item(data, "sha256")
        sha512_hash = protocol._hash_item(data, "sha512")
        blake2b_hash = protocol._hash_item(data, "blake2b")
        
        assert sha256_hash != sha512_hash
        assert sha256_hash != blake2b_hash
        assert sha512_hash != blake2b_hash
        
        assert len(sha256_hash) == 64
        assert len(sha512_hash) == 128
        assert len(blake2b_hash) == 128
    
    def test_compute_intersection_basic(self):
        """测试基本交集计算"""
        protocol = PSIProtocol()
        
        party_a = ["item1", "item2", "item3"]
        party_b = ["item2", "item3", "item4"]
        
        result = protocol.compute_intersection(party_a, party_b)
        
        assert "intersection" in result
        assert "intersection_size" in result
        assert "computation_time" in result
        assert "privacy_budget_used" in result
        
        assert set(result["intersection"]) == {"item2", "item3"}
        assert result["intersection_size"] == 2
        assert result["computation_time"] > 0
    
    def test_compute_intersection_empty(self):
        """测试空交集计算"""
        protocol = PSIProtocol()
        
        party_a = ["item1", "item2"]
        party_b = ["item3", "item4"]
        
        result = protocol.compute_intersection(party_a, party_b)
        
        assert result["intersection"] == []
        assert result["intersection_size"] == 0
    
    def test_compute_intersection_identical(self):
        """测试相同数据集交集计算"""
        protocol = PSIProtocol()
        
        party_a = ["item1", "item2", "item3"]
        party_b = ["item1", "item2", "item3"]
        
        result = protocol.compute_intersection(party_a, party_b)
        
        assert set(result["intersection"]) == {"item1", "item2", "item3"}
        assert result["intersection_size"] == 3
    
    def test_compute_intersection_with_duplicates(self):
        """测试包含重复项的交集计算"""
        protocol = PSIProtocol()
        
        party_a = ["item1", "item2", "item2", "item3"]
        party_b = ["item2", "item3", "item3", "item4"]
        
        result = protocol.compute_intersection(party_a, party_b)
        
        # 结果应该去重
        assert set(result["intersection"]) == {"item2", "item3"}
        assert result["intersection_size"] == 2
    
    def test_privacy_budget_calculation(self):
        """测试隐私预算计算"""
        protocol = PSIProtocol()
        
        party_a = ["item1", "item2"]
        party_b = ["item2", "item3"]
        
        result = protocol.compute_intersection(
            party_a, party_b, privacy_budget=1.0
        )
        
        assert 0 < result["privacy_budget_used"] <= 1.0
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        protocol = PSIProtocol()
        
        # 创建较大的数据集
        party_a = [f"item_a_{i}" for i in range(1000)]
        party_b = [f"item_b_{i}" for i in range(500)] + [f"item_a_{i}" for i in range(100)]
        
        result = protocol.compute_intersection(party_a, party_b)
        
        assert result["intersection_size"] == 100
        assert result["computation_time"] < 10.0  # 应该在10秒内完成


if __name__ == "__main__":
    pytest.main([__file__, "-v"])