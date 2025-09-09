#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSI服务单元测试
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, PSI_SESSIONS, PSI_RESULTS

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_storage():
    """每个测试前清理存储"""
    PSI_SESSIONS.clear()
    PSI_RESULTS.clear()
    yield
    PSI_SESSIONS.clear()
    PSI_RESULTS.clear()

def test_health_check():
    """测试健康检查"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_start_psi_session():
    """测试启动PSI会话"""
    request_data = {
        "session_name": "test_session",
        "description": "测试会话",
        "ttl_minutes": 60
    }
    
    response = client.post("/psi/start", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "created"
    assert "created_at" in data
    assert "expires_at" in data
    
    # 验证会话已存储
    session_id = data["session_id"]
    assert session_id in PSI_SESSIONS

def test_upload_psi_data():
    """测试上传PSI数据"""
    # 先创建会话
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 生成测试tokens
    test_tokens = [
        "a" * 64,  # 模拟SHA256哈希
        "b" * 64,
        "c" * 64
    ]
    
    upload_data = {
        "session_id": session_id,
        "side": "A",
        "psi_tokens": test_tokens,
        "metadata": {"source": "test"}
    }
    
    response = client.post("/psi/upload/A", json=upload_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["session_id"] == session_id
    assert data["side"] == "A"
    assert data["token_count"] == 3
    assert data["unique_count"] == 3

def test_upload_invalid_side():
    """测试上传无效方"""
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    upload_data = {
        "session_id": session_id,
        "side": "C",  # 无效的side
        "psi_tokens": ["a" * 64]
    }
    
    response = client.post("/psi/upload/C", json=upload_data)
    assert response.status_code == 400

def test_upload_invalid_tokens():
    """测试上传无效tokens"""
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 测试空tokens
    upload_data = {
        "session_id": session_id,
        "side": "A",
        "psi_tokens": []
    }
    
    response = client.post("/psi/upload/A", json=upload_data)
    assert response.status_code == 400
    
    # 测试无效长度的token
    upload_data["psi_tokens"] = ["short_token"]
    response = client.post("/psi/upload/A", json=upload_data)
    assert response.status_code == 400

def test_compute_psi_empty_intersection():
    """测试计算空交集"""
    # 创建会话
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 上传A方数据
    tokens_a = ["a" * 64, "b" * 64]
    upload_a = {
        "session_id": session_id,
        "side": "A",
        "psi_tokens": tokens_a
    }
    client.post("/psi/upload/A", json=upload_a)
    
    # 上传B方数据（无交集）
    tokens_b = ["c" * 64, "d" * 64]
    upload_b = {
        "session_id": session_id,
        "side": "B",
        "psi_tokens": tokens_b
    }
    client.post("/psi/upload/B", json=upload_b)
    
    # 计算PSI
    compute_request = {
        "session_id": session_id,
        "compute_preview": True,
        "preview_limit": 5
    }
    
    response = client.post("/psi/compute", json=compute_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["intersect_size"] == 0
    assert data["side_a_size"] == 2
    assert data["side_b_size"] == 2
    assert data["intersection_rate_a"] == 0.0
    assert data["intersection_rate_b"] == 0.0
    assert len(data["sample_preview"]) == 0
    assert "mapping_store_key" in data

def test_compute_psi_with_intersection():
    """测试计算有交集的PSI"""
    # 创建会话
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 共同的tokens
    common_tokens = ["x" * 64, "y" * 64]
    
    # 上传A方数据
    tokens_a = common_tokens + ["a" * 64]
    upload_a = {
        "session_id": session_id,
        "side": "A",
        "psi_tokens": tokens_a
    }
    client.post("/psi/upload/A", json=upload_a)
    
    # 上传B方数据
    tokens_b = common_tokens + ["b" * 64]
    upload_b = {
        "session_id": session_id,
        "side": "B",
        "psi_tokens": tokens_b
    }
    client.post("/psi/upload/B", json=upload_b)
    
    # 计算PSI
    compute_request = {
        "session_id": session_id,
        "compute_preview": True,
        "preview_limit": 5
    }
    
    response = client.post("/psi/compute", json=compute_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["intersect_size"] == 2
    assert data["side_a_size"] == 3
    assert data["side_b_size"] == 3
    assert abs(data["intersection_rate_a"] - 2/3) < 0.001
    assert abs(data["intersection_rate_b"] - 2/3) < 0.001
    assert len(data["sample_preview"]) == 2
    
    # 验证预览样本是交集的子集
    for token in data["sample_preview"]:
        assert token in common_tokens

def test_get_psi_result():
    """测试获取PSI结果"""
    # 创建会话并计算PSI
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 上传数据
    common_token = "z" * 64
    tokens_a = [common_token, "a" * 64]
    tokens_b = [common_token, "b" * 64]
    
    client.post("/psi/upload/A", json={
        "session_id": session_id,
        "side": "A",
        "psi_tokens": tokens_a
    })
    
    client.post("/psi/upload/B", json={
        "session_id": session_id,
        "side": "B",
        "psi_tokens": tokens_b
    })
    
    # 计算PSI
    compute_response = client.post("/psi/compute", json={
        "session_id": session_id
    })
    mapping_key = compute_response.json()["mapping_store_key"]
    
    # 获取结果
    result_response = client.get(f"/psi/result/{mapping_key}")
    assert result_response.status_code == 200
    
    data = result_response.json()
    assert data["session_id"] == session_id
    assert data["mapping_store_key"] == mapping_key
    assert common_token in data["intersection_tokens"]
    assert len(data["intersection_tokens"]) == 1

def test_compute_without_both_sides():
    """测试只有一方数据时计算PSI"""
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 只上传A方数据
    client.post("/psi/upload/A", json={
        "session_id": session_id,
        "side": "A",
        "psi_tokens": ["a" * 64]
    })
    
    # 尝试计算PSI
    response = client.post("/psi/compute", json={
        "session_id": session_id
    })
    assert response.status_code == 400

def test_duplicate_tokens():
    """测试重复tokens处理"""
    session_response = client.post("/psi/start", json={"ttl_minutes": 60})
    session_id = session_response.json()["session_id"]
    
    # 上传包含重复tokens的数据
    duplicate_tokens = ["a" * 64, "a" * 64, "b" * 64]
    
    response = client.post("/psi/upload/A", json={
        "session_id": session_id,
        "side": "A",
        "psi_tokens": duplicate_tokens
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["token_count"] == 3  # 原始数量
    assert data["unique_count"] == 2  # 去重后数量

def test_list_sessions():
    """测试列出会话"""
    # 创建几个会话
    session1 = client.post("/psi/start", json={"session_name": "session1"})
    session2 = client.post("/psi/start", json={"session_name": "session2"})
    
    response = client.get("/psi/sessions")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total"] == 2
    assert len(data["sessions"]) == 2
    
    session_names = [s["session_name"] for s in data["sessions"]]
    assert "session1" in session_names
    assert "session2" in session_names

def test_delete_session():
    """测试删除会话"""
    session_response = client.post("/psi/start", json={"session_name": "to_delete"})
    session_id = session_response.json()["session_id"]
    
    # 验证会话存在
    assert session_id in PSI_SESSIONS
    
    # 删除会话
    delete_response = client.delete(f"/psi/sessions/{session_id}")
    assert delete_response.status_code == 200
    
    # 验证会话已删除
    assert session_id not in PSI_SESSIONS

def test_metrics():
    """测试指标接口"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_sessions" in data
    assert "computed_sessions" in data
    assert "pending_sessions" in data
    assert "total_results" in data
    assert "timestamp" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])