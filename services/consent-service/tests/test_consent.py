#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同意管理服务单元测试
"""

import os
import sys
import json
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

class TestConsentService:
    """同意管理服务测试类"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_issue_consent_success(self):
        """测试成功签发同意票据"""
        request_data = {
            "subject": "user123",
            "purpose": "credit_scoring",
            "scope_features": ["age", "income", "credit_history"],
            "ttl_hours": 24,
            "issuer": "bank_partner",
            "metadata": {"channel": "mobile_app"}
        }
        
        response = client.post("/consent/issue", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "consent_jwt" in data
        assert "consent_id" in data
        assert data["subject"] == "user123"
        assert data["purpose"] == "credit_scoring"
        assert data["scope_features"] == ["age", "income", "credit_history"]
        assert data["issuer"] == "bank_partner"
        assert "fingerprint" in data
        
        return data["consent_jwt"]
    
    def test_issue_consent_invalid_purpose(self):
        """测试无效目的的同意票据签发"""
        request_data = {
            "subject": "user123",
            "purpose": "invalid_purpose",
            "scope_features": ["age", "income"],
            "issuer": "bank_partner"
        }
        
        response = client.post("/consent/issue", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_issue_consent_empty_features(self):
        """测试空特征列表的同意票据签发"""
        request_data = {
            "subject": "user123",
            "purpose": "credit_scoring",
            "scope_features": [],
            "issuer": "bank_partner"
        }
        
        response = client.post("/consent/issue", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_verify_consent_success(self):
        """测试成功验证同意票据"""
        # 先签发一个同意票据
        issue_request = {
            "subject": "user456",
            "purpose": "credit_scoring",
            "scope_features": ["age", "income", "credit_history", "employment_status"],
            "ttl_hours": 24,
            "issuer": "bank_partner"
        }
        
        issue_response = client.post("/consent/issue", json=issue_request)
        assert issue_response.status_code == 200
        consent_jwt = issue_response.json()["consent_jwt"]
        
        # 验证同意票据
        verify_request = {
            "consent_jwt": consent_jwt,
            "requested_purpose": "credit_scoring",
            "requested_features": ["age", "income"],
            "requester_issuer": "bank_partner"
        }
        
        response = client.post("/consent/verify", json=verify_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "valid" in data
        assert "consent_id" in data
        assert "subject" in data
        assert "granted_features" in data
        assert "denied_features" in data
        assert "policy_decisions" in data
    
    def test_verify_consent_purpose_mismatch(self):
        """测试目的不匹配的同意验证"""
        # 先签发一个同意票据
        issue_request = {
            "subject": "user789",
            "purpose": "credit_scoring",
            "scope_features": ["age", "income"],
            "issuer": "bank_partner"
        }
        
        issue_response = client.post("/consent/issue", json=issue_request)
        assert issue_response.status_code == 200
        consent_jwt = issue_response.json()["consent_jwt"]
        
        # 验证不同目的
        verify_request = {
            "consent_jwt": consent_jwt,
            "requested_purpose": "marketing",  # 不同的目的
            "requested_features": ["age"],
            "requester_issuer": "bank_partner"
        }
        
        response = client.post("/consent/verify", json=verify_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["valid"] == False
        assert len(data["granted_features"]) == 0
        assert len(data["denied_features"]) > 0
    
    def test_verify_consent_feature_out_of_scope(self):
        """测试请求超出授权范围的特征"""
        # 先签发一个同意票据
        issue_request = {
            "subject": "user101",
            "purpose": "credit_scoring",
            "scope_features": ["age", "income"],  # 只授权这两个特征
            "issuer": "bank_partner"
        }
        
        issue_response = client.post("/consent/issue", json=issue_request)
        assert issue_response.status_code == 200
        consent_jwt = issue_response.json()["consent_jwt"]
        
        # 请求超出范围的特征
        verify_request = {
            "consent_jwt": consent_jwt,
            "requested_purpose": "credit_scoring",
            "requested_features": ["age", "income", "credit_history"],  # credit_history未授权
            "requester_issuer": "bank_partner"
        }
        
        response = client.post("/consent/verify", json=verify_request)
        assert response.status_code == 200
        
        data = response.json()
        # 应该有部分特征被拒绝
        assert "credit_history" in data["denied_features"]
    
    def test_revoke_consent(self):
        """测试撤回同意"""
        # 先签发一个同意票据
        issue_request = {
            "subject": "user202",
            "purpose": "credit_scoring",
            "scope_features": ["age", "income"],
            "issuer": "bank_partner"
        }
        
        issue_response = client.post("/consent/issue", json=issue_request)
        assert issue_response.status_code == 200
        consent_data = issue_response.json()
        consent_jwt = consent_data["consent_jwt"]
        consent_id = consent_data["consent_id"]
        
        # 撤回同意
        revoke_request = {
            "subject": "user202",
            "consent_id": consent_id,
            "reason": "用户主动撤回"
        }
        
        response = client.post("/consent/revoke", json=revoke_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "revocation_id" in data
        assert data["subject"] == "user202"
        assert data["status"] == "revoked"
        
        # 验证撤回后的同意票据应该无效
        verify_request = {
            "consent_jwt": consent_jwt,
            "requested_purpose": "credit_scoring",
            "requested_features": ["age"],
            "requester_issuer": "bank_partner"
        }
        
        verify_response = client.post("/consent/verify", json=verify_request)
        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        assert verify_data["valid"] == False
    
    def test_audit_record_and_query(self):
        """测试审计记录和查询"""
        # 记录审计信息
        audit_request = {
            "request_id": "req_12345",
            "consent_fingerprint": "fp_abcdef",
            "model_hash": "model_hash_123",
            "threshold": 0.7,
            "policy_version": "v1.0",
            "decision": "approved",
            "score": 0.85,
            "metadata": {"processing_time_ms": 150}
        }
        
        response = client.post("/audit/record", json=audit_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "audit_id" in data
        assert data["request_id"] == "req_12345"
        assert data["status"] == "recorded"
        
        # 查询审计记录
        query_response = client.get("/audit/req_12345")
        assert query_response.status_code == 200
        
        query_data = query_response.json()
        assert query_data["request_id"] == "req_12345"
        assert "records" in query_data
        assert len(query_data["records"]) > 0
        
        record = query_data["records"][0]
        assert record["consent_fingerprint"] == "fp_abcdef"
        assert record["model_hash"] == "model_hash_123"
        assert record["decision"] == "approved"
        assert record["score"] == 0.85
    
    def test_get_policies(self):
        """测试获取策略"""
        response = client.get("/policies")
        assert response.status_code == 200
        
        data = response.json()
        assert "policies" in data
        assert "grouping_policies" in data
        assert "total_policies" in data
        assert "total_grouping_policies" in data
    
    def test_verify_invalid_jwt(self):
        """测试验证无效JWT"""
        verify_request = {
            "consent_jwt": "invalid.jwt.token",
            "requested_purpose": "credit_scoring",
            "requested_features": ["age"],
            "requester_issuer": "bank_partner"
        }
        
        response = client.post("/consent/verify", json=verify_request)
        assert response.status_code == 401  # Unauthorized
    
    def test_audit_by_subject(self):
        """测试按主体查询审计记录"""
        # 先记录一些审计信息
        for i in range(3):
            audit_request = {
                "request_id": f"req_subject_{i}",
                "consent_fingerprint": f"fp_subject_{i}",
                "policy_version": "v1.0",
                "decision": "approved",
                "metadata": {"test": True}
            }
            
            response = client.post("/audit/record", json=audit_request)
            assert response.status_code == 200
        
        # 查询主体的审计记录
        response = client.get("/audit/subject/test_subject")
        assert response.status_code == 200
        
        data = response.json()
        assert "subject" in data
        assert "records" in data
        assert "total" in data

if __name__ == "__main__":
    pytest.main(["-v", __file__])