import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta
import jwt

from main import app
from models import ConsentRequest, ConsentPolicy, User, ConsentStatus
from auth import create_access_token, verify_token
from database import get_db

client = TestClient(app)

# 测试用户数据
test_user = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123"
}

test_admin = {
    "username": "admin",
    "email": "admin@example.com",
    "password": "admin123"
}

class TestConsentService:
    """同意管理服务测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "database_status" in data
    
    def test_metrics_endpoint(self):
        """测试指标接口"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "active_consents" in data
        assert "pending_requests" in data
        assert "approved_requests" in data
        assert "rejected_requests" in data
    
    def test_user_registration(self):
        """测试用户注册"""
        response = client.post("/auth/register", json=test_user)
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
        assert "id" in data
        assert "password" not in data  # 密码不应该返回
    
    def test_user_registration_duplicate(self):
        """测试重复用户注册"""
        # 第一次注册
        client.post("/auth/register", json=test_user)
        
        # 第二次注册相同用户
        response = client.post("/auth/register", json=test_user)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_user_login(self):
        """测试用户登录"""
        # 先注册用户
        client.post("/auth/register", json=test_user)
        
        # 登录
        login_data = {
            "username": test_user["username"],
            "password": test_user["password"]
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_user_login_invalid(self):
        """测试无效登录"""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpass"
        }
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]
    
    def test_protected_endpoint_without_token(self):
        """测试未授权访问受保护端点"""
        response = client.get("/consent/requests")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self):
        """测试使用无效token访问受保护端点"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/consent/requests", headers=headers)
        assert response.status_code == 401
    
    def get_auth_headers(self, user_data=None):
        """获取认证头"""
        if user_data is None:
            user_data = test_user
        
        # 注册并登录用户
        client.post("/auth/register", json=user_data)
        login_response = client.post("/auth/login", json={
            "username": user_data["username"],
            "password": user_data["password"]
        })
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_consent_request(self):
        """测试创建同意请求"""
        headers = self.get_auth_headers()
        
        request_data = {
            "data_type": "personal_info",
            "purpose": "risk_assessment",
            "description": "用于信用风险评估的个人信息使用",
            "data_fields": ["name", "age", "income"],
            "retention_period": 365,
            "third_party_sharing": False
        }
        
        response = client.post("/consent/requests", json=request_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert data["data_type"] == request_data["data_type"]
        assert data["purpose"] == request_data["purpose"]
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data
    
    def test_get_consent_requests(self):
        """测试获取同意请求列表"""
        headers = self.get_auth_headers()
        
        # 创建几个请求
        for i in range(3):
            request_data = {
                "data_type": f"data_type_{i}",
                "purpose": f"purpose_{i}",
                "description": f"Description {i}",
                "data_fields": ["field1", "field2"],
                "retention_period": 365
            }
            client.post("/consent/requests", json=request_data, headers=headers)
        
        # 获取请求列表
        response = client.get("/consent/requests", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert all("id" in item for item in data)
        assert all("status" in item for item in data)
    
    def test_get_consent_request_by_id(self):
        """测试根据ID获取同意请求"""
        headers = self.get_auth_headers()
        
        # 创建请求
        request_data = {
            "data_type": "personal_info",
            "purpose": "risk_assessment",
            "description": "测试请求",
            "data_fields": ["name", "age"],
            "retention_period": 365
        }
        create_response = client.post("/consent/requests", json=request_data, headers=headers)
        request_id = create_response.json()["id"]
        
        # 获取特定请求
        response = client.get(f"/consent/requests/{request_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == request_id
        assert data["data_type"] == request_data["data_type"]
    
    def test_get_nonexistent_consent_request(self):
        """测试获取不存在的同意请求"""
        headers = self.get_auth_headers()
        
        response = client.get("/consent/requests/999999", headers=headers)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_approve_consent_request(self):
        """测试批准同意请求"""
        # 使用管理员账户
        admin_headers = self.get_auth_headers(test_admin)
        user_headers = self.get_auth_headers(test_user)
        
        # 创建请求
        request_data = {
            "data_type": "personal_info",
            "purpose": "risk_assessment",
            "description": "测试请求",
            "data_fields": ["name", "age"],
            "retention_period": 365
        }
        create_response = client.post("/consent/requests", json=request_data, headers=user_headers)
        request_id = create_response.json()["id"]
        
        # 批准请求
        approval_data = {
            "status": "approved",
            "comments": "请求已批准"
        }
        response = client.put(f"/consent/requests/{request_id}/status", 
                            json=approval_data, headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["comments"] == approval_data["comments"]
    
    def test_reject_consent_request(self):
        """测试拒绝同意请求"""
        admin_headers = self.get_auth_headers(test_admin)
        user_headers = self.get_auth_headers(test_user)
        
        # 创建请求
        request_data = {
            "data_type": "sensitive_data",
            "purpose": "marketing",
            "description": "营销用途",
            "data_fields": ["email", "phone"],
            "retention_period": 365
        }
        create_response = client.post("/consent/requests", json=request_data, headers=user_headers)
        request_id = create_response.json()["id"]
        
        # 拒绝请求
        rejection_data = {
            "status": "rejected",
            "comments": "不符合隐私政策"
        }
        response = client.put(f"/consent/requests/{request_id}/status", 
                            json=rejection_data, headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["comments"] == rejection_data["comments"]
    
    def test_revoke_consent(self):
        """测试撤销同意"""
        headers = self.get_auth_headers()
        
        # 创建并批准请求
        request_data = {
            "data_type": "personal_info",
            "purpose": "risk_assessment",
            "description": "测试请求",
            "data_fields": ["name", "age"],
            "retention_period": 365
        }
        create_response = client.post("/consent/requests", json=request_data, headers=headers)
        request_id = create_response.json()["id"]
        
        # 撤销同意
        response = client.delete(f"/consent/requests/{request_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "revoked" in data["message"]
    
    def test_create_consent_policy(self):
        """测试创建同意政策"""
        admin_headers = self.get_auth_headers(test_admin)
        
        policy_data = {
            "name": "数据使用政策",
            "description": "规定数据使用的基本原则",
            "data_types": ["personal_info", "financial_data"],
            "allowed_purposes": ["risk_assessment", "fraud_detection"],
            "max_retention_period": 1095,
            "require_explicit_consent": True,
            "allow_third_party_sharing": False
        }
        
        response = client.post("/consent/policies", json=policy_data, headers=admin_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == policy_data["name"]
        assert data["require_explicit_consent"] == policy_data["require_explicit_consent"]
        assert "id" in data
    
    def test_get_consent_policies(self):
        """测试获取同意政策列表"""
        admin_headers = self.get_auth_headers(test_admin)
        
        # 创建几个政策
        for i in range(2):
            policy_data = {
                "name": f"政策 {i}",
                "description": f"政策描述 {i}",
                "data_types": ["type1", "type2"],
                "allowed_purposes": ["purpose1"],
                "max_retention_period": 365
            }
            client.post("/consent/policies", json=policy_data, headers=admin_headers)
        
        # 获取政策列表
        response = client.get("/consent/policies", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        assert all("id" in item for item in data)
        assert all("name" in item for item in data)
    
    def test_update_consent_policy(self):
        """测试更新同意政策"""
        admin_headers = self.get_auth_headers(test_admin)
        
        # 创建政策
        policy_data = {
            "name": "原始政策",
            "description": "原始描述",
            "data_types": ["type1"],
            "allowed_purposes": ["purpose1"],
            "max_retention_period": 365
        }
        create_response = client.post("/consent/policies", json=policy_data, headers=admin_headers)
        policy_id = create_response.json()["id"]
        
        # 更新政策
        update_data = {
            "name": "更新后的政策",
            "description": "更新后的描述",
            "max_retention_period": 730
        }
        response = client.put(f"/consent/policies/{policy_id}", 
                            json=update_data, headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["max_retention_period"] == update_data["max_retention_period"]
    
    def test_delete_consent_policy(self):
        """测试删除同意政策"""
        admin_headers = self.get_auth_headers(test_admin)
        
        # 创建政策
        policy_data = {
            "name": "待删除政策",
            "description": "这个政策将被删除",
            "data_types": ["type1"],
            "allowed_purposes": ["purpose1"],
            "max_retention_period": 365
        }
        create_response = client.post("/consent/policies", json=policy_data, headers=admin_headers)
        policy_id = create_response.json()["id"]
        
        # 删除政策
        response = client.delete(f"/consent/policies/{policy_id}", headers=admin_headers)
        assert response.status_code == 200
        
        # 验证政策已删除
        get_response = client.get(f"/consent/policies/{policy_id}", headers=admin_headers)
        assert get_response.status_code == 404
    
    def test_consent_request_validation(self):
        """测试同意请求参数验证"""
        headers = self.get_auth_headers()
        
        # 测试缺少必需字段
        invalid_data = {
            "purpose": "risk_assessment"
            # 缺少 data_type
        }
        response = client.post("/consent/requests", json=invalid_data, headers=headers)
        assert response.status_code == 422
        
        # 测试无效的保留期
        invalid_data = {
            "data_type": "personal_info",
            "purpose": "risk_assessment",
            "description": "测试",
            "retention_period": -1  # 无效值
        }
        response = client.post("/consent/requests", json=invalid_data, headers=headers)
        assert response.status_code == 422
    
    def test_user_permissions(self):
        """测试用户权限控制"""
        user_headers = self.get_auth_headers(test_user)
        
        # 普通用户不能创建政策
        policy_data = {
            "name": "测试政策",
            "description": "测试",
            "data_types": ["type1"],
            "allowed_purposes": ["purpose1"],
            "max_retention_period": 365
        }
        response = client.post("/consent/policies", json=policy_data, headers=user_headers)
        assert response.status_code == 403
    
    def test_token_expiration(self):
        """测试token过期"""
        # 创建一个过期的token
        expired_payload = {
            "sub": "testuser",
            "exp": datetime.utcnow() - timedelta(hours=1)  # 1小时前过期
        }
        expired_token = jwt.encode(expired_payload, "secret", algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.get("/consent/requests", headers=headers)
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()


class TestAuthModule:
    """认证模块测试类"""
    
    def test_create_access_token(self):
        """测试创建访问token"""
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 验证token可以被解码
        decoded = jwt.decode(token, "secret", algorithms=["HS256"])
        assert decoded["sub"] == "testuser"
        assert decoded["role"] == "user"
        assert "exp" in decoded
    
    def test_verify_token_valid(self):
        """测试验证有效token"""
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        
        payload = verify_token(token)
        assert payload["sub"] == "testuser"
        assert payload["role"] == "user"
    
    def test_verify_token_invalid(self):
        """测试验证无效token"""
        with pytest.raises(Exception):
            verify_token("invalid_token")
    
    def test_verify_token_expired(self):
        """测试验证过期token"""
        expired_payload = {
            "sub": "testuser",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        expired_token = jwt.encode(expired_payload, "secret", algorithm="HS256")
        
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_token(expired_token)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])