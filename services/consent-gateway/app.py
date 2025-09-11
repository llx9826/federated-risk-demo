#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同意网关服务 (Consent Gateway Service)

功能:
1. 同意票据签发/撤销/验证 (JWT/SD-JWT)
2. OPA策略引擎集成
3. 目的绑定同意管理
4. 审计日志记录
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from uuid import uuid4

import jwt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import asyncpg
import redis.asyncio as redis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 环境变量配置
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRE_HOURS = int(os.getenv('JWT_EXPIRE_HOURS', '24'))
OPA_URL = os.getenv('OPA_URL', 'http://opa:8181')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/consent_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')

# FastAPI应用初始化
app = FastAPI(
    title="同意网关服务",
    description="处理同意票据的签发、验证和撤销，集成OPA策略引擎",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
db_pool = None
redis_client = None
security = HTTPBearer()

# Pydantic模型定义
class ConsentIssueRequest(BaseModel):
    """同意签发请求"""
    user_id: str = Field(..., description="用户ID")
    organization: str = Field(..., description="申请机构")
    purpose: str = Field(..., description="使用目的")
    data_categories: List[str] = Field(..., description="数据类别")
    retention_period: str = Field(..., description="保留期限(ISO 8601)")
    processing_methods: List[str] = Field(..., description="处理方式")
    third_parties: Optional[List[str]] = Field(None, description="第三方共享")
    user_signature: str = Field(..., description="用户数字签名")
    
    @validator('purpose')
    def validate_purpose(cls, v):
        allowed_purposes = [
            'credit_scoring', 'risk_assessment', 'fraud_detection',
            'marketing', 'research', 'compliance'
        ]
        if v not in allowed_purposes:
            raise ValueError(f'目的必须是以下之一: {allowed_purposes}')
        return v
    
    @validator('data_categories')
    def validate_data_categories(cls, v):
        allowed_categories = [
            'identity', 'financial', 'behavioral', 'demographic',
            'transaction', 'credit_history', 'device_info'
        ]
        for category in v:
            if category not in allowed_categories:
                raise ValueError(f'数据类别必须是以下之一: {allowed_categories}')
        return v

class ConsentRevokeRequest(BaseModel):
    """同意撤销请求"""
    consent_token: str = Field(..., description="同意票据")
    user_id: str = Field(..., description="用户ID")
    revoke_reason: str = Field(..., description="撤销原因")
    user_signature: str = Field(..., description="用户数字签名")

class ConsentVerifyRequest(BaseModel):
    """同意验证请求"""
    consent_token: str = Field(..., description="同意票据")
    purpose: str = Field(..., description="使用目的")
    data_categories: List[str] = Field(..., description="数据类别")
    requesting_service: str = Field(..., description="请求服务")

class ConsentResponse(BaseModel):
    """同意响应"""
    consent_id: str
    consent_token: str
    status: str
    expires_at: datetime
    fingerprint: str

class ConsentVerifyResponse(BaseModel):
    """同意验证响应"""
    valid: bool
    consent_id: str
    user_id: str
    purpose: str
    data_categories: List[str]
    expires_at: datetime
    policy_decision: Dict[str, Any]
    restrictions: Optional[Dict[str, Any]] = None

class PolicyEvaluationRequest(BaseModel):
    """策略评估请求"""
    input: Dict[str, Any]

# 数据库初始化
async def init_database():
    """初始化数据库连接池"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # 创建表结构
        async with db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consent_records (
                    id SERIAL PRIMARY KEY,
                    consent_id VARCHAR(64) UNIQUE NOT NULL,
                    user_id VARCHAR(64) NOT NULL,
                    organization VARCHAR(64) NOT NULL,
                    purpose VARCHAR(64) NOT NULL,
                    data_categories TEXT[] NOT NULL,
                    retention_period VARCHAR(32) NOT NULL,
                    processing_methods TEXT[] NOT NULL,
                    third_parties TEXT[],
                    status VARCHAR(16) NOT NULL DEFAULT 'active',
                    consent_token TEXT NOT NULL,
                    fingerprint VARCHAR(64) NOT NULL,
                    user_signature TEXT NOT NULL,
                    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL,
                    revoked_at TIMESTAMPTZ,
                    revoke_reason TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_consent_user_id ON consent_records (user_id);
                CREATE INDEX IF NOT EXISTS idx_consent_status ON consent_records (status);
                CREATE INDEX IF NOT EXISTS idx_consent_purpose ON consent_records (purpose);
                CREATE INDEX IF NOT EXISTS idx_consent_expires ON consent_records (expires_at);
            """)
        
        logger.info("数据库连接池初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise

# Redis初始化
async def init_redis():
    """初始化Redis连接"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis连接初始化成功")
    except Exception as e:
        logger.error(f"Redis初始化失败: {e}")
        raise

# 工具函数
def generate_consent_id() -> str:
    """生成同意ID"""
    return f"consent_{uuid4().hex[:16]}"

def generate_fingerprint(consent_data: dict) -> str:
    """生成同意指纹"""
    # 排除时间戳等动态字段
    stable_data = {
        'user_id': consent_data['user_id'],
        'organization': consent_data['organization'],
        'purpose': consent_data['purpose'],
        'data_categories': sorted(consent_data['data_categories']),
        'retention_period': consent_data['retention_period']
    }
    
    canonical_json = json.dumps(stable_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode()).hexdigest()

def create_jwt_token(payload: dict) -> str:
    """创建JWT令牌"""
    payload['iat'] = datetime.utcnow()
    payload['exp'] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效令牌")

async def evaluate_opa_policy(policy_input: dict) -> dict:
    """调用OPA策略引擎"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPA_URL}/v1/data/consent/allow",
                json={"input": policy_input},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"OPA策略评估失败: {e}")
        return {"result": False, "error": str(e)}

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{uuid4().hex[:16]}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "HIGH",
            "source": {
                "service": "consent-gateway",
                "version": "1.0.0"
            },
            "operation": event_data,
            "compliance": {
                "regulation": ["PIPL", "JR/T_0196"],
                "data_classification": "CONFIDENTIAL"
            }
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{AUDIT_SERVICE_URL}/audit/records",
                json=audit_record,
                timeout=5.0
            )
    except Exception as e:
        logger.error(f"审计日志发送失败: {e}")

# API路由
@app.post("/consent/issue", response_model=ConsentResponse)
async def issue_consent(request: ConsentIssueRequest):
    """签发同意票据"""
    try:
        # 生成同意ID和指纹
        consent_id = generate_consent_id()
        fingerprint = generate_fingerprint(request.dict())
        
        # 计算过期时间
        expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
        
        # 创建JWT载荷
        jwt_payload = {
            "consent_id": consent_id,
            "user_id": request.user_id,
            "organization": request.organization,
            "purpose": request.purpose,
            "data_categories": request.data_categories,
            "retention_period": request.retention_period,
            "processing_methods": request.processing_methods,
            "third_parties": request.third_parties,
            "fingerprint": fingerprint,
            "type": "consent_token"
        }
        
        # 生成JWT令牌
        consent_token = create_jwt_token(jwt_payload)
        
        # OPA策略评估
        policy_input = {
            "user_id": request.user_id,
            "organization": request.organization,
            "purpose": request.purpose,
            "data_categories": request.data_categories,
            "action": "issue_consent"
        }
        
        policy_result = await evaluate_opa_policy(policy_input)
        if not policy_result.get("result", False):
            raise HTTPException(
                status_code=403,
                detail=f"策略评估失败: {policy_result.get('error', '未知错误')}"
            )
        
        # 存储到数据库
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO consent_records (
                    consent_id, user_id, organization, purpose, data_categories,
                    retention_period, processing_methods, third_parties,
                    consent_token, fingerprint, user_signature, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, 
                consent_id, request.user_id, request.organization, request.purpose,
                request.data_categories, request.retention_period, request.processing_methods,
                request.third_parties, consent_token, fingerprint, request.user_signature,
                expires_at
            )
        
        # 缓存到Redis
        await redis_client.setex(
            f"consent:{consent_id}",
            JWT_EXPIRE_HOURS * 3600,
            json.dumps(jwt_payload)
        )
        
        # 发送审计日志
        await send_audit_log("CONSENT_ISSUE", {
            "consent_id": consent_id,
            "user_id": request.user_id,
            "organization": request.organization,
            "purpose": request.purpose,
            "fingerprint": fingerprint
        })
        
        return ConsentResponse(
            consent_id=consent_id,
            consent_token=consent_token,
            status="active",
            expires_at=expires_at,
            fingerprint=fingerprint
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同意签发失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/consent/revoke")
async def revoke_consent(request: ConsentRevokeRequest):
    """撤销同意票据"""
    try:
        # 验证JWT令牌
        payload = verify_jwt_token(request.consent_token)
        consent_id = payload.get('consent_id')
        
        # 验证用户身份
        if payload.get('user_id') != request.user_id:
            raise HTTPException(status_code=403, detail="用户身份不匹配")
        
        # 更新数据库状态
        async with db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE consent_records 
                SET status = 'revoked', revoked_at = NOW(), revoke_reason = $1, updated_at = NOW()
                WHERE consent_id = $2 AND user_id = $3 AND status = 'active'
            """, request.revoke_reason, consent_id, request.user_id)
            
            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="同意记录未找到或已撤销")
        
        # 从Redis删除缓存
        await redis_client.delete(f"consent:{consent_id}")
        
        # 添加到撤销黑名单
        await redis_client.sadd("revoked_consents", consent_id)
        
        # 发送审计日志
        await send_audit_log("CONSENT_REVOKE", {
            "consent_id": consent_id,
            "user_id": request.user_id,
            "revoke_reason": request.revoke_reason
        })
        
        return {"message": "同意已成功撤销", "consent_id": consent_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同意撤销失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/consent/verify", response_model=ConsentVerifyResponse)
async def verify_consent(request: ConsentVerifyRequest):
    """验证同意票据"""
    try:
        # 验证JWT令牌
        payload = verify_jwt_token(request.consent_token)
        consent_id = payload.get('consent_id')
        
        # 检查撤销黑名单
        is_revoked = await redis_client.sismember("revoked_consents", consent_id)
        if is_revoked:
            return ConsentVerifyResponse(
                valid=False,
                consent_id=consent_id,
                user_id=payload.get('user_id', ''),
                purpose='',
                data_categories=[],
                expires_at=datetime.utcnow(),
                policy_decision={"allow": False, "reason": "同意已撤销"}
            )
        
        # 验证目的和数据类别
        token_purpose = payload.get('purpose')
        token_categories = payload.get('data_categories', [])
        
        if request.purpose != token_purpose:
            return ConsentVerifyResponse(
                valid=False,
                consent_id=consent_id,
                user_id=payload.get('user_id', ''),
                purpose=token_purpose,
                data_categories=token_categories,
                expires_at=datetime.fromisoformat(payload.get('exp')),
                policy_decision={"allow": False, "reason": "使用目的不匹配"}
            )
        
        # 检查数据类别是否在授权范围内
        unauthorized_categories = set(request.data_categories) - set(token_categories)
        if unauthorized_categories:
            return ConsentVerifyResponse(
                valid=False,
                consent_id=consent_id,
                user_id=payload.get('user_id', ''),
                purpose=token_purpose,
                data_categories=token_categories,
                expires_at=datetime.fromisoformat(payload.get('exp')),
                policy_decision={
                    "allow": False, 
                    "reason": f"未授权的数据类别: {list(unauthorized_categories)}"
                }
            )
        
        # OPA策略评估
        policy_input = {
            "user_id": payload.get('user_id'),
            "organization": payload.get('organization'),
            "purpose": request.purpose,
            "data_categories": request.data_categories,
            "requesting_service": request.requesting_service,
            "action": "access_data",
            "consent_id": consent_id
        }
        
        policy_result = await evaluate_opa_policy(policy_input)
        
        # 发送审计日志
        await send_audit_log("CONSENT_VERIFY", {
            "consent_id": consent_id,
            "user_id": payload.get('user_id'),
            "purpose": request.purpose,
            "requesting_service": request.requesting_service,
            "verification_result": policy_result.get("result", False)
        })
        
        return ConsentVerifyResponse(
            valid=policy_result.get("result", False),
            consent_id=consent_id,
            user_id=payload.get('user_id', ''),
            purpose=token_purpose,
            data_categories=token_categories,
            expires_at=datetime.fromisoformat(payload.get('exp')),
            policy_decision=policy_result,
            restrictions=policy_result.get("restrictions")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同意验证失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/consent/{consent_id}")
async def get_consent_details(consent_id: str):
    """获取同意详情"""
    try:
        # 先从Redis缓存查询
        cached_data = await redis_client.get(f"consent:{consent_id}")
        if cached_data:
            return json.loads(cached_data)
        
        # 从数据库查询
        async with db_pool.acquire() as conn:
            record = await conn.fetchrow("""
                SELECT * FROM consent_records WHERE consent_id = $1
            """, consent_id)
            
            if not record:
                raise HTTPException(status_code=404, detail="同意记录未找到")
            
            return dict(record)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取同意详情失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/consent/user/{user_id}")
async def get_user_consents(user_id: str, status: Optional[str] = None):
    """获取用户的所有同意记录"""
    try:
        query = "SELECT * FROM consent_records WHERE user_id = $1"
        params = [user_id]
        
        if status:
            query += " AND status = $2"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
            
    except Exception as e:
        logger.error(f"获取用户同意记录失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/policy/evaluate")
async def evaluate_policy(request: PolicyEvaluationRequest):
    """策略评估接口"""
    try:
        result = await evaluate_opa_policy(request.input)
        return result
    except Exception as e:
        logger.error(f"策略评估失败: {e}")
        raise HTTPException(status_code=500, detail="策略评估失败")

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # 检查Redis连接
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不可用")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动同意网关服务...")
    await init_database()
    await init_redis()
    logger.info("同意网关服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("关闭同意网关服务...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    logger.info("同意网关服务已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )