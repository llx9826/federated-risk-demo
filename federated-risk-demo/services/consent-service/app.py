#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同意管理服务 (Consent Service)
基于Casbin的策略即代码 + JWT的Purpose-Bound Consent实现
"""

import os
import uuid
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import jwt
import casbin
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# JWT配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

# Casbin配置
CASBIN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "casbin_model.conf")
CASBIN_POLICY_PATH = os.path.join(os.path.dirname(__file__), "casbin_policy.csv")

# 审计数据库配置
AUDIT_DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")

app = FastAPI(
    title="Consent Service",
    description="同意管理服务 - Purpose-Bound Consent + Casbin策略引擎",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全配置
security = HTTPBearer()

# 全局变量
casbin_enforcer = None
audit_db_conn = None

# Pydantic模型
class ConsentIssueRequest(BaseModel):
    """同意票据签发请求"""
    subject: str = Field(..., description="数据主体标识")
    purpose: str = Field(..., description="数据使用目的")
    scope_features: List[str] = Field(..., description="授权特征列表")
    ttl_hours: Optional[int] = Field(24, description="票据有效期(小时)")
    issuer: str = Field(..., description="签发方标识")
    metadata: Optional[Dict] = Field(default_factory=dict, description="额外元数据")
    
    @validator('purpose')
    def validate_purpose(cls, v):
        allowed_purposes = ['credit_scoring', 'risk_assessment', 'marketing', 'research']
        if v not in allowed_purposes:
            raise ValueError(f'目的必须是以下之一: {allowed_purposes}')
        return v
    
    @validator('scope_features')
    def validate_features(cls, v):
        if not v:
            raise ValueError('授权特征列表不能为空')
        return v

class ConsentIssueResponse(BaseModel):
    """同意票据签发响应"""
    consent_jwt: str
    consent_id: str
    subject: str
    purpose: str
    scope_features: List[str]
    issued_at: str
    expires_at: str
    issuer: str
    fingerprint: str

class ConsentRevokeRequest(BaseModel):
    """同意撤回请求"""
    subject: str = Field(..., description="数据主体标识")
    consent_id: Optional[str] = Field(None, description="特定同意ID")
    revoke_all: bool = Field(False, description="是否撤回该主体的所有同意")
    reason: Optional[str] = Field(None, description="撤回原因")

class ConsentVerifyRequest(BaseModel):
    """同意验证请求"""
    consent_jwt: str = Field(..., description="同意票据JWT")
    requested_purpose: str = Field(..., description="请求的数据使用目的")
    requested_features: List[str] = Field(..., description="请求的特征列表")
    requester_issuer: str = Field(..., description="请求方标识")
    context: Optional[Dict] = Field(default_factory=dict, description="请求上下文")

class ConsentVerifyResponse(BaseModel):
    """同意验证响应"""
    valid: bool
    consent_id: str
    subject: str
    granted_features: List[str]
    denied_features: List[str]
    policy_decisions: List[Dict]
    verification_time: str
    fingerprint: str

class AuditRecordRequest(BaseModel):
    """审计记录请求"""
    request_id: str = Field(..., description="请求唯一标识")
    consent_fingerprint: str = Field(..., description="同意票据指纹")
    model_hash: Optional[str] = Field(None, description="模型哈希")
    threshold: Optional[float] = Field(None, description="决策阈值")
    policy_version: str = Field(..., description="策略版本")
    decision: str = Field(..., description="决策结果")
    score: Optional[float] = Field(None, description="风险评分")
    metadata: Optional[Dict] = Field(default_factory=dict, description="额外元数据")

class AuditRecordResponse(BaseModel):
    """审计记录响应"""
    audit_id: str
    request_id: str
    timestamp: str
    status: str

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    casbin_status: str
    audit_db_status: str

def init_casbin_enforcer():
    """初始化Casbin执行器"""
    global casbin_enforcer
    try:
        if not os.path.exists(CASBIN_MODEL_PATH):
            raise FileNotFoundError(f"Casbin模型文件不存在: {CASBIN_MODEL_PATH}")
        if not os.path.exists(CASBIN_POLICY_PATH):
            raise FileNotFoundError(f"Casbin策略文件不存在: {CASBIN_POLICY_PATH}")
        
        casbin_enforcer = casbin.Enforcer(CASBIN_MODEL_PATH, CASBIN_POLICY_PATH)
        logger.info("Casbin执行器初始化成功")
        return True
    except Exception as e:
        logger.error(f"Casbin执行器初始化失败: {str(e)}")
        return False

def init_audit_database():
    """初始化审计数据库"""
    global audit_db_conn
    try:
        audit_db_conn = sqlite3.connect(AUDIT_DB_PATH, check_same_thread=False)
        audit_db_conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_records (
                audit_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                consent_fingerprint TEXT NOT NULL,
                model_hash TEXT,
                threshold REAL,
                policy_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                decision TEXT NOT NULL,
                score REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        audit_db_conn.execute('''
            CREATE TABLE IF NOT EXISTS consent_revocations (
                revocation_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                consent_id TEXT,
                reason TEXT,
                revoked_at TEXT NOT NULL,
                revoked_by TEXT
            )
        ''')
        
        audit_db_conn.commit()
        logger.info("审计数据库初始化成功")
        return True
    except Exception as e:
        logger.error(f"审计数据库初始化失败: {str(e)}")
        return False

def generate_consent_fingerprint(payload: Dict) -> str:
    """生成同意票据指纹"""
    import hashlib
    # 使用关键字段生成指纹
    key_fields = {
        'subject': payload.get('subject'),
        'purpose': payload.get('purpose'),
        'scope_features': sorted(payload.get('scope_features', [])),
        'issuer': payload.get('issuer')
    }
    fingerprint_str = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

def verify_jwt_token(token: str) -> Dict:
    """验证JWT令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="同意票据已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的同意票据")

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动同意管理服务...")
    
    # 初始化Casbin
    if not init_casbin_enforcer():
        logger.error("Casbin初始化失败，服务可能无法正常工作")
    
    # 初始化审计数据库
    if not init_audit_database():
        logger.error("审计数据库初始化失败，审计功能可能无法正常工作")
    
    logger.info("同意管理服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    global audit_db_conn
    if audit_db_conn:
        audit_db_conn.close()
    logger.info("同意管理服务已关闭")

@app.post("/consent/issue", response_model=ConsentIssueResponse)
async def issue_consent(request: ConsentIssueRequest):
    """签发同意票据"""
    consent_id = str(uuid.uuid4())
    issued_at = datetime.now()
    expires_at = issued_at + timedelta(hours=request.ttl_hours)
    
    # 构建JWT payload
    payload = {
        'consent_id': consent_id,
        'subject': request.subject,
        'purpose': request.purpose,
        'scope_features': request.scope_features,
        'issuer': request.issuer,
        'iat': int(issued_at.timestamp()),
        'exp': int(expires_at.timestamp()),
        'metadata': request.metadata
    }
    
    # 生成指纹
    fingerprint = generate_consent_fingerprint(payload)
    payload['fingerprint'] = fingerprint
    
    # 签发JWT
    try:
        consent_jwt = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"JWT签发失败: {str(e)}")
        raise HTTPException(status_code=500, detail="同意票据签发失败")
    
    logger.info(f"签发同意票据: {consent_id} for {request.subject}")
    
    return ConsentIssueResponse(
        consent_jwt=consent_jwt,
        consent_id=consent_id,
        subject=request.subject,
        purpose=request.purpose,
        scope_features=request.scope_features,
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
        issuer=request.issuer,
        fingerprint=fingerprint
    )

@app.post("/consent/revoke")
async def revoke_consent(request: ConsentRevokeRequest):
    """撤回同意"""
    global audit_db_conn
    
    revocation_id = str(uuid.uuid4())
    revoked_at = datetime.now().isoformat()
    
    try:
        # 记录撤回操作
        audit_db_conn.execute(
            "INSERT INTO consent_revocations (revocation_id, subject, consent_id, reason, revoked_at, revoked_by) VALUES (?, ?, ?, ?, ?, ?)",
            (revocation_id, request.subject, request.consent_id, request.reason, revoked_at, "user")
        )
        audit_db_conn.commit()
        
        logger.info(f"撤回同意: {request.subject} - {request.consent_id}")
        
        return {
            'revocation_id': revocation_id,
            'subject': request.subject,
            'consent_id': request.consent_id,
            'revoked_at': revoked_at,
            'status': 'revoked'
        }
        
    except Exception as e:
        logger.error(f"撤回同意失败: {str(e)}")
        raise HTTPException(status_code=500, detail="撤回同意失败")

@app.post("/consent/verify", response_model=ConsentVerifyResponse)
async def verify_consent(request: ConsentVerifyRequest):
    """验证同意票据和策略"""
    global casbin_enforcer
    
    if not casbin_enforcer:
        raise HTTPException(status_code=500, detail="策略引擎未初始化")
    
    # 验证JWT
    try:
        payload = verify_jwt_token(request.consent_jwt)
    except HTTPException:
        raise
    
    consent_id = payload.get('consent_id')
    subject = payload.get('subject')
    consent_purpose = payload.get('purpose')
    consent_features = payload.get('scope_features', [])
    consent_issuer = payload.get('issuer')
    fingerprint = payload.get('fingerprint', '')
    
    # 检查目的匹配
    if request.requested_purpose != consent_purpose:
        return ConsentVerifyResponse(
            valid=False,
            consent_id=consent_id,
            subject=subject,
            granted_features=[],
            denied_features=request.requested_features,
            policy_decisions=[{
                'feature': 'purpose_check',
                'decision': 'deny',
                'reason': f'目的不匹配: 请求{request.requested_purpose}, 授权{consent_purpose}'
            }],
            verification_time=datetime.now().isoformat(),
            fingerprint=fingerprint
        )
    
    # 检查撤回状态
    global audit_db_conn
    try:
        cursor = audit_db_conn.execute(
            "SELECT COUNT(*) FROM consent_revocations WHERE subject = ? AND (consent_id = ? OR consent_id IS NULL)",
            (subject, consent_id)
        )
        revocation_count = cursor.fetchone()[0]
        if revocation_count > 0:
            return ConsentVerifyResponse(
                valid=False,
                consent_id=consent_id,
                subject=subject,
                granted_features=[],
                denied_features=request.requested_features,
                policy_decisions=[{
                    'feature': 'revocation_check',
                    'decision': 'deny',
                    'reason': '同意已被撤回'
                }],
                verification_time=datetime.now().isoformat(),
                fingerprint=fingerprint
            )
    except Exception as e:
        logger.warning(f"撤回状态检查失败: {str(e)}")
    
    # Casbin策略验证
    granted_features = []
    denied_features = []
    policy_decisions = []
    
    for feature in request.requested_features:
        # 检查特征是否在同意范围内
        if feature not in consent_features:
            denied_features.append(feature)
            policy_decisions.append({
                'feature': feature,
                'decision': 'deny',
                'reason': '特征不在同意范围内'
            })
            continue
        
        # Casbin策略检查
        try:
            # 构建Casbin请求参数
            casbin_request = [
                subject,  # sub
                'data',   # obj
                'read',   # act
                request.requested_purpose,  # purpose
                feature,  # feature
                request.requester_issuer    # issuer
            ]
            
            allowed = casbin_enforcer.enforce(*casbin_request)
            
            if allowed:
                granted_features.append(feature)
                policy_decisions.append({
                    'feature': feature,
                    'decision': 'allow',
                    'reason': 'Casbin策略允许'
                })
            else:
                denied_features.append(feature)
                policy_decisions.append({
                    'feature': feature,
                    'decision': 'deny',
                    'reason': 'Casbin策略拒绝'
                })
                
        except Exception as e:
            logger.error(f"Casbin策略检查失败 {feature}: {str(e)}")
            denied_features.append(feature)
            policy_decisions.append({
                'feature': feature,
                'decision': 'deny',
                'reason': f'策略检查错误: {str(e)}'
            })
    
    verification_time = datetime.now().isoformat()
    is_valid = len(granted_features) > 0
    
    logger.info(f"同意验证完成: {consent_id}, 授权特征: {len(granted_features)}, 拒绝特征: {len(denied_features)}")
    
    return ConsentVerifyResponse(
        valid=is_valid,
        consent_id=consent_id,
        subject=subject,
        granted_features=granted_features,
        denied_features=denied_features,
        policy_decisions=policy_decisions,
        verification_time=verification_time,
        fingerprint=fingerprint
    )

@app.post("/audit/record", response_model=AuditRecordResponse)
async def record_audit(request: AuditRecordRequest):
    """记录审计信息"""
    global audit_db_conn
    
    audit_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        audit_db_conn.execute(
            "INSERT INTO audit_records (audit_id, request_id, consent_fingerprint, model_hash, threshold, policy_version, timestamp, decision, score, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                audit_id,
                request.request_id,
                request.consent_fingerprint,
                request.model_hash,
                request.threshold,
                request.policy_version,
                timestamp,
                request.decision,
                request.score,
                json.dumps(request.metadata) if request.metadata else None
            )
        )
        audit_db_conn.commit()
        
        logger.info(f"审计记录已保存: {audit_id}")
        
        return AuditRecordResponse(
            audit_id=audit_id,
            request_id=request.request_id,
            timestamp=timestamp,
            status='recorded'
        )
        
    except Exception as e:
        logger.error(f"审计记录保存失败: {str(e)}")
        raise HTTPException(status_code=500, detail="审计记录保存失败")

@app.get("/audit/{request_id}")
async def get_audit_record(request_id: str):
    """查询审计记录"""
    global audit_db_conn
    
    try:
        cursor = audit_db_conn.execute(
            "SELECT * FROM audit_records WHERE request_id = ? ORDER BY created_at DESC",
            (request_id,)
        )
        
        records = []
        for row in cursor.fetchall():
            record = {
                'audit_id': row[0],
                'request_id': row[1],
                'consent_fingerprint': row[2],
                'model_hash': row[3],
                'threshold': row[4],
                'policy_version': row[5],
                'timestamp': row[6],
                'decision': row[7],
                'score': row[8],
                'metadata': json.loads(row[9]) if row[9] else {},
                'created_at': row[10]
            }
            records.append(record)
        
        return {
            'request_id': request_id,
            'records': records,
            'total': len(records)
        }
        
    except Exception as e:
        logger.error(f"审计记录查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail="审计记录查询失败")

@app.get("/audit/subject/{subject}")
async def get_audit_by_subject(subject: str, limit: int = 100):
    """按主体查询审计记录"""
    global audit_db_conn
    
    try:
        # 通过consent_fingerprint关联查询
        cursor = audit_db_conn.execute(
            "SELECT * FROM audit_records ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        
        records = []
        for row in cursor.fetchall():
            record = {
                'audit_id': row[0],
                'request_id': row[1],
                'consent_fingerprint': row[2],
                'model_hash': row[3],
                'threshold': row[4],
                'policy_version': row[5],
                'timestamp': row[6],
                'decision': row[7],
                'score': row[8],
                'metadata': json.loads(row[9]) if row[9] else {},
                'created_at': row[10]
            }
            records.append(record)
        
        return {
            'subject': subject,
            'records': records,
            'total': len(records)
        }
        
    except Exception as e:
        logger.error(f"主体审计记录查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail="主体审计记录查询失败")

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    global casbin_enforcer, audit_db_conn
    
    casbin_status = "healthy" if casbin_enforcer else "error"
    audit_db_status = "healthy" if audit_db_conn else "error"
    
    overall_status = "healthy" if casbin_status == "healthy" and audit_db_status == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        casbin_status=casbin_status,
        audit_db_status=audit_db_status
    )

@app.get("/policies")
async def get_policies():
    """获取当前策略"""
    global casbin_enforcer
    
    if not casbin_enforcer:
        raise HTTPException(status_code=500, detail="策略引擎未初始化")
    
    try:
        policies = casbin_enforcer.get_policy()
        grouping_policies = casbin_enforcer.get_grouping_policy()
        
        return {
            'policies': policies,
            'grouping_policies': grouping_policies,
            'total_policies': len(policies),
            'total_grouping_policies': len(grouping_policies)
        }
    except Exception as e:
        logger.error(f"获取策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取策略失败")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7002))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )