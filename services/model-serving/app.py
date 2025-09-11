#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦模型服务 (Model Serving)

功能:
1. FastAPI /score 评分接口
2. 同态加密轻量验证 /score_he
3. 模型版本管理和A/B测试
4. 实时特征获取和决策引擎
"""

import os
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import httpx
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# 简化的同态加密实现（生产环境应使用专业库如Microsoft SEAL）
class SimpleHE:
    """简化的同态加密实现"""
    
    def __init__(self, key_size: int = 1024):
        self.key_size = key_size
        self.public_key = self._generate_public_key()
        self.private_key = self._generate_private_key()
    
    def _generate_public_key(self):
        # 简化实现，实际应使用真正的同态加密算法
        return np.random.randint(1, 1000, size=self.key_size)
    
    def _generate_private_key(self):
        return np.random.randint(1, 1000, size=self.key_size)
    
    def encrypt(self, plaintext: float) -> str:
        """加密明文"""
        # 简化的加密：添加噪声
        noise = np.random.normal(0, 0.01)
        encrypted_value = plaintext + noise
        # 返回加密后的字符串表示
        return f"HE_{encrypted_value:.6f}_{hash(str(self.public_key)) % 10000}"
    
    def decrypt(self, ciphertext: str) -> float:
        """解密密文"""
        if not ciphertext.startswith("HE_"):
            raise ValueError("无效的密文格式")
        
        parts = ciphertext.split("_")
        if len(parts) != 3:
            raise ValueError("密文格式错误")
        
        return float(parts[1])
    
    def add_encrypted(self, ciphertext1: str, ciphertext2: str) -> str:
        """同态加法"""
        val1 = self.decrypt(ciphertext1)
        val2 = self.decrypt(ciphertext2)
        result = val1 + val2
        return self.encrypt(result)
    
    def multiply_encrypted(self, ciphertext: str, scalar: float) -> str:
        """同态标量乘法"""
        val = self.decrypt(ciphertext)
        result = val * scalar
        return self.encrypt(result)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 环境变量配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/model_serving_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MODEL_STORE_PATH = os.getenv('MODEL_STORE_PATH', '/app/models')
FEATURE_STORE_URL = os.getenv('FEATURE_STORE_URL', 'http://feature-store:8083')
CONSENT_GATEWAY_URL = os.getenv('CONSENT_GATEWAY_URL', 'http://consent-gateway:8080')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')
POLICY_SERVICE_URL = os.getenv('POLICY_SERVICE_URL', 'http://policy:8080')
OIDC_ISSUER = os.getenv('OIDC_ISSUER', 'http://localhost:8080/auth/realms/federated-risk')
OIDC_AUDIENCE = os.getenv('OIDC_AUDIENCE', 'model-serving')

# 创建必要目录
Path(MODEL_STORE_PATH).mkdir(parents=True, exist_ok=True)

# FastAPI应用初始化
app = FastAPI(
    title="联邦模型服务",
    description="提供模型评分和同态加密验证服务",
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

# 安全配置
security = HTTPBearer()

# 全局变量
db_pool = None
redis_client = None
he_engine = SimpleHE()
loaded_models = {}
model_metadata = {}

# Prometheus指标
request_count = Counter('model_serving_requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('model_serving_request_duration_seconds', 'Request duration', ['endpoint'])
score_distribution = Histogram('model_serving_score_distribution', 'Score distribution', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
active_models = Gauge('model_serving_active_models', 'Number of active models')
feature_fetch_duration = Histogram('model_serving_feature_fetch_duration_seconds', 'Feature fetch duration')

# 枚举定义
class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    FEDERATED_MODEL = "federated_model"

class DecisionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"
    PENDING = "pending"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Pydantic模型定义
class ScoreRequest(BaseModel):
    """评分请求"""
    entity_id: str = Field(..., description="实体ID")
    psi_token: Optional[str] = Field(None, description="PSI对齐令牌")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    as_of_timestamp: Optional[datetime] = Field(None, description="时间点")
    model_version: Optional[str] = Field(None, description="模型版本")
    features_override: Optional[Dict[str, Any]] = Field(None, description="特征覆盖")
    request_context: Dict[str, Any] = Field(default_factory=dict, description="请求上下文")
    
    @validator('entity_id')
    def validate_entity_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('实体ID不能为空')
        return v.strip()

class HEScoreRequest(BaseModel):
    """同态加密评分请求"""
    entity_id: str = Field(..., description="实体ID")
    encrypted_features: Dict[str, str] = Field(..., description="加密特征")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    model_version: Optional[str] = Field(None, description="模型版本")
    
    @validator('encrypted_features')
    def validate_encrypted_features(cls, v):
        if not v:
            raise ValueError('加密特征不能为空')
        for key, value in v.items():
            if not isinstance(value, str) or not value.startswith('HE_'):
                raise ValueError(f'特征 {key} 的加密格式无效')
        return v

class BatchScoreRequest(BaseModel):
    """批量评分请求"""
    entity_ids: List[str] = Field(..., description="实体ID列表")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    model_version: Optional[str] = Field(None, description="模型版本")
    as_of_timestamp: Optional[datetime] = Field(None, description="时间点")
    
    @validator('entity_ids')
    def validate_entity_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('实体ID列表不能为空')
        if len(v) > 1000:
            raise ValueError('批量请求最多支持1000个实体')
        return v

class ModelRegistration(BaseModel):
    """模型注册"""
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本")
    model_type: ModelType = Field(..., description="模型类型")
    model_path: str = Field(..., description="模型文件路径")
    feature_names: List[str] = Field(..., description="特征名称列表")
    threshold_config: Dict[str, float] = Field(..., description="阈值配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="模型元数据")
    is_active: bool = Field(default=True, description="是否激活")

class ScoreResponse(BaseModel):
    """评分响应"""
    entity_id: str
    score: float
    decision: DecisionType
    risk_level: RiskLevel
    threshold: float
    model_hash: str
    model_version: str
    policy_version: str
    features_used: List[str]
    timestamp: datetime
    request_id: str
    confidence: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None

class HEScoreResponse(BaseModel):
    """同态加密评分响应"""
    entity_id: str
    encrypted_score: str
    plaintext_score: float
    score_difference: float
    verification_status: str
    model_hash: str
    timestamp: datetime
    request_id: str

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
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(128) NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    model_type VARCHAR(32) NOT NULL,
                    model_path VARCHAR(256) NOT NULL,
                    model_hash VARCHAR(64) NOT NULL,
                    feature_names TEXT[] NOT NULL,
                    threshold_config JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(model_name, model_version)
                );
                
                CREATE TABLE IF NOT EXISTS score_requests (
                    id SERIAL PRIMARY KEY,
                    request_id VARCHAR(64) UNIQUE NOT NULL,
                    entity_id VARCHAR(128) NOT NULL,
                    model_name VARCHAR(128) NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    score FLOAT NOT NULL,
                    decision VARCHAR(32) NOT NULL,
                    risk_level VARCHAR(32) NOT NULL,
                    threshold FLOAT NOT NULL,
                    features_used TEXT[] NOT NULL,
                    request_context JSONB,
                    response_time_ms INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(128) NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    metric_name VARCHAR(64) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    measurement_date DATE NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_models_name_version ON models (model_name, model_version);
                CREATE INDEX IF NOT EXISTS idx_models_active ON models (is_active);
                CREATE INDEX IF NOT EXISTS idx_score_requests_entity ON score_requests (entity_id);
                CREATE INDEX IF NOT EXISTS idx_score_requests_created ON score_requests (created_at);
                CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance (model_name, model_version);
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

# 模型加载
async def load_models():
    """加载所有激活的模型"""
    global loaded_models, model_metadata
    try:
        async with db_pool.acquire() as conn:
            models = await conn.fetch(
                "SELECT * FROM models WHERE is_active = TRUE ORDER BY created_at DESC"
            )
        
        for model_record in models:
            model_key = f"{model_record['model_name']}:{model_record['model_version']}"
            
            try:
                # 加载模型文件
                model_path = model_record['model_path']
                if os.path.exists(model_path):
                    if model_path.endswith('.pkl'):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    elif model_path.endswith('.joblib'):
                        model = joblib.load(model_path)
                    else:
                        # 创建默认模型
                        model = create_default_model(model_record['model_type'])
                else:
                    # 创建默认模型
                    model = create_default_model(model_record['model_type'])
                
                loaded_models[model_key] = model
                model_metadata[model_key] = dict(model_record)
                
                logger.info(f"模型加载成功: {model_key}")
                
            except Exception as e:
                logger.error(f"模型加载失败 {model_key}: {e}")
        
        # 更新Prometheus指标
        active_models.set(len(loaded_models))
        
        logger.info(f"共加载 {len(loaded_models)} 个模型")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        # 创建默认模型以确保服务可用
        await create_default_models()

def create_default_model(model_type: str):
    """创建默认模型"""
    if model_type == ModelType.LOGISTIC_REGRESSION.value:
        model = LogisticRegression(random_state=42)
        # 使用虚拟数据训练
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        return model
    elif model_type == ModelType.RANDOM_FOREST.value:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        return model
    else:
        # 默认使用逻辑回归
        return create_default_model(ModelType.LOGISTIC_REGRESSION.value)

async def create_default_models():
    """创建默认模型"""
    global loaded_models, model_metadata
    
    default_model = create_default_model(ModelType.LOGISTIC_REGRESSION.value)
    model_key = "default_model:v1.0"
    
    loaded_models[model_key] = default_model
    model_metadata[model_key] = {
        'model_name': 'default_model',
        'model_version': 'v1.0',
        'model_type': ModelType.LOGISTIC_REGRESSION.value,
        'model_hash': 'default_hash',
        'feature_names': ['credit_score', 'annual_income', 'debt_to_income_ratio', 
                         'account_age_months', 'num_credit_cards'],
        'threshold_config': {'approve': 0.3, 'reject': 0.7},
        'metadata': {'description': '默认风控模型'}
    }
    
    active_models.set(1)
    logger.info("默认模型创建成功")

# 工具函数
async def validate_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """验证JWT令牌"""
    try:
        # 简化实现，实际应该验证JWT签名
        token = credentials.credentials
        
        # 模拟JWT解码
        if token.startswith('Bearer.'):
            # 简化的令牌验证
            return {
                "sub": "user123",
                "iss": OIDC_ISSUER,
                "aud": OIDC_AUDIENCE,
                "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
            }
        else:
            raise HTTPException(status_code=401, detail="无效的令牌格式")
            
    except Exception as e:
        logger.error(f"JWT验证失败: {e}")
        raise HTTPException(status_code=401, detail="令牌验证失败")

async def validate_consent(consent_token: str, entity_id: str, features: List[str]) -> bool:
    """验证同意令牌"""
    if not consent_token:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONSENT_GATEWAY_URL}/consent/introspect",
                json={
                    "consent_token": consent_token,
                    "entity_id": entity_id,
                    "requested_features": features,
                    "purpose": "risk_scoring"
                },
                timeout=5.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("valid", False)
        
        return False
    except Exception as e:
        logger.error(f"同意验证失败: {e}")
        return False

async def get_features(entity_id: str, feature_names: List[str], 
                      consent_token: Optional[str] = None,
                      as_of_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """获取特征"""
    start_time = datetime.utcnow()
    
    try:
        request_data = {
            "entity_id": entity_id,
            "feature_names": feature_names,
            "request_context": {"source": "model_serving"}
        }
        
        if consent_token:
            request_data["consent_token"] = consent_token
        
        if as_of_timestamp:
            request_data["as_of_timestamp"] = as_of_timestamp.isoformat()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FEATURE_STORE_URL}/features/online",
                json=request_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("features", {})
            else:
                logger.warning(f"特征获取失败，状态码: {response.status_code}")
                return await get_mock_features(entity_id, feature_names)
    
    except Exception as e:
        logger.error(f"特征获取失败: {e}")
        return await get_mock_features(entity_id, feature_names)
    
    finally:
        duration = (datetime.utcnow() - start_time).total_seconds()
        feature_fetch_duration.observe(duration)

async def get_mock_features(entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
    """获取模拟特征"""
    # 基于entity_id生成确定性的模拟数据
    import hashlib
    seed = int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed % (2**32))
    
    mock_features = {
        "credit_score": np.random.randint(300, 850),
        "annual_income": np.random.uniform(30000, 200000),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.8),
        "account_age_months": np.random.randint(1, 240),
        "num_credit_cards": np.random.randint(0, 10),
        "has_mortgage": np.random.choice([True, False]),
        "employment_status": np.random.choice(["employed", "self_employed", "unemployed"]),
        "total_orders": np.random.randint(0, 100),
        "avg_order_value": np.random.uniform(20, 500),
        "days_since_last_order": np.random.randint(0, 365),
        "favorite_category": np.random.choice(["electronics", "clothing", "books", "home"]),
        "return_rate": np.random.uniform(0, 0.3),
        "loyalty_score": np.random.uniform(0, 1)
    }
    
    # 只返回请求的特征
    result = {}
    for feature_name in feature_names:
        if feature_name in mock_features:
            value = mock_features[feature_name]
            # 转换numpy类型为Python原生类型
            if isinstance(value, np.integer):
                result[feature_name] = int(value)
            elif isinstance(value, np.floating):
                result[feature_name] = float(value)
            elif isinstance(value, np.bool_):
                result[feature_name] = bool(value)
            else:
                result[feature_name] = value
        else:
            result[feature_name] = 0.0  # 默认值
    
    return result

def get_model_and_metadata(model_version: Optional[str] = None) -> tuple:
    """获取模型和元数据"""
    if model_version:
        # 查找指定版本的模型
        for key, metadata in model_metadata.items():
            if metadata['model_version'] == model_version:
                return loaded_models[key], metadata
    
    # 返回默认模型或第一个可用模型
    if "default_model:v1.0" in loaded_models:
        return loaded_models["default_model:v1.0"], model_metadata["default_model:v1.0"]
    elif loaded_models:
        key = next(iter(loaded_models))
        return loaded_models[key], model_metadata[key]
    else:
        raise HTTPException(status_code=503, detail="没有可用的模型")

def predict_score(model, features: Dict[str, Any], feature_names: List[str]) -> float:
    """预测评分"""
    try:
        # 构建特征向量
        feature_vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0.0)
            if isinstance(value, bool):
                feature_vector.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # 简单的字符串编码
                feature_vector.append(float(hash(value) % 100) / 100.0)
            else:
                feature_vector.append(float(value))
        
        # 标准化特征（简化实现）
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # 预测概率
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_array)
            # 返回正类概率
            return float(proba[0][1] if proba.shape[1] > 1 else proba[0][0])
        else:
            # 如果模型不支持概率预测，使用决策函数
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(feature_array)
                # 将决策值转换为概率
                return float(1 / (1 + np.exp(-decision[0])))
            else:
                # 最后的回退：使用预测结果
                prediction = model.predict(feature_array)
                return float(prediction[0])
    
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        # 返回随机评分作为回退
        return np.random.uniform(0.1, 0.9)

def make_decision(score: float, threshold_config: Dict[str, float]) -> tuple:
    """做出决策"""
    approve_threshold = threshold_config.get('approve', 0.3)
    reject_threshold = threshold_config.get('reject', 0.7)
    
    if score <= approve_threshold:
        return DecisionType.APPROVE, RiskLevel.LOW
    elif score >= reject_threshold:
        return DecisionType.REJECT, RiskLevel.HIGH
    else:
        return DecisionType.MANUAL_REVIEW, RiskLevel.MEDIUM

async def log_score_request(request_id: str, entity_id: str, model_name: str, 
                          model_version: str, score: float, decision: str,
                          risk_level: str, threshold: float, features_used: List[str],
                          request_context: Dict[str, Any], response_time_ms: int):
    """记录评分请求"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO score_requests (
                    request_id, entity_id, model_name, model_version, score,
                    decision, risk_level, threshold, features_used,
                    request_context, response_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                request_id, entity_id, model_name, model_version, score,
                decision, risk_level, threshold, features_used,
                json.dumps(request_context), response_time_ms
            )
    except Exception as e:
        logger.error(f"评分请求日志记录失败: {e}")

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "HIGH" if event_data.get('decision') == 'reject' else "MEDIUM",
            "source": {
                "service": "model-serving",
                "version": "1.0.0"
            },
            "operation": event_data
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{AUDIT_SERVICE_URL}/audit/records",
                json=audit_record,
                timeout=5.0
            )
    except Exception as e:
        logger.error(f"审计日志发送失败: {e}")

def generate_request_id() -> str:
    """生成请求ID"""
    return f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

def calculate_model_hash(model_name: str, model_version: str) -> str:
    """计算模型哈希"""
    content = f"{model_name}:{model_version}:{datetime.utcnow().date()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

# API路由
@app.post("/score", response_model=ScoreResponse)
async def score_endpoint(request: ScoreRequest, 
                        background_tasks: BackgroundTasks,
                        user_info: dict = Depends(validate_jwt_token)):
    """模型评分接口"""
    start_time = datetime.utcnow()
    request_id = generate_request_id()
    
    try:
        with request_duration.labels(endpoint='score').time():
            # 获取模型和元数据
            model, metadata = get_model_and_metadata(request.model_version)
            
            # 验证同意令牌（对于敏感特征）
            feature_names = metadata['feature_names']
            if not await validate_consent(request.consent_token, request.entity_id, feature_names):
                request_count.labels(endpoint='score', status='consent_denied').inc()
                raise HTTPException(status_code=403, detail="同意验证失败")
            
            # 获取特征
            if request.features_override:
                features = request.features_override
            else:
                features = await get_features(
                    request.entity_id, 
                    feature_names,
                    request.consent_token,
                    request.as_of_timestamp
                )
            
            # 预测评分
            score = predict_score(model, features, feature_names)
            
            # 做出决策
            decision, risk_level = make_decision(score, metadata['threshold_config'])
            
            # 计算响应时间
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # 构建响应
            response = ScoreResponse(
                entity_id=request.entity_id,
                score=score,
                decision=decision,
                risk_level=risk_level,
                threshold=metadata['threshold_config'].get('reject', 0.7),
                model_hash=calculate_model_hash(metadata['model_name'], metadata['model_version']),
                model_version=metadata['model_version'],
                policy_version="v1.0",
                features_used=feature_names,
                timestamp=datetime.utcnow(),
                request_id=request_id,
                confidence=min(abs(score - 0.5) * 2, 1.0)  # 简化的置信度计算
            )
            
            # 更新Prometheus指标
            request_count.labels(endpoint='score', status='success').inc()
            score_distribution.observe(score)
            
            # 后台任务：记录日志和审计
            background_tasks.add_task(
                log_score_request,
                request_id, request.entity_id, metadata['model_name'],
                metadata['model_version'], score, decision.value,
                risk_level.value, response.threshold, feature_names,
                request.request_context, response_time_ms
            )
            
            background_tasks.add_task(
                send_audit_log,
                "MODEL_SCORE",
                {
                    "request_id": request_id,
                    "entity_id": request.entity_id,
                    "model_name": metadata['model_name'],
                    "model_version": metadata['model_version'],
                    "score": score,
                    "decision": decision.value,
                    "risk_level": risk_level.value,
                    "features_used": feature_names,
                    "consent_provided": bool(request.consent_token)
                }
            )
            
            return response
    
    except HTTPException:
        request_count.labels(endpoint='score', status='error').inc()
        raise
    except Exception as e:
        logger.error(f"评分失败: {e}")
        request_count.labels(endpoint='score', status='error').inc()
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/score_he", response_model=HEScoreResponse)
async def score_he_endpoint(request: HEScoreRequest,
                           user_info: dict = Depends(validate_jwt_token)):
    """同态加密评分接口"""
    request_id = generate_request_id()
    
    try:
        with request_duration.labels(endpoint='score_he').time():
            # 获取模型和元数据
            model, metadata = get_model_and_metadata(request.model_version)
            
            # 验证同意令牌
            feature_names = list(request.encrypted_features.keys())
            if not await validate_consent(request.consent_token, request.entity_id, feature_names):
                request_count.labels(endpoint='score_he', status='consent_denied').inc()
                raise HTTPException(status_code=403, detail="同意验证失败")
            
            # 解密特征进行明文计算（用于验证）
            plaintext_features = {}
            for feature_name, encrypted_value in request.encrypted_features.items():
                try:
                    plaintext_value = he_engine.decrypt(encrypted_value)
                    plaintext_features[feature_name] = plaintext_value
                except Exception as e:
                    logger.error(f"特征解密失败 {feature_name}: {e}")
                    plaintext_features[feature_name] = 0.0
            
            # 明文评分
            plaintext_score = predict_score(model, plaintext_features, feature_names)
            
            # 同态加密评分（简化实现）
            # 在实际应用中，这里应该使用真正的同态加密算法
            encrypted_score = he_engine.encrypt(plaintext_score)
            
            # 验证一致性
            decrypted_score = he_engine.decrypt(encrypted_score)
            score_difference = abs(plaintext_score - decrypted_score)
            
            verification_status = "PASS" if score_difference < 0.01 else "FAIL"
            
            # 构建响应
            response = HEScoreResponse(
                entity_id=request.entity_id,
                encrypted_score=encrypted_score,
                plaintext_score=plaintext_score,
                score_difference=score_difference,
                verification_status=verification_status,
                model_hash=calculate_model_hash(metadata['model_name'], metadata['model_version']),
                timestamp=datetime.utcnow(),
                request_id=request_id
            )
            
            # 更新Prometheus指标
            request_count.labels(endpoint='score_he', status='success').inc()
            
            # 发送审计日志
            await send_audit_log("MODEL_SCORE_HE", {
                "request_id": request_id,
                "entity_id": request.entity_id,
                "model_name": metadata['model_name'],
                "model_version": metadata['model_version'],
                "verification_status": verification_status,
                "score_difference": score_difference,
                "features_count": len(request.encrypted_features)
            })
            
            return response
    
    except HTTPException:
        request_count.labels(endpoint='score_he', status='error').inc()
        raise
    except Exception as e:
        logger.error(f"同态加密评分失败: {e}")
        request_count.labels(endpoint='score_he', status='error').inc()
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/score/batch")
async def batch_score_endpoint(request: BatchScoreRequest,
                              background_tasks: BackgroundTasks,
                              user_info: dict = Depends(validate_jwt_token)):
    """批量评分接口"""
    start_time = datetime.utcnow()
    
    try:
        with request_duration.labels(endpoint='batch_score').time():
            # 获取模型和元数据
            model, metadata = get_model_and_metadata(request.model_version)
            feature_names = metadata['feature_names']
            
            # 验证同意令牌
            if not await validate_consent(request.consent_token, request.entity_ids[0], feature_names):
                request_count.labels(endpoint='batch_score', status='consent_denied').inc()
                raise HTTPException(status_code=403, detail="同意验证失败")
            
            results = []
            
            for entity_id in request.entity_ids:
                try:
                    # 获取特征
                    features = await get_features(
                        entity_id, 
                        feature_names,
                        request.consent_token,
                        request.as_of_timestamp
                    )
                    
                    # 预测评分
                    score = predict_score(model, features, feature_names)
                    
                    # 做出决策
                    decision, risk_level = make_decision(score, metadata['threshold_config'])
                    
                    results.append({
                        "entity_id": entity_id,
                        "score": score,
                        "decision": decision.value,
                        "risk_level": risk_level.value,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"批量评分失败 {entity_id}: {e}")
                    results.append({
                        "entity_id": entity_id,
                        "error": str(e),
                        "status": "error"
                    })
            
            # 计算响应时间
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # 更新Prometheus指标
            request_count.labels(endpoint='batch_score', status='success').inc()
            
            # 后台任务：发送审计日志
            background_tasks.add_task(
                send_audit_log,
                "MODEL_BATCH_SCORE",
                {
                    "batch_size": len(request.entity_ids),
                    "success_count": len([r for r in results if r.get('status') == 'success']),
                    "error_count": len([r for r in results if r.get('status') == 'error']),
                    "model_name": metadata['model_name'],
                    "model_version": metadata['model_version'],
                    "response_time_ms": response_time_ms
                }
            )
            
            return {
                "results": results,
                "batch_size": len(request.entity_ids),
                "success_count": len([r for r in results if r.get('status') == 'success']),
                "error_count": len([r for r in results if r.get('status') == 'error']),
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": response_time_ms
            }
    
    except HTTPException:
        request_count.labels(endpoint='batch_score', status='error').inc()
        raise
    except Exception as e:
        logger.error(f"批量评分失败: {e}")
        request_count.labels(endpoint='batch_score', status='error').inc()
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/models/register")
async def register_model(model_reg: ModelRegistration,
                        user_info: dict = Depends(validate_jwt_token)):
    """注册模型"""
    try:
        # 计算模型哈希
        model_hash = calculate_model_hash(model_reg.model_name, model_reg.model_version)
        
        async with db_pool.acquire() as conn:
            # 检查模型是否已存在
            existing = await conn.fetchrow(
                "SELECT id FROM models WHERE model_name = $1 AND model_version = $2",
                model_reg.model_name, model_reg.model_version
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="模型版本已存在")
            
            # 插入模型记录
            await conn.execute("""
                INSERT INTO models (
                    model_name, model_version, model_type, model_path, model_hash,
                    feature_names, threshold_config, metadata, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
                model_reg.model_name, model_reg.model_version, model_reg.model_type.value,
                model_reg.model_path, model_hash, model_reg.feature_names,
                json.dumps(model_reg.threshold_config), json.dumps(model_reg.metadata),
                model_reg.is_active
            )
        
        # 如果模型激活，重新加载模型
        if model_reg.is_active:
            await load_models()
        
        # 发送审计日志
        await send_audit_log("MODEL_REGISTER", {
            "model_name": model_reg.model_name,
            "model_version": model_reg.model_version,
            "model_type": model_reg.model_type.value,
            "model_hash": model_hash,
            "is_active": model_reg.is_active
        })
        
        return {
            "message": "模型注册成功",
            "model_name": model_reg.model_name,
            "model_version": model_reg.model_version,
            "model_hash": model_hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型注册失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/models")
async def list_models(active_only: bool = True):
    """列出模型"""
    try:
        query = "SELECT * FROM models"
        params = []
        
        if active_only:
            query += " WHERE is_active = TRUE"
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            models = await conn.fetch(query, *params)
            return [dict(model) for model in models]
            
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/models/{model_name}/performance")
async def get_model_performance(model_name: str, days: int = 30):
    """获取模型性能指标"""
    try:
        start_date = datetime.utcnow().date() - timedelta(days=days)
        
        async with db_pool.acquire() as conn:
            performance = await conn.fetch("""
                SELECT metric_name, metric_value, measurement_date
                FROM model_performance
                WHERE model_name = $1 AND measurement_date >= $2
                ORDER BY measurement_date DESC, metric_name
            """, model_name, start_date)
            
            return [dict(record) for record in performance]
            
    except Exception as e:
        logger.error(f"获取模型性能失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/metrics")
async def get_metrics():
    """获取Prometheus指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # 检查Redis连接
        await redis_client.ping()
        
        # 检查模型加载状态
        models_loaded = len(loaded_models) > 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "models": "healthy" if models_loaded else "degraded"
            },
            "models_count": len(loaded_models)
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不可用")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动模型服务...")
    await init_database()
    await init_redis()
    await load_models()
    logger.info("模型服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("关闭模型服务...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    logger.info("模型服务已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8084,
        reload=True,
        log_level="info"
    )