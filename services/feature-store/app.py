#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦特征存储 (Feature Store)

功能:
1. Feast特征存储集成
2. 离线和在线特征一致性
3. 特征版本管理和血缘追踪
4. 特征安全和隐私保护
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
from feast import FeatureStore, Entity, FeatureView, Field, FileSource, RedisSource
from feast.types import Float32, Float64, Int32, Int64, String, Bool, UnixTimestamp

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String as SQLString, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 环境变量配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/feature_store_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
FEAST_REPO_PATH = os.getenv('FEAST_REPO_PATH', '/app/feast_repo')
OFFLINE_STORE_PATH = os.getenv('OFFLINE_STORE_PATH', '/app/data/offline')
CONSENT_GATEWAY_URL = os.getenv('CONSENT_GATEWAY_URL', 'http://consent-gateway:8080')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')

# 创建必要目录
Path(FEAST_REPO_PATH).mkdir(parents=True, exist_ok=True)
Path(OFFLINE_STORE_PATH).mkdir(parents=True, exist_ok=True)

# FastAPI应用初始化
app = FastAPI(
    title="联邦特征存储",
    description="基于Feast的特征存储服务，支持离在线一致性",
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
feature_store = None
sql_engine = None

# 枚举定义
class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    TEXT = "text"

class DataSource(str, Enum):
    BANK = "bank"
    ECOMMERCE = "ecommerce"
    TELECOM = "telecom"
    EXTERNAL = "external"

class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

# Pydantic模型定义
class FeatureDefinition(BaseModel):
    """特征定义"""
    feature_name: str = Field(..., description="特征名称")
    feature_type: FeatureType = Field(..., description="特征类型")
    data_source: DataSource = Field(..., description="数据源")
    description: str = Field(..., description="特征描述")
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.INTERNAL, description="隐私级别")
    owner: str = Field(..., description="特征负责人")
    tags: List[str] = Field(default_factory=list, description="特征标签")
    
    @validator('feature_name')
    def validate_feature_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('特征名称只能包含字母、数字和下划线')
        return v

class FeatureViewDefinition(BaseModel):
    """特征视图定义"""
    view_name: str = Field(..., description="视图名称")
    entity_name: str = Field(..., description="实体名称")
    features: List[FeatureDefinition] = Field(..., description="特征列表")
    source_table: str = Field(..., description="源表名称")
    timestamp_field: str = Field(default="event_timestamp", description="时间戳字段")
    ttl_days: int = Field(default=30, ge=1, le=365, description="TTL天数")
    batch_source_config: Dict[str, Any] = Field(default_factory=dict, description="批处理源配置")
    stream_source_config: Dict[str, Any] = Field(default_factory=dict, description="流处理源配置")

class FeatureRequest(BaseModel):
    """特征请求"""
    entity_id: str = Field(..., description="实体ID")
    feature_names: List[str] = Field(..., description="特征名称列表")
    as_of_timestamp: Optional[datetime] = Field(None, description="时间点")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    request_context: Dict[str, Any] = Field(default_factory=dict, description="请求上下文")

class BatchFeatureRequest(BaseModel):
    """批量特征请求"""
    entity_ids: List[str] = Field(..., description="实体ID列表")
    feature_names: List[str] = Field(..., description="特征名称列表")
    start_timestamp: datetime = Field(..., description="开始时间")
    end_timestamp: datetime = Field(..., description="结束时间")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    
    @validator('end_timestamp')
    def validate_timestamps(cls, v, values):
        if 'start_timestamp' in values and v <= values['start_timestamp']:
            raise ValueError('结束时间必须大于开始时间')
        return v

class FeatureIngestionRequest(BaseModel):
    """特征摄取请求"""
    view_name: str = Field(..., description="视图名称")
    data: List[Dict[str, Any]] = Field(..., description="特征数据")
    timestamp_field: str = Field(default="event_timestamp", description="时间戳字段")
    entity_field: str = Field(default="entity_id", description="实体字段")
    source_info: Dict[str, Any] = Field(default_factory=dict, description="数据源信息")

# 数据库初始化
async def init_database():
    """初始化数据库连接池"""
    global db_pool, sql_engine
    try:
        # AsyncPG连接池
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # SQLAlchemy引擎（用于Feast）
        sql_engine = create_engine(DATABASE_URL)
        
        # 创建表结构
        async with db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_definitions (
                    id SERIAL PRIMARY KEY,
                    feature_name VARCHAR(128) UNIQUE NOT NULL,
                    feature_type VARCHAR(32) NOT NULL,
                    data_source VARCHAR(32) NOT NULL,
                    description TEXT,
                    privacy_level VARCHAR(32) NOT NULL,
                    owner VARCHAR(64) NOT NULL,
                    tags TEXT[],
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS feature_views (
                    id SERIAL PRIMARY KEY,
                    view_name VARCHAR(128) UNIQUE NOT NULL,
                    entity_name VARCHAR(128) NOT NULL,
                    source_table VARCHAR(128) NOT NULL,
                    timestamp_field VARCHAR(64) NOT NULL,
                    ttl_days INTEGER NOT NULL,
                    config JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    id SERIAL PRIMARY KEY,
                    feature_name VARCHAR(128) NOT NULL,
                    source_table VARCHAR(128) NOT NULL,
                    transformation_logic TEXT,
                    upstream_features TEXT[],
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS feature_access_log (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(128) NOT NULL,
                    feature_names TEXT[] NOT NULL,
                    request_timestamp TIMESTAMPTZ NOT NULL,
                    consent_token VARCHAR(256),
                    request_context JSONB,
                    response_status VARCHAR(32) NOT NULL,
                    privacy_level VARCHAR(32) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_feature_definitions_name ON feature_definitions (feature_name);
                CREATE INDEX IF NOT EXISTS idx_feature_views_name ON feature_views (view_name);
                CREATE INDEX IF NOT EXISTS idx_feature_lineage_name ON feature_lineage (feature_name);
                CREATE INDEX IF NOT EXISTS idx_feature_access_log_entity ON feature_access_log (entity_id);
                CREATE INDEX IF NOT EXISTS idx_feature_access_log_timestamp ON feature_access_log (request_timestamp);
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

# Feast初始化
async def init_feast():
    """初始化Feast特征存储"""
    global feature_store
    try:
        # 创建Feast配置文件
        feast_config = {
            "project": "federated_risk_demo",
            "registry": f"{FEAST_REPO_PATH}/data/registry.db",
            "provider": "local",
            "offline_store": {
                "type": "file"
            },
            "online_store": {
                "type": "redis",
                "connection_string": REDIS_URL
            },
            "entity_key_serialization_version": 2
        }
        
        # 写入feature_store.yaml
        config_path = f"{FEAST_REPO_PATH}/feature_store.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(feast_config, f)
        
        # 初始化Feast存储
        os.chdir(FEAST_REPO_PATH)
        feature_store = FeatureStore(repo_path=FEAST_REPO_PATH)
        
        # 创建基础实体和特征视图
        await create_default_entities_and_views()
        
        logger.info("Feast特征存储初始化成功")
    except Exception as e:
        logger.error(f"Feast初始化失败: {e}")
        raise

async def create_default_entities_and_views():
    """创建默认实体和特征视图"""
    try:
        # 创建客户实体
        customer_entity = Entity(
            name="customer",
            description="客户实体",
            join_keys=["customer_id"]
        )
        
        # 创建银行特征视图
        bank_features_source = FileSource(
            name="bank_features_source",
            path=f"{OFFLINE_STORE_PATH}/bank_features.parquet",
            timestamp_field="event_timestamp"
        )
        
        bank_features_view = FeatureView(
            name="bank_features",
            entities=[customer_entity],
            ttl=timedelta(days=30),
            schema=[
                Field(name="credit_score", dtype=Int32),
                Field(name="annual_income", dtype=Float64),
                Field(name="debt_to_income_ratio", dtype=Float32),
                Field(name="account_age_months", dtype=Int32),
                Field(name="num_credit_cards", dtype=Int32),
                Field(name="has_mortgage", dtype=Bool),
                Field(name="employment_status", dtype=String),
            ],
            source=bank_features_source,
            tags={"team": "risk", "source": "bank"}
        )
        
        # 创建电商特征视图
        ecommerce_features_source = FileSource(
            name="ecommerce_features_source",
            path=f"{OFFLINE_STORE_PATH}/ecommerce_features.parquet",
            timestamp_field="event_timestamp"
        )
        
        ecommerce_features_view = FeatureView(
            name="ecommerce_features",
            entities=[customer_entity],
            ttl=timedelta(days=30),
            schema=[
                Field(name="total_orders", dtype=Int32),
                Field(name="avg_order_value", dtype=Float64),
                Field(name="days_since_last_order", dtype=Int32),
                Field(name="favorite_category", dtype=String),
                Field(name="return_rate", dtype=Float32),
                Field(name="loyalty_score", dtype=Float32),
            ],
            source=ecommerce_features_source,
            tags={"team": "growth", "source": "ecommerce"}
        )
        
        # 应用到Feast
        feature_store.apply([customer_entity, bank_features_view, ecommerce_features_view])
        
        logger.info("默认实体和特征视图创建成功")
        
    except Exception as e:
        logger.error(f"创建默认实体和特征视图失败: {e}")
        # 不抛出异常，允许服务继续运行

# 工具函数
async def validate_consent(consent_token: str, feature_names: List[str]) -> bool:
    """验证同意令牌"""
    if not consent_token:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONSENT_GATEWAY_URL}/consent/introspect",
                json={
                    "consent_token": consent_token,
                    "requested_features": feature_names
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

async def log_feature_access(entity_id: str, feature_names: List[str], 
                           consent_token: Optional[str], request_context: Dict[str, Any],
                           response_status: str, privacy_level: str):
    """记录特征访问日志"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO feature_access_log (
                    entity_id, feature_names, request_timestamp, consent_token,
                    request_context, response_status, privacy_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                entity_id, feature_names, datetime.utcnow(), consent_token,
                json.dumps(request_context), response_status, privacy_level
            )
    except Exception as e:
        logger.error(f"特征访问日志记录失败: {e}")

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "MEDIUM",
            "source": {
                "service": "feature-store",
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

def get_feature_privacy_level(feature_names: List[str]) -> str:
    """获取特征隐私级别"""
    # 简化实现，实际应该查询数据库
    sensitive_features = ['credit_score', 'annual_income', 'debt_to_income_ratio']
    
    for feature in feature_names:
        if any(sensitive in feature.lower() for sensitive in sensitive_features):
            return PrivacyLevel.CONFIDENTIAL.value
    
    return PrivacyLevel.INTERNAL.value

# API路由
@app.post("/features/define")
async def define_feature(feature_def: FeatureDefinition):
    """定义新特征"""
    try:
        async with db_pool.acquire() as conn:
            # 检查特征是否已存在
            existing = await conn.fetchrow(
                "SELECT feature_name FROM feature_definitions WHERE feature_name = $1",
                feature_def.feature_name
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="特征已存在")
            
            # 插入特征定义
            await conn.execute("""
                INSERT INTO feature_definitions (
                    feature_name, feature_type, data_source, description,
                    privacy_level, owner, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                feature_def.feature_name, feature_def.feature_type.value,
                feature_def.data_source.value, feature_def.description,
                feature_def.privacy_level.value, feature_def.owner, feature_def.tags
            )
        
        # 发送审计日志
        await send_audit_log("FEATURE_DEFINE", {
            "feature_name": feature_def.feature_name,
            "feature_type": feature_def.feature_type.value,
            "data_source": feature_def.data_source.value,
            "privacy_level": feature_def.privacy_level.value
        })
        
        return {"message": "特征定义成功", "feature_name": feature_def.feature_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征定义失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/features/views/create")
async def create_feature_view(view_def: FeatureViewDefinition):
    """创建特征视图"""
    try:
        async with db_pool.acquire() as conn:
            # 检查视图是否已存在
            existing = await conn.fetchrow(
                "SELECT view_name FROM feature_views WHERE view_name = $1",
                view_def.view_name
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="特征视图已存在")
            
            # 插入特征视图
            config = {
                "features": [f.dict() for f in view_def.features],
                "batch_source_config": view_def.batch_source_config,
                "stream_source_config": view_def.stream_source_config
            }
            
            await conn.execute("""
                INSERT INTO feature_views (
                    view_name, entity_name, source_table, timestamp_field,
                    ttl_days, config
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                view_def.view_name, view_def.entity_name, view_def.source_table,
                view_def.timestamp_field, view_def.ttl_days, json.dumps(config)
            )
        
        # TODO: 在Feast中创建实际的特征视图
        
        return {"message": "特征视图创建成功", "view_name": view_def.view_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征视图创建失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/features/online")
async def get_online_features(request: FeatureRequest):
    """获取在线特征"""
    try:
        # 验证同意令牌
        privacy_level = get_feature_privacy_level(request.feature_names)
        if privacy_level in [PrivacyLevel.CONFIDENTIAL.value, PrivacyLevel.RESTRICTED.value]:
            if not await validate_consent(request.consent_token, request.feature_names):
                await log_feature_access(
                    request.entity_id, request.feature_names, request.consent_token,
                    request.request_context, "DENIED", privacy_level
                )
                raise HTTPException(status_code=403, detail="同意验证失败")
        
        # 从Feast获取在线特征
        entity_dict = {"customer_id": [request.entity_id]}
        
        try:
            feature_vector = feature_store.get_online_features(
                features=request.feature_names,
                entity_rows=entity_dict
            )
            
            # 转换为字典格式
            features_dict = feature_vector.to_dict()
            
            # 移除实体键
            result = {}
            for feature_name in request.feature_names:
                if feature_name in features_dict:
                    values = features_dict[feature_name]
                    result[feature_name] = values[0] if values else None
            
        except Exception as feast_error:
            logger.warning(f"Feast获取特征失败，使用模拟数据: {feast_error}")
            # 使用模拟数据
            result = await get_mock_features(request.entity_id, request.feature_names)
        
        # 记录访问日志
        await log_feature_access(
            request.entity_id, request.feature_names, request.consent_token,
            request.request_context, "SUCCESS", privacy_level
        )
        
        # 发送审计日志
        await send_audit_log("FEATURE_ACCESS", {
            "entity_id": request.entity_id,
            "feature_names": request.feature_names,
            "privacy_level": privacy_level,
            "consent_provided": bool(request.consent_token)
        })
        
        return {
            "entity_id": request.entity_id,
            "features": result,
            "timestamp": datetime.utcnow().isoformat(),
            "privacy_level": privacy_level
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取在线特征失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

async def get_mock_features(entity_id: str, feature_names: List[str]) -> Dict[str, Any]:
    """获取模拟特征数据"""
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
            result[feature_name] = None
    
    return result

@app.post("/features/batch")
async def get_batch_features(request: BatchFeatureRequest):
    """获取批量特征"""
    try:
        # 验证同意令牌
        privacy_level = get_feature_privacy_level(request.feature_names)
        if privacy_level in [PrivacyLevel.CONFIDENTIAL.value, PrivacyLevel.RESTRICTED.value]:
            if not await validate_consent(request.consent_token, request.feature_names):
                raise HTTPException(status_code=403, detail="同意验证失败")
        
        # 构建实体DataFrame
        entity_df = pd.DataFrame({
            "customer_id": request.entity_ids,
            "event_timestamp": [request.end_timestamp] * len(request.entity_ids)
        })
        
        try:
            # 从Feast获取历史特征
            training_df = feature_store.get_historical_features(
                entity_df=entity_df,
                features=request.feature_names
            ).to_df()
            
            # 转换为字典格式
            result = training_df.to_dict('records')
            
        except Exception as feast_error:
            logger.warning(f"Feast获取批量特征失败，使用模拟数据: {feast_error}")
            # 使用模拟数据
            result = []
            for entity_id in request.entity_ids:
                features = await get_mock_features(entity_id, request.feature_names)
                features["customer_id"] = entity_id
                features["event_timestamp"] = request.end_timestamp.isoformat()
                result.append(features)
        
        # 记录批量访问日志
        for entity_id in request.entity_ids:
            await log_feature_access(
                entity_id, request.feature_names, request.consent_token,
                {"batch_request": True, "batch_size": len(request.entity_ids)},
                "SUCCESS", privacy_level
            )
        
        return {
            "features": result,
            "timestamp": datetime.utcnow().isoformat(),
            "privacy_level": privacy_level,
            "batch_size": len(request.entity_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取批量特征失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/features/ingest")
async def ingest_features(request: FeatureIngestionRequest):
    """摄取特征数据"""
    try:
        # 验证特征视图存在
        async with db_pool.acquire() as conn:
            view = await conn.fetchrow(
                "SELECT * FROM feature_views WHERE view_name = $1",
                request.view_name
            )
            
            if not view:
                raise HTTPException(status_code=404, detail="特征视图不存在")
        
        # 转换数据为DataFrame
        df = pd.DataFrame(request.data)
        
        # 确保必要字段存在
        if request.timestamp_field not in df.columns:
            df[request.timestamp_field] = datetime.utcnow()
        
        if request.entity_field not in df.columns:
            raise HTTPException(status_code=400, detail=f"缺少实体字段: {request.entity_field}")
        
        # 保存到离线存储
        offline_path = f"{OFFLINE_STORE_PATH}/{request.view_name}_features.parquet"
        
        # 如果文件已存在，追加数据
        if os.path.exists(offline_path):
            existing_df = pd.read_parquet(offline_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(offline_path, index=False)
        
        # 推送到在线存储（Redis）
        for _, row in df.iterrows():
            entity_id = row[request.entity_field]
            timestamp = row[request.timestamp_field]
            
            # 构建Redis键
            for column in df.columns:
                if column not in [request.entity_field, request.timestamp_field]:
                    redis_key = f"feast:{request.view_name}:{column}:{entity_id}"
                    value = row[column]
                    
                    # 存储特征值和时间戳
                    feature_data = {
                        "value": str(value),
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                    }
                    
                    await redis_client.hset(redis_key, mapping=feature_data)
                    # 设置TTL
                    await redis_client.expire(redis_key, view['ttl_days'] * 24 * 3600)
        
        # 发送审计日志
        await send_audit_log("FEATURE_INGEST", {
            "view_name": request.view_name,
            "records_count": len(request.data),
            "source_info": request.source_info
        })
        
        return {
            "message": "特征数据摄取成功",
            "view_name": request.view_name,
            "records_ingested": len(request.data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征数据摄取失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/features/definitions")
async def list_feature_definitions(data_source: Optional[DataSource] = None, 
                                 privacy_level: Optional[PrivacyLevel] = None):
    """列出特征定义"""
    try:
        query = "SELECT * FROM feature_definitions WHERE 1=1"
        params = []
        
        if data_source:
            query += " AND data_source = $" + str(len(params) + 1)
            params.append(data_source.value)
        
        if privacy_level:
            query += " AND privacy_level = $" + str(len(params) + 1)
            params.append(privacy_level.value)
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            features = await conn.fetch(query, *params)
            return [dict(feature) for feature in features]
            
    except Exception as e:
        logger.error(f"列出特征定义失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/features/views")
async def list_feature_views():
    """列出特征视图"""
    try:
        async with db_pool.acquire() as conn:
            views = await conn.fetch("SELECT * FROM feature_views ORDER BY created_at DESC")
            return [dict(view) for view in views]
            
    except Exception as e:
        logger.error(f"列出特征视图失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/features/lineage/{feature_name}")
async def get_feature_lineage(feature_name: str):
    """获取特征血缘"""
    try:
        async with db_pool.acquire() as conn:
            lineage = await conn.fetch(
                "SELECT * FROM feature_lineage WHERE feature_name = $1 ORDER BY created_at DESC",
                feature_name
            )
            return [dict(record) for record in lineage]
            
    except Exception as e:
        logger.error(f"获取特征血缘失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/features/access-log")
async def get_access_log(entity_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = Query(default=100, le=1000)):
    """获取特征访问日志"""
    try:
        query = "SELECT * FROM feature_access_log WHERE 1=1"
        params = []
        
        if entity_id:
            query += " AND entity_id = $" + str(len(params) + 1)
            params.append(entity_id)
        
        if start_time:
            query += " AND request_timestamp >= $" + str(len(params) + 1)
            params.append(start_time)
        
        if end_time:
            query += " AND request_timestamp <= $" + str(len(params) + 1)
            params.append(end_time)
        
        query += " ORDER BY request_timestamp DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with db_pool.acquire() as conn:
            logs = await conn.fetch(query, *params)
            return [dict(log) for log in logs]
            
    except Exception as e:
        logger.error(f"获取访问日志失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # 检查Redis连接
        await redis_client.ping()
        
        # 检查Feast存储
        feast_healthy = feature_store is not None
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "feast": "healthy" if feast_healthy else "degraded"
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不可用")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动特征存储服务...")
    await init_database()
    await init_redis()
    await init_feast()
    logger.info("特征存储服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("关闭特征存储服务...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    logger.info("特征存储服务已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )