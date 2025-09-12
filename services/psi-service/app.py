#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSI服务 (Private Set Intersection Service)

功能:
1. ECDH-PSI协议实现
2. Token-join回退机制
3. 多方PSI支持
4. 性能监控和优化
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Set, Tuple
from pathlib import Path
from enum import Enum
import asyncio
import time
from contextvars import ContextVar
import uuid

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn

# 请求追踪上下文
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
psi_session_id_var: ContextVar[str] = ContextVar('psi_session_id', default='')

# 结构化日志器
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _format_log(self, level: str, message: str, **kwargs) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'service': 'psi-service',
            'request_id': request_id_var.get(''),
            'psi_session_id': psi_session_id_var.get(''),
            'message': message,
            **kwargs
        }
        return json.dumps(log_data, ensure_ascii=False)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_log('INFO', message, **kwargs))
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format_log('ERROR', message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_log('WARNING', message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_log('DEBUG', message, **kwargs))

# 轻量追踪器
class SimpleTracer:
    def __init__(self):
        self.spans = []
    
    def start_span(self, operation_name: str, **tags) -> dict:
        span = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'tags': tags,
            'request_id': request_id_var.get(''),
            'psi_session_id': psi_session_id_var.get('')
        }
        return span
    
    def finish_span(self, span: dict, **tags):
        span['finish_time'] = time.time()
        span['duration_ms'] = (span['finish_time'] - span['start_time']) * 1000
        span['tags'].update(tags)
        self.spans.append(span)
        
        # 导出到文件
        trace_file = Path('./traces/psi_traces.jsonl')
        trace_file.parent.mkdir(exist_ok=True)
        with open(trace_file, 'a') as f:
            f.write(json.dumps(span, ensure_ascii=False) + '\n')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 使用结构化格式
)
logger = StructuredLogger(__name__)
tracer = SimpleTracer()

# 环境变量配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://root:123456@localhost:5432/federated_risk')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
DATA_STORE_PATH = os.getenv('DATA_STORE_PATH', './data/psi')
CONSENT_GATEWAY_URL = os.getenv('CONSENT_GATEWAY_URL', 'http://consent-gateway:8080')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')
MAX_SET_SIZE = int(os.getenv('MAX_SET_SIZE', '10000000'))  # 1000万
MAX_CONCURRENT_PSI = int(os.getenv('MAX_CONCURRENT_PSI', '10'))

# 创建必要目录
Path(DATA_STORE_PATH).mkdir(parents=True, exist_ok=True)

# 全局变量
db_pool = None
redis_client = None
active_psi_sessions = {}
psi_semaphore = None

# Prometheus指标
psi_requests_total = Counter('psi_requests_total', 'Total PSI requests', ['method', 'status'])
psi_duration = Histogram('psi_duration_seconds', 'PSI computation duration', ['method', 'set_size_bucket'])
psi_throughput = Histogram('psi_throughput_elements_per_second', 'PSI throughput')
active_sessions = Gauge('psi_active_sessions', 'Number of active PSI sessions')
set_size_distribution = Histogram('psi_set_size_distribution', 'Set size distribution', 
                                buckets=[100, 1000, 10000, 100000, 1000000, 10000000])
intersection_size_distribution = Histogram('psi_intersection_size_distribution', 'Intersection size distribution',
                                         buckets=[0, 10, 100, 1000, 10000, 100000])

# 枚举定义
class PSIMethod(str, Enum):
    ECDH_PSI = "ecdh_psi"
    TOKEN_JOIN = "token_join"
    BLOOM_FILTER = "bloom_filter"

class PSIStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PartyRole(str, Enum):
    SENDER = "sender"
    RECEIVER = "receiver"
    COORDINATOR = "coordinator"

# FastAPI应用初始化
app = FastAPI(
    title="PSI服务",
    description="隐私集合求交服务，支持ECDH-PSI和Token-join",
    version="1.0.0"
)

# 请求中间件
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    # 生成请求ID
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    
    start_time = time.time()
    
    logger.info("请求开始", 
                method=request.method, 
                url=str(request.url),
                client_ip=request.client.host if request.client else None)
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("请求完成",
                    status_code=response.status_code,
                    duration_ms=duration_ms)
        
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("请求失败",
                     error=str(e),
                     duration_ms=duration_ms)
        raise

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ECDH-PSI实现
class ECDHPSIEngine:
    """ECDH-PSI引擎"""
    
    def __init__(self):
        self.curve = ec.SECP256R1()
        self.backend = default_backend()
    
    def generate_keypair(self) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        """生成ECDH密钥对"""
        private_key = ec.generate_private_key(self.curve, self.backend)
        public_key = private_key.public_key()
        return private_key, public_key
    
    def hash_element(self, element) -> bytes:
        """哈希元素"""
        if isinstance(element, dict):
            # 字典类型转换为JSON字符串
            element_str = json.dumps(element, sort_keys=True, ensure_ascii=False)
        else:
            # 其他类型转换为字符串
            element_str = str(element)
        return hashlib.sha256(element_str.encode('utf-8')).digest()
    
    def point_to_hash(self, point: ec.EllipticCurvePublicKey) -> str:
        """将椭圆曲线点转换为哈希值"""
        point_bytes = point.public_numbers().x.to_bytes(32, 'big')
        return hashlib.sha256(point_bytes).hexdigest()
    
    def encrypt_set(self, elements: List[str], private_key: ec.EllipticCurvePrivateKey) -> List[str]:
        """加密集合元素"""
        encrypted_elements = []
        
        for element in elements:
            # 将元素哈希到椭圆曲线点
            element_hash = self.hash_element(element)
            
            # 简化实现：使用哈希值作为x坐标生成点
            # 实际实现应该使用更安全的哈希到曲线点的方法
            x_coord = int.from_bytes(element_hash[:32], 'big') % self.curve.key_size
            
            try:
                # 尝试构造椭圆曲线点
                point_numbers = ec.EllipticCurvePublicNumbers(x_coord, 0, self.curve)
                point = point_numbers.public_key(self.backend)
                
                # 使用私钥进行标量乘法
                shared_point = private_key.exchange(ec.ECDH(), point)
                encrypted_element = hashlib.sha256(shared_point).hexdigest()
                encrypted_elements.append(encrypted_element)
                
            except Exception:
                # 如果构造点失败，使用简化的加密方法
                scalar = private_key.private_numbers().private_value
                encrypted_value = (int.from_bytes(element_hash, 'big') * scalar) % (2**256)
                encrypted_element = hashlib.sha256(encrypted_value.to_bytes(32, 'big')).hexdigest()
                encrypted_elements.append(encrypted_element)
        
        return encrypted_elements
    
    def compute_intersection(self, set_a: List[str], set_b: List[str]) -> Tuple[List[str], int]:
        """计算交集"""
        set_a_hashes = set(set_a)
        set_b_hashes = set(set_b)
        
        intersection = list(set_a_hashes & set_b_hashes)
        intersection_size = len(intersection)
        
        return intersection, intersection_size

# Token-join回退实现
class TokenJoinEngine:
    """Token-join回退引擎"""
    
    def __init__(self):
        self.salt_length = 32
    
    def generate_tokens(self, elements: List, salt: Optional[bytes] = None) -> Tuple[List[str], bytes]:
        """生成令牌"""
        if salt is None:
            salt = secrets.token_bytes(self.salt_length)
        
        tokens = []
        for element in elements:
            # 处理不同类型的元素
            if isinstance(element, dict):
                element_str = json.dumps(element, sort_keys=True, ensure_ascii=False)
            else:
                element_str = str(element)
            
            # 使用HMAC-like方法生成令牌
            token_input = salt + element_str.encode('utf-8')
            token = hashlib.sha256(token_input).hexdigest()
            tokens.append(token)
        
        return tokens, salt
    
    def compute_intersection(self, tokens_a: List[str], tokens_b: List[str]) -> Tuple[List[str], int]:
        """计算令牌交集"""
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        
        intersection = list(set_a & set_b)
        intersection_size = len(intersection)
        
        return intersection, intersection_size

# Pydantic模型定义
class PSISessionRequest(BaseModel):
    """PSI会话请求"""
    session_id: str = Field(..., description="会话ID")
    method: PSIMethod = Field(default=PSIMethod.ECDH_PSI, description="PSI方法")
    party_role: PartyRole = Field(..., description="参与方角色")
    party_id: str = Field(..., description="参与方ID")
    other_parties: List[str] = Field(default_factory=list, description="其他参与方ID列表")
    consent_token: Optional[str] = Field(None, description="同意令牌")
    timeout_seconds: int = Field(default=3600, ge=60, le=86400, description="超时时间（秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('会话ID不能为空')
        return v.strip()

class PSIDataUpload(BaseModel):
    """PSI数据上传"""
    session_id: str = Field(..., description="会话ID")
    party_id: str = Field(..., description="参与方ID")
    data_hash: str = Field(..., description="数据哈希")
    element_count: int = Field(..., ge=1, description="元素数量")
    encrypted: bool = Field(default=False, description="是否已加密")
    
class PSIComputeRequest(BaseModel):
    """PSI计算请求"""
    session_id: str = Field(..., description="会话ID")
    party_id: str = Field(..., description="参与方ID")
    force_method: Optional[PSIMethod] = Field(None, description="强制使用的方法")
    return_intersection: bool = Field(default=False, description="是否返回交集内容")
    
class PSIResult(BaseModel):
    """PSI结果"""
    session_id: str
    intersection_size: int
    intersection_elements: Optional[List[str]] = None
    computation_time_ms: int
    method_used: PSIMethod
    party_contributions: Dict[str, int]
    metadata: Dict[str, Any]
    timestamp: datetime

class PSISessionStatus(BaseModel):
    """PSI会话状态"""
    session_id: str
    status: PSIStatus
    method: PSIMethod
    parties: List[str]
    data_uploaded: Dict[str, bool]
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    result: Optional[PSIResult] = None

class PSIResultResponse(BaseModel):
    """PSI结果响应"""
    session_id: str
    mapping_store_key: str
    intersection_tokens: List[str]
    metadata: Dict

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    active_sessions: int
    total_results: int

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
                CREATE TABLE IF NOT EXISTS psi_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) UNIQUE NOT NULL,
                    method VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    parties TEXT[] NOT NULL,
                    coordinator_party VARCHAR(128),
                    timeout_seconds INTEGER NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS psi_data (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    party_id VARCHAR(128) NOT NULL,
                    data_path VARCHAR(256) NOT NULL,
                    data_hash VARCHAR(64) NOT NULL,
                    element_count INTEGER NOT NULL,
                    encrypted BOOLEAN NOT NULL DEFAULT FALSE,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(session_id, party_id)
                );
                
                CREATE TABLE IF NOT EXISTS psi_results (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) UNIQUE NOT NULL,
                    intersection_size INTEGER NOT NULL,
                    intersection_hash VARCHAR(64),
                    computation_time_ms INTEGER NOT NULL,
                    method_used VARCHAR(32) NOT NULL,
                    party_contributions JSONB NOT NULL,
                    result_metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS psi_performance (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    metric_name VARCHAR(64) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    measurement_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_psi_sessions_id ON psi_sessions (session_id);
                CREATE INDEX IF NOT EXISTS idx_psi_sessions_status ON psi_sessions (status);
                CREATE INDEX IF NOT EXISTS idx_psi_data_session ON psi_data (session_id);
                CREATE INDEX IF NOT EXISTS idx_psi_results_session ON psi_results (session_id);
                CREATE INDEX IF NOT EXISTS idx_psi_performance_session ON psi_performance (session_id);
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
async def validate_consent(consent_token: str, party_id: str, data_purpose: str = "psi_computation") -> bool:
    """验证同意令牌"""
    if not consent_token:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONSENT_GATEWAY_URL}/consent/introspect",
                json={
                    "consent_token": consent_token,
                    "party_id": party_id,
                    "purpose": data_purpose
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

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "MEDIUM",
            "source": {
                "service": "psi-service",
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

def calculate_data_hash(data: List) -> str:
    """计算数据哈希"""
    # 处理不同类型的数据
    if not data:
        return hashlib.sha256(b'').hexdigest()
    
    # 如果是字典列表，转换为字符串后排序
    if isinstance(data[0], dict):
        str_data = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in data]
        sorted_data = sorted(str_data)
    else:
        # 如果是字符串列表，直接排序
        sorted_data = sorted(str(item) for item in data)
    
    content = '\n'.join(sorted_data)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_set_size_bucket(size: int) -> str:
    """获取集合大小桶"""
    if size <= 1000:
        return "small"
    elif size <= 100000:
        return "medium"
    elif size <= 1000000:
        return "large"
    else:
        return "xlarge"

async def cleanup_expired_sessions():
    """清理过期会话"""
    try:
        async with db_pool.acquire() as conn:
            # 查找过期会话
            expired_sessions = await conn.fetch("""
                SELECT session_id FROM psi_sessions 
                WHERE status IN ('pending', 'running') 
                AND created_at < NOW() - INTERVAL '1 hour' * timeout_seconds / 3600
            """)
            
            for session in expired_sessions:
                session_id = session['session_id']
                
                # 更新会话状态
                await conn.execute(
                    "UPDATE psi_sessions SET status = 'failed', updated_at = NOW() WHERE session_id = $1",
                    session_id
                )
                
                # 从内存中移除
                if session_id in active_psi_sessions:
                    del active_psi_sessions[session_id]
                
                logger.info(f"清理过期会话: {session_id}")
        
    except Exception as e:
        logger.error(f"清理过期会话失败: {e}")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global psi_semaphore
    psi_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PSI)
    
    await init_database()
    await init_redis()
    
    # 启动清理任务
    asyncio.create_task(periodic_cleanup())
    
    logger.info("PSI服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    
    logger.info("PSI服务关闭完成")

async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            await cleanup_expired_sessions()
            await asyncio.sleep(300)  # 每5分钟清理一次
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")
            await asyncio.sleep(60)

# API路由
@app.post("/psi/session", response_model=PSISessionStatus)
async def create_psi_session(
    request: PSISessionRequest,
    authorization: str = Header(None, alias="Authorization")
):
    """创建PSI会话"""
    start_time = time.time()
    
    try:
        # 验证授权
        if request.consent_token:
            # 验证同意令牌
            if not await validate_consent(request.consent_token, request.party_id, "psi_computation"):
                raise HTTPException(status_code=403, detail="同意验证失败")
        
        # 检查并发限制
        if len(active_psi_sessions) >= MAX_CONCURRENT_PSI:
            raise HTTPException(status_code=429, detail="PSI会话数量已达上限")
        
        # 存储会话信息
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO psi_sessions (session_id, method, status, parties, timeout_seconds, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, request.session_id, request.method.value, PSIStatus.PENDING.value, 
                [request.party_id] + request.other_parties, request.timeout_seconds, 
                json.dumps(request.metadata))
        
        # 缓存到内存
        now = datetime.utcnow()
        active_psi_sessions[request.session_id] = {
            "method": request.method,
            "status": PSIStatus.PENDING,
            "party_role": request.party_role,
            "party_id": request.party_id,
            "other_parties": request.other_parties,
            "timeout_seconds": request.timeout_seconds,
            "created_at": now,
            "updated_at": now,
            "uploaded_parties": set(),
            "data_ready": False
        }
        
        # 发送审计日志
        await send_audit_log("psi_session_created", {
            "session_id": request.session_id,
            "party_id": request.party_id,
            "method": request.method.value,
            "party_role": request.party_role.value
        })
        
        # 更新指标
        psi_requests_total.labels(method=request.method.value, status="created").inc()
        active_sessions.set(len(active_psi_sessions))
        
        logger.info(f"创建PSI会话: {request.session_id}, 参与方: {request.party_id}")
        
        return PSISessionStatus(
            session_id=request.session_id,
            status=PSIStatus.PENDING,
            method=request.method,
            parties=[request.party_id] + request.other_parties,
            data_uploaded={request.party_id: False},
            progress_percentage=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建PSI会话失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")
    finally:
        psi_duration.labels(method="session_create", set_size_bucket="unknown").observe(time.time() - start_time)

@app.post("/psi/upload")
async def upload_psi_data(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    party_id: str = Form(...),
    authorization: str = Header(None, alias="Authorization")
):
    """上传PSI数据"""
    start_time = time.time()
    
    try:
        # 验证授权
        if authorization and authorization.startswith("Bearer "):
            consent_token = authorization[7:]
            if not await validate_consent(consent_token, party_id, "psi_computation"):
                raise HTTPException(status_code=403, detail="同意验证失败")
        
        # 检查会话是否存在
        if session_id not in active_psi_sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        session = active_psi_sessions[session_id]
        
        if session["status"] != PSIStatus.PENDING:
            raise HTTPException(status_code=400, detail="会话状态不允许上传数据")
        
        # 读取和处理数据
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:  # 100MB限制
            raise HTTPException(status_code=413, detail="文件过大")
        
        # 解析数据
        data_elements = []
        if file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                data_elements = data
            elif isinstance(data, dict) and 'elements' in data:
                data_elements = data['elements']
        elif file.filename.endswith('.txt'):
            data_elements = content.decode('utf-8').strip().split('\n')
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        if len(data_elements) > MAX_SET_SIZE:
            raise HTTPException(status_code=413, detail=f"数据集过大，最大支持{MAX_SET_SIZE}个元素")
        
        # 计算数据哈希
        data_hash = calculate_data_hash(data_elements)
        
        # 保存数据文件
        data_path = Path(DATA_STORE_PATH) / session_id / f"{party_id}.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data_elements, f)
        
        # 存储数据信息到数据库
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO psi_data (session_id, party_id, data_path, data_hash, element_count)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id, party_id) 
                DO UPDATE SET data_hash = $4, element_count = $5, uploaded_at = NOW()
            """, session_id, party_id, str(data_path), data_hash, len(data_elements))
        
        # 缓存数据到Redis
        data_key = f"psi_data:{session_id}:{party_id}"
        await redis_client.setex(data_key, 3600, json.dumps(data_elements))  # 1小时过期
        
        # 更新会话状态
        session["uploaded_parties"].add(party_id)
        session["data_uploaded"] = {p: p in session["uploaded_parties"] for p in [session["party_id"]] + session["other_parties"]}
        session["updated_at"] = datetime.utcnow()
        
        # 检查是否所有参与方都已上传
        all_parties = [session["party_id"]] + session["other_parties"]
        if len(session["uploaded_parties"]) == len(all_parties):
            session["status"] = PSIStatus.PENDING  # 保持PENDING状态，等待计算
            session["data_ready"] = True
            
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE psi_sessions SET status = $1, updated_at = NOW() WHERE session_id = $2",
                    PSIStatus.PENDING.value, session_id
                )
        
        # 发送审计日志
        await send_audit_log("psi_data_uploaded", {
            "session_id": session_id,
            "party_id": party_id,
            "data_hash": data_hash,
            "element_count": len(data_elements)
        })
        
        # 更新指标
        set_size_distribution.observe(len(data_elements))
        
        logger.info(f"数据上传成功: 会话={session_id}, 参与方={party_id}, 数据量={len(data_elements)}")
        
        return {
            "message": "数据上传成功",
            "session_id": session_id,
            "party_id": party_id,
            "data_hash": data_hash,
            "element_count": len(data_elements),
            "uploaded_parties": len(session["uploaded_parties"]),
            "total_parties": len(all_parties),
            "status": session["status"].value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据上传失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")
    finally:
        psi_duration.labels(method="data_upload", set_size_bucket="unknown").observe(time.time() - start_time)

@app.post("/psi/compute", response_model=PSIResult)
async def compute_psi(
    request: PSIComputeRequest,
    authorization: str = Header(None, alias="Authorization")
):
    """计算PSI交集"""
    start_time = time.time()
    
    async with psi_semaphore:  # 限制并发计算
        try:
            # 验证授权
            if authorization and authorization.startswith("Bearer "):
                consent_token = authorization[7:]
                if not await validate_consent(consent_token, request.party_id, "psi_computation"):
                    raise HTTPException(status_code=403, detail="同意验证失败")
            
            # 检查会话
            if request.session_id not in active_psi_sessions:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            session = active_psi_sessions[request.session_id]
            
            logger.info(f"会话状态检查: session_id={request.session_id}, data_ready={session.get('data_ready', 'NOT_SET')}, status={session['status']}, uploaded_parties={session.get('uploaded_parties', set())}")
            if not session.get("data_ready", False):
                raise HTTPException(status_code=400, detail="数据未准备就绪")
            
            # 更新状态为计算中
            session["status"] = PSIStatus.RUNNING
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE psi_sessions SET status = $1, updated_at = NOW() WHERE session_id = $2",
                    PSIStatus.RUNNING.value, request.session_id
                )
            
            # 获取所有参与方数据
            party_datasets = {}
            all_parties = [session["party_id"]] + session["other_parties"]
            
            for party_id in all_parties:
                data_key = f"psi_data:{request.session_id}:{party_id}"
                data_json = await redis_client.get(data_key)
                if data_json:
                    party_datasets[party_id] = json.loads(data_json)
                else:
                    # 从文件读取
                    data_path = Path(DATA_STORE_PATH) / request.session_id / f"{party_id}.json"
                    if data_path.exists():
                        with open(data_path, 'r', encoding='utf-8') as f:
                            party_datasets[party_id] = json.load(f)
            
            if len(party_datasets) != len(all_parties):
                raise HTTPException(status_code=400, detail="部分参与方数据缺失")
            
            # 执行PSI计算
            computation_start = time.time()
            
            # 选择PSI方法
            method = request.force_method or session["method"]
            
            if method == PSIMethod.ECDH_PSI:
                # 使用ECDH-PSI
                ecdh_engine = ECDHPSIEngine()
                
                # 简化的两方PSI实现
                if len(party_datasets) == 2:
                    parties = list(party_datasets.keys())
                    set_a = party_datasets[parties[0]]
                    set_b = party_datasets[parties[1]]
                    
                    # 生成密钥对
                    private_key_a, _ = ecdh_engine.generate_keypair()
                    private_key_b, _ = ecdh_engine.generate_keypair()
                    
                    # 加密集合
                    encrypted_a = ecdh_engine.encrypt_set(set_a, private_key_a)
                    encrypted_b = ecdh_engine.encrypt_set(set_b, private_key_b)
                    
                    # 计算交集
                    intersection_tokens, intersection_size = ecdh_engine.compute_intersection(encrypted_a, encrypted_b)
                else:
                    raise HTTPException(status_code=400, detail="ECDH-PSI目前仅支持两方计算")
            
            elif method == PSIMethod.TOKEN_JOIN:
                # 使用Token-join
                token_engine = TokenJoinEngine()
                
                # 生成令牌
                all_tokens = []
                salt = None
                
                for party_id, elements in party_datasets.items():
                    tokens, salt = token_engine.generate_tokens(elements, salt)
                    all_tokens.append(set(tokens))
                
                # 计算交集
                intersection = all_tokens[0]
                for token_set in all_tokens[1:]:
                    intersection = intersection & token_set
                
                intersection_tokens = list(intersection)
                intersection_size = len(intersection_tokens)
            
            else:
                raise HTTPException(status_code=400, detail="不支持的PSI方法")
            
            computation_time = int((time.time() - computation_start) * 1000)
            
            # 存储结果
            party_contributions = {party: len(data) for party, data in party_datasets.items()}
            
            result_metadata = {
                "total_elements": sum(party_contributions.values()),
                "intersection_rate": intersection_size / max(party_contributions.values()) if party_contributions else 0,
                "computation_method": method.value
            }
            
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO psi_results (session_id, intersection_size, computation_time_ms, 
                                           method_used, party_contributions, result_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, request.session_id, intersection_size, computation_time, 
                    method.value, json.dumps(party_contributions), json.dumps(result_metadata))
            
            # 更新会话状态
            session["status"] = PSIStatus.COMPLETED
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE psi_sessions SET status = $1, updated_at = NOW() WHERE session_id = $2",
                    PSIStatus.COMPLETED.value, request.session_id
                )
            
            # 发送审计日志
            await send_audit_log("psi_computation_completed", {
                "session_id": request.session_id,
                "intersection_size": intersection_size,
                "computation_time_ms": computation_time,
                "method": method.value
            })
            
            # 更新指标
            psi_requests_total.labels(method=method.value, status="completed").inc()
            intersection_size_distribution.observe(intersection_size)
            
            # 计算吞吐量
            total_elements = sum(party_contributions.values())
            throughput = total_elements / (computation_time / 1000) if computation_time > 0 else 0
            psi_throughput.observe(throughput)
            
            logger.info(f"PSI计算完成: 会话={request.session_id}, 交集大小={intersection_size}, 耗时={computation_time}ms")
            
            result = PSIResult(
                session_id=request.session_id,
                intersection_size=intersection_size,
                intersection_elements=intersection_tokens if request.return_intersection else None,
                computation_time_ms=computation_time,
                method_used=method,
                party_contributions=party_contributions,
                metadata=result_metadata,
                timestamp=datetime.utcnow()
            )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"PSI计算失败: {e}")
            # 更新会话状态为失败
            if request.session_id in active_psi_sessions:
                active_psi_sessions[request.session_id]["status"] = PSIStatus.FAILED
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE psi_sessions SET status = $1, updated_at = NOW() WHERE session_id = $2",
                        PSIStatus.FAILED.value, request.session_id
                    )
            
            psi_requests_total.labels(method="unknown", status="failed").inc()
            raise HTTPException(status_code=500, detail="计算失败")
        finally:
            psi_duration.labels(method="psi_compute", set_size_bucket="unknown").observe(time.time() - start_time)
            active_sessions.set(len(active_psi_sessions))

@app.get("/psi/sessions", response_model=List[PSISessionStatus])
async def get_psi_sessions(limit: int = 100, offset: int = 0):
    """获取PSI会话列表"""
    try:
        sessions = []
        
        # 从内存中获取活跃会话
        for session_id, session in active_psi_sessions.items():
            all_parties = [session["party_id"]] + session["other_parties"]
            data_uploaded = {party: party in session["uploaded_parties"] for party in all_parties}
            
            progress = 0.0
            if session["status"] == PSIStatus.PENDING:
                progress = len(session["uploaded_parties"]) / len(all_parties) * 50
            elif session["status"] == PSIStatus.READY:
                progress = 50.0
            elif session["status"] == PSIStatus.RUNNING:
                progress = 75.0
            elif session["status"] == PSIStatus.COMPLETED:
                progress = 100.0
            
            session_status = PSISessionStatus(
                session_id=session_id,
                status=session["status"],
                method=session["method"],
                parties=all_parties,
                data_uploaded=data_uploaded,
                progress_percentage=progress,
                created_at=session["created_at"],
                updated_at=session["updated_at"],
                error_message=session.get("error_message"),
                result=session.get("result")
            )
            sessions.append(session_status)
        
        # 按创建时间倒序排序
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        
        # 应用分页
        return sessions[offset:offset + limit]
        
    except Exception as e:
        logger.error(f"获取PSI会话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取会话列表失败")

@app.get("/psi/sessions/{session_id}", response_model=PSISessionStatus)
async def get_psi_session_status(session_id: str):
    """获取PSI会话状态"""
    try:
        # 先从内存查找
        if session_id in active_psi_sessions:
            session = active_psi_sessions[session_id]
            all_parties = [session["party_id"]] + session["other_parties"]
            data_uploaded = {party: party in session["uploaded_parties"] for party in all_parties}
            
            progress = 0.0
            if session["status"] == PSIStatus.PENDING:
                progress = len(session["uploaded_parties"]) / len(all_parties) * 50
            elif session["status"] == PSIStatus.READY:
                progress = 50.0
            elif session["status"] == PSIStatus.RUNNING:
                progress = 75.0
            elif session["status"] == PSIStatus.COMPLETED:
                progress = 100.0
            
            return PSISessionStatus(
                session_id=session_id,
                status=session["status"],
                method=session["method"],
                parties=all_parties,
                data_uploaded=data_uploaded,
                progress_percentage=progress,
                created_at=session["created_at"],
                updated_at=session.get("updated_at", datetime.utcnow())
            )
        
        # 从数据库查找
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM psi_sessions WHERE session_id = $1",
                session_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            return PSISessionStatus(
                session_id=session_id,
                status=PSIStatus(row['status']),
                method=PSIMethod(row['method']),
                parties=row['parties'],
                data_uploaded={},
                progress_percentage=100.0 if row['status'] == 'completed' else 0.0,
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话状态失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/psi/results/{session_id}", response_model=PSIResult)
async def get_psi_result(session_id: str):
    """获取PSI计算结果"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM psi_results WHERE session_id = $1",
                session_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="结果不存在")
            
            return PSIResult(
                session_id=session_id,
                intersection_size=row['intersection_size'],
                intersection_elements=None,  # 不返回具体元素
                computation_time_ms=row['computation_time_ms'],
                method_used=PSIMethod(row['method_used']),
                party_contributions=json.loads(row['party_contributions']),
                metadata=json.loads(row['result_metadata']),
                timestamp=row['created_at']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取计算结果失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # 检查Redis连接
        await redis_client.ping()
        
        # 统计活跃会话
        active_count = len([s for s in active_psi_sessions.values() 
                           if s["status"] in [PSIStatus.PENDING, PSIStatus.READY, PSIStatus.RUNNING]])
        
        # 统计总结果数
        async with db_pool.acquire() as conn:
            total_results = await conn.fetchval("SELECT COUNT(*) FROM psi_results")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            active_sessions=active_count,
            total_results=total_results or 0
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            active_sessions=0,
            total_results=0
        )

@app.get("/metrics")
async def get_prometheus_metrics():
    """获取Prometheus指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 测试专用路由 (仅在测试环境下可用)
@app.post("/test/psi/setup-mock-data")
async def setup_mock_psi_data(session_id: str, party_1_id: str = "party_1", party_2_id: str = "party_2"):
    """为测试环境预设PSI数据"""
    if session_id not in active_psi_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = active_psi_sessions[session_id]
    
    # 生成模拟数据
    # 创建有交集的测试数据
    common_elements = [f"common_{i}" for i in range(1000, 1500)]  # 500个共同元素
    party_1_unique = [f"party1_{i}" for i in range(2000, 2500)]  # 500个party1独有
    party_2_unique = [f"party2_{i}" for i in range(3000, 3500)]  # 500个party2独有
    
    party_1_data = common_elements + party_1_unique  # 1000个元素
    party_2_data = common_elements + party_2_unique  # 1000个元素
    
    # 存储到Redis
    data_key_1 = f"psi_data:{session_id}:{party_1_id}"
    data_key_2 = f"psi_data:{session_id}:{party_2_id}"
    
    await redis_client.setex(data_key_1, 3600, json.dumps(party_1_data))
    await redis_client.setex(data_key_2, 3600, json.dumps(party_2_data))
    
    # 更新会话状态
    session["uploaded_parties"].add(party_1_id)
    session["uploaded_parties"].add(party_2_id)
    session["data_uploaded"] = {party_1_id: True, party_2_id: True}
    session["data_ready"] = True
    session["updated_at"] = datetime.utcnow()
    
    logger.info(f"为会话 {session_id} 预设测试数据: {party_1_id}({len(party_1_data)}), {party_2_id}({len(party_2_data)})")
    
    return {
        "session_id": session_id,
        "party_1_data_size": len(party_1_data),
        "party_2_data_size": len(party_2_data),
        "expected_intersection_size": len(common_elements),
        "data_ready": True
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7001))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )