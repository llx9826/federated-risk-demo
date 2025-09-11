#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习编排器 (Federated Orchestrator)

功能:
1. 支持Flower/NVFlare/FATE框架启动脚手架
2. 多方参与者注册和管理
3. 训练任务调度和监控
4. 模型聚合和版本管理
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from uuid import uuid4
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import asyncpg
import redis.asyncio as redis
from kubernetes import client, config
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 环境变量配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/orchestrator_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
KUBERNETES_NAMESPACE = os.getenv('KUBERNETES_NAMESPACE', 'federated-learning')
FLOWER_SERVER_IMAGE = os.getenv('FLOWER_SERVER_IMAGE', 'flwr/flower:latest')
FATE_IMAGE = os.getenv('FATE_IMAGE', 'federatedai/fate:latest')
NVFLARE_IMAGE = os.getenv('NVFLARE_IMAGE', 'nvflare/nvflare:latest')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')

# FastAPI应用初始化
app = FastAPI(
    title="联邦学习编排器",
    description="管理联邦学习训练任务的编排和调度",
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
k8s_apps_v1 = None
k8s_core_v1 = None
active_connections: Dict[str, WebSocket] = {}

# 枚举定义
class FrameworkType(str, Enum):
    FLOWER = "flower"
    FATE = "fate"
    NVFLARE = "nvflare"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ParticipantStatus(str, Enum):
    REGISTERED = "registered"
    CONNECTED = "connected"
    TRAINING = "training"
    DISCONNECTED = "disconnected"

# Pydantic模型定义
class ParticipantRegistration(BaseModel):
    """参与者注册"""
    participant_id: str = Field(..., description="参与者ID")
    organization: str = Field(..., description="组织名称")
    endpoint: str = Field(..., description="参与者端点")
    capabilities: Dict[str, Any] = Field(..., description="计算能力")
    data_schema: Dict[str, Any] = Field(..., description="数据模式")
    security_config: Dict[str, str] = Field(..., description="安全配置")
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('端点必须是有效的HTTP(S) URL')
        return v

class TrainingTaskRequest(BaseModel):
    """训练任务请求"""
    task_name: str = Field(..., description="任务名称")
    framework: FrameworkType = Field(..., description="联邦学习框架")
    algorithm: str = Field(..., description="训练算法")
    participants: List[str] = Field(..., description="参与者列表")
    model_config: Dict[str, Any] = Field(..., description="模型配置")
    training_config: Dict[str, Any] = Field(..., description="训练配置")
    privacy_config: Dict[str, Any] = Field(default_factory=dict, description="隐私配置")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="资源需求")
    
    @validator('participants')
    def validate_participants(cls, v):
        if len(v) < 2:
            raise ValueError('至少需要2个参与者')
        return v

class ModelAggregationRequest(BaseModel):
    """模型聚合请求"""
    task_id: str = Field(..., description="任务ID")
    round_number: int = Field(..., description="轮次号")
    participant_models: Dict[str, str] = Field(..., description="参与者模型")
    aggregation_strategy: str = Field(default="fedavg", description="聚合策略")
    weights: Optional[Dict[str, float]] = Field(None, description="聚合权重")

class TaskStatusUpdate(BaseModel):
    """任务状态更新"""
    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0, le=1)
    current_round: int
    total_rounds: int
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

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
                CREATE TABLE IF NOT EXISTS participants (
                    id SERIAL PRIMARY KEY,
                    participant_id VARCHAR(64) UNIQUE NOT NULL,
                    organization VARCHAR(128) NOT NULL,
                    endpoint VARCHAR(256) NOT NULL,
                    capabilities JSONB NOT NULL,
                    data_schema JSONB NOT NULL,
                    security_config JSONB NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'registered',
                    last_heartbeat TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS training_tasks (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(64) UNIQUE NOT NULL,
                    task_name VARCHAR(128) NOT NULL,
                    framework VARCHAR(32) NOT NULL,
                    algorithm VARCHAR(64) NOT NULL,
                    participants TEXT[] NOT NULL,
                    model_config JSONB NOT NULL,
                    training_config JSONB NOT NULL,
                    privacy_config JSONB,
                    resource_requirements JSONB,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    progress FLOAT DEFAULT 0.0,
                    current_round INTEGER DEFAULT 0,
                    total_rounds INTEGER DEFAULT 10,
                    metrics JSONB,
                    error_message TEXT,
                    k8s_deployment VARCHAR(128),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                );
                
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(64) NOT NULL,
                    version INTEGER NOT NULL,
                    round_number INTEGER NOT NULL,
                    model_hash VARCHAR(64) NOT NULL,
                    model_path VARCHAR(256) NOT NULL,
                    metrics JSONB,
                    aggregation_info JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(task_id, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_participants_status ON participants (status);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON training_tasks (status);
                CREATE INDEX IF NOT EXISTS idx_tasks_framework ON training_tasks (framework);
                CREATE INDEX IF NOT EXISTS idx_model_versions_task ON model_versions (task_id);
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

# Kubernetes初始化
async def init_kubernetes():
    """初始化Kubernetes客户端"""
    global k8s_apps_v1, k8s_core_v1
    try:
        # 尝试加载集群内配置
        try:
            config.load_incluster_config()
        except:
            # 如果失败，加载本地配置
            config.load_kube_config()
        
        k8s_apps_v1 = client.AppsV1Api()
        k8s_core_v1 = client.CoreV1Api()
        logger.info("Kubernetes客户端初始化成功")
    except Exception as e:
        logger.error(f"Kubernetes初始化失败: {e}")
        # 不抛出异常，允许在非K8s环境中运行

# 工具函数
def generate_task_id() -> str:
    """生成任务ID"""
    return f"task_{uuid4().hex[:16]}"

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{uuid4().hex[:16]}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "HIGH",
            "source": {
                "service": "federated-orchestrator",
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

async def broadcast_task_update(task_update: TaskStatusUpdate):
    """广播任务状态更新"""
    message = task_update.dict()
    
    # 发送到WebSocket连接
    disconnected = []
    for connection_id, websocket in active_connections.items():
        try:
            await websocket.send_json(message)
        except:
            disconnected.append(connection_id)
    
    # 清理断开的连接
    for connection_id in disconnected:
        active_connections.pop(connection_id, None)
    
    # 发送到Redis发布订阅
    await redis_client.publish("task_updates", json.dumps(message))

# Kubernetes部署函数
async def create_flower_deployment(task_id: str, config: dict) -> str:
    """创建Flower训练部署"""
    deployment_name = f"flower-{task_id}"
    
    deployment_spec = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "namespace": KUBERNETES_NAMESPACE,
            "labels": {
                "app": "flower-server",
                "task-id": task_id
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "flower-server",
                    "task-id": task_id
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "flower-server",
                        "task-id": task_id
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "flower-server",
                        "image": FLOWER_SERVER_IMAGE,
                        "ports": [{"containerPort": 8080}],
                        "env": [
                            {"name": "TASK_ID", "value": task_id},
                            {"name": "CONFIG", "value": json.dumps(config)}
                        ],
                        "resources": config.get("resource_requirements", {
                            "requests": {"cpu": "500m", "memory": "1Gi"},
                            "limits": {"cpu": "2", "memory": "4Gi"}
                        })
                    }]
                }
            }
        }
    }
    
    try:
        k8s_apps_v1.create_namespaced_deployment(
            namespace=KUBERNETES_NAMESPACE,
            body=deployment_spec
        )
        logger.info(f"Flower部署创建成功: {deployment_name}")
        return deployment_name
    except Exception as e:
        logger.error(f"Flower部署创建失败: {e}")
        raise

async def create_fate_deployment(task_id: str, config: dict) -> str:
    """创建FATE训练部署"""
    deployment_name = f"fate-{task_id}"
    
    # FATE需要更复杂的配置
    deployment_spec = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "namespace": KUBERNETES_NAMESPACE,
            "labels": {
                "app": "fate-server",
                "task-id": task_id
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "fate-server",
                    "task-id": task_id
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "fate-server",
                        "task-id": task_id
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "fate-server",
                        "image": FATE_IMAGE,
                        "ports": [{"containerPort": 9380}],
                        "env": [
                            {"name": "TASK_ID", "value": task_id},
                            {"name": "FATE_CONFIG", "value": json.dumps(config)}
                        ],
                        "resources": config.get("resource_requirements", {
                            "requests": {"cpu": "1", "memory": "2Gi"},
                            "limits": {"cpu": "4", "memory": "8Gi"}
                        })
                    }]
                }
            }
        }
    }
    
    try:
        k8s_apps_v1.create_namespaced_deployment(
            namespace=KUBERNETES_NAMESPACE,
            body=deployment_spec
        )
        logger.info(f"FATE部署创建成功: {deployment_name}")
        return deployment_name
    except Exception as e:
        logger.error(f"FATE部署创建失败: {e}")
        raise

# API路由
@app.post("/participants/register")
async def register_participant(participant: ParticipantRegistration):
    """注册参与者"""
    try:
        async with db_pool.acquire() as conn:
            # 检查是否已存在
            existing = await conn.fetchrow(
                "SELECT participant_id FROM participants WHERE participant_id = $1",
                participant.participant_id
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="参与者已存在")
            
            # 插入新参与者
            await conn.execute("""
                INSERT INTO participants (
                    participant_id, organization, endpoint, capabilities,
                    data_schema, security_config
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                participant.participant_id, participant.organization,
                participant.endpoint, json.dumps(participant.capabilities),
                json.dumps(participant.data_schema), json.dumps(participant.security_config)
            )
        
        # 发送审计日志
        await send_audit_log("PARTICIPANT_REGISTER", {
            "participant_id": participant.participant_id,
            "organization": participant.organization,
            "endpoint": participant.endpoint
        })
        
        return {"message": "参与者注册成功", "participant_id": participant.participant_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"参与者注册失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/tasks/create")
async def create_training_task(task_request: TrainingTaskRequest, background_tasks: BackgroundTasks):
    """创建训练任务"""
    try:
        task_id = generate_task_id()
        
        # 验证参与者是否存在
        async with db_pool.acquire() as conn:
            for participant_id in task_request.participants:
                participant = await conn.fetchrow(
                    "SELECT participant_id FROM participants WHERE participant_id = $1 AND status = 'registered'",
                    participant_id
                )
                if not participant:
                    raise HTTPException(
                        status_code=400,
                        detail=f"参与者 {participant_id} 不存在或状态异常"
                    )
            
            # 创建训练任务记录
            await conn.execute("""
                INSERT INTO training_tasks (
                    task_id, task_name, framework, algorithm, participants,
                    model_config, training_config, privacy_config, resource_requirements,
                    total_rounds
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                task_id, task_request.task_name, task_request.framework.value,
                task_request.algorithm, task_request.participants,
                json.dumps(task_request.model_config), json.dumps(task_request.training_config),
                json.dumps(task_request.privacy_config), json.dumps(task_request.resource_requirements),
                task_request.training_config.get("rounds", 10)
            )
        
        # 后台启动训练任务
        background_tasks.add_task(start_training_task, task_id, task_request)
        
        # 发送审计日志
        await send_audit_log("TRAINING_TASK_CREATE", {
            "task_id": task_id,
            "task_name": task_request.task_name,
            "framework": task_request.framework.value,
            "participants": task_request.participants
        })
        
        return {
            "message": "训练任务创建成功",
            "task_id": task_id,
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"训练任务创建失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

async def start_training_task(task_id: str, task_request: TrainingTaskRequest):
    """启动训练任务"""
    try:
        # 更新任务状态为运行中
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE training_tasks SET status = 'running', updated_at = NOW() WHERE task_id = $1",
                task_id
            )
        
        # 根据框架类型创建相应的部署
        deployment_name = None
        if task_request.framework == FrameworkType.FLOWER:
            deployment_name = await create_flower_deployment(task_id, task_request.dict())
        elif task_request.framework == FrameworkType.FATE:
            deployment_name = await create_fate_deployment(task_id, task_request.dict())
        elif task_request.framework == FrameworkType.NVFLARE:
            # TODO: 实现NVFlare部署
            pass
        
        # 更新部署信息
        if deployment_name:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE training_tasks SET k8s_deployment = $1 WHERE task_id = $2",
                    deployment_name, task_id
                )
        
        # 广播任务状态更新
        await broadcast_task_update(TaskStatusUpdate(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=0.0,
            current_round=0,
            total_rounds=task_request.training_config.get("rounds", 10)
        ))
        
        logger.info(f"训练任务启动成功: {task_id}")
        
    except Exception as e:
        logger.error(f"训练任务启动失败: {e}")
        
        # 更新任务状态为失败
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE training_tasks SET status = 'failed', error_message = $1, updated_at = NOW() WHERE task_id = $2",
                str(e), task_id
            )

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        async with db_pool.acquire() as conn:
            task = await conn.fetchrow(
                "SELECT * FROM training_tasks WHERE task_id = $1",
                task_id
            )
            
            if not task:
                raise HTTPException(status_code=404, detail="任务未找到")
            
            return dict(task)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/tasks")
async def list_tasks(status: Optional[TaskStatus] = None, framework: Optional[FrameworkType] = None):
    """列出训练任务"""
    try:
        query = "SELECT * FROM training_tasks WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = $" + str(len(params) + 1)
            params.append(status.value)
        
        if framework:
            query += " AND framework = $" + str(len(params) + 1)
            params.append(framework.value)
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            tasks = await conn.fetch(query, *params)
            return [dict(task) for task in tasks]
            
    except Exception as e:
        logger.error(f"列出任务失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/tasks/{task_id}/aggregate")
async def aggregate_models(task_id: str, request: ModelAggregationRequest):
    """聚合模型"""
    try:
        # 验证任务存在
        async with db_pool.acquire() as conn:
            task = await conn.fetchrow(
                "SELECT * FROM training_tasks WHERE task_id = $1",
                task_id
            )
            
            if not task:
                raise HTTPException(status_code=404, detail="任务未找到")
            
            if task['status'] != 'running':
                raise HTTPException(status_code=400, detail="任务状态不允许聚合")
        
        # TODO: 实现具体的模型聚合逻辑
        # 这里应该调用相应框架的聚合算法
        
        # 创建新的模型版本记录
        model_hash = f"model_{uuid4().hex[:16]}"
        model_path = f"/models/{task_id}/v{request.round_number}"
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO model_versions (
                    task_id, version, round_number, model_hash, model_path,
                    aggregation_info
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                task_id, request.round_number, request.round_number,
                model_hash, model_path, json.dumps({
                    "strategy": request.aggregation_strategy,
                    "weights": request.weights,
                    "participants": list(request.participant_models.keys())
                })
            )
            
            # 更新任务进度
            progress = request.round_number / task['total_rounds']
            await conn.execute(
                "UPDATE training_tasks SET current_round = $1, progress = $2, updated_at = NOW() WHERE task_id = $3",
                request.round_number, progress, task_id
            )
        
        # 广播进度更新
        await broadcast_task_update(TaskStatusUpdate(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=progress,
            current_round=request.round_number,
            total_rounds=task['total_rounds']
        ))
        
        return {
            "message": "模型聚合成功",
            "model_hash": model_hash,
            "model_path": model_path,
            "round_number": request.round_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型聚合失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket连接端点"""
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(client_id, None)
        logger.info(f"WebSocket连接断开: {client_id}")

@app.get("/participants")
async def list_participants(status: Optional[ParticipantStatus] = None):
    """列出参与者"""
    try:
        query = "SELECT * FROM participants WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = $" + str(len(params) + 1)
            params.append(status.value)
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            participants = await conn.fetch(query, *params)
            return [dict(participant) for participant in participants]
            
    except Exception as e:
        logger.error(f"列出参与者失败: {e}")
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
    logger.info("启动联邦学习编排器...")
    await init_database()
    await init_redis()
    await init_kubernetes()
    logger.info("联邦学习编排器启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("关闭联邦学习编排器...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    logger.info("联邦学习编排器已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )