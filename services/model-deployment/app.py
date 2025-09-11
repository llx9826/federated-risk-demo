#!/usr/bin/env python3
"""
模型部署服务 - 联邦学习模型部署和版本管理

实现功能：
1. 模型版本管理和发布
2. A/B测试和灰度发布
3. 模型性能监控
4. 自动回滚机制
5. 负载均衡和扩缩容
6. 部署环境管理
"""

import os
import json
import time
import uuid
import pickle
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
import hashlib

import asyncpg
import redis.asyncio as redis
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
import httpx

# 容器和部署相关
import docker
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# 环境配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/federated_risk")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONSENT_GATEWAY_URL = os.getenv("CONSENT_GATEWAY_URL", "http://localhost:8001")
MODEL_REGISTRY_URL = os.getenv("MODEL_REGISTRY_URL", "http://localhost:8004")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./data/models")
DEPLOYMENT_STORAGE_PATH = os.getenv("DEPLOYMENT_STORAGE_PATH", "./data/deployments")
DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY", "localhost:5000")
KUBERNETES_NAMESPACE = os.getenv("KUBERNETES_NAMESPACE", "federated-risk")
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "60"))  # 秒

# 创建必要目录
Path("./logs").mkdir(exist_ok=True)
Path(MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(DEPLOYMENT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path("./data/configs").mkdir(parents=True, exist_ok=True)
Path("./data/metrics").mkdir(parents=True, exist_ok=True)

# 全局变量
db_pool = None
redis_client = None
docker_client = None
k8s_apps_v1 = None
k8s_core_v1 = None

# Prometheus指标
deployment_requests_total = Counter('deployment_requests_total', 'Total deployment requests', ['action', 'environment'])
deployment_duration = Histogram('deployment_duration_seconds', 'Deployment duration', ['action'])
active_deployments = Gauge('active_deployments', 'Number of active deployments', ['environment', 'version'])
model_prediction_requests = Counter('model_prediction_requests_total', 'Total prediction requests', ['model_id', 'version'])
model_prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency', ['model_id'])
model_prediction_errors = Counter('model_prediction_errors_total', 'Prediction errors', ['model_id', 'error_type'])
deployment_health_score = Gauge('deployment_health_score', 'Deployment health score', ['deployment_id'])
ab_test_traffic_ratio = Gauge('ab_test_traffic_ratio', 'A/B test traffic ratio', ['deployment_id', 'variant'])

# 枚举定义
from enum import Enum

class DeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"

class DeploymentStrategy(str, Enum):
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"
    AB_TEST = "ab_test"

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    TERMINATED = "terminated"

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class TrafficSplitType(str, Enum):
    PERCENTAGE = "percentage"
    USER_ATTRIBUTE = "user_attribute"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"

# FastAPI应用初始化
app = FastAPI(
    title="联邦风控模型部署服务",
    description="提供机器学习模型的部署、版本管理和监控",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic模型定义
class ModelDeploymentRequest(BaseModel):
    """模型部署请求"""
    model_id: str = Field(..., description="模型ID")
    model_version: str = Field(..., description="模型版本")
    environment: DeploymentEnvironment = Field(..., description="部署环境")
    strategy: DeploymentStrategy = Field(default=DeploymentStrategy.ROLLING_UPDATE, description="部署策略")
    replicas: int = Field(default=3, description="副本数量")
    resource_limits: Dict[str, str] = Field(default_factory=lambda: {"cpu": "1000m", "memory": "2Gi"}, description="资源限制")
    resource_requests: Dict[str, str] = Field(default_factory=lambda: {"cpu": "500m", "memory": "1Gi"}, description="资源请求")
    health_check_path: str = Field(default="/health", description="健康检查路径")
    auto_scaling: bool = Field(default=True, description="自动扩缩容")
    min_replicas: int = Field(default=2, description="最小副本数")
    max_replicas: int = Field(default=10, description="最大副本数")
    target_cpu_utilization: int = Field(default=70, description="目标CPU使用率")
    rollback_on_failure: bool = Field(default=True, description="失败时自动回滚")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="部署元数据")

class ABTestConfig(BaseModel):
    """A/B测试配置"""
    test_name: str = Field(..., description="测试名称")
    control_version: str = Field(..., description="对照组版本")
    treatment_version: str = Field(..., description="实验组版本")
    traffic_split: Dict[str, float] = Field(..., description="流量分配")
    split_type: TrafficSplitType = Field(default=TrafficSplitType.PERCENTAGE, description="分流类型")
    split_criteria: Dict[str, Any] = Field(default_factory=dict, description="分流条件")
    duration_hours: int = Field(default=24, description="测试持续时间（小时）")
    success_metrics: List[str] = Field(default_factory=list, description="成功指标")
    auto_promote: bool = Field(default=False, description="自动提升")
    significance_threshold: float = Field(default=0.05, description="显著性阈值")

class CanaryDeploymentConfig(BaseModel):
    """金丝雀部署配置"""
    initial_traffic_percentage: float = Field(default=5.0, description="初始流量百分比")
    increment_percentage: float = Field(default=10.0, description="递增百分比")
    increment_interval_minutes: int = Field(default=30, description="递增间隔（分钟）")
    success_threshold: float = Field(default=99.0, description="成功率阈值")
    error_threshold: float = Field(default=1.0, description="错误率阈值")
    latency_threshold_ms: float = Field(default=500.0, description="延迟阈值（毫秒）")
    auto_rollback: bool = Field(default=True, description="自动回滚")
    monitoring_duration_minutes: int = Field(default=60, description="监控持续时间（分钟）")

class DeploymentResponse(BaseModel):
    """部署响应"""
    deployment_id: str
    model_id: str
    model_version: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    status: DeploymentStatus
    endpoint_url: str
    replicas: int
    health_status: HealthStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

class DeploymentStatusResponse(BaseModel):
    """部署状态响应"""
    deployment_id: str
    status: DeploymentStatus
    health_status: HealthStatus
    replicas_ready: int
    replicas_total: int
    traffic_percentage: float
    metrics: Dict[str, float]
    last_health_check: datetime
    error_message: Optional[str]

class PredictionRequest(BaseModel):
    """预测请求"""
    features: Dict[str, Any] = Field(..., description="特征数据")
    model_version: Optional[str] = Field(None, description="指定模型版本")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="请求元数据")

class PredictionResponse(BaseModel):
    """预测响应"""
    prediction: Union[float, int, str, List[float]]
    confidence: Optional[float]
    model_id: str
    model_version: str
    prediction_id: str
    latency_ms: float
    timestamp: datetime

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    redis_status: str
    kubernetes_status: str
    active_deployments: int
    total_predictions: int

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
                CREATE TABLE IF NOT EXISTS deployments (
                    id SERIAL PRIMARY KEY,
                    deployment_id VARCHAR(128) UNIQUE NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    environment VARCHAR(32) NOT NULL,
                    strategy VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    health_status VARCHAR(32) NOT NULL DEFAULT 'unknown',
                    endpoint_url VARCHAR(512),
                    replicas INTEGER NOT NULL DEFAULT 1,
                    replicas_ready INTEGER NOT NULL DEFAULT 0,
                    traffic_percentage FLOAT NOT NULL DEFAULT 100.0,
                    resource_config JSONB NOT NULL,
                    deployment_config JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    terminated_at TIMESTAMPTZ
                );
                
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id SERIAL PRIMARY KEY,
                    test_id VARCHAR(128) UNIQUE NOT NULL,
                    test_name VARCHAR(256) NOT NULL,
                    control_deployment_id VARCHAR(128) NOT NULL,
                    treatment_deployment_id VARCHAR(128) NOT NULL,
                    traffic_split JSONB NOT NULL,
                    split_config JSONB NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ,
                    results JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS deployment_metrics (
                    id SERIAL PRIMARY KEY,
                    deployment_id VARCHAR(128) NOT NULL,
                    metric_name VARCHAR(128) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_timestamp TIMESTAMPTZ NOT NULL,
                    labels JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    prediction_id VARCHAR(128) UNIQUE NOT NULL,
                    deployment_id VARCHAR(128) NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    model_version VARCHAR(64) NOT NULL,
                    features JSONB NOT NULL,
                    prediction JSONB NOT NULL,
                    confidence FLOAT,
                    latency_ms FLOAT NOT NULL,
                    user_id VARCHAR(128),
                    session_id VARCHAR(128),
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS deployment_events (
                    id SERIAL PRIMARY KEY,
                    deployment_id VARCHAR(128) NOT NULL,
                    event_type VARCHAR(64) NOT NULL,
                    event_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_deployments_model_id ON deployments (model_id);
                CREATE INDEX IF NOT EXISTS idx_deployments_environment ON deployments (environment);
                CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments (status);
                CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests (status);
                CREATE INDEX IF NOT EXISTS idx_deployment_metrics_deployment_id ON deployment_metrics (deployment_id);
                CREATE INDEX IF NOT EXISTS idx_deployment_metrics_timestamp ON deployment_metrics (metric_timestamp);
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_deployment_id ON prediction_logs (deployment_id);
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs (created_at);
                CREATE INDEX IF NOT EXISTS idx_deployment_events_deployment_id ON deployment_events (deployment_id);
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

# Docker和Kubernetes初始化
async def init_container_clients():
    """初始化容器和Kubernetes客户端"""
    global docker_client, k8s_apps_v1, k8s_core_v1
    
    try:
        # 初始化Docker客户端
        docker_client = docker.from_env()
        logger.info("Docker客户端初始化成功")
        
        # 初始化Kubernetes客户端
        try:
            config.load_incluster_config()  # 集群内配置
        except:
            config.load_kube_config()  # 本地配置
        
        k8s_apps_v1 = client.AppsV1Api()
        k8s_core_v1 = client.CoreV1Api()
        logger.info("Kubernetes客户端初始化成功")
        
    except Exception as e:
        logger.warning(f"容器客户端初始化失败: {e}")
        # 在开发环境中可以继续运行

# 工具函数
async def generate_deployment_id() -> str:
    """生成部署ID"""
    return f"deploy_{uuid.uuid4().hex[:16]}"

async def generate_prediction_id() -> str:
    """生成预测ID"""
    return f"pred_{uuid.uuid4().hex[:16]}"

async def log_deployment_event(deployment_id: str, event_type: str, event_data: Dict[str, Any]):
    """记录部署事件"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO deployment_events (deployment_id, event_type, event_data)
                VALUES ($1, $2, $3)
            """, deployment_id, event_type, json.dumps(event_data))
    except Exception as e:
        logger.error(f"记录部署事件失败: {e}")

async def record_metric(deployment_id: str, metric_name: str, metric_value: float, 
                       labels: Optional[Dict[str, str]] = None):
    """记录部署指标"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO deployment_metrics (deployment_id, metric_name, metric_value, metric_timestamp, labels)
                VALUES ($1, $2, $3, NOW(), $4)
            """, deployment_id, metric_name, metric_value, json.dumps(labels or {}))
    except Exception as e:
        logger.error(f"记录指标失败: {e}")

class ModelDeploymentManager:
    """模型部署管理器"""
    
    def __init__(self):
        self.active_deployments = {}
        self.deployment_configs = {}
    
    async def deploy_model(self, request: ModelDeploymentRequest) -> str:
        """部署模型"""
        deployment_id = await generate_deployment_id()
        
        try:
            # 记录部署开始事件
            await log_deployment_event(deployment_id, "deployment_started", {
                "model_id": request.model_id,
                "model_version": request.model_version,
                "environment": request.environment.value,
                "strategy": request.strategy.value
            })
            
            # 创建部署记录
            endpoint_url = f"http://{request.model_id}-{request.environment.value}.{KUBERNETES_NAMESPACE}.svc.cluster.local"
            
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO deployments (
                        deployment_id, model_id, model_version, environment, strategy,
                        status, endpoint_url, replicas, resource_config, deployment_config, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                    deployment_id, request.model_id, request.model_version,
                    request.environment.value, request.strategy.value,
                    DeploymentStatus.PENDING.value, endpoint_url, request.replicas,
                    json.dumps({
                        "limits": request.resource_limits,
                        "requests": request.resource_requests
                    }),
                    json.dumps({
                        "health_check_path": request.health_check_path,
                        "auto_scaling": request.auto_scaling,
                        "min_replicas": request.min_replicas,
                        "max_replicas": request.max_replicas,
                        "target_cpu_utilization": request.target_cpu_utilization,
                        "rollback_on_failure": request.rollback_on_failure
                    }),
                    json.dumps(request.metadata)
                )
            
            # 根据策略执行部署
            if request.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(deployment_id, request)
            elif request.strategy == DeploymentStrategy.ROLLING_UPDATE:
                await self._deploy_rolling_update(deployment_id, request)
            elif request.strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(deployment_id, request)
            else:
                await self._deploy_standard(deployment_id, request)
            
            # 更新部署状态
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # 启动健康检查
            asyncio.create_task(self._monitor_deployment_health(deployment_id))
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"部署失败 {deployment_id}: {e}")
            await self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
            await log_deployment_event(deployment_id, "deployment_failed", {"error": str(e)})
            raise
    
    async def _deploy_standard(self, deployment_id: str, request: ModelDeploymentRequest):
        """标准部署"""
        try:
            if k8s_apps_v1 is None:
                # 模拟部署（开发环境）
                await asyncio.sleep(2)
                logger.info(f"模拟部署完成: {deployment_id}")
                return
            
            # 创建Kubernetes Deployment
            deployment_manifest = self._create_deployment_manifest(deployment_id, request)
            
            k8s_apps_v1.create_namespaced_deployment(
                namespace=KUBERNETES_NAMESPACE,
                body=deployment_manifest
            )
            
            # 创建Service
            service_manifest = self._create_service_manifest(deployment_id, request)
            
            k8s_core_v1.create_namespaced_service(
                namespace=KUBERNETES_NAMESPACE,
                body=service_manifest
            )
            
            # 如果启用自动扩缩容，创建HPA
            if request.auto_scaling:
                hpa_manifest = self._create_hpa_manifest(deployment_id, request)
                k8s_apps_v1.create_namespaced_horizontal_pod_autoscaler(
                    namespace=KUBERNETES_NAMESPACE,
                    body=hpa_manifest
                )
            
            logger.info(f"Kubernetes部署创建成功: {deployment_id}")
            
        except Exception as e:
            logger.error(f"Kubernetes部署失败 {deployment_id}: {e}")
            raise
    
    async def _deploy_blue_green(self, deployment_id: str, request: ModelDeploymentRequest):
        """蓝绿部署"""
        try:
            # 创建绿色环境（新版本）
            green_deployment_id = f"{deployment_id}-green"
            await self._deploy_standard(green_deployment_id, request)
            
            # 等待绿色环境就绪
            await self._wait_for_deployment_ready(green_deployment_id)
            
            # 切换流量到绿色环境
            await self._switch_traffic(deployment_id, green_deployment_id)
            
            # 清理蓝色环境（旧版本）
            await self._cleanup_old_deployment(deployment_id)
            
            logger.info(f"蓝绿部署完成: {deployment_id}")
            
        except Exception as e:
            logger.error(f"蓝绿部署失败 {deployment_id}: {e}")
            # 回滚到蓝色环境
            await self._rollback_deployment(deployment_id)
            raise
    
    async def _deploy_rolling_update(self, deployment_id: str, request: ModelDeploymentRequest):
        """滚动更新部署"""
        try:
            # 获取现有部署
            existing_deployment = await self._get_existing_deployment(request.model_id, request.environment)
            
            if existing_deployment:
                # 更新现有部署
                await self._update_existing_deployment(existing_deployment['deployment_id'], request)
            else:
                # 创建新部署
                await self._deploy_standard(deployment_id, request)
            
            logger.info(f"滚动更新部署完成: {deployment_id}")
            
        except Exception as e:
            logger.error(f"滚动更新部署失败 {deployment_id}: {e}")
            raise
    
    async def _deploy_canary(self, deployment_id: str, request: ModelDeploymentRequest):
        """金丝雀部署"""
        try:
            # 创建金丝雀版本（少量副本）
            canary_request = request.copy()
            canary_request.replicas = 1  # 金丝雀版本只部署1个副本
            
            await self._deploy_standard(deployment_id, canary_request)
            
            # 配置流量分割（5%流量到金丝雀版本）
            await self._configure_traffic_split(deployment_id, 5.0)
            
            # 启动金丝雀监控
            asyncio.create_task(self._monitor_canary_deployment(deployment_id))
            
            logger.info(f"金丝雀部署完成: {deployment_id}")
            
        except Exception as e:
            logger.error(f"金丝雀部署失败 {deployment_id}: {e}")
            raise
    
    def _create_deployment_manifest(self, deployment_id: str, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """创建Kubernetes Deployment清单"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{request.model_id}-{request.environment.value}",
                "namespace": KUBERNETES_NAMESPACE,
                "labels": {
                    "app": request.model_id,
                    "environment": request.environment.value,
                    "version": request.model_version,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "replicas": request.replicas,
                "selector": {
                    "matchLabels": {
                        "app": request.model_id,
                        "environment": request.environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": request.model_id,
                            "environment": request.environment.value,
                            "version": request.model_version,
                            "deployment-id": deployment_id
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": f"{DOCKER_REGISTRY}/model-server:{request.model_version}",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "MODEL_ID", "value": request.model_id},
                                {"name": "MODEL_VERSION", "value": request.model_version},
                                {"name": "ENVIRONMENT", "value": request.environment.value}
                            ],
                            "resources": {
                                "limits": request.resource_limits,
                                "requests": request.resource_requests
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": request.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": request.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
    
    def _create_service_manifest(self, deployment_id: str, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """创建Kubernetes Service清单"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{request.model_id}-{request.environment.value}",
                "namespace": KUBERNETES_NAMESPACE,
                "labels": {
                    "app": request.model_id,
                    "environment": request.environment.value,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "selector": {
                    "app": request.model_id,
                    "environment": request.environment.value
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
    
    def _create_hpa_manifest(self, deployment_id: str, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """创建Kubernetes HPA清单"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{request.model_id}-{request.environment.value}-hpa",
                "namespace": KUBERNETES_NAMESPACE
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{request.model_id}-{request.environment.value}"
                },
                "minReplicas": request.min_replicas,
                "maxReplicas": request.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": request.target_cpu_utilization
                        }
                    }
                }]
            }
        }
    
    async def _update_deployment_status(self, deployment_id: str, status: DeploymentStatus):
        """更新部署状态"""
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE deployments SET status = $1, updated_at = NOW() WHERE deployment_id = $2",
                    status.value, deployment_id
                )
            
            await log_deployment_event(deployment_id, "status_changed", {"new_status": status.value})
            
        except Exception as e:
            logger.error(f"更新部署状态失败 {deployment_id}: {e}")
    
    async def _monitor_deployment_health(self, deployment_id: str):
        """监控部署健康状态"""
        try:
            while True:
                await asyncio.sleep(MONITORING_INTERVAL)
                
                # 检查部署状态
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT status, endpoint_url FROM deployments WHERE deployment_id = $1",
                        deployment_id
                    )
                    
                    if not row or row['status'] in [DeploymentStatus.TERMINATED.value, DeploymentStatus.FAILED.value]:
                        break
                
                # 执行健康检查
                health_status = await self._check_deployment_health(deployment_id, row['endpoint_url'])
                
                # 更新健康状态
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE deployments SET health_status = $1, updated_at = NOW() WHERE deployment_id = $2",
                        health_status.value, deployment_id
                    )
                
                # 记录健康指标
                health_score = 1.0 if health_status == HealthStatus.HEALTHY else 0.0
                deployment_health_score.labels(deployment_id=deployment_id).set(health_score)
                
        except Exception as e:
            logger.error(f"健康监控失败 {deployment_id}: {e}")
    
    async def _check_deployment_health(self, deployment_id: str, endpoint_url: str) -> HealthStatus:
        """检查部署健康状态"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{endpoint_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        return HealthStatus.HEALTHY
                    else:
                        return HealthStatus.DEGRADED
                else:
                    return HealthStatus.UNHEALTHY
                    
        except Exception as e:
            logger.warning(f"健康检查失败 {deployment_id}: {e}")
            return HealthStatus.UNKNOWN
    
    async def _get_existing_deployment(self, model_id: str, environment: DeploymentEnvironment) -> Optional[Dict[str, Any]]:
        """获取现有部署"""
        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM deployments 
                    WHERE model_id = $1 AND environment = $2 AND status = $3
                    ORDER BY created_at DESC LIMIT 1
                """, model_id, environment.value, DeploymentStatus.ACTIVE.value)
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"获取现有部署失败: {e}")
            return None
    
    async def _wait_for_deployment_ready(self, deployment_id: str, timeout: int = 300):
        """等待部署就绪"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT health_status, replicas, replicas_ready FROM deployments WHERE deployment_id = $1",
                        deployment_id
                    )
                    
                    if row and row['health_status'] == HealthStatus.HEALTHY.value and row['replicas_ready'] >= row['replicas']:
                        return True
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.warning(f"检查部署就绪状态失败 {deployment_id}: {e}")
                await asyncio.sleep(10)
        
        raise TimeoutError(f"部署超时未就绪: {deployment_id}")
    
    async def _switch_traffic(self, old_deployment_id: str, new_deployment_id: str):
        """切换流量"""
        try:
            # 更新流量分配
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE deployments SET traffic_percentage = 0.0 WHERE deployment_id = $1",
                    old_deployment_id
                )
                await conn.execute(
                    "UPDATE deployments SET traffic_percentage = 100.0 WHERE deployment_id = $1",
                    new_deployment_id
                )
            
            await log_deployment_event(new_deployment_id, "traffic_switched", {
                "from_deployment": old_deployment_id,
                "to_deployment": new_deployment_id
            })
            
        except Exception as e:
            logger.error(f"切换流量失败: {e}")
            raise
    
    async def _configure_traffic_split(self, deployment_id: str, percentage: float):
        """配置流量分割"""
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE deployments SET traffic_percentage = $1 WHERE deployment_id = $2",
                    percentage, deployment_id
                )
            
            ab_test_traffic_ratio.labels(
                deployment_id=deployment_id,
                variant="canary"
            ).set(percentage / 100.0)
            
        except Exception as e:
            logger.error(f"配置流量分割失败: {e}")
            raise
    
    async def _monitor_canary_deployment(self, deployment_id: str):
        """监控金丝雀部署"""
        try:
            # 获取金丝雀配置
            canary_config = CanaryDeploymentConfig()
            
            current_percentage = canary_config.initial_traffic_percentage
            
            while current_percentage < 100.0:
                await asyncio.sleep(canary_config.increment_interval_minutes * 60)
                
                # 检查金丝雀指标
                metrics = await self._get_deployment_metrics(deployment_id)
                
                success_rate = metrics.get('success_rate', 0.0)
                error_rate = metrics.get('error_rate', 100.0)
                avg_latency = metrics.get('avg_latency_ms', 1000.0)
                
                # 检查是否满足推广条件
                if (success_rate >= canary_config.success_threshold and 
                    error_rate <= canary_config.error_threshold and 
                    avg_latency <= canary_config.latency_threshold_ms):
                    
                    # 增加流量
                    current_percentage = min(100.0, current_percentage + canary_config.increment_percentage)
                    await self._configure_traffic_split(deployment_id, current_percentage)
                    
                    await log_deployment_event(deployment_id, "canary_promoted", {
                        "new_traffic_percentage": current_percentage,
                        "success_rate": success_rate,
                        "error_rate": error_rate,
                        "avg_latency_ms": avg_latency
                    })
                    
                else:
                    # 回滚金丝雀部署
                    if canary_config.auto_rollback:
                        await self._rollback_deployment(deployment_id)
                        
                        await log_deployment_event(deployment_id, "canary_rollback", {
                            "reason": "metrics_threshold_exceeded",
                            "success_rate": success_rate,
                            "error_rate": error_rate,
                            "avg_latency_ms": avg_latency
                        })
                    
                    break
            
            # 金丝雀部署成功，更新为活跃状态
            if current_percentage >= 100.0:
                await self._update_deployment_status(deployment_id, DeploymentStatus.ACTIVE)
                
        except Exception as e:
            logger.error(f"金丝雀监控失败 {deployment_id}: {e}")
    
    async def _get_deployment_metrics(self, deployment_id: str) -> Dict[str, float]:
        """获取部署指标"""
        try:
            async with db_pool.acquire() as conn:
                # 获取最近1小时的指标
                rows = await conn.fetch("""
                    SELECT metric_name, AVG(metric_value) as avg_value
                    FROM deployment_metrics 
                    WHERE deployment_id = $1 AND metric_timestamp > NOW() - INTERVAL '1 hour'
                    GROUP BY metric_name
                """, deployment_id)
                
                metrics = {}
                for row in rows:
                    metrics[row['metric_name']] = float(row['avg_value'])
                
                return metrics
                
        except Exception as e:
            logger.error(f"获取部署指标失败 {deployment_id}: {e}")
            return {}
    
    async def _rollback_deployment(self, deployment_id: str):
        """回滚部署"""
        try:
            await self._update_deployment_status(deployment_id, DeploymentStatus.ROLLING_BACK)
            
            # 获取前一个稳定版本
            async with db_pool.acquire() as conn:
                current_deployment = await conn.fetchrow(
                    "SELECT model_id, environment FROM deployments WHERE deployment_id = $1",
                    deployment_id
                )
                
                if current_deployment:
                    previous_deployment = await conn.fetchrow("""
                        SELECT deployment_id FROM deployments 
                        WHERE model_id = $1 AND environment = $2 AND status = $3 AND deployment_id != $4
                        ORDER BY created_at DESC LIMIT 1
                    """, 
                        current_deployment['model_id'], current_deployment['environment'],
                        DeploymentStatus.ACTIVE.value, deployment_id
                    )
                    
                    if previous_deployment:
                        # 切换流量到前一个版本
                        await self._switch_traffic(deployment_id, previous_deployment['deployment_id'])
            
            # 终止当前部署
            await self._terminate_deployment(deployment_id)
            
            await log_deployment_event(deployment_id, "deployment_rollback", {})
            
        except Exception as e:
            logger.error(f"回滚部署失败 {deployment_id}: {e}")
            raise
    
    async def _terminate_deployment(self, deployment_id: str):
        """终止部署"""
        try:
            # 更新数据库状态
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE deployments 
                    SET status = $1, terminated_at = NOW(), updated_at = NOW() 
                    WHERE deployment_id = $2
                """, DeploymentStatus.TERMINATED.value, deployment_id)
            
            # 删除Kubernetes资源
            if k8s_apps_v1 and k8s_core_v1:
                try:
                    # 获取部署信息
                    async with db_pool.acquire() as conn:
                        row = await conn.fetchrow(
                            "SELECT model_id, environment FROM deployments WHERE deployment_id = $1",
                            deployment_id
                        )
                    
                    if row:
                        deployment_name = f"{row['model_id']}-{row['environment']}"
                        
                        # 删除Deployment
                        k8s_apps_v1.delete_namespaced_deployment(
                            name=deployment_name,
                            namespace=KUBERNETES_NAMESPACE
                        )
                        
                        # 删除Service
                        k8s_core_v1.delete_namespaced_service(
                            name=deployment_name,
                            namespace=KUBERNETES_NAMESPACE
                        )
                        
                        # 删除HPA
                        try:
                            k8s_apps_v1.delete_namespaced_horizontal_pod_autoscaler(
                                name=f"{deployment_name}-hpa",
                                namespace=KUBERNETES_NAMESPACE
                            )
                        except:
                            pass  # HPA可能不存在
                        
                except Exception as e:
                    logger.warning(f"删除Kubernetes资源失败 {deployment_id}: {e}")
            
            await log_deployment_event(deployment_id, "deployment_terminated", {})
            
        except Exception as e:
            logger.error(f"终止部署失败 {deployment_id}: {e}")
            raise
    
    async def _cleanup_old_deployment(self, deployment_id: str):
        """清理旧部署"""
        try:
            # 等待一段时间确保新部署稳定
            await asyncio.sleep(300)  # 5分钟
            
            # 终止旧部署
            await self._terminate_deployment(deployment_id)
            
        except Exception as e:
            logger.error(f"清理旧部署失败 {deployment_id}: {e}")
    
    async def _update_existing_deployment(self, deployment_id: str, request: ModelDeploymentRequest):
        """更新现有部署"""
        try:
            if k8s_apps_v1 is None:
                # 模拟更新（开发环境）
                await asyncio.sleep(2)
                logger.info(f"模拟更新完成: {deployment_id}")
                return
            
            # 获取现有Deployment
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT model_id, environment FROM deployments WHERE deployment_id = $1",
                    deployment_id
                )
            
            if row:
                deployment_name = f"{row['model_id']}-{row['environment']}"
                
                # 更新Deployment镜像
                deployment = k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=KUBERNETES_NAMESPACE
                )
                
                deployment.spec.template.spec.containers[0].image = f"{DOCKER_REGISTRY}/model-server:{request.model_version}"
                
                k8s_apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=KUBERNETES_NAMESPACE,
                    body=deployment
                )
                
                # 更新数据库记录
                async with db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE deployments 
                        SET model_version = $1, updated_at = NOW() 
                        WHERE deployment_id = $2
                    """, request.model_version, deployment_id)
                
                logger.info(f"部署更新成功: {deployment_id}")
            
        except Exception as e:
            logger.error(f"更新部署失败 {deployment_id}: {e}")
            raise

# 全局部署管理器实例
deployment_manager = ModelDeploymentManager()

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self):
        self.active_tests = {}
    
    async def create_ab_test(self, config: ABTestConfig, 
                           control_deployment_id: str, 
                           treatment_deployment_id: str) -> str:
        """创建A/B测试"""
        test_id = f"abtest_{uuid.uuid4().hex[:16]}"
        
        try:
            # 保存A/B测试配置
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ab_tests (
                        test_id, test_name, control_deployment_id, treatment_deployment_id,
                        traffic_split, split_config, status, start_time, end_time
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8)
                """, 
                    test_id, config.test_name, control_deployment_id, treatment_deployment_id,
                    json.dumps(config.traffic_split), json.dumps({
                        "split_type": config.split_type.value,
                        "split_criteria": config.split_criteria,
                        "success_metrics": config.success_metrics,
                        "auto_promote": config.auto_promote,
                        "significance_threshold": config.significance_threshold
                    }),
                    "active", datetime.utcnow() + timedelta(hours=config.duration_hours)
                )
            
            # 配置流量分割
            for variant, percentage in config.traffic_split.items():
                deployment_id = control_deployment_id if variant == "control" else treatment_deployment_id
                await deployment_manager._configure_traffic_split(deployment_id, percentage)
                
                ab_test_traffic_ratio.labels(
                    deployment_id=deployment_id,
                    variant=variant
                ).set(percentage / 100.0)
            
            # 启动A/B测试监控
            asyncio.create_task(self._monitor_ab_test(test_id))
            
            self.active_tests[test_id] = config
            
            logger.info(f"A/B测试创建成功: {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"创建A/B测试失败: {e}")
            raise
    
    async def _monitor_ab_test(self, test_id: str):
        """监控A/B测试"""
        try:
            while True:
                await asyncio.sleep(3600)  # 每小时检查一次
                
                # 检查测试是否仍然活跃
                async with db_pool.acquire() as conn:
                    test_row = await conn.fetchrow(
                        "SELECT * FROM ab_tests WHERE test_id = $1 AND status = 'active'",
                        test_id
                    )
                    
                    if not test_row:
                        break
                    
                    # 检查是否超时
                    if datetime.utcnow() > test_row['end_time']:
                        await self._finalize_ab_test(test_id)
                        break
                
                # 收集测试指标
                results = await self._collect_ab_test_metrics(test_id)
                
                # 检查是否可以自动提升
                config = json.loads(test_row['split_config'])
                if config.get('auto_promote', False):
                    if await self._should_promote_treatment(test_id, results):
                        await self._promote_treatment(test_id)
                        break
                
        except Exception as e:
            logger.error(f"A/B测试监控失败 {test_id}: {e}")
    
    async def _collect_ab_test_metrics(self, test_id: str) -> Dict[str, Any]:
        """收集A/B测试指标"""
        try:
            async with db_pool.acquire() as conn:
                test_row = await conn.fetchrow(
                    "SELECT control_deployment_id, treatment_deployment_id FROM ab_tests WHERE test_id = $1",
                    test_id
                )
                
                if not test_row:
                    return {}
                
                # 获取对照组和实验组的指标
                control_metrics = await deployment_manager._get_deployment_metrics(test_row['control_deployment_id'])
                treatment_metrics = await deployment_manager._get_deployment_metrics(test_row['treatment_deployment_id'])
                
                return {
                    "control": control_metrics,
                    "treatment": treatment_metrics,
                    "comparison": self._compare_metrics(control_metrics, treatment_metrics)
                }
                
        except Exception as e:
            logger.error(f"收集A/B测试指标失败 {test_id}: {e}")
            return {}
    
    def _compare_metrics(self, control: Dict[str, float], treatment: Dict[str, float]) -> Dict[str, Any]:
        """比较指标"""
        comparison = {}
        
        for metric in set(control.keys()) | set(treatment.keys()):
            control_value = control.get(metric, 0.0)
            treatment_value = treatment.get(metric, 0.0)
            
            if control_value > 0:
                improvement = (treatment_value - control_value) / control_value * 100
            else:
                improvement = 0.0
            
            comparison[metric] = {
                "control_value": control_value,
                "treatment_value": treatment_value,
                "improvement_percent": improvement,
                "is_significant": abs(improvement) > 5.0  # 简化的显著性检验
            }
        
        return comparison
    
    async def _should_promote_treatment(self, test_id: str, results: Dict[str, Any]) -> bool:
        """判断是否应该提升实验组"""
        try:
            comparison = results.get("comparison", {})
            
            # 检查关键指标是否有显著改善
            success_rate_improvement = comparison.get("success_rate", {}).get("improvement_percent", 0.0)
            latency_improvement = comparison.get("avg_latency_ms", {}).get("improvement_percent", 0.0)
            
            # 成功率提升且延迟没有显著增加
            if success_rate_improvement > 5.0 and latency_improvement < 10.0:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"判断提升条件失败 {test_id}: {e}")
            return False
    
    async def _promote_treatment(self, test_id: str):
        """提升实验组为主版本"""
        try:
            async with db_pool.acquire() as conn:
                test_row = await conn.fetchrow(
                    "SELECT treatment_deployment_id FROM ab_tests WHERE test_id = $1",
                    test_id
                )
                
                if test_row:
                    # 将实验组流量设置为100%
                    await deployment_manager._configure_traffic_split(
                        test_row['treatment_deployment_id'], 100.0
                    )
                    
                    # 更新测试状态
                    await conn.execute(
                        "UPDATE ab_tests SET status = 'promoted', end_time = NOW() WHERE test_id = $1",
                        test_id
                    )
            
            logger.info(f"A/B测试实验组提升成功: {test_id}")
            
        except Exception as e:
            logger.error(f"提升实验组失败 {test_id}: {e}")
    
    async def _finalize_ab_test(self, test_id: str):
        """结束A/B测试"""
        try:
            # 收集最终结果
            final_results = await self._collect_ab_test_metrics(test_id)
            
            # 更新测试状态和结果
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE ab_tests 
                    SET status = 'completed', results = $1, end_time = NOW() 
                    WHERE test_id = $2
                """, json.dumps(final_results), test_id)
            
            # 从活跃测试中移除
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            logger.info(f"A/B测试结束: {test_id}")
            
        except Exception as e:
            logger.error(f"结束A/B测试失败 {test_id}: {e}")

# 全局A/B测试管理器实例
ab_test_manager = ABTestManager()

# API路由实现
@app.post("/deployments", response_model=DeploymentResponse, summary="部署模型")
async def deploy_model(request: ModelDeploymentRequest, background_tasks: BackgroundTasks):
    """部署模型到指定环境"""
    try:
        deployment_requests_total.labels(
            action="deploy",
            environment=request.environment.value
        ).inc()
        
        with deployment_duration.labels(action="deploy").time():
            deployment_id = await deployment_manager.deploy_model(request)
        
        # 获取部署信息
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM deployments WHERE deployment_id = $1",
                deployment_id
            )
        
        if not row:
            raise HTTPException(status_code=404, detail="部署记录未找到")
        
        return DeploymentResponse(
            deployment_id=row['deployment_id'],
            model_id=row['model_id'],
            model_version=row['model_version'],
            environment=DeploymentEnvironment(row['environment']),
            strategy=DeploymentStrategy(row['strategy']),
            status=DeploymentStatus(row['status']),
            endpoint_url=row['endpoint_url'],
            replicas=row['replicas'],
            health_status=HealthStatus(row['health_status']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            metadata=json.loads(row['metadata'])
        )
        
    except Exception as e:
        logger.error(f"部署模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署失败: {str(e)}")

@app.get("/deployments/{deployment_id}", response_model=DeploymentStatusResponse, summary="获取部署状态")
async def get_deployment_status(deployment_id: str):
    """获取部署状态和健康信息"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM deployments WHERE deployment_id = $1",
                deployment_id
            )
        
        if not row:
            raise HTTPException(status_code=404, detail="部署未找到")
        
        # 获取最新指标
        metrics = await deployment_manager._get_deployment_metrics(deployment_id)
        
        return DeploymentStatusResponse(
            deployment_id=row['deployment_id'],
            status=DeploymentStatus(row['status']),
            health_status=HealthStatus(row['health_status']),
            replicas_ready=row['replicas_ready'],
            replicas_total=row['replicas'],
            traffic_percentage=row['traffic_percentage'],
            metrics=metrics,
            last_health_check=row['updated_at'],
            error_message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取部署状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/deployments", summary="列出部署")
async def list_deployments(
    model_id: Optional[str] = None,
    environment: Optional[DeploymentEnvironment] = None,
    status: Optional[DeploymentStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """列出部署记录"""
    try:
        conditions = []
        params = []
        param_count = 0
        
        if model_id:
            param_count += 1
            conditions.append(f"model_id = ${param_count}")
            params.append(model_id)
        
        if environment:
            param_count += 1
            conditions.append(f"environment = ${param_count}")
            params.append(environment.value)
        
        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM deployments 
                {where_clause}
                ORDER BY created_at DESC 
                LIMIT {limit_param} OFFSET {offset_param}
            """, *params)
            
            deployments = []
            for row in rows:
                deployments.append({
                    "deployment_id": row['deployment_id'],
                    "model_id": row['model_id'],
                    "model_version": row['model_version'],
                    "environment": row['environment'],
                    "strategy": row['strategy'],
                    "status": row['status'],
                    "health_status": row['health_status'],
                    "endpoint_url": row['endpoint_url'],
                    "replicas": row['replicas'],
                    "traffic_percentage": row['traffic_percentage'],
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at']
                })
            
            return {
                "deployments": deployments,
                "total": len(deployments),
                "limit": limit,
                "offset": offset
            }
        
    except Exception as e:
        logger.error(f"列出部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出部署失败: {str(e)}")

@app.post("/deployments/{deployment_id}/rollback", summary="回滚部署")
async def rollback_deployment(deployment_id: str):
    """回滚部署到前一个版本"""
    try:
        deployment_requests_total.labels(
            action="rollback",
            environment="unknown"
        ).inc()
        
        with deployment_duration.labels(action="rollback").time():
            await deployment_manager._rollback_deployment(deployment_id)
        
        return {"message": "回滚请求已提交", "deployment_id": deployment_id}
        
    except Exception as e:
        logger.error(f"回滚部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚失败: {str(e)}")

@app.delete("/deployments/{deployment_id}", summary="终止部署")
async def terminate_deployment(deployment_id: str):
    """终止部署"""
    try:
        deployment_requests_total.labels(
            action="terminate",
            environment="unknown"
        ).inc()
        
        with deployment_duration.labels(action="terminate").time():
            await deployment_manager._terminate_deployment(deployment_id)
        
        return {"message": "部署已终止", "deployment_id": deployment_id}
        
    except Exception as e:
        logger.error(f"终止部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"终止失败: {str(e)}")

@app.post("/deployments/{deployment_id}/scale", summary="扩缩容部署")
async def scale_deployment(deployment_id: str, replicas: int):
    """扩缩容部署"""
    try:
        if replicas < 1:
            raise HTTPException(status_code=400, detail="副本数必须大于0")
        
        # 更新数据库中的副本数
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE deployments SET replicas = $1, updated_at = NOW() WHERE deployment_id = $2",
                replicas, deployment_id
            )
        
        # 如果有Kubernetes客户端，更新实际部署
        if k8s_apps_v1:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT model_id, environment FROM deployments WHERE deployment_id = $1",
                    deployment_id
                )
            
            if row:
                deployment_name = f"{row['model_id']}-{row['environment']}"
                
                # 更新Deployment副本数
                k8s_apps_v1.patch_namespaced_deployment_scale(
                    name=deployment_name,
                    namespace=KUBERNETES_NAMESPACE,
                    body={"spec": {"replicas": replicas}}
                )
        
        await log_deployment_event(deployment_id, "scaled", {"new_replicas": replicas})
        
        return {"message": "扩缩容请求已提交", "deployment_id": deployment_id, "replicas": replicas}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"扩缩容失败: {e}")
        raise HTTPException(status_code=500, detail=f"扩缩容失败: {str(e)}")

@app.post("/ab-tests", summary="创建A/B测试")
async def create_ab_test(
    config: ABTestConfig,
    control_deployment_id: str,
    treatment_deployment_id: str
):
    """创建A/B测试"""
    try:
        test_id = await ab_test_manager.create_ab_test(
            config, control_deployment_id, treatment_deployment_id
        )
        
        return {
            "test_id": test_id,
            "message": "A/B测试创建成功",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"创建A/B测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建A/B测试失败: {str(e)}")

@app.get("/ab-tests/{test_id}", summary="获取A/B测试结果")
async def get_ab_test_results(test_id: str):
    """获取A/B测试结果"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ab_tests WHERE test_id = $1",
                test_id
            )
        
        if not row:
            raise HTTPException(status_code=404, detail="A/B测试未找到")
        
        # 如果测试仍在进行中，获取实时指标
        if row['status'] == 'active':
            results = await ab_test_manager._collect_ab_test_metrics(test_id)
        else:
            results = json.loads(row['results']) if row['results'] else {}
        
        return {
            "test_id": row['test_id'],
            "test_name": row['test_name'],
            "status": row['status'],
            "start_time": row['start_time'],
            "end_time": row['end_time'],
            "traffic_split": json.loads(row['traffic_split']),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取A/B测试结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")

@app.post("/predict/{deployment_id}", response_model=PredictionResponse, summary="模型预测")
async def predict(deployment_id: str, request: PredictionRequest):
    """使用指定部署进行预测"""
    start_time = time.time()
    prediction_id = await generate_prediction_id()
    
    try:
        # 获取部署信息
        async with db_pool.acquire() as conn:
            deployment_row = await conn.fetchrow(
                "SELECT * FROM deployments WHERE deployment_id = $1 AND status = $2",
                deployment_id, DeploymentStatus.ACTIVE.value
            )
        
        if not deployment_row:
            raise HTTPException(status_code=404, detail="活跃部署未找到")
        
        model_prediction_requests.labels(
            model_id=deployment_row['model_id'],
            version=deployment_row['model_version']
        ).inc()
        
        # 调用模型服务进行预测
        endpoint_url = deployment_row['endpoint_url']
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{endpoint_url}/predict",
                    json=request.dict()
                )
                
                if response.status_code != 200:
                    model_prediction_errors.labels(
                        model_id=deployment_row['model_id'],
                        error_type="http_error"
                    ).inc()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"预测服务错误: {response.text}"
                    )
                
                prediction_result = response.json()
                
        except httpx.TimeoutException:
            model_prediction_errors.labels(
                model_id=deployment_row['model_id'],
                error_type="timeout"
            ).inc()
            raise HTTPException(status_code=504, detail="预测请求超时")
        except httpx.RequestError as e:
            model_prediction_errors.labels(
                model_id=deployment_row['model_id'],
                error_type="connection_error"
            ).inc()
            raise HTTPException(status_code=503, detail=f"连接预测服务失败: {str(e)}")
        
        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        model_prediction_latency.labels(
            model_id=deployment_row['model_id']
        ).observe(latency_ms / 1000)
        
        # 记录预测日志
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO prediction_logs (
                    prediction_id, deployment_id, model_id, model_version,
                    features, prediction, confidence, latency_ms,
                    user_id, session_id, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                prediction_id, deployment_id, deployment_row['model_id'], deployment_row['model_version'],
                json.dumps(request.features), json.dumps(prediction_result),
                prediction_result.get('confidence'), latency_ms,
                request.user_id, request.session_id, json.dumps(request.metadata)
            )
        
        return PredictionResponse(
            prediction=prediction_result['prediction'],
            confidence=prediction_result.get('confidence'),
            model_id=deployment_row['model_id'],
            model_version=deployment_row['model_version'],
            prediction_id=prediction_id,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        model_prediction_errors.labels(
            model_id="unknown",
            error_type="internal_error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/deployments/{deployment_id}/metrics", summary="获取部署指标")
async def get_deployment_metrics(deployment_id: str, hours: int = 24):
    """获取部署指标"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT metric_name, metric_value, metric_timestamp, labels
                FROM deployment_metrics 
                WHERE deployment_id = $1 AND metric_timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY metric_timestamp DESC
            """, deployment_id, hours)
            
            metrics = []
            for row in rows:
                metrics.append({
                    "metric_name": row['metric_name'],
                    "metric_value": row['metric_value'],
                    "timestamp": row['metric_timestamp'],
                    "labels": json.loads(row['labels']) if row['labels'] else {}
                })
            
            return {
                "deployment_id": deployment_id,
                "metrics": metrics,
                "time_range_hours": hours
            }
        
    except Exception as e:
        logger.error(f"获取部署指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

@app.get("/health", response_model=HealthCheckResponse, summary="健康检查")
async def health_check():
    """服务健康检查"""
    try:
        # 检查数据库连接
        db_status = "healthy"
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except:
            db_status = "unhealthy"
        
        # 检查Redis连接
        redis_status = "healthy"
        try:
            await redis_client.ping()
        except:
            redis_status = "unhealthy"
        
        # 检查Kubernetes连接
        k8s_status = "healthy"
        try:
            if k8s_core_v1:
                k8s_core_v1.list_namespace(limit=1)
            else:
                k8s_status = "not_configured"
        except:
            k8s_status = "unhealthy"
        
        # 获取活跃部署数量
        async with db_pool.acquire() as conn:
            active_count = await conn.fetchval(
                "SELECT COUNT(*) FROM deployments WHERE status = $1",
                DeploymentStatus.ACTIVE.value
            )
            
            total_predictions = await conn.fetchval(
                "SELECT COUNT(*) FROM prediction_logs WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
        
        overall_status = "healthy"
        if db_status != "healthy" or redis_status != "healthy":
            overall_status = "unhealthy"
        elif k8s_status == "unhealthy":
            overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status=db_status,
            redis_status=redis_status,
            kubernetes_status=k8s_status,
            active_deployments=active_count,
            total_predictions=total_predictions
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status="unknown",
            redis_status="unknown",
            kubernetes_status="unknown",
            active_deployments=0,
            total_predictions=0
        )

@app.get("/metrics", summary="获取Prometheus指标")
async def get_metrics():
    """获取Prometheus格式的指标"""
    try:
        # 更新活跃部署指标
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT environment, model_version, COUNT(*) as count
                FROM deployments 
                WHERE status = $1
                GROUP BY environment, model_version
            """, DeploymentStatus.ACTIVE.value)
            
            # 清除旧指标
            active_deployments.clear()
            
            # 设置新指标
            for row in rows:
                active_deployments.labels(
                    environment=row['environment'],
                    version=row['model_version']
                ).set(row['count'])
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        return Response(
            content="# 指标获取失败\n",
            media_type=CONTENT_TYPE_LATEST
        )

# 后台任务
async def cleanup_old_deployments():
    """清理旧的已终止部署记录"""
    try:
        async with db_pool.acquire() as conn:
            # 删除30天前的已终止部署记录
            deleted_count = await conn.fetchval("""
                DELETE FROM deployments 
                WHERE status = $1 AND terminated_at < NOW() - INTERVAL '30 days'
                RETURNING COUNT(*)
            """, DeploymentStatus.TERMINATED.value)
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧部署记录")
    
    except Exception as e:
        logger.error(f"清理旧部署记录失败: {e}")

async def cleanup_old_metrics():
    """清理旧的指标数据"""
    try:
        async with db_pool.acquire() as conn:
            # 删除7天前的指标数据
            deleted_count = await conn.fetchval("""
                DELETE FROM deployment_metrics 
                WHERE metric_timestamp < NOW() - INTERVAL '7 days'
                RETURNING COUNT(*)
            """)
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧指标记录")
    
    except Exception as e:
        logger.error(f"清理旧指标数据失败: {e}")

async def cleanup_old_prediction_logs():
    """清理旧的预测日志"""
    try:
        async with db_pool.acquire() as conn:
            # 删除30天前的预测日志
            deleted_count = await conn.fetchval("""
                DELETE FROM prediction_logs 
                WHERE created_at < NOW() - INTERVAL '30 days'
                RETURNING COUNT(*)
            """)
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧预测日志")
    
    except Exception as e:
        logger.error(f"清理旧预测日志失败: {e}")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        logger.info("启动模型部署服务...")
        
        # 初始化数据库
        await init_database()
        
        # 初始化Redis
        await init_redis()
        
        # 初始化容器客户端
        await init_container_clients()
        
        # 启动定期清理任务
        asyncio.create_task(periodic_cleanup())
        
        logger.info("模型部署服务启动完成")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    try:
        logger.info("关闭模型部署服务...")
        
        # 关闭数据库连接池
        if db_pool:
            await db_pool.close()
        
        # 关闭Redis连接
        if redis_client:
            await redis_client.close()
        
        # 关闭Docker客户端
        if docker_client:
            docker_client.close()
        
        logger.info("模型部署服务已关闭")
        
    except Exception as e:
        logger.error(f"服务关闭失败: {e}")

async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时执行一次
            
            # 执行清理任务
            await cleanup_old_deployments()
            await cleanup_old_metrics()
            await cleanup_old_prediction_logs()
            
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info"
    )