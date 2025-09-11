#!/usr/bin/env python3
"""
API网关服务 - 联邦风控六步闭环统一入口

实现功能：
1. 统一API入口和路由
2. 六步闭环流程编排：同意→对齐→联训→解释→上线→审计
3. 服务发现和负载均衡
4. 认证授权和限流
5. 请求日志和监控
6. 错误处理和熔断
"""

import os
import json
import time
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import httpx
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

# 环境配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/federated_risk")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONSENT_GATEWAY_URL = os.getenv("CONSENT_GATEWAY_URL", "http://localhost:8001")
PSI_SERVICE_URL = os.getenv("PSI_SERVICE_URL", "http://localhost:8002")
FEDERATED_ORCHESTRATOR_URL = os.getenv("FEDERATED_ORCHESTRATOR_URL", "http://localhost:8003")
MODEL_TRAINER_URL = os.getenv("MODEL_TRAINER_URL", "http://localhost:8004")
FEATURE_STORE_URL = os.getenv("FEATURE_STORE_URL", "http://localhost:8005")
MODEL_SERVING_URL = os.getenv("MODEL_SERVING_URL", "http://localhost:8006")
AUDIT_SERVICE_URL = os.getenv("AUDIT_SERVICE_URL", "http://localhost:8007")

# 创建必要目录
Path("./logs").mkdir(exist_ok=True)
Path("./data/workflows").mkdir(parents=True, exist_ok=True)

# 全局变量
db_pool = None
redis_client = None
active_workflows = {}
service_registry = {}

# Prometheus指标
api_requests_total = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_gateway_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
workflow_executions_total = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_type', 'status'])
workflow_step_duration = Histogram('workflow_step_duration_seconds', 'Workflow step duration', ['workflow_type', 'step'])
active_workflows_gauge = Gauge('active_workflows', 'Number of active workflows')
service_health_status = Gauge('service_health_status', 'Service health status', ['service_name'])

# 枚举定义
from enum import Enum

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowStep(str, Enum):
    CONSENT = "consent"          # 1. 同意收集
    ALIGNMENT = "alignment"      # 2. 数据对齐(PSI)
    TRAINING = "training"        # 3. 联邦训练
    EXPLANATION = "explanation"  # 4. 模型解释
    DEPLOYMENT = "deployment"    # 5. 模型上线
    AUDIT = "audit"             # 6. 审计记录

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# FastAPI应用初始化
app = FastAPI(
    title="联邦风控API网关",
    description="统一管理联邦风控六步闭环流程的API网关服务",
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

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Pydantic模型定义
class WorkflowRequest(BaseModel):
    """工作流启动请求"""
    workflow_id: str = Field(..., description="工作流ID")
    workflow_type: str = Field(default="federated_risk_control", description="工作流类型")
    parties: List[str] = Field(..., description="参与方列表")
    coordinator_party: str = Field(..., description="协调方")
    data_purpose: str = Field(default="risk_assessment", description="数据使用目的")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="模型配置")
    privacy_config: Dict[str, Any] = Field(default_factory=dict, description="隐私配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class WorkflowStatus(BaseModel):
    """工作流状态"""
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[WorkflowStep]
    completed_steps: List[WorkflowStep]
    failed_steps: List[WorkflowStep]
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime]
    error_message: Optional[str]
    step_results: Dict[str, Any]

class ConsentRequest(BaseModel):
    """同意收集请求"""
    workflow_id: str
    party_id: str
    data_types: List[str]
    purposes: List[str]
    retention_period: int = Field(default=30, description="数据保留期(天)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AlignmentRequest(BaseModel):
    """数据对齐请求"""
    workflow_id: str
    psi_method: str = Field(default="ecdh_psi", description="PSI方法")
    timeout_seconds: int = Field(default=3600, description="超时时间")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TrainingRequest(BaseModel):
    """联邦训练请求"""
    workflow_id: str
    algorithm: str = Field(default="secure_boost", description="训练算法")
    rounds: int = Field(default=10, description="训练轮数")
    privacy_budget: float = Field(default=1.0, description="隐私预算")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExplanationRequest(BaseModel):
    """模型解释请求"""
    workflow_id: str
    model_id: str
    explanation_method: str = Field(default="federated_shap", description="解释方法")
    sample_size: int = Field(default=1000, description="样本大小")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DeploymentRequest(BaseModel):
    """模型部署请求"""
    workflow_id: str
    model_id: str
    deployment_env: str = Field(default="production", description="部署环境")
    scaling_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuditRequest(BaseModel):
    """审计记录请求"""
    workflow_id: str
    audit_scope: str = Field(default="full", description="审计范围")
    compliance_standards: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ServiceHealthResponse(BaseModel):
    """服务健康状态响应"""
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str]

class GatewayHealthResponse(BaseModel):
    """网关健康检查响应"""
    status: str
    timestamp: datetime
    version: str
    active_workflows: int
    service_statuses: Dict[str, ServiceHealthResponse]

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
                CREATE TABLE IF NOT EXISTS workflows (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(128) UNIQUE NOT NULL,
                    workflow_type VARCHAR(64) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    current_step VARCHAR(32),
                    parties TEXT[] NOT NULL,
                    coordinator_party VARCHAR(128) NOT NULL,
                    data_purpose VARCHAR(128) NOT NULL,
                    model_config JSONB NOT NULL,
                    privacy_config JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                );
                
                CREATE TABLE IF NOT EXISTS workflow_steps (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(128) NOT NULL,
                    step_name VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    duration_ms INTEGER,
                    result_data JSONB,
                    error_message TEXT,
                    UNIQUE(workflow_id, step_name)
                );
                
                CREATE TABLE IF NOT EXISTS service_registry (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(64) UNIQUE NOT NULL,
                    service_url VARCHAR(256) NOT NULL,
                    health_endpoint VARCHAR(128) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    last_health_check TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    response_time_ms FLOAT,
                    error_message TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_workflows_id ON workflows (workflow_id);
                CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows (status);
                CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow ON workflow_steps (workflow_id);
                CREATE INDEX IF NOT EXISTS idx_workflow_steps_status ON workflow_steps (status);
                CREATE INDEX IF NOT EXISTS idx_service_registry_name ON service_registry (service_name);
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

# 服务注册
async def register_services():
    """注册后端服务"""
    services = {
        "consent-gateway": {"url": CONSENT_GATEWAY_URL, "health": "/health"},
        "psi-service": {"url": PSI_SERVICE_URL, "health": "/health"},
        "federated-orchestrator": {"url": FEDERATED_ORCHESTRATOR_URL, "health": "/health"},
        "model-trainer": {"url": MODEL_TRAINER_URL, "health": "/health"},
        "feature-store": {"url": FEATURE_STORE_URL, "health": "/health"},
        "model-serving": {"url": MODEL_SERVING_URL, "health": "/health"},
        "audit-service": {"url": AUDIT_SERVICE_URL, "health": "/health"}
    }
    
    async with db_pool.acquire() as conn:
        for service_name, config in services.items():
            await conn.execute("""
                INSERT INTO service_registry (service_name, service_url, health_endpoint, status)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (service_name) 
                DO UPDATE SET service_url = $2, health_endpoint = $3, status = $4
            """, service_name, config["url"], config["health"], ServiceStatus.UNKNOWN.value)
    
    logger.info("服务注册完成")

# 工具函数
async def call_service(service_url: str, endpoint: str, method: str = "GET", 
                      data: Optional[Dict] = None, headers: Optional[Dict] = None,
                      timeout: float = 30.0) -> Dict[str, Any]:
    """调用后端服务"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"服务调用失败 {service_url}/{endpoint}: {e}")
        raise HTTPException(status_code=502, detail=f"服务调用失败: {str(e)}")

async def check_service_health(service_name: str) -> ServiceHealthResponse:
    """检查服务健康状态"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT service_url, health_endpoint FROM service_registry WHERE service_name = $1",
                service_name
            )
            
            if not row:
                return ServiceHealthResponse(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    response_time_ms=0.0,
                    last_check=datetime.utcnow(),
                    error_message="服务未注册"
                )
            
            start_time = time.time()
            
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    url = f"{row['service_url']}{row['health_endpoint']}"
                    response = await client.get(url)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        status = ServiceStatus.HEALTHY
                        error_message = None
                    else:
                        status = ServiceStatus.UNHEALTHY
                        error_message = f"HTTP {response.status_code}"
                        
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                status = ServiceStatus.UNHEALTHY
                error_message = str(e)
            
            # 更新数据库记录
            await conn.execute("""
                UPDATE service_registry 
                SET status = $1, last_health_check = NOW(), response_time_ms = $2, error_message = $3
                WHERE service_name = $4
            """, status.value, response_time, error_message, service_name)
            
            # 更新Prometheus指标
            service_health_status.labels(service_name=service_name).set(
                1 if status == ServiceStatus.HEALTHY else 0
            )
            
            return ServiceHealthResponse(
                service_name=service_name,
                status=status,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                error_message=error_message
            )
            
    except Exception as e:
        logger.error(f"健康检查失败 {service_name}: {e}")
        return ServiceHealthResponse(
            service_name=service_name,
            status=ServiceStatus.UNKNOWN,
            response_time_ms=0.0,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )

async def update_workflow_status(workflow_id: str, status: WorkflowStatus, 
                               current_step: Optional[WorkflowStep] = None,
                               error_message: Optional[str] = None):
    """更新工作流状态"""
    try:
        async with db_pool.acquire() as conn:
            if status == WorkflowStatus.COMPLETED:
                await conn.execute("""
                    UPDATE workflows 
                    SET status = $1, current_step = $2, updated_at = NOW(), completed_at = NOW()
                    WHERE workflow_id = $3
                """, status.value, current_step.value if current_step else None, workflow_id)
            else:
                await conn.execute("""
                    UPDATE workflows 
                    SET status = $1, current_step = $2, updated_at = NOW()
                    WHERE workflow_id = $3
                """, status.value, current_step.value if current_step else None, workflow_id)
        
        # 更新内存缓存
        if workflow_id in active_workflows:
            active_workflows[workflow_id]["status"] = status
            active_workflows[workflow_id]["current_step"] = current_step
            active_workflows[workflow_id]["updated_at"] = datetime.utcnow()
            if error_message:
                active_workflows[workflow_id]["error_message"] = error_message
        
        # 更新指标
        active_workflows_gauge.set(len([w for w in active_workflows.values() 
                                      if w["status"] in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]]))
        
    except Exception as e:
        logger.error(f"更新工作流状态失败 {workflow_id}: {e}")

async def record_step_result(workflow_id: str, step: WorkflowStep, status: str,
                           duration_ms: int, result_data: Optional[Dict] = None,
                           error_message: Optional[str] = None):
    """记录步骤执行结果"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_steps (workflow_id, step_name, status, completed_at, duration_ms, result_data, error_message)
                VALUES ($1, $2, $3, NOW(), $4, $5, $6)
                ON CONFLICT (workflow_id, step_name)
                DO UPDATE SET status = $2, completed_at = NOW(), duration_ms = $4, result_data = $5, error_message = $6
            """, workflow_id, step.value, status, duration_ms, 
                json.dumps(result_data) if result_data else None, error_message)
        
        # 更新Prometheus指标
        workflow_step_duration.labels(workflow_type="federated_risk_control", step=step.value).observe(duration_ms / 1000)
        
    except Exception as e:
        logger.error(f"记录步骤结果失败 {workflow_id}/{step.value}: {e}")

# 六步闭环工作流实现
class FederatedRiskWorkflow:
    """联邦风控六步闭环工作流"""
    
    def __init__(self, workflow_id: str, request: WorkflowRequest):
        self.workflow_id = workflow_id
        self.request = request
        self.step_results = {}
    
    async def execute(self):
        """执行完整工作流"""
        try:
            await update_workflow_status(self.workflow_id, WorkflowStatus.RUNNING)
            
            # 步骤1: 同意收集
            await self._execute_step(WorkflowStep.CONSENT, self._step_consent)
            
            # 步骤2: 数据对齐
            await self._execute_step(WorkflowStep.ALIGNMENT, self._step_alignment)
            
            # 步骤3: 联邦训练
            await self._execute_step(WorkflowStep.TRAINING, self._step_training)
            
            # 步骤4: 模型解释
            await self._execute_step(WorkflowStep.EXPLANATION, self._step_explanation)
            
            # 步骤5: 模型部署
            await self._execute_step(WorkflowStep.DEPLOYMENT, self._step_deployment)
            
            # 步骤6: 审计记录
            await self._execute_step(WorkflowStep.AUDIT, self._step_audit)
            
            # 工作流完成
            await update_workflow_status(self.workflow_id, WorkflowStatus.COMPLETED, WorkflowStep.AUDIT)
            workflow_executions_total.labels(workflow_type="federated_risk_control", status="completed").inc()
            
            logger.info(f"工作流执行完成: {self.workflow_id}")
            
        except Exception as e:
            logger.error(f"工作流执行失败 {self.workflow_id}: {e}")
            await update_workflow_status(self.workflow_id, WorkflowStatus.FAILED, error_message=str(e))
            workflow_executions_total.labels(workflow_type="federated_risk_control", status="failed").inc()
            raise
    
    async def _execute_step(self, step: WorkflowStep, step_func):
        """执行单个步骤"""
        start_time = time.time()
        
        try:
            await update_workflow_status(self.workflow_id, WorkflowStatus.RUNNING, step)
            
            result = await step_func()
            duration_ms = int((time.time() - start_time) * 1000)
            
            await record_step_result(self.workflow_id, step, "completed", duration_ms, result)
            self.step_results[step.value] = result
            
            logger.info(f"步骤完成 {self.workflow_id}/{step.value}: {duration_ms}ms")
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            await record_step_result(self.workflow_id, step, "failed", duration_ms, error_message=str(e))
            raise
    
    async def _step_consent(self) -> Dict[str, Any]:
        """步骤1: 同意收集"""
        consent_tokens = {}
        
        for party_id in self.request.parties:
            # 为每个参与方收集同意
            consent_data = {
                "party_id": party_id,
                "data_types": ["transaction_data", "user_profile"],
                "purposes": [self.request.data_purpose],
                "retention_period": 30,
                "workflow_id": self.workflow_id
            }
            
            result = await call_service(
                CONSENT_GATEWAY_URL,
                "consent/issue",
                "POST",
                consent_data
            )
            
            consent_tokens[party_id] = result.get("consent_token")
        
        return {
            "consent_tokens": consent_tokens,
            "parties": self.request.parties,
            "data_purpose": self.request.data_purpose
        }
    
    async def _step_alignment(self) -> Dict[str, Any]:
        """步骤2: 数据对齐(PSI)"""
        # 创建PSI会话
        psi_data = {
            "session_id": f"psi_{self.workflow_id}",
            "method": "ecdh_psi",
            "parties": self.request.parties,
            "coordinator_party": self.request.coordinator_party,
            "timeout_seconds": 3600,
            "metadata": {"workflow_id": self.workflow_id}
        }
        
        session_result = await call_service(
            PSI_SERVICE_URL,
            "psi/session",
            "POST",
            psi_data
        )
        
        # 等待PSI计算完成(简化实现)
        await asyncio.sleep(2)
        
        # 获取PSI结果
        psi_result = await call_service(
            PSI_SERVICE_URL,
            f"psi/results/{psi_data['session_id']}",
            "GET"
        )
        
        return {
            "psi_session_id": psi_data["session_id"],
            "intersection_size": psi_result.get("intersection_size", 0),
            "alignment_quality": "high" if psi_result.get("intersection_size", 0) > 1000 else "medium"
        }
    
    async def _step_training(self) -> Dict[str, Any]:
        """步骤3: 联邦训练"""
        training_data = {
            "task_id": f"train_{self.workflow_id}",
            "algorithm": self.request.model_config.get("algorithm", "secure_boost"),
            "parties": self.request.parties,
            "coordinator_party": self.request.coordinator_party,
            "rounds": self.request.model_config.get("rounds", 10),
            "privacy_budget": self.request.privacy_config.get("budget", 1.0),
            "model_params": self.request.model_config,
            "metadata": {"workflow_id": self.workflow_id}
        }
        
        # 启动训练任务
        training_result = await call_service(
            MODEL_TRAINER_URL,
            "training/start",
            "POST",
            training_data
        )
        
        # 等待训练完成(简化实现)
        await asyncio.sleep(5)
        
        # 获取训练结果
        task_result = await call_service(
            MODEL_TRAINER_URL,
            f"training/tasks/{training_data['task_id']}",
            "GET"
        )
        
        return {
            "task_id": training_data["task_id"],
            "model_id": task_result.get("model_id"),
            "training_metrics": task_result.get("metrics", {}),
            "model_performance": task_result.get("performance", {})
        }
    
    async def _step_explanation(self) -> Dict[str, Any]:
        """步骤4: 模型解释"""
        model_id = self.step_results.get("training", {}).get("model_id")
        if not model_id:
            raise ValueError("训练步骤未产生有效模型ID")
        
        explanation_data = {
            "model_id": model_id,
            "explanation_method": "federated_shap",
            "sample_size": 1000,
            "workflow_id": self.workflow_id
        }
        
        explanation_result = await call_service(
            MODEL_TRAINER_URL,
            "explanation/generate",
            "POST",
            explanation_data
        )
        
        return {
            "model_id": model_id,
            "explanation_id": explanation_result.get("explanation_id"),
            "feature_importance": explanation_result.get("feature_importance", {}),
            "explanation_quality": explanation_result.get("quality_score", 0.0)
        }
    
    async def _step_deployment(self) -> Dict[str, Any]:
        """步骤5: 模型部署"""
        model_id = self.step_results.get("training", {}).get("model_id")
        if not model_id:
            raise ValueError("训练步骤未产生有效模型ID")
        
        deployment_data = {
            "model_id": model_id,
            "deployment_env": "production",
            "scaling_config": {
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            },
            "workflow_id": self.workflow_id
        }
        
        deployment_result = await call_service(
            MODEL_SERVING_URL,
            "models/deploy",
            "POST",
            deployment_data
        )
        
        return {
            "model_id": model_id,
            "deployment_id": deployment_result.get("deployment_id"),
            "endpoint_url": deployment_result.get("endpoint_url"),
            "deployment_status": deployment_result.get("status", "deployed")
        }
    
    async def _step_audit(self) -> Dict[str, Any]:
        """步骤6: 审计记录"""
        audit_data = {
            "workflow_id": self.workflow_id,
            "audit_scope": "full",
            "compliance_standards": ["GDPR", "CCPA", "PCI_DSS"],
            "step_results": self.step_results,
            "parties": self.request.parties,
            "data_purpose": self.request.data_purpose
        }
        
        audit_result = await call_service(
            AUDIT_SERVICE_URL,
            "audit/records",
            "POST",
            audit_data
        )
        
        return {
            "audit_id": audit_result.get("audit_id"),
            "compliance_score": audit_result.get("compliance_score", 0.0),
            "audit_report_url": audit_result.get("report_url"),
            "compliance_status": audit_result.get("status", "compliant")
        }

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    await init_database()
    await init_redis()
    await register_services()
    
    # 启动健康检查任务
    asyncio.create_task(periodic_health_check())
    
    logger.info("API网关服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    
    logger.info("API网关服务关闭完成")

async def periodic_health_check():
    """定期健康检查"""
    while True:
        try:
            services = ["consent-gateway", "psi-service", "federated-orchestrator", 
                       "model-trainer", "feature-store", "model-serving", "audit-service"]
            
            for service_name in services:
                await check_service_health(service_name)
            
            await asyncio.sleep(30)  # 每30秒检查一次
        except Exception as e:
            logger.error(f"定期健康检查失败: {e}")
            await asyncio.sleep(60)

# 注册路由
from routes import workflow_router, step_router, health_router, metrics_router, monitor_request

app.include_router(workflow_router)
app.include_router(step_router)
app.include_router(health_router)
app.include_router(metrics_router)

# 添加请求监控中间件
app.middleware("http")(monitor_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)