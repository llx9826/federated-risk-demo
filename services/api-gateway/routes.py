#!/usr/bin/env python3
"""
API网关路由实现 - 六步闭环API端点

实现功能：
1. 工作流管理API
2. 六步闭环各步骤API
3. 服务健康检查API
4. 监控指标API
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app import (
    db_pool, redis_client, active_workflows,
    api_requests_total, api_request_duration, workflow_executions_total,
    WorkflowRequest, WorkflowStatus, ConsentRequest, AlignmentRequest,
    TrainingRequest, ExplanationRequest, DeploymentRequest, AuditRequest,
    ServiceHealthResponse, GatewayHealthResponse,
    WorkflowStatus as WStatus, WorkflowStep, ServiceStatus,
    FederatedRiskWorkflow, check_service_health, update_workflow_status
)

# 创建路由器
workflow_router = APIRouter(prefix="/api/v1/workflows", tags=["工作流管理"])
step_router = APIRouter(prefix="/api/v1/steps", tags=["工作流步骤"])
health_router = APIRouter(prefix="/health", tags=["健康检查"])
metrics_router = APIRouter(prefix="/metrics", tags=["监控指标"])

# 中间件：请求监控
async def monitor_request(request: Request, call_next):
    """请求监控中间件"""
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    try:
        response = await call_next(request)
        status = response.status_code
        
        # 记录指标
        api_requests_total.labels(method=method, endpoint=path, status=status).inc()
        api_request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        # 记录错误指标
        api_requests_total.labels(method=method, endpoint=path, status=500).inc()
        api_request_duration.labels(method=method, endpoint=path).observe(time.time() - start_time)
        raise

# 工作流管理API
@workflow_router.post("/start", summary="启动联邦风控工作流")
async def start_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """启动六步闭环工作流"""
    try:
        # 检查工作流是否已存在
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT workflow_id FROM workflows WHERE workflow_id = $1",
                request.workflow_id
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="工作流ID已存在")
        
        # 创建工作流记录
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows (
                    workflow_id, workflow_type, status, parties, coordinator_party,
                    data_purpose, model_config, privacy_config, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
                request.workflow_id, request.workflow_type, WStatus.PENDING.value,
                request.parties, request.coordinator_party, request.data_purpose,
                request.model_config, request.privacy_config, request.metadata
            )
        
        # 添加到内存缓存
        active_workflows[request.workflow_id] = {
            "status": WStatus.PENDING,
            "current_step": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "request": request
        }
        
        # 后台执行工作流
        workflow = FederatedRiskWorkflow(request.workflow_id, request)
        background_tasks.add_task(workflow.execute)
        
        # 更新指标
        workflow_executions_total.labels(workflow_type="federated_risk_control", status="started").inc()
        
        return {
            "workflow_id": request.workflow_id,
            "status": "started",
            "message": "工作流已启动，正在后台执行",
            "estimated_duration_minutes": 15
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动工作流失败: {str(e)}")

@workflow_router.get("/{workflow_id}/status", summary="获取工作流状态")
async def get_workflow_status(workflow_id: str):
    """获取工作流执行状态"""
    try:
        async with db_pool.acquire() as conn:
            # 获取工作流基本信息
            workflow_row = await conn.fetchrow(
                "SELECT * FROM workflows WHERE workflow_id = $1",
                workflow_id
            )
            
            if not workflow_row:
                raise HTTPException(status_code=404, detail="工作流不存在")
            
            # 获取步骤执行情况
            step_rows = await conn.fetch(
                "SELECT * FROM workflow_steps WHERE workflow_id = $1 ORDER BY started_at",
                workflow_id
            )
            
            completed_steps = [row['step_name'] for row in step_rows if row['status'] == 'completed']
            failed_steps = [row['step_name'] for row in step_rows if row['status'] == 'failed']
            
            # 计算进度百分比
            total_steps = 6  # 六步闭环
            progress_percentage = (len(completed_steps) / total_steps) * 100
            
            # 估算完成时间
            estimated_completion = None
            if workflow_row['status'] == WStatus.RUNNING.value and completed_steps:
                avg_step_duration = sum(row['duration_ms'] or 0 for row in step_rows) / len(step_rows)
                remaining_steps = total_steps - len(completed_steps)
                estimated_seconds = (remaining_steps * avg_step_duration) / 1000
                estimated_completion = datetime.utcnow().timestamp() + estimated_seconds
            
            # 收集步骤结果
            step_results = {}
            for row in step_rows:
                if row['result_data']:
                    step_results[row['step_name']] = row['result_data']
            
            return {
                "workflow_id": workflow_id,
                "status": workflow_row['status'],
                "current_step": workflow_row['current_step'],
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "progress_percentage": progress_percentage,
                "created_at": workflow_row['created_at'].isoformat(),
                "updated_at": workflow_row['updated_at'].isoformat(),
                "estimated_completion": estimated_completion,
                "error_message": None,
                "step_results": step_results
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工作流状态失败: {str(e)}")

@workflow_router.get("/", summary="列出所有工作流")
async def list_workflows(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    """列出工作流"""
    try:
        async with db_pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    "SELECT * FROM workflows WHERE status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                    status, limit, offset
                )
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM workflows WHERE status = $1",
                    status
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM workflows ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
                total = await conn.fetchval("SELECT COUNT(*) FROM workflows")
            
            workflows = []
            for row in rows:
                workflows.append({
                    "workflow_id": row['workflow_id'],
                    "workflow_type": row['workflow_type'],
                    "status": row['status'],
                    "current_step": row['current_step'],
                    "parties": row['parties'],
                    "coordinator_party": row['coordinator_party'],
                    "data_purpose": row['data_purpose'],
                    "created_at": row['created_at'].isoformat(),
                    "updated_at": row['updated_at'].isoformat(),
                    "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None
                })
            
            return {
                "workflows": workflows,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出工作流失败: {str(e)}")

@workflow_router.delete("/{workflow_id}", summary="取消工作流")
async def cancel_workflow(workflow_id: str):
    """取消正在执行的工作流"""
    try:
        async with db_pool.acquire() as conn:
            # 检查工作流是否存在
            workflow_row = await conn.fetchrow(
                "SELECT status FROM workflows WHERE workflow_id = $1",
                workflow_id
            )
            
            if not workflow_row:
                raise HTTPException(status_code=404, detail="工作流不存在")
            
            if workflow_row['status'] in [WStatus.COMPLETED.value, WStatus.FAILED.value, WStatus.CANCELLED.value]:
                raise HTTPException(status_code=400, detail="工作流已完成，无法取消")
            
            # 更新状态为已取消
            await update_workflow_status(workflow_id, WStatus.CANCELLED)
            
            # 从内存缓存中移除
            if workflow_id in active_workflows:
                del active_workflows[workflow_id]
            
            return {
                "workflow_id": workflow_id,
                "status": "cancelled",
                "message": "工作流已取消"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消工作流失败: {str(e)}")

# 工作流步骤API
@step_router.post("/consent", summary="执行同意收集步骤")
async def execute_consent_step(request: ConsentRequest):
    """单独执行同意收集步骤"""
    try:
        # 这里可以单独执行某个步骤，用于调试或重试
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_consent()
        
        return {
            "step": "consent",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同意收集步骤失败: {str(e)}")

@step_router.post("/alignment", summary="执行数据对齐步骤")
async def execute_alignment_step(request: AlignmentRequest):
    """单独执行数据对齐步骤"""
    try:
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_alignment()
        
        return {
            "step": "alignment",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据对齐步骤失败: {str(e)}")

@step_router.post("/training", summary="执行联邦训练步骤")
async def execute_training_step(request: TrainingRequest):
    """单独执行联邦训练步骤"""
    try:
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_training()
        
        return {
            "step": "training",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"联邦训练步骤失败: {str(e)}")

@step_router.post("/explanation", summary="执行模型解释步骤")
async def execute_explanation_step(request: ExplanationRequest):
    """单独执行模型解释步骤"""
    try:
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_explanation()
        
        return {
            "step": "explanation",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型解释步骤失败: {str(e)}")

@step_router.post("/deployment", summary="执行模型部署步骤")
async def execute_deployment_step(request: DeploymentRequest):
    """单独执行模型部署步骤"""
    try:
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_deployment()
        
        return {
            "step": "deployment",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型部署步骤失败: {str(e)}")

@step_router.post("/audit", summary="执行审计记录步骤")
async def execute_audit_step(request: AuditRequest):
    """单独执行审计记录步骤"""
    try:
        workflow = FederatedRiskWorkflow(request.workflow_id, None)
        result = await workflow._step_audit()
        
        return {
            "step": "audit",
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"审计记录步骤失败: {str(e)}")

# 健康检查API
@health_router.get("/", summary="网关健康检查")
async def gateway_health():
    """API网关健康检查"""
    try:
        # 检查数据库连接
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # 检查Redis连接
        await redis_client.ping()
        
        # 检查所有后端服务
        services = ["consent-gateway", "psi-service", "federated-orchestrator", 
                   "model-trainer", "feature-store", "model-serving", "audit-service"]
        
        service_statuses = {}
        for service_name in services:
            health_result = await check_service_health(service_name)
            service_statuses[service_name] = health_result
        
        # 计算活跃工作流数量
        active_count = len([w for w in active_workflows.values() 
                          if w["status"] in [WStatus.PENDING, WStatus.RUNNING]])
        
        return GatewayHealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            active_workflows=active_count,
            service_statuses=service_statuses
        )
        
    except Exception as e:
        return GatewayHealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            active_workflows=0,
            service_statuses={}
        )

@health_router.get("/services", summary="后端服务健康状态")
async def services_health():
    """获取所有后端服务健康状态"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM service_registry ORDER BY service_name"
            )
            
            services = []
            for row in rows:
                services.append({
                    "service_name": row['service_name'],
                    "service_url": row['service_url'],
                    "status": row['status'],
                    "last_health_check": row['last_health_check'].isoformat(),
                    "response_time_ms": row['response_time_ms'],
                    "error_message": row['error_message']
                })
            
            return {
                "services": services,
                "total_services": len(services),
                "healthy_services": len([s for s in services if s["status"] == ServiceStatus.HEALTHY.value]),
                "unhealthy_services": len([s for s in services if s["status"] == ServiceStatus.UNHEALTHY.value])
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取服务健康状态失败: {str(e)}")

# 监控指标API
@metrics_router.get("/prometheus", summary="Prometheus指标")
async def prometheus_metrics():
    """返回Prometheus格式的指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@metrics_router.get("/summary", summary="指标摘要")
async def metrics_summary():
    """返回指标摘要"""
    try:
        async with db_pool.acquire() as conn:
            # 工作流统计
            workflow_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_workflows,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_workflows,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_workflows,
                    COUNT(*) FILTER (WHERE status = 'running') as running_workflows,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending_workflows
                FROM workflows
            """)
            
            # 步骤统计
            step_stats = await conn.fetch("""
                SELECT 
                    step_name,
                    COUNT(*) as total_executions,
                    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                    AVG(duration_ms) as avg_duration_ms
                FROM workflow_steps
                GROUP BY step_name
            """)
            
            # 服务健康统计
            service_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_services,
                    COUNT(*) FILTER (WHERE status = 'healthy') as healthy_services,
                    AVG(response_time_ms) as avg_response_time_ms
                FROM service_registry
            """)
        
        return {
            "workflow_metrics": dict(workflow_stats),
            "step_metrics": [dict(row) for row in step_stats],
            "service_metrics": dict(service_stats),
            "active_workflows": len(active_workflows),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取指标摘要失败: {str(e)}")

# 导出路由器
__all__ = ["workflow_router", "step_router", "health_router", "metrics_router", "monitor_request"]