from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid
import time
import json
import logging
from datetime import datetime
import numpy as np
import joblib
import os
import hashlib
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="联邦学习推理服务",
    description="联邦学习模型推理和审计服务",
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

# 数据模型
class InferenceRequest(BaseModel):
    model_id: str
    features: List[float]
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class InferenceResponse(BaseModel):
    request_id: str
    model_id: str
    prediction: Any
    confidence: Optional[float] = None
    probability: Optional[List[float]] = None
    processing_time: float
    timestamp: str
    audit_id: str

class ModelRegistry(BaseModel):
    id: str
    name: str
    version: str
    algorithm: str
    file_path: str
    created_at: str
    updated_at: str
    status: str  # active, inactive, deprecated
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    privacy_budget: float
    participants: List[str]
    checksum: str

class AuditLog(BaseModel):
    id: str
    request_id: str
    model_id: str
    user_id: Optional[str]
    input_hash: str
    output_hash: str
    processing_time: float
    timestamp: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: str  # success, error
    error_message: Optional[str] = None
    privacy_cost: float

class ModelPerformance(BaseModel):
    model_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time: float
    average_confidence: float
    last_updated: str
    performance_trend: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    active_models: int
    total_requests: int
    cache_hit_rate: float

# 全局状态
start_time = time.time()
model_registry: Dict[str, ModelRegistry] = {}
audit_logs: Dict[str, AuditLog] = {}
model_performance: Dict[str, ModelPerformance] = {}
model_cache: Dict[str, Any] = {}  # 模型缓存
prediction_cache: Dict[str, InferenceResponse] = {}  # 预测结果缓存

# 确保模型存储目录存在
MODEL_STORAGE_PATH = "./models"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# 工具函数
def calculate_hash(data: Any) -> str:
    """计算数据哈希值"""
    return hashlib.sha256(str(data).encode()).hexdigest()

def calculate_file_checksum(file_path: str) -> str:
    """计算文件校验和"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return ""

def load_model(model_id: str):
    """加载模型到缓存"""
    if model_id in model_cache:
        return model_cache[model_id]
    
    if model_id not in model_registry:
        raise ValueError(f"模型不存在: {model_id}")
    
    model_info = model_registry[model_id]
    
    if not os.path.exists(model_info.file_path):
        raise ValueError(f"模型文件不存在: {model_info.file_path}")
    
    try:
        model = joblib.load(model_info.file_path)
        model_cache[model_id] = model
        logger.info(f"模型加载成功: {model_id}")
        return model
    except Exception as e:
        raise ValueError(f"模型加载失败: {str(e)}")

def update_model_performance(model_id: str, processing_time: float, success: bool, confidence: Optional[float] = None):
    """更新模型性能统计"""
    if model_id not in model_performance:
        model_performance[model_id] = ModelPerformance(
            model_id=model_id,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_processing_time=0.0,
            average_confidence=0.0,
            last_updated=datetime.now().isoformat(),
            performance_trend=[]
        )
    
    perf = model_performance[model_id]
    perf.total_requests += 1
    
    if success:
        perf.successful_requests += 1
        # 更新平均处理时间
        perf.average_processing_time = (
            (perf.average_processing_time * (perf.successful_requests - 1) + processing_time) / 
            perf.successful_requests
        )
        
        # 更新平均置信度
        if confidence is not None:
            perf.average_confidence = (
                (perf.average_confidence * (perf.successful_requests - 1) + confidence) / 
                perf.successful_requests
            )
    else:
        perf.failed_requests += 1
    
    perf.last_updated = datetime.now().isoformat()
    
    # 添加性能趋势数据点
    perf.performance_trend.append({
        "timestamp": datetime.now().isoformat(),
        "processing_time": processing_time,
        "success": success,
        "confidence": confidence
    })
    
    # 保持趋势数据在合理范围内
    if len(perf.performance_trend) > 1000:
        perf.performance_trend = perf.performance_trend[-1000:]

# API端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    active_models = len([m for m in model_registry.values() if m.status == "active"])
    total_requests = sum(perf.total_requests for perf in model_performance.values())
    
    # 计算缓存命中率
    cache_hits = len(prediction_cache)
    cache_hit_rate = cache_hits / max(total_requests, 1) * 100
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=time.time() - start_time,
        active_models=active_models,
        total_requests=total_requests,
        cache_hit_rate=cache_hit_rate
    )

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    total_requests = sum(perf.total_requests for perf in model_performance.values())
    successful_requests = sum(perf.successful_requests for perf in model_performance.values())
    failed_requests = sum(perf.failed_requests for perf in model_performance.values())
    
    return {
        "total_models": len(model_registry),
        "active_models": len([m for m in model_registry.values() if m.status == "active"]),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate": successful_requests / max(total_requests, 1) * 100,
        "cache_size": len(model_cache),
        "audit_logs_count": len(audit_logs),
        "uptime": time.time() - start_time
    }

@app.post("/models/register", response_model=ModelRegistry)
async def register_model(model_data: Dict[str, Any]):
    """注册模型"""
    model_id = str(uuid.uuid4())
    
    # 验证模型文件
    file_path = model_data.get("file_path", "")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="模型文件不存在")
    
    checksum = calculate_file_checksum(file_path)
    
    model_registry_entry = ModelRegistry(
        id=model_id,
        name=model_data["name"],
        version=model_data.get("version", "1.0.0"),
        algorithm=model_data["algorithm"],
        file_path=file_path,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        status="active",
        metadata=model_data.get("metadata", {}),
        performance_metrics=model_data.get("performance_metrics", {}),
        privacy_budget=model_data.get("privacy_budget", 0.0),
        participants=model_data.get("participants", []),
        checksum=checksum
    )
    
    model_registry[model_id] = model_registry_entry
    
    logger.info(f"模型注册成功: {model_id}")
    return model_registry_entry

@app.get("/models", response_model=List[ModelRegistry])
async def list_models(status: Optional[str] = None, algorithm: Optional[str] = None):
    """列出模型"""
    models = list(model_registry.values())
    
    if status:
        models = [m for m in models if m.status == status]
    
    if algorithm:
        models = [m for m in models if m.algorithm == algorithm]
    
    return models

@app.get("/models/{model_id}", response_model=ModelRegistry)
async def get_model_info(model_id: str):
    """获取模型信息"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    return model_registry[model_id]

@app.post("/models/{model_id}/predict", response_model=InferenceResponse)
async def predict(model_id: str, request: InferenceRequest):
    """模型推理"""
    start_time_inference = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    audit_id = str(uuid.uuid4())
    
    try:
        # 检查模型是否存在且激活
        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        model_info = model_registry[model_id]
        if model_info.status != "active":
            raise HTTPException(status_code=400, detail="模型未激活")
        
        # 检查缓存
        cache_key = f"{model_id}_{calculate_hash(request.features)}"
        if cache_key in prediction_cache:
            cached_response = prediction_cache[cache_key]
            logger.info(f"使用缓存结果: {request_id}")
            return cached_response
        
        # 加载模型
        model = load_model(model_id)
        
        # 进行预测
        features = np.array([request.features])
        prediction = model.predict(features)
        
        # 计算置信度和概率
        confidence = None
        probability = None
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = proba.tolist()
            confidence = float(np.max(proba))
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(features)[0]
            confidence = float(abs(decision))
        
        processing_time = time.time() - start_time_inference
        
        # 创建响应
        response = InferenceResponse(
            request_id=request_id,
            model_id=model_id,
            prediction=prediction[0].tolist() if hasattr(prediction[0], 'tolist') else prediction[0],
            confidence=confidence,
            probability=probability,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            audit_id=audit_id
        )
        
        # 缓存结果
        prediction_cache[cache_key] = response
        
        # 创建审计日志
        audit_log = AuditLog(
            id=audit_id,
            request_id=request_id,
            model_id=model_id,
            user_id=request.user_id,
            input_hash=calculate_hash(request.features),
            output_hash=calculate_hash(response.prediction),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            status="success",
            privacy_cost=0.1  # 简化的隐私成本
        )
        
        audit_logs[audit_id] = audit_log
        
        # 更新性能统计
        update_model_performance(model_id, processing_time, True, confidence)
        
        logger.info(f"推理完成: {request_id}, 模型: {model_id}")
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time_inference
        
        # 创建错误审计日志
        audit_log = AuditLog(
            id=audit_id,
            request_id=request_id,
            model_id=model_id,
            user_id=request.user_id,
            input_hash=calculate_hash(request.features),
            output_hash="",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            status="error",
            error_message=str(e),
            privacy_cost=0.0
        )
        
        audit_logs[audit_id] = audit_log
        
        # 更新性能统计
        update_model_performance(model_id, processing_time, False)
        
        logger.error(f"推理失败: {request_id}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.post("/models/{model_id}/batch_predict")
async def batch_predict(model_id: str, requests: List[InferenceRequest]):
    """批量推理"""
    results = []
    
    for request in requests:
        try:
            result = await predict(model_id, request)
            results.append(result)
        except Exception as e:
            results.append({
                "request_id": request.request_id or str(uuid.uuid4()),
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/models/{model_id}/performance", response_model=ModelPerformance)
async def get_model_performance(model_id: str):
    """获取模型性能统计"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    if model_id not in model_performance:
        return ModelPerformance(
            model_id=model_id,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_processing_time=0.0,
            average_confidence=0.0,
            last_updated=datetime.now().isoformat(),
            performance_trend=[]
        )
    
    return model_performance[model_id]

@app.put("/models/{model_id}/status")
async def update_model_status(model_id: str, status_data: Dict[str, str]):
    """更新模型状态"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    new_status = status_data.get("status")
    if new_status not in ["active", "inactive", "deprecated"]:
        raise HTTPException(status_code=400, detail="无效的状态值")
    
    model_registry[model_id].status = new_status
    model_registry[model_id].updated_at = datetime.now().isoformat()
    
    # 如果模型被停用，从缓存中移除
    if new_status != "active" and model_id in model_cache:
        del model_cache[model_id]
    
    logger.info(f"模型状态更新: {model_id} -> {new_status}")
    return {"message": "模型状态更新成功"}

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    # 从注册表中移除
    del model_registry[model_id]
    
    # 从缓存中移除
    if model_id in model_cache:
        del model_cache[model_id]
    
    # 移除性能统计
    if model_id in model_performance:
        del model_performance[model_id]
    
    logger.info(f"模型删除成功: {model_id}")
    return {"message": "模型删除成功"}

@app.get("/audit/logs", response_model=List[AuditLog])
async def list_audit_logs(model_id: Optional[str] = None, user_id: Optional[str] = None, limit: int = 100):
    """列出审计日志"""
    logs = list(audit_logs.values())
    
    if model_id:
        logs = [log for log in logs if log.model_id == model_id]
    
    if user_id:
        logs = [log for log in logs if log.user_id == user_id]
    
    # 按时间戳排序并限制数量
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    return logs[:limit]

@app.get("/audit/logs/{audit_id}", response_model=AuditLog)
async def get_audit_log(audit_id: str):
    """获取审计日志详情"""
    if audit_id not in audit_logs:
        raise HTTPException(status_code=404, detail="审计日志不存在")
    
    return audit_logs[audit_id]

@app.get("/audit/summary")
async def get_audit_summary():
    """获取审计摘要"""
    total_logs = len(audit_logs)
    successful_logs = len([log for log in audit_logs.values() if log.status == "success"])
    failed_logs = total_logs - successful_logs
    
    total_privacy_cost = sum(log.privacy_cost for log in audit_logs.values())
    
    return {
        "total_logs": total_logs,
        "successful_requests": successful_logs,
        "failed_requests": failed_logs,
        "success_rate": successful_logs / max(total_logs, 1) * 100,
        "total_privacy_cost": total_privacy_cost,
        "average_privacy_cost": total_privacy_cost / max(total_logs, 1)
    }

@app.delete("/cache/clear")
async def clear_cache():
    """清空缓存"""
    model_cache.clear()
    prediction_cache.clear()
    
    logger.info("缓存已清空")
    return {"message": "缓存已清空"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)