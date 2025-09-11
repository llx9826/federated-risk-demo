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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="联邦学习训练服务",
    description="联邦学习模型训练和管理服务",
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
class TrainingRequest(BaseModel):
    job_name: str
    algorithm: str  # logistic_regression, random_forest
    participants: List[str]  # 参与方ID列表
    data_schema: Dict[str, str]  # 数据字段和类型
    target_column: str
    privacy_budget: float = 1.0
    max_iterations: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = True
    differential_privacy: bool = True
    noise_multiplier: float = 1.0

class TrainingJob(BaseModel):
    id: str
    job_name: str
    algorithm: str
    participants: List[str]
    status: str  # pending, running, completed, failed
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    privacy_budget_used: float = 0.0
    iterations_completed: int = 0

class ModelInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    training_job_id: str
    version: str
    created_at: str
    metrics: Dict[str, float]
    participants: List[str]
    privacy_budget_used: float
    file_path: str
    is_active: bool = True

class ParticipantData(BaseModel):
    participant_id: str
    data_hash: str
    feature_count: int
    sample_count: int
    data_quality_score: float
    privacy_contribution: float

class FederatedUpdate(BaseModel):
    participant_id: str
    iteration: int
    model_weights: List[float]
    gradient_norm: float
    loss: float
    privacy_cost: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    active_jobs: int
    total_models: int

# 全局状态
start_time = time.time()
training_jobs: Dict[str, TrainingJob] = {}
models: Dict[str, ModelInfo] = {}
participant_data: Dict[str, ParticipantData] = {}
federated_updates: Dict[str, List[FederatedUpdate]] = {}

# 确保模型存储目录存在
MODEL_STORAGE_PATH = "./models"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# 差分隐私相关函数
def add_noise(data: np.ndarray, noise_multiplier: float, sensitivity: float = 1.0) -> np.ndarray:
    """添加高斯噪声实现差分隐私"""
    noise = np.random.normal(0, noise_multiplier * sensitivity, data.shape)
    return data + noise

def calculate_privacy_cost(noise_multiplier: float, iterations: int, delta: float = 1e-5) -> float:
    """计算隐私成本（epsilon值）"""
    # 简化的隐私成本计算
    epsilon = iterations / (noise_multiplier ** 2)
    return epsilon

# 联邦学习相关函数
def aggregate_weights(updates: List[FederatedUpdate]) -> List[float]:
    """聚合参与方的模型权重"""
    if not updates:
        return []
    
    # 简单平均聚合
    aggregated = np.mean([update.model_weights for update in updates], axis=0)
    return aggregated.tolist()

def simulate_federated_training(job: TrainingJob) -> None:
    """模拟联邦学习训练过程"""
    try:
        logger.info(f"开始训练任务: {job.id}")
        job.status = "running"
        job.started_at = datetime.now().isoformat()
        
        # 生成模拟数据
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 根据算法选择模型
        if job.algorithm == "logistic_regression":
            model = LogisticRegression(random_state=42)
        elif job.algorithm == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的算法: {job.algorithm}")
        
        # 模拟联邦学习迭代
        for iteration in range(min(job.max_iterations, 10)):  # 限制迭代次数以加快演示
            # 模拟参与方更新
            updates = []
            for participant in job.participants:
                # 添加差分隐私噪声
                noisy_weights = add_noise(
                    np.random.randn(10), 
                    noise_multiplier=1.0
                )
                
                update = FederatedUpdate(
                    participant_id=participant,
                    iteration=iteration,
                    model_weights=noisy_weights.tolist(),
                    gradient_norm=np.linalg.norm(noisy_weights),
                    loss=np.random.uniform(0.3, 0.7),
                    privacy_cost=calculate_privacy_cost(1.0, iteration + 1)
                )
                updates.append(update)
            
            # 聚合更新
            aggregated_weights = aggregate_weights(updates)
            
            # 更新进度
            job.progress = (iteration + 1) / min(job.max_iterations, 10) * 100
            job.iterations_completed = iteration + 1
            
            # 存储更新历史
            if job.id not in federated_updates:
                federated_updates[job.id] = []
            federated_updates[job.id].extend(updates)
            
            time.sleep(0.5)  # 模拟训练时间
        
        # 训练最终模型
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # 保存模型
        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.joblib")
        joblib.dump(model, model_path)
        
        # 创建模型信息
        model_info = ModelInfo(
            id=model_id,
            name=f"{job.job_name}_model",
            algorithm=job.algorithm,
            training_job_id=job.id,
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            participants=job.participants,
            privacy_budget_used=calculate_privacy_cost(1.0, job.iterations_completed),
            file_path=model_path
        )
        
        models[model_id] = model_info
        
        # 更新训练任务状态
        job.status = "completed"
        job.completed_at = datetime.now().isoformat()
        job.model_id = model_id
        job.metrics = metrics
        job.privacy_budget_used = model_info.privacy_budget_used
        job.progress = 100.0
        
        logger.info(f"训练任务完成: {job.id}, 模型ID: {model_id}")
        
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.now().isoformat()
        logger.error(f"训练任务失败: {job.id}, 错误: {str(e)}")

# API端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    active_jobs = len([job for job in training_jobs.values() if job.status == "running"])
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=time.time() - start_time,
        active_jobs=active_jobs,
        total_models=len(models)
    )

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    job_status_counts = {}
    for job in training_jobs.values():
        job_status_counts[job.status] = job_status_counts.get(job.status, 0) + 1
    
    return {
        "total_jobs": len(training_jobs),
        "job_status_counts": job_status_counts,
        "total_models": len(models),
        "total_participants": len(participant_data),
        "uptime": time.time() - start_time
    }

@app.post("/training/jobs", response_model=TrainingJob)
async def create_training_job(request: TrainingRequest, background_tasks: BackgroundTasks):
    """创建训练任务"""
    job_id = str(uuid.uuid4())
    
    job = TrainingJob(
        id=job_id,
        job_name=request.job_name,
        algorithm=request.algorithm,
        participants=request.participants,
        status="pending",
        progress=0.0,
        created_at=datetime.now().isoformat()
    )
    
    training_jobs[job_id] = job
    
    # 在后台启动训练
    background_tasks.add_task(simulate_federated_training, job)
    
    logger.info(f"训练任务创建成功: {job_id}")
    return job

@app.get("/training/jobs", response_model=List[TrainingJob])
async def list_training_jobs(status: Optional[str] = None):
    """列出训练任务"""
    jobs = list(training_jobs.values())
    
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    return jobs

@app.get("/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """获取训练任务详情"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    return training_jobs[job_id]

@app.delete("/training/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """取消训练任务"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    job = training_jobs[job_id]
    if job.status == "running":
        job.status = "cancelled"
        job.completed_at = datetime.now().isoformat()
        logger.info(f"训练任务已取消: {job_id}")
    
    return {"message": "训练任务已取消"}

@app.get("/training/jobs/{job_id}/updates")
async def get_training_updates(job_id: str):
    """获取训练更新历史"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    return federated_updates.get(job_id, [])

@app.get("/models", response_model=List[ModelInfo])
async def list_models(algorithm: Optional[str] = None):
    """列出模型"""
    model_list = list(models.values())
    
    if algorithm:
        model_list = [model for model in model_list if model.algorithm == algorithm]
    
    return model_list

@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """获取模型详情"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    return models[model_id]

@app.post("/models/{model_id}/predict")
async def predict_with_model(model_id: str, data: Dict[str, Any]):
    """使用模型进行预测"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    model_info = models[model_id]
    
    try:
        # 加载模型
        model = joblib.load(model_info.file_path)
        
        # 准备预测数据
        features = data.get("features", [])
        if not features:
            raise ValueError("缺少特征数据")
        
        # 进行预测
        prediction = model.predict([features])
        probability = None
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([features])[0].tolist()
        
        return {
            "model_id": model_id,
            "prediction": prediction[0].tolist() if hasattr(prediction[0], 'tolist') else prediction[0],
            "probability": probability,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/participants/{participant_id}/data")
async def register_participant_data(participant_id: str, data_info: Dict[str, Any]):
    """注册参与方数据信息"""
    participant_data[participant_id] = ParticipantData(
        participant_id=participant_id,
        data_hash=data_info.get("data_hash", ""),
        feature_count=data_info.get("feature_count", 0),
        sample_count=data_info.get("sample_count", 0),
        data_quality_score=data_info.get("data_quality_score", 0.0),
        privacy_contribution=data_info.get("privacy_contribution", 0.0)
    )
    
    logger.info(f"参与方数据注册成功: {participant_id}")
    return {"message": "参与方数据注册成功"}

@app.get("/participants")
async def list_participants():
    """列出参与方"""
    return list(participant_data.values())

@app.get("/participants/{participant_id}")
async def get_participant(participant_id: str):
    """获取参与方详情"""
    if participant_id not in participant_data:
        raise HTTPException(status_code=404, detail="参与方不存在")
    
    return participant_data[participant_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)