#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习模型训练器 (Model Trainer)

功能:
1. SecureBoost / Fed-XGBoost 训练与评估
2. 支持SecAgg、差分隐私(DP)
3. 联邦SHAP解释
4. 模型注册和版本管理
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx
import asyncpg
import redis.asyncio as redis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 环境变量配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/trainer_db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MODEL_STORAGE_PATH = os.getenv('MODEL_STORAGE_PATH', '/app/models')
REPORTS_PATH = os.getenv('REPORTS_PATH', '/app/reports')
FEATURE_STORE_URL = os.getenv('FEATURE_STORE_URL', 'http://feature-store:8080')
MODEL_SERVING_URL = os.getenv('MODEL_SERVING_URL', 'http://model-serving:8080')
AUDIT_SERVICE_URL = os.getenv('AUDIT_SERVICE_URL', 'http://audit-ledger:8080')

# 创建必要目录
Path(MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)

# FastAPI应用初始化
app = FastAPI(
    title="联邦学习模型训练器",
    description="SecureBoost和Fed-XGBoost训练与评估服务",
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

# 枚举定义
class AlgorithmType(str, Enum):
    SECURE_BOOST = "secure_boost"
    FED_XGBOOST = "fed_xgboost"
    VERTICAL_LR = "vertical_lr"

class PrivacyLevel(str, Enum):
    NONE = "none"  # ε=∞
    LOW = "low"    # ε=8
    MEDIUM = "medium"  # ε=5
    HIGH = "high"  # ε=3

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic模型定义
class TrainingConfig(BaseModel):
    """训练配置"""
    algorithm: AlgorithmType = Field(..., description="训练算法")
    privacy_level: PrivacyLevel = Field(default=PrivacyLevel.NONE, description="隐私保护级别")
    enable_secure_agg: bool = Field(default=False, description="启用安全聚合")
    num_rounds: int = Field(default=10, ge=1, le=100, description="训练轮数")
    learning_rate: float = Field(default=0.1, gt=0, le=1, description="学习率")
    max_depth: int = Field(default=6, ge=1, le=20, description="最大深度")
    subsample: float = Field(default=0.8, gt=0, le=1, description="子采样率")
    colsample_bytree: float = Field(default=0.8, gt=0, le=1, description="特征采样率")
    reg_alpha: float = Field(default=0.0, ge=0, description="L1正则化")
    reg_lambda: float = Field(default=1.0, ge=0, description="L2正则化")
    early_stopping_rounds: int = Field(default=10, ge=1, description="早停轮数")
    
class TrainingRequest(BaseModel):
    """训练请求"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    participants: List[str] = Field(..., description="参与方列表")
    target_column: str = Field(..., description="目标列名")
    feature_columns: List[str] = Field(..., description="特征列名")
    config: TrainingConfig = Field(..., description="训练配置")
    data_sources: Dict[str, str] = Field(..., description="数据源配置")
    
    @validator('participants')
    def validate_participants(cls, v):
        if len(v) < 2:
            raise ValueError('至少需要2个参与方')
        return v

class ModelEvaluationResult(BaseModel):
    """模型评估结果"""
    auc: float = Field(..., description="AUC值")
    ks: float = Field(..., description="KS值")
    precision: float = Field(..., description="精确率")
    recall: float = Field(..., description="召回率")
    f1_score: float = Field(..., description="F1分数")
    training_time: float = Field(..., description="训练时间(秒)")
    communication_cost: float = Field(..., description="通信开销(MB)")
    privacy_budget_used: float = Field(..., description="隐私预算消耗")

class SHAPExplanation(BaseModel):
    """SHAP解释结果"""
    global_importance: Dict[str, float] = Field(..., description="全局特征重要性")
    feature_interactions: Dict[str, float] = Field(..., description="特征交互")
    privacy_preserving: bool = Field(..., description="是否隐私保护")
    explanation_quality: float = Field(..., description="解释质量分数")

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
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(64) UNIQUE NOT NULL,
                    task_name VARCHAR(128) NOT NULL,
                    algorithm VARCHAR(32) NOT NULL,
                    privacy_level VARCHAR(16) NOT NULL,
                    participants TEXT[] NOT NULL,
                    config JSONB NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    progress FLOAT DEFAULT 0.0,
                    current_round INTEGER DEFAULT 0,
                    total_rounds INTEGER DEFAULT 10,
                    model_hash VARCHAR(64),
                    model_path VARCHAR(256),
                    evaluation_results JSONB,
                    shap_results JSONB,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                );
                
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(64) NOT NULL,
                    round_number INTEGER NOT NULL,
                    participant_id VARCHAR(64) NOT NULL,
                    metrics JSONB NOT NULL,
                    privacy_budget_used FLOAT DEFAULT 0.0,
                    communication_bytes BIGINT DEFAULT 0,
                    computation_time FLOAT DEFAULT 0.0,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(task_id, round_number, participant_id)
                );
                
                CREATE TABLE IF NOT EXISTS model_registry (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(64) UNIQUE NOT NULL,
                    task_id VARCHAR(64) NOT NULL,
                    model_name VARCHAR(128) NOT NULL,
                    algorithm VARCHAR(32) NOT NULL,
                    version INTEGER NOT NULL,
                    model_hash VARCHAR(64) NOT NULL,
                    model_path VARCHAR(256) NOT NULL,
                    performance_metrics JSONB NOT NULL,
                    feature_importance JSONB,
                    privacy_guarantees JSONB,
                    deployment_config JSONB,
                    status VARCHAR(32) NOT NULL DEFAULT 'trained',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deployed_at TIMESTAMPTZ
                );
                
                CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs (status);
                CREATE INDEX IF NOT EXISTS idx_training_jobs_algorithm ON training_jobs (algorithm);
                CREATE INDEX IF NOT EXISTS idx_training_metrics_task ON training_metrics (task_id);
                CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry (status);
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
def calculate_model_hash(model_data: bytes) -> str:
    """计算模型哈希"""
    return hashlib.sha256(model_data).hexdigest()

def get_privacy_epsilon(privacy_level: PrivacyLevel) -> float:
    """获取隐私预算epsilon值"""
    epsilon_map = {
        PrivacyLevel.NONE: float('inf'),
        PrivacyLevel.LOW: 8.0,
        PrivacyLevel.MEDIUM: 5.0,
        PrivacyLevel.HIGH: 3.0
    }
    return epsilon_map[privacy_level]

def add_differential_privacy_noise(data: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
    """添加差分隐私噪声"""
    if epsilon == float('inf'):
        return data
    
    # 拉普拉斯噪声
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

def calculate_ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """计算KS统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks = np.max(tpr - fpr)
    return ks

async def send_audit_log(event_type: str, event_data: dict):
    """发送审计日志"""
    try:
        audit_record = {
            "audit_id": f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": "HIGH",
            "source": {
                "service": "model-trainer",
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

# 联邦学习算法实现
class SecureBoostTrainer:
    """SecureBoost训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.epsilon = get_privacy_epsilon(config.privacy_level)
        self.models = {}
        self.feature_importance = {}
        
    async def train(self, task_id: str, participants_data: Dict[str, pd.DataFrame]) -> Tuple[Any, ModelEvaluationResult]:
        """训练SecureBoost模型"""
        start_time = datetime.utcnow()
        communication_cost = 0.0
        
        try:
            # 合并参与方数据（模拟纵向联邦学习）
            combined_data = self._combine_vertical_data(participants_data)
            
            # 分割训练和测试集
            X = combined_data.drop('target', axis=1)
            y = combined_data['target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 创建XGBoost模型
            model = xgb.XGBClassifier(
                n_estimators=self.config.num_rounds,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=42,
                eval_metric='auc'
            )
            
            # 训练模型（模拟安全聚合）
            if self.config.enable_secure_agg:
                model = await self._secure_aggregation_training(
                    model, X_train, y_train, X_test, y_test, participants_data
                )
                communication_cost += len(participants_data) * 10.0  # 模拟通信开销
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            
            # 预测和评估
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 添加差分隐私噪声
            if self.epsilon != float('inf'):
                y_pred_proba = add_differential_privacy_noise(
                    y_pred_proba, self.epsilon, sensitivity=0.1
                )
                y_pred_proba = np.clip(y_pred_proba, 0, 1)
            
            # 计算评估指标
            auc = roc_auc_score(y_test, y_pred_proba)
            ks = calculate_ks_statistic(y_test, y_pred_proba)
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            f1 = 2 * (precision.mean() * recall.mean()) / (precision.mean() + recall.mean())
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            privacy_budget_used = 1.0 / self.epsilon if self.epsilon != float('inf') else 0.0
            
            # 保存模型
            model_path = f"{MODEL_STORAGE_PATH}/{task_id}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存特征重要性
            self.feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            evaluation_result = ModelEvaluationResult(
                auc=auc,
                ks=ks,
                precision=precision.mean(),
                recall=recall.mean(),
                f1_score=f1,
                training_time=training_time,
                communication_cost=communication_cost,
                privacy_budget_used=privacy_budget_used
            )
            
            return model, evaluation_result
            
        except Exception as e:
            logger.error(f"SecureBoost训练失败: {e}")
            raise
    
    def _combine_vertical_data(self, participants_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并纵向联邦数据"""
        # 假设第一个参与方有标签
        main_participant = list(participants_data.keys())[0]
        combined_data = participants_data[main_participant].copy()
        
        # 合并其他参与方的特征
        for participant_id, data in participants_data.items():
            if participant_id != main_participant:
                # 基于ID合并（实际应该通过PSI对齐）
                if 'id' in data.columns and 'id' in combined_data.columns:
                    combined_data = combined_data.merge(
                        data.drop('target', axis=1, errors='ignore'),
                        on='id',
                        how='inner'
                    )
                else:
                    # 简单拼接（仅用于演示）
                    for col in data.columns:
                        if col not in combined_data.columns:
                            combined_data[col] = data[col]
        
        return combined_data
    
    async def _secure_aggregation_training(self, model, X_train, y_train, X_test, y_test, participants_data):
        """安全聚合训练（模拟）"""
        # 模拟多轮安全聚合
        for round_num in range(self.config.num_rounds):
            # 每个参与方本地训练
            local_models = {}
            for participant_id in participants_data.keys():
                local_model = xgb.XGBClassifier(
                    n_estimators=1,  # 每轮只训练一棵树
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    random_state=42 + round_num
                )
                
                # 模拟本地训练
                local_model.fit(X_train, y_train)
                local_models[participant_id] = local_model
            
            # 模拟安全聚合（实际应该使用密码学协议）
            # 这里简化为模型参数平均
            await asyncio.sleep(0.1)  # 模拟通信延迟
        
        # 最终训练完整模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False
        )
        
        return model

class FederatedSHAPExplainer:
    """联邦SHAP解释器"""
    
    def __init__(self, model, feature_names: List[str], privacy_level: PrivacyLevel):
        self.model = model
        self.feature_names = feature_names
        self.privacy_level = privacy_level
        self.epsilon = get_privacy_epsilon(privacy_level)
    
    async def explain(self, X_sample: pd.DataFrame, num_samples: int = 100) -> SHAPExplanation:
        """生成联邦SHAP解释"""
        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(self.model)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample.iloc[:num_samples])
            
            # 计算全局特征重要性
            global_importance = np.abs(shap_values).mean(axis=0)
            
            # 添加差分隐私噪声
            if self.epsilon != float('inf'):
                global_importance = add_differential_privacy_noise(
                    global_importance, self.epsilon, sensitivity=0.1
                )
                global_importance = np.abs(global_importance)  # 确保非负
            
            # 归一化
            global_importance = global_importance / global_importance.sum()
            
            # 创建特征重要性字典
            importance_dict = dict(zip(self.feature_names, global_importance))
            
            # 计算特征交互（简化版本）
            feature_interactions = {}
            for i, feat1 in enumerate(self.feature_names[:5]):  # 只计算前5个特征的交互
                for j, feat2 in enumerate(self.feature_names[:5]):
                    if i < j:
                        interaction_key = f"{feat1}_x_{feat2}"
                        # 简化的交互计算
                        interaction_value = np.corrcoef(
                            shap_values[:, i], shap_values[:, j]
                        )[0, 1]
                        if not np.isnan(interaction_value):
                            feature_interactions[interaction_key] = abs(interaction_value)
            
            # 计算解释质量分数
            explanation_quality = min(1.0, len(X_sample) / 1000.0)  # 基于样本数量
            
            return SHAPExplanation(
                global_importance=importance_dict,
                feature_interactions=feature_interactions,
                privacy_preserving=(self.epsilon != float('inf')),
                explanation_quality=explanation_quality
            )
            
        except Exception as e:
            logger.error(f"SHAP解释生成失败: {e}")
            raise

# API路由
@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """启动训练任务"""
    try:
        # 检查任务是否已存在
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT task_id FROM training_jobs WHERE task_id = $1",
                request.task_id
            )
            
            if existing:
                raise HTTPException(status_code=409, detail="训练任务已存在")
            
            # 创建训练任务记录
            await conn.execute("""
                INSERT INTO training_jobs (
                    task_id, task_name, algorithm, privacy_level, participants,
                    config, total_rounds
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                request.task_id, request.task_name, request.config.algorithm.value,
                request.config.privacy_level.value, request.participants,
                json.dumps(request.config.dict()), request.config.num_rounds
            )
        
        # 后台启动训练
        background_tasks.add_task(execute_training, request)
        
        # 发送审计日志
        await send_audit_log("TRAINING_START", {
            "task_id": request.task_id,
            "algorithm": request.config.algorithm.value,
            "privacy_level": request.config.privacy_level.value,
            "participants": request.participants
        })
        
        return {
            "message": "训练任务已启动",
            "task_id": request.task_id,
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

async def execute_training(request: TrainingRequest):
    """执行训练任务"""
    try:
        # 更新状态为运行中
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE training_jobs SET status = 'running', updated_at = NOW() WHERE task_id = $1",
                request.task_id
            )
        
        # 模拟加载参与方数据
        participants_data = await load_participants_data(request)
        
        # 根据算法类型选择训练器
        if request.config.algorithm == AlgorithmType.SECURE_BOOST:
            trainer = SecureBoostTrainer(request.config)
            model, evaluation_result = await trainer.train(request.task_id, participants_data)
        else:
            raise ValueError(f"不支持的算法: {request.config.algorithm}")
        
        # 计算模型哈希
        model_path = f"{MODEL_STORAGE_PATH}/{request.task_id}_model.pkl"
        with open(model_path, 'rb') as f:
            model_data = f.read()
        model_hash = calculate_model_hash(model_data)
        
        # 生成SHAP解释
        sample_data = list(participants_data.values())[0]
        explainer = FederatedSHAPExplainer(
            model, request.feature_columns, request.config.privacy_level
        )
        shap_explanation = await explainer.explain(sample_data)
        
        # 更新训练任务状态
        async with db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE training_jobs SET 
                    status = 'completed',
                    progress = 1.0,
                    current_round = $1,
                    model_hash = $2,
                    model_path = $3,
                    evaluation_results = $4,
                    shap_results = $5,
                    completed_at = NOW(),
                    updated_at = NOW()
                WHERE task_id = $6
            """, 
                request.config.num_rounds, model_hash, model_path,
                json.dumps(evaluation_result.dict()), json.dumps(shap_explanation.dict()),
                request.task_id
            )
        
        # 注册模型到模型服务
        await register_model_to_serving(request.task_id, model_hash, model_path, evaluation_result)
        
        # 生成训练报告
        await generate_training_report(request.task_id, evaluation_result, shap_explanation)
        
        logger.info(f"训练任务完成: {request.task_id}")
        
    except Exception as e:
        logger.error(f"训练任务执行失败: {e}")
        
        # 更新状态为失败
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE training_jobs SET status = 'failed', error_message = $1, updated_at = NOW() WHERE task_id = $2",
                str(e), request.task_id
            )

async def load_participants_data(request: TrainingRequest) -> Dict[str, pd.DataFrame]:
    """加载参与方数据（模拟）"""
    participants_data = {}
    
    for participant_id in request.participants:
        # 模拟生成数据
        np.random.seed(hash(participant_id) % 2**32)
        n_samples = 10000
        n_features = len(request.feature_columns) // len(request.participants)
        
        # 生成特征数据
        features = np.random.randn(n_samples, n_features)
        feature_names = request.feature_columns[:n_features]
        
        data = pd.DataFrame(features, columns=feature_names)
        data['id'] = range(n_samples)
        
        # 第一个参与方有标签
        if participant_id == request.participants[0]:
            # 生成目标变量
            target = (features.sum(axis=1) + np.random.randn(n_samples) * 0.1) > 0
            data['target'] = target.astype(int)
        
        participants_data[participant_id] = data
    
    return participants_data

async def register_model_to_serving(task_id: str, model_hash: str, model_path: str, evaluation_result: ModelEvaluationResult):
    """注册模型到模型服务"""
    try:
        model_info = {
            "model_id": f"model_{task_id}",
            "task_id": task_id,
            "model_hash": model_hash,
            "model_path": model_path,
            "performance_metrics": evaluation_result.dict(),
            "algorithm": "secure_boost",
            "version": 1
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MODEL_SERVING_URL}/models/register",
                json=model_info,
                timeout=10.0
            )
            response.raise_for_status()
            
        logger.info(f"模型注册成功: {task_id}")
        
    except Exception as e:
        logger.error(f"模型注册失败: {e}")

async def generate_training_report(task_id: str, evaluation_result: ModelEvaluationResult, shap_explanation: SHAPExplanation):
    """生成训练报告"""
    try:
        report = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": evaluation_result.dict(),
            "feature_importance": shap_explanation.global_importance,
            "privacy_guarantees": {
                "differential_privacy": shap_explanation.privacy_preserving,
                "privacy_budget_used": evaluation_result.privacy_budget_used
            },
            "training_summary": {
                "training_time": evaluation_result.training_time,
                "communication_cost": evaluation_result.communication_cost,
                "model_quality": {
                    "auc": evaluation_result.auc,
                    "ks": evaluation_result.ks,
                    "f1_score": evaluation_result.f1_score
                }
            }
        }
        
        # 保存报告
        report_path = f"{REPORTS_PATH}/train_report_{task_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告生成完成: {report_path}")
        
    except Exception as e:
        logger.error(f"训练报告生成失败: {e}")

@app.get("/tasks/{task_id}")
async def get_training_status(task_id: str):
    """获取训练状态"""
    try:
        async with db_pool.acquire() as conn:
            task = await conn.fetchrow(
                "SELECT * FROM training_jobs WHERE task_id = $1",
                task_id
            )
            
            if not task:
                raise HTTPException(status_code=404, detail="训练任务未找到")
            
            return dict(task)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/tasks")
async def list_training_tasks(status: Optional[TrainingStatus] = None, algorithm: Optional[AlgorithmType] = None):
    """列出训练任务"""
    try:
        query = "SELECT * FROM training_jobs WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = $" + str(len(params) + 1)
            params.append(status.value)
        
        if algorithm:
            query += " AND algorithm = $" + str(len(params) + 1)
            params.append(algorithm.value)
        
        query += " ORDER BY created_at DESC"
        
        async with db_pool.acquire() as conn:
            tasks = await conn.fetch(query, *params)
            return [dict(task) for task in tasks]
            
    except Exception as e:
        logger.error(f"列出训练任务失败: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.get("/tasks/{task_id}/report")
async def get_training_report(task_id: str):
    """获取训练报告"""
    try:
        report_path = f"{REPORTS_PATH}/train_report_{task_id}.json"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="训练报告未找到")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练报告失败: {e}")
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
    logger.info("启动模型训练器...")
    await init_database()
    await init_redis()
    logger.info("模型训练器启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("关闭模型训练器...")
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    logger.info("模型训练器已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8082,
        reload=True,
        log_level="info"
    )