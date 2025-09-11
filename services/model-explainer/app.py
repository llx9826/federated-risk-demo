#!/usr/bin/env python3
"""
模型解释服务 - 联邦学习模型可解释性分析

实现功能：
1. SHAP值计算和分析
2. LIME局部解释
3. 特征重要性分析
4. 模型决策边界可视化
5. 全局和局部解释报告
6. 公平性和偏见检测
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
import io

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

# 机器学习和解释性库
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer

# 环境配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/federated_risk")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONSENT_GATEWAY_URL = os.getenv("CONSENT_GATEWAY_URL", "http://localhost:8001")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./data/models")
EXPLANATION_STORAGE_PATH = os.getenv("EXPLANATION_STORAGE_PATH", "./data/explanations")
VISUALIZATION_STORAGE_PATH = os.getenv("VISUALIZATION_STORAGE_PATH", "./data/visualizations")

# 创建必要目录
Path("./logs").mkdir(exist_ok=True)
Path(MODEL_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(EXPLANATION_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(VISUALIZATION_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path("./data/reports").mkdir(parents=True, exist_ok=True)

# 全局变量
db_pool = None
redis_client = None

# Prometheus指标
explanation_requests_total = Counter('explanation_requests_total', 'Total explanation requests', ['method', 'model_type'])
explanation_processing_duration = Histogram('explanation_processing_duration_seconds', 'Explanation processing duration', ['method'])
model_fairness_score = Gauge('model_fairness_score', 'Model fairness score', ['model_id', 'metric'])
feature_importance_gauge = Gauge('feature_importance', 'Feature importance scores', ['model_id', 'feature'])
explanation_quality_score = Gauge('explanation_quality_score', 'Explanation quality score', ['model_id', 'method'])

# 枚举定义
from enum import Enum

class ExplanationMethod(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"
    FEDERATED_MODEL = "federated_model"

class ExplanationScope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    COHORT = "cohort"

class FairnessMetric(str, Enum):
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"

# FastAPI应用初始化
app = FastAPI(
    title="联邦风控模型解释服务",
    description="提供机器学习模型的可解释性分析和公平性评估",
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
class ModelUploadRequest(BaseModel):
    """模型上传请求"""
    model_id: str = Field(..., description="模型ID")
    model_type: ModelType = Field(..., description="模型类型")
    model_name: str = Field(..., description="模型名称")
    feature_names: List[str] = Field(..., description="特征名称列表")
    target_names: Optional[List[str]] = Field(None, description="目标类别名称")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="模型元数据")

class ExplanationRequest(BaseModel):
    """解释请求"""
    model_id: str = Field(..., description="模型ID")
    method: ExplanationMethod = Field(..., description="解释方法")
    scope: ExplanationScope = Field(default=ExplanationScope.GLOBAL, description="解释范围")
    sample_indices: Optional[List[int]] = Field(None, description="样本索引（局部解释用）")
    background_samples: Optional[int] = Field(100, description="背景样本数量")
    max_features: Optional[int] = Field(10, description="最大特征数量")
    generate_visualization: bool = Field(default=True, description="生成可视化")
    fairness_analysis: bool = Field(default=False, description="公平性分析")
    protected_attributes: Optional[List[str]] = Field(None, description="受保护属性")

class FairnessAnalysisRequest(BaseModel):
    """公平性分析请求"""
    model_id: str = Field(..., description="模型ID")
    protected_attributes: List[str] = Field(..., description="受保护属性")
    fairness_metrics: List[FairnessMetric] = Field(..., description="公平性指标")
    reference_group: Optional[str] = Field(None, description="参考组")
    generate_report: bool = Field(default=True, description="生成报告")

class ExplanationResponse(BaseModel):
    """解释响应"""
    explanation_id: str
    model_id: str
    method: ExplanationMethod
    scope: ExplanationScope
    feature_importance: Dict[str, float]
    explanation_values: Optional[Dict[str, Any]]
    visualization_urls: List[str]
    quality_score: float
    created_at: datetime
    expires_at: datetime

class FairnessAnalysisResponse(BaseModel):
    """公平性分析响应"""
    analysis_id: str
    model_id: str
    fairness_scores: Dict[str, float]
    bias_indicators: Dict[str, Any]
    recommendations: List[str]
    mitigation_strategies: List[str]
    report_url: Optional[str]
    created_at: datetime

class ModelSummaryResponse(BaseModel):
    """模型摘要响应"""
    model_id: str
    model_type: ModelType
    feature_count: int
    explanation_count: int
    latest_explanation: Optional[Dict[str, Any]]
    fairness_status: str
    interpretability_score: float

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    redis_status: str
    active_explanations: int

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
                    model_id VARCHAR(128) UNIQUE NOT NULL,
                    model_type VARCHAR(64) NOT NULL,
                    model_name VARCHAR(256) NOT NULL,
                    feature_names TEXT[] NOT NULL,
                    target_names TEXT[],
                    model_data BYTEA NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS explanations (
                    id SERIAL PRIMARY KEY,
                    explanation_id VARCHAR(128) UNIQUE NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    method VARCHAR(64) NOT NULL,
                    scope VARCHAR(32) NOT NULL,
                    feature_importance JSONB NOT NULL,
                    explanation_values JSONB,
                    visualization_paths TEXT[],
                    quality_score FLOAT NOT NULL DEFAULT 0.0,
                    processing_time FLOAT NOT NULL DEFAULT 0.0,
                    sample_indices INTEGER[],
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS fairness_analyses (
                    id SERIAL PRIMARY KEY,
                    analysis_id VARCHAR(128) UNIQUE NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    protected_attributes TEXT[] NOT NULL,
                    fairness_metrics TEXT[] NOT NULL,
                    fairness_scores JSONB NOT NULL,
                    bias_indicators JSONB NOT NULL,
                    recommendations TEXT[] NOT NULL,
                    mitigation_strategies TEXT[] NOT NULL,
                    report_path VARCHAR(512),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS explanation_datasets (
                    id SERIAL PRIMARY KEY,
                    dataset_id VARCHAR(128) UNIQUE NOT NULL,
                    model_id VARCHAR(128) NOT NULL,
                    data BYTEA NOT NULL,
                    feature_names TEXT[] NOT NULL,
                    target_column VARCHAR(128),
                    sample_count INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_models_model_id ON models (model_id);
                CREATE INDEX IF NOT EXISTS idx_explanations_model_id ON explanations (model_id);
                CREATE INDEX IF NOT EXISTS idx_explanations_method ON explanations (method);
                CREATE INDEX IF NOT EXISTS idx_fairness_analyses_model_id ON fairness_analyses (model_id);
                CREATE INDEX IF NOT EXISTS idx_explanation_datasets_model_id ON explanation_datasets (model_id);
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
async def generate_explanation_id() -> str:
    """生成解释ID"""
    return f"exp_{uuid.uuid4().hex[:16]}"

async def serialize_model(model) -> bytes:
    """序列化模型"""
    return pickle.dumps(model)

async def deserialize_model(model_data: bytes):
    """反序列化模型"""
    return pickle.loads(model_data)

async def calculate_explanation_quality(explanation_values: Dict[str, Any], 
                                      method: ExplanationMethod) -> float:
    """计算解释质量评分"""
    quality_score = 0.0
    
    if method == ExplanationMethod.SHAP:
        # SHAP解释质量基于值的一致性和覆盖度
        if 'shap_values' in explanation_values:
            shap_values = explanation_values['shap_values']
            if isinstance(shap_values, list) and len(shap_values) > 0:
                # 计算SHAP值的稳定性
                values_array = np.array(shap_values)
                if values_array.size > 0:
                    stability = 1.0 - np.std(values_array) / (np.mean(np.abs(values_array)) + 1e-8)
                    quality_score = max(0.0, min(1.0, stability))
    
    elif method == ExplanationMethod.LIME:
        # LIME解释质量基于局部拟合度
        if 'local_score' in explanation_values:
            quality_score = explanation_values['local_score']
    
    elif method == ExplanationMethod.FEATURE_IMPORTANCE:
        # 特征重要性质量基于重要性分布
        if 'importance_values' in explanation_values:
            importance = list(explanation_values['importance_values'].values())
            if importance:
                # 计算重要性分布的熵
                importance_array = np.array(importance)
                importance_array = importance_array / np.sum(importance_array)
                entropy = -np.sum(importance_array * np.log(importance_array + 1e-8))
                normalized_entropy = entropy / np.log(len(importance))
                quality_score = 1.0 - normalized_entropy  # 熵越低，质量越高
    
    return max(0.0, min(1.0, quality_score))

class ModelExplainer:
    """模型解释器类"""
    
    def __init__(self, model, feature_names: List[str], model_type: ModelType):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer_cache = {}
    
    async def explain_with_shap(self, X: np.ndarray, 
                              background_samples: int = 100,
                              max_features: int = 10) -> Dict[str, Any]:
        """使用SHAP进行解释"""
        try:
            # 选择合适的SHAP解释器
            if self.model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                if 'tree_explainer' not in self.explainer_cache:
                    self.explainer_cache['tree_explainer'] = shap.TreeExplainer(self.model)
                explainer = self.explainer_cache['tree_explainer']
            else:
                # 使用KernelExplainer作为通用解释器
                if 'kernel_explainer' not in self.explainer_cache:
                    background = X[:min(background_samples, len(X))]
                    self.explainer_cache['kernel_explainer'] = shap.KernelExplainer(
                        self.model.predict_proba, background
                    )
                explainer = self.explainer_cache['kernel_explainer']
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X[:min(100, len(X))])
            
            # 处理多类分类情况
            if isinstance(shap_values, list):
                # 多类分类，取第一类的SHAP值
                shap_values_processed = shap_values[0]
            else:
                shap_values_processed = shap_values
            
            # 计算特征重要性
            feature_importance = {}
            if len(shap_values_processed.shape) == 2:
                # 全局重要性：所有样本的SHAP值绝对值平均
                global_importance = np.mean(np.abs(shap_values_processed), axis=0)
                for i, feature in enumerate(self.feature_names[:len(global_importance)]):
                    feature_importance[feature] = float(global_importance[i])
            
            # 排序并取前N个特征
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:max_features]
            feature_importance = dict(sorted_features)
            
            return {
                'shap_values': shap_values_processed.tolist() if hasattr(shap_values_processed, 'tolist') else shap_values_processed,
                'feature_importance': feature_importance,
                'expected_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                'method': 'shap'
            }
            
        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            raise
    
    async def explain_with_lime(self, X: np.ndarray, 
                              sample_indices: List[int],
                              max_features: int = 10) -> Dict[str, Any]:
        """使用LIME进行局部解释"""
        try:
            # 创建LIME解释器
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                class_names=['0', '1'],  # 假设二分类
                mode='classification'
            )
            
            explanations = []
            feature_importance = {}
            
            for idx in sample_indices[:10]:  # 限制解释样本数量
                if idx < len(X):
                    # 解释单个样本
                    exp = explainer.explain_instance(
                        X[idx], 
                        self.model.predict_proba,
                        num_features=max_features
                    )
                    
                    # 提取特征重要性
                    local_importance = dict(exp.as_list())
                    explanations.append({
                        'sample_index': idx,
                        'local_importance': local_importance,
                        'local_score': exp.score
                    })
                    
                    # 累积全局重要性
                    for feature, importance in local_importance.items():
                        if feature in feature_importance:
                            feature_importance[feature] += abs(importance)
                        else:
                            feature_importance[feature] = abs(importance)
            
            # 归一化全局重要性
            if feature_importance:
                total_importance = sum(feature_importance.values())
                feature_importance = {k: v/total_importance 
                                    for k, v in feature_importance.items()}
            
            # 排序特征重要性
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:max_features]
            feature_importance = dict(sorted_features)
            
            return {
                'explanations': explanations,
                'feature_importance': feature_importance,
                'local_score': np.mean([exp['local_score'] for exp in explanations]) if explanations else 0.0,
                'method': 'lime'
            }
            
        except Exception as e:
            logger.error(f"LIME解释失败: {e}")
            raise
    
    async def explain_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                       max_features: int = 10) -> Dict[str, Any]:
        """计算特征重要性"""
        try:
            feature_importance = {}
            
            if hasattr(self.model, 'feature_importances_'):
                # 树模型的内置特征重要性
                importances = self.model.feature_importances_
                for i, feature in enumerate(self.feature_names[:len(importances)]):
                    feature_importance[feature] = float(importances[i])
            
            elif hasattr(self.model, 'coef_'):
                # 线性模型的系数作为重要性
                coef = self.model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # 取第一类的系数
                
                for i, feature in enumerate(self.feature_names[:len(coef)]):
                    feature_importance[feature] = float(abs(coef[i]))
            
            else:
                # 使用排列重要性
                from sklearn.inspection import permutation_importance
                
                perm_importance = permutation_importance(
                    self.model, X, y, n_repeats=10, random_state=42
                )
                
                for i, feature in enumerate(self.feature_names[:len(perm_importance.importances_mean)]):
                    feature_importance[feature] = float(perm_importance.importances_mean[i])
            
            # 归一化重要性
            if feature_importance:
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v/total_importance 
                                        for k, v in feature_importance.items()}
            
            # 排序并取前N个特征
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:max_features]
            feature_importance = dict(sorted_features)
            
            return {
                'importance_values': feature_importance,
                'method': 'feature_importance'
            }
            
        except Exception as e:
            logger.error(f"特征重要性计算失败: {e}")
            raise

class FairnessAnalyzer:
    """公平性分析器类"""
    
    def __init__(self, model, X: np.ndarray, y: np.ndarray, 
                 protected_attributes: List[str], feature_names: List[str]):
        self.model = model
        self.X = X
        self.y = y
        self.protected_attributes = protected_attributes
        self.feature_names = feature_names
        self.predictions = model.predict(X)
        self.prediction_probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    async def analyze_demographic_parity(self, sensitive_feature: np.ndarray) -> Dict[str, float]:
        """分析人口统计学平等性"""
        try:
            # 计算不同组的正预测率
            unique_groups = np.unique(sensitive_feature)
            selection_rates = {}
            
            for group in unique_groups:
                group_mask = sensitive_feature == group
                group_predictions = self.predictions[group_mask]
                selection_rate = np.mean(group_predictions)
                selection_rates[f'group_{group}'] = float(selection_rate)
            
            # 计算差异
            rates = list(selection_rates.values())
            parity_difference = max(rates) - min(rates)
            parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 0.0
            
            return {
                'selection_rates': selection_rates,
                'parity_difference': float(parity_difference),
                'parity_ratio': float(parity_ratio),
                'is_fair': parity_difference < 0.1  # 10%阈值
            }
            
        except Exception as e:
            logger.error(f"人口统计学平等性分析失败: {e}")
            raise
    
    async def analyze_equalized_odds(self, sensitive_feature: np.ndarray) -> Dict[str, float]:
        """分析机会均等性"""
        try:
            unique_groups = np.unique(sensitive_feature)
            tpr_scores = {}  # True Positive Rate
            fpr_scores = {}  # False Positive Rate
            
            for group in unique_groups:
                group_mask = sensitive_feature == group
                group_y_true = self.y[group_mask]
                group_y_pred = self.predictions[group_mask]
                
                # 计算TPR和FPR
                tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                
                tpr_scores[f'group_{group}'] = float(tpr)
                fpr_scores[f'group_{group}'] = float(fpr)
            
            # 计算差异
            tpr_values = list(tpr_scores.values())
            fpr_values = list(fpr_scores.values())
            
            tpr_difference = max(tpr_values) - min(tpr_values)
            fpr_difference = max(fpr_values) - min(fpr_values)
            
            return {
                'tpr_scores': tpr_scores,
                'fpr_scores': fpr_scores,
                'tpr_difference': float(tpr_difference),
                'fpr_difference': float(fpr_difference),
                'is_fair': tpr_difference < 0.1 and fpr_difference < 0.1
            }
            
        except Exception as e:
            logger.error(f"机会均等性分析失败: {e}")
            raise
    
    async def generate_bias_report(self, protected_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """生成偏见分析报告"""
        try:
            bias_indicators = {}
            overall_fairness = True
            
            for attr_name, attr_values in protected_features.items():
                # 人口统计学平等性
                demo_parity = await self.analyze_demographic_parity(attr_values)
                
                # 机会均等性
                eq_odds = await self.analyze_equalized_odds(attr_values)
                
                bias_indicators[attr_name] = {
                    'demographic_parity': demo_parity,
                    'equalized_odds': eq_odds
                }
                
                # 更新整体公平性
                if not demo_parity['is_fair'] or not eq_odds['is_fair']:
                    overall_fairness = False
            
            # 生成建议
            recommendations = []
            mitigation_strategies = []
            
            if not overall_fairness:
                recommendations.extend([
                    "检测到模型存在公平性问题，建议进行偏见缓解",
                    "考虑重新平衡训练数据",
                    "使用公平性约束的训练算法",
                    "实施后处理公平性调整"
                ])
                
                mitigation_strategies.extend([
                    "数据预处理：重采样、合成数据生成",
                    "算法内处理：公平性约束优化",
                    "后处理：阈值优化、校准",
                    "持续监控：部署后公平性监测"
                ])
            else:
                recommendations.append("模型通过公平性检查，建议定期监控")
                mitigation_strategies.append("维持当前公平性水平，定期重新评估")
            
            return {
                'bias_indicators': bias_indicators,
                'overall_fairness': overall_fairness,
                'recommendations': recommendations,
                'mitigation_strategies': mitigation_strategies
            }
            
        except Exception as e:
            logger.error(f"偏见分析报告生成失败: {e}")
            raise

# API路由实现
@app.post("/models/upload", summary="上传模型")
async def upload_model(request: ModelUploadRequest, model_file: UploadFile = File(...)):
    """上传机器学习模型"""
    try:
        # 读取模型文件
        model_data = await model_file.read()
        
        # 验证模型
        try:
            model = pickle.loads(model_data)
            if not hasattr(model, 'predict'):
                raise ValueError("上传的对象不是有效的机器学习模型")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"模型文件无效: {str(e)}")
        
        # 保存到数据库
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO models (
                    model_id, model_type, model_name, feature_names, 
                    target_names, model_data, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (model_id) 
                DO UPDATE SET model_data = $6, updated_at = NOW()
            """, 
                request.model_id, request.model_type.value, request.model_name,
                request.feature_names, request.target_names or [],
                model_data, json.dumps(request.metadata)
            )
        
        logger.info(f"模型上传成功: {request.model_id}")
        
        return {
            "status": "success",
            "message": "模型上传成功",
            "model_id": request.model_id,
            "model_type": request.model_type.value,
            "feature_count": len(request.feature_names)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型上传失败: {str(e)}")

@app.post("/models/{model_id}/data", summary="上传解释数据")
async def upload_explanation_data(model_id: str, data_file: UploadFile = File(...)):
    """上传用于解释的数据集"""
    try:
        # 检查模型是否存在
        async with db_pool.acquire() as conn:
            model_row = await conn.fetchrow(
                "SELECT model_id, feature_names FROM models WHERE model_id = $1",
                model_id
            )
            
            if not model_row:
                raise HTTPException(status_code=404, detail="模型不存在")
        
        # 读取数据文件
        file_content = await data_file.read()
        
        # 解析CSV数据
        try:
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"数据文件格式错误: {str(e)}")
        
        # 验证特征列
        feature_names = model_row['feature_names']
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"数据缺少必要特征: {list(missing_features)}"
            )
        
        # 序列化数据
        data_bytes = pickle.dumps(df)
        dataset_id = f"dataset_{uuid.uuid4().hex[:16]}"
        
        # 保存到数据库
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO explanation_datasets (
                    dataset_id, model_id, data, feature_names, 
                    target_column, sample_count
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                dataset_id, model_id, data_bytes, list(df.columns),
                'target' if 'target' in df.columns else None, len(df)
            )
        
        return {
            "status": "success",
            "message": "数据上传成功",
            "dataset_id": dataset_id,
            "sample_count": len(df),
            "feature_count": len(df.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据上传失败: {str(e)}")

@app.post("/explanations/generate", response_model=ExplanationResponse, summary="生成模型解释")
async def generate_explanation(request: ExplanationRequest, background_tasks: BackgroundTasks):
    """生成模型解释"""
    start_time = time.time()
    
    try:
        # 获取模型和数据
        async with db_pool.acquire() as conn:
            model_row = await conn.fetchrow(
                "SELECT * FROM models WHERE model_id = $1",
                request.model_id
            )
            
            if not model_row:
                raise HTTPException(status_code=404, detail="模型不存在")
            
            # 获取最新的数据集
            dataset_row = await conn.fetchrow("""
                SELECT * FROM explanation_datasets 
                WHERE model_id = $1 
                ORDER BY created_at DESC LIMIT 1
            """, request.model_id)
            
            if not dataset_row:
                raise HTTPException(status_code=404, detail="未找到解释数据集")
        
        # 反序列化模型和数据
        model = await deserialize_model(model_row['model_data'])
        df = pickle.loads(dataset_row['data'])
        
        # 准备数据
        feature_names = model_row['feature_names']
        X = df[feature_names].values
        y = df['target'].values if 'target' in df.columns else None
        
        # 创建解释器
        explainer = ModelExplainer(
            model, feature_names, ModelType(model_row['model_type'])
        )
        
        # 根据方法生成解释
        explanation_values = {}
        feature_importance = {}
        
        if request.method == ExplanationMethod.SHAP:
            result = await explainer.explain_with_shap(
                X, request.background_samples, request.max_features
            )
            explanation_values = result
            feature_importance = result['feature_importance']
        
        elif request.method == ExplanationMethod.LIME:
            if not request.sample_indices:
                # 默认选择前10个样本
                request.sample_indices = list(range(min(10, len(X))))
            
            result = await explainer.explain_with_lime(
                X, request.sample_indices, request.max_features
            )
            explanation_values = result
            feature_importance = result['feature_importance']
        
        elif request.method == ExplanationMethod.FEATURE_IMPORTANCE:
            if y is None:
                raise HTTPException(status_code=400, detail="特征重要性分析需要目标变量")
            
            result = await explainer.explain_feature_importance(
                X, y, request.max_features
            )
            explanation_values = result
            feature_importance = result['importance_values']
        
        # 计算解释质量
        quality_score = await calculate_explanation_quality(
            explanation_values, request.method
        )
        
        # 生成解释ID
        explanation_id = await generate_explanation_id()
        
        # 生成可视化（后台任务）
        visualization_urls = []
        if request.generate_visualization:
            background_tasks.add_task(
                generate_explanation_visualizations,
                explanation_id, request.method, explanation_values, feature_importance
            )
            visualization_urls = [
                f"/explanations/{explanation_id}/visualizations/importance.png",
                f"/explanations/{explanation_id}/visualizations/summary.png"
            ]
        
        # 保存解释结果
        expires_at = datetime.utcnow() + timedelta(days=7)  # 7天后过期
        processing_time = time.time() - start_time
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO explanations (
                    explanation_id, model_id, method, scope, feature_importance,
                    explanation_values, visualization_paths, quality_score,
                    processing_time, sample_indices, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                explanation_id, request.model_id, request.method.value, request.scope.value,
                json.dumps(feature_importance), json.dumps(explanation_values),
                visualization_urls, quality_score, processing_time,
                request.sample_indices or [], expires_at
            )
        
        # 更新指标
        explanation_requests_total.labels(
            method=request.method.value,
            model_type=model_row['model_type']
        ).inc()
        
        explanation_processing_duration.labels(
            method=request.method.value
        ).observe(processing_time)
        
        explanation_quality_score.labels(
            model_id=request.model_id,
            method=request.method.value
        ).set(quality_score)
        
        # 更新特征重要性指标
        for feature, importance in feature_importance.items():
            feature_importance_gauge.labels(
                model_id=request.model_id,
                feature=feature
            ).set(importance)
        
        logger.info(f"解释生成成功: {explanation_id}")
        
        return ExplanationResponse(
            explanation_id=explanation_id,
            model_id=request.model_id,
            method=request.method,
            scope=request.scope,
            feature_importance=feature_importance,
            explanation_values=explanation_values,
            visualization_urls=visualization_urls,
            quality_score=quality_score,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成解释失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成解释失败: {str(e)}")

async def generate_explanation_visualizations(explanation_id: str, method: ExplanationMethod,
                                            explanation_values: Dict[str, Any],
                                            feature_importance: Dict[str, float]):
    """生成解释可视化图表"""
    try:
        viz_dir = Path(VISUALIZATION_STORAGE_PATH) / explanation_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 特征重要性条形图
        if feature_importance:
            plt.figure(figsize=(10, 6))
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            plt.barh(features, importances)
            plt.xlabel('重要性')
            plt.title(f'特征重要性 ({method.value.upper()})')
            plt.tight_layout()
            plt.savefig(viz_dir / 'importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 方法特定的可视化
        if method == ExplanationMethod.SHAP and 'shap_values' in explanation_values:
            # SHAP摘要图
            plt.figure(figsize=(10, 8))
            shap_values = np.array(explanation_values['shap_values'])
            
            if len(shap_values.shape) == 2:
                # 创建简化的摘要图
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                feature_names = list(feature_importance.keys())
                
                plt.barh(feature_names, mean_shap[:len(feature_names)])
                plt.xlabel('平均|SHAP值|')
                plt.title('SHAP值摘要')
                plt.tight_layout()
                plt.savefig(viz_dir / 'summary.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        elif method == ExplanationMethod.LIME and 'explanations' in explanation_values:
            # LIME局部解释图
            explanations = explanation_values['explanations']
            if explanations:
                plt.figure(figsize=(12, 8))
                
                # 绘制多个样本的局部解释
                for i, exp in enumerate(explanations[:5]):  # 最多5个样本
                    local_imp = exp['local_importance']
                    features = list(local_imp.keys())
                    values = list(local_imp.values())
                    
                    plt.subplot(2, 3, i+1)
                    colors = ['red' if v < 0 else 'blue' for v in values]
                    plt.barh(features, values, color=colors)
                    plt.title(f'样本 {exp["sample_index"]}')
                    plt.xlabel('LIME重要性')
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'summary.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"可视化生成成功: {explanation_id}")
        
    except Exception as e:
        logger.error(f"生成可视化失败 {explanation_id}: {e}")

@app.get("/explanations/{explanation_id}", summary="获取解释结果")
async def get_explanation(explanation_id: str):
    """获取解释结果"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM explanations WHERE explanation_id = $1",
                explanation_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="解释结果不存在")
            
            if datetime.utcnow() > row['expires_at']:
                raise HTTPException(status_code=410, detail="解释结果已过期")
            
            return {
                "explanation_id": row['explanation_id'],
                "model_id": row['model_id'],
                "method": row['method'],
                "scope": row['scope'],
                "feature_importance": row['feature_importance'],
                "explanation_values": row['explanation_values'],
                "visualization_urls": row['visualization_paths'],
                "quality_score": row['quality_score'],
                "processing_time": row['processing_time'],
                "sample_indices": row['sample_indices'],
                "created_at": row['created_at'].isoformat(),
                "expires_at": row['expires_at'].isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取解释结果失败: {str(e)}")

@app.get("/explanations/{explanation_id}/visualizations/{filename}", summary="获取可视化图片")
async def get_visualization(explanation_id: str, filename: str):
    """获取解释可视化图片"""
    try:
        file_path = Path(VISUALIZATION_STORAGE_PATH) / explanation_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="可视化文件不存在")
        
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可视化失败: {str(e)}")

@app.post("/fairness/analyze", response_model=FairnessAnalysisResponse, summary="公平性分析")
async def analyze_fairness(request: FairnessAnalysisRequest, background_tasks: BackgroundTasks):
    """执行模型公平性分析"""
    try:
        # 获取模型和数据
        async with db_pool.acquire() as conn:
            model_row = await conn.fetchrow(
                "SELECT * FROM models WHERE model_id = $1",
                request.model_id
            )
            
            if not model_row:
                raise HTTPException(status_code=404, detail="模型不存在")
            
            dataset_row = await conn.fetchrow("""
                SELECT * FROM explanation_datasets 
                WHERE model_id = $1 
                ORDER BY created_at DESC LIMIT 1
            """, request.model_id)
            
            if not dataset_row:
                raise HTTPException(status_code=404, detail="未找到数据集")
        
        # 反序列化模型和数据
        model = await deserialize_model(model_row['model_data'])
        df = pickle.loads(dataset_row['data'])
        
        # 验证受保护属性
        missing_attrs = set(request.protected_attributes) - set(df.columns)
        if missing_attrs:
            raise HTTPException(
                status_code=400,
                detail=f"数据缺少受保护属性: {list(missing_attrs)}"
            )
        
        # 准备数据
        feature_names = model_row['feature_names']
        X = df[feature_names].values
        y = df['target'].values if 'target' in df.columns else None
        
        if y is None:
            raise HTTPException(status_code=400, detail="公平性分析需要目标变量")
        
        # 准备受保护属性数据
        protected_features = {}
        for attr in request.protected_attributes:
            protected_features[attr] = df[attr].values
        
        # 创建公平性分析器
        analyzer = FairnessAnalyzer(
            model, X, y, request.protected_attributes, feature_names
        )
        
        # 执行偏见分析
        bias_report = await analyzer.generate_bias_report(protected_features)
        
        # 计算公平性评分
        fairness_scores = {}
        for attr_name, indicators in bias_report['bias_indicators'].items():
            demo_score = 1.0 if indicators['demographic_parity']['is_fair'] else 0.0
            eq_odds_score = 1.0 if indicators['equalized_odds']['is_fair'] else 0.0
            fairness_scores[f'{attr_name}_demographic_parity'] = demo_score
            fairness_scores[f'{attr_name}_equalized_odds'] = eq_odds_score
        
        # 生成分析ID
        analysis_id = f"fairness_{uuid.uuid4().hex[:16]}"
        
        # 保存分析结果
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO fairness_analyses (
                    analysis_id, model_id, protected_attributes, fairness_metrics,
                    fairness_scores, bias_indicators, recommendations, mitigation_strategies
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 
                analysis_id, request.model_id, request.protected_attributes,
                [m.value for m in request.fairness_metrics],
                json.dumps(fairness_scores), json.dumps(bias_report['bias_indicators']),
                bias_report['recommendations'], bias_report['mitigation_strategies']
            )
        
        # 更新公平性指标
        for metric_name, score in fairness_scores.items():
            model_fairness_score.labels(
                model_id=request.model_id,
                metric=metric_name
            ).set(score)
        
        # 后台生成报告
        report_url = None
        if request.generate_report:
            background_tasks.add_task(
                generate_fairness_report, analysis_id, bias_report
            )
            report_url = f"/fairness/{analysis_id}/report"
        
        logger.info(f"公平性分析完成: {analysis_id}")
        
        return FairnessAnalysisResponse(
            analysis_id=analysis_id,
            model_id=request.model_id,
            fairness_scores=fairness_scores,
            bias_indicators=bias_report['bias_indicators'],
            recommendations=bias_report['recommendations'],
            mitigation_strategies=bias_report['mitigation_strategies'],
            report_url=report_url,
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"公平性分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"公平性分析失败: {str(e)}")

async def generate_fairness_report(analysis_id: str, bias_report: Dict[str, Any]):
    """生成公平性分析报告"""
    try:
        report_path = Path("./data/reports") / f"fairness_{analysis_id}.html"
        
        # 生成HTML报告
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>模型公平性分析报告</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .fair {{ color: green; font-weight: bold; }}
        .unfair {{ color: red; font-weight: bold; }}
        .metric {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation {{ background-color: #e6f3ff; padding: 10px; margin: 10px 0; border-left: 4px solid #0066cc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>模型公平性分析报告</h1>
        <p>分析ID: {analysis_id}</p>
        <p>生成时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>整体公平性评估</h2>
        <p class="{'fair' if bias_report['overall_fairness'] else 'unfair'}">
            整体评估: {'通过' if bias_report['overall_fairness'] else '未通过'}
        </p>
    </div>
    
    <div class="section">
        <h2>偏见指标详情</h2>
        {generate_bias_indicators_html(bias_report['bias_indicators'])}
    </div>
    
    <div class="section">
        <h2>改进建议</h2>
        {''.join([f'<div class="recommendation">{rec}</div>' for rec in bias_report['recommendations']])}
    </div>
    
    <div class="section">
        <h2>缓解策略</h2>
        {''.join([f'<div class="recommendation">{strategy}</div>' for strategy in bias_report['mitigation_strategies']])}
    </div>
</body>
</html>
        """
        
        report_path.write_text(html_content, encoding='utf-8')
        
        # 更新数据库记录
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE fairness_analyses SET report_path = $1 WHERE analysis_id = $2",
                str(report_path), analysis_id
            )
        
        logger.info(f"公平性报告生成成功: {analysis_id}")
        
    except Exception as e:
        logger.error(f"生成公平性报告失败 {analysis_id}: {e}")

def generate_bias_indicators_html(bias_indicators: Dict[str, Any]) -> str:
    """生成偏见指标HTML"""
    html = ""
    for attr_name, indicators in bias_indicators.items():
        html += f"<h3>受保护属性: {attr_name}</h3>"
        
        # 人口统计学平等性
        demo_parity = indicators['demographic_parity']
        html += f"""
        <div class="metric">
            <h4>人口统计学平等性</h4>
            <p>状态: <span class="{'fair' if demo_parity['is_fair'] else 'unfair'}">
                {'公平' if demo_parity['is_fair'] else '不公平'}
            </span></p>
            <p>选择率差异: {demo_parity['parity_difference']:.3f}</p>
            <p>选择率比例: {demo_parity['parity_ratio']:.3f}</p>
        </div>
        """
        
        # 机会均等性
        eq_odds = indicators['equalized_odds']
        html += f"""
        <div class="metric">
            <h4>机会均等性</h4>
            <p>状态: <span class="{'fair' if eq_odds['is_fair'] else 'unfair'}">
                {'公平' if eq_odds['is_fair'] else '不公平'}
            </span></p>
            <p>TPR差异: {eq_odds['tpr_difference']:.3f}</p>
            <p>FPR差异: {eq_odds['fpr_difference']:.3f}</p>
        </div>
        """
    
    return html

@app.get("/fairness/{analysis_id}/report", summary="获取公平性报告")
async def get_fairness_report(analysis_id: str):
    """获取公平性分析报告"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT report_path FROM fairness_analyses WHERE analysis_id = $1",
                analysis_id
            )
            
            if not row or not row['report_path']:
                raise HTTPException(status_code=404, detail="报告不存在")
            
            report_path = Path(row['report_path'])
            if not report_path.exists():
                raise HTTPException(status_code=404, detail="报告文件不存在")
            
            return FileResponse(
                path=str(report_path),
                media_type="text/html",
                filename=f"fairness_report_{analysis_id}.html"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取报告失败: {str(e)}")

@app.get("/models/{model_id}/summary", response_model=ModelSummaryResponse, summary="获取模型摘要")
async def get_model_summary(model_id: str):
    """获取模型解释摘要"""
    try:
        async with db_pool.acquire() as conn:
            # 获取模型信息
            model_row = await conn.fetchrow(
                "SELECT * FROM models WHERE model_id = $1",
                model_id
            )
            
            if not model_row:
                raise HTTPException(status_code=404, detail="模型不存在")
            
            # 获取解释统计
            explanation_count = await conn.fetchval(
                "SELECT COUNT(*) FROM explanations WHERE model_id = $1",
                model_id
            )
            
            # 获取最新解释
            latest_explanation = await conn.fetchrow("""
                SELECT explanation_id, method, quality_score, created_at
                FROM explanations 
                WHERE model_id = $1 
                ORDER BY created_at DESC LIMIT 1
            """, model_id)
            
            # 获取公平性状态
            fairness_row = await conn.fetchrow("""
                SELECT fairness_scores FROM fairness_analyses 
                WHERE model_id = $1 
                ORDER BY created_at DESC LIMIT 1
            """, model_id)
            
            fairness_status = "未评估"
            if fairness_row:
                scores = fairness_row['fairness_scores']
                if all(score >= 0.8 for score in scores.values()):
                    fairness_status = "公平"
                else:
                    fairness_status = "存在偏见"
            
            # 计算可解释性评分
            interpretability_score = 0.0
            if latest_explanation:
                interpretability_score = latest_explanation['quality_score']
            
            latest_exp_dict = None
            if latest_explanation:
                latest_exp_dict = {
                    "explanation_id": latest_explanation['explanation_id'],
                    "method": latest_explanation['method'],
                    "quality_score": latest_explanation['quality_score'],
                    "created_at": latest_explanation['created_at'].isoformat()
                }
            
            return ModelSummaryResponse(
                model_id=model_id,
                model_type=ModelType(model_row['model_type']),
                feature_count=len(model_row['feature_names']),
                explanation_count=explanation_count,
                latest_explanation=latest_exp_dict,
                fairness_status=fairness_status,
                interpretability_score=interpretability_score
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型摘要失败: {str(e)}")

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """健康检查端点"""
    try:
        # 检查数据库连接
        db_status = "healthy"
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception:
            db_status = "unhealthy"
        
        # 检查Redis连接
        redis_status = "healthy"
        try:
            await redis_client.ping()
        except Exception:
            redis_status = "unhealthy"
        
        # 获取活跃解释数量
        active_explanations = 0
        try:
            async with db_pool.acquire() as conn:
                active_explanations = await conn.fetchval("""
                    SELECT COUNT(*) FROM explanations 
                    WHERE expires_at > NOW()
                """)
        except Exception:
            pass
        
        overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status=db_status,
            redis_status=redis_status,
            active_explanations=active_explanations
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status="unknown",
            redis_status="unknown",
            active_explanations=0
        )

@app.get("/metrics", summary="获取Prometheus指标")
async def get_metrics():
    """获取Prometheus监控指标"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# 清理过期解释的后台任务
async def cleanup_expired_explanations():
    """清理过期的解释结果"""
    try:
        async with db_pool.acquire() as conn:
            # 获取过期的解释
            expired_rows = await conn.fetch("""
                SELECT explanation_id, visualization_paths 
                FROM explanations 
                WHERE expires_at < NOW()
            """)
            
            for row in expired_rows:
                # 删除可视化文件
                for viz_path in row['visualization_paths'] or []:
                    try:
                        file_path = Path(VISUALIZATION_STORAGE_PATH) / row['explanation_id']
                        if file_path.exists():
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"删除可视化文件失败 {row['explanation_id']}: {e}")
            
            # 删除过期记录
            deleted_count = await conn.fetchval("""
                DELETE FROM explanations 
                WHERE expires_at < NOW()
                RETURNING COUNT(*)
            """)
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个过期解释")
                
    except Exception as e:
        logger.error(f"清理过期解释失败: {e}")

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        await init_database()
        await init_redis()
        
        # 启动定期清理任务
        import asyncio
        asyncio.create_task(periodic_cleanup())
        
        logger.info("模型解释服务启动成功")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    try:
        if db_pool:
            await db_pool.close()
        if redis_client:
            await redis_client.close()
        logger.info("模型解释服务关闭")
    except Exception as e:
        logger.error(f"服务关闭失败: {e}")

async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时执行一次
            await cleanup_expired_explanations()
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )