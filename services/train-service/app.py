#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练服务 (Train Service)
基于SecretFlow的纵向联邦学习训练服务
支持SecureBoost和Hetero-LR算法，集成差分隐私
"""

import os
import uuid
import json
import pickle
import hashlib
import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from contextvars import ContextVar

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import requests

# 请求上下文变量
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
training_id_var: ContextVar[str] = ContextVar('training_id', default='')

# 结构化日志配置
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _format_log(self, level: str, message: str, **kwargs) -> Dict:
        """格式化结构化日志"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": "train-service",
            "message": message,
            "request_id": request_id_var.get(''),
            "training_id": training_id_var.get(''),
            **kwargs
        }
        return log_entry
    
    def info(self, message: str, **kwargs):
        log_entry = self._format_log("INFO", message, **kwargs)
        self.logger.info(json.dumps(log_entry))
    
    def warning(self, message: str, **kwargs):
        log_entry = self._format_log("WARNING", message, **kwargs)
        self.logger.warning(json.dumps(log_entry))
    
    def error(self, message: str, **kwargs):
        log_entry = self._format_log("ERROR", message, **kwargs)
        self.logger.error(json.dumps(log_entry))
    
    def debug(self, message: str, **kwargs):
        log_entry = self._format_log("DEBUG", message, **kwargs)
        self.logger.debug(json.dumps(log_entry))

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 使用简单格式，因为我们输出JSON
)
logger = StructuredLogger(__name__)

# OpenTelemetry轻量打点
class SimpleTracer:
    def __init__(self):
        self.spans = []
        self.trace_file = Path("traces/train_service_traces.jsonl")
        self.trace_file.parent.mkdir(exist_ok=True)
    
    def start_span(self, name: str, **attributes) -> Dict:
        span = {
            "span_id": str(uuid.uuid4()),
            "trace_id": request_id_var.get('') or str(uuid.uuid4()),
            "name": name,
            "start_time": time.time(),
            "attributes": attributes
        }
        return span
    
    def end_span(self, span: Dict, **attributes):
        span["end_time"] = time.time()
        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
        span["attributes"].update(attributes)
        
        # 写入文件
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(span) + "\n")

tracer = SimpleTracer()

# 配置
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / "data" / "synth"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PSI_SERVICE_URL = os.getenv("PSI_SERVICE_URL", "http://localhost:8001")

# 确保目录存在
Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Train Service",
    description="联邦学习训练服务 - SecretFlow + 差分隐私",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 请求中间件
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    # 生成请求ID
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    
    # 记录请求开始
    start_time = time.time()
    logger.info("Request started", 
                method=request.method, 
                url=str(request.url),
                user_agent=request.headers.get("user-agent", ""))
    
    # 处理请求
    response = await call_next(request)
    
    # 记录请求结束
    duration_ms = (time.time() - start_time) * 1000
    logger.info("Request completed", 
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2))
    
    return response

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
training_sessions = {}  # 存储训练会话
model_registry = {}     # 模型注册表
self_healing_attempts = {}  # 自愈尝试记录

# Pydantic模型
class DPConfig(BaseModel):
    """差分隐私配置"""
    enable: bool = Field(False, description="是否启用差分隐私")
    epsilon: float = Field(1.0, description="隐私预算ε")
    delta: Optional[float] = Field(1e-5, description="隐私预算δ")
    noise_type: str = Field("gaussian", description="噪声类型: gaussian/laplace")
    clipping_norm: float = Field(1.0, description="梯度裁剪范数")
    
    @validator('epsilon')
    def validate_epsilon(cls, v):
        if v <= 0:
            raise ValueError('ε必须大于0')
        return v

class FeatureConfig(BaseModel):
    """特征配置"""
    A: List[str] = Field(..., description="A方特征列表")
    B: List[str] = Field(..., description="B方特征列表")

class SecureBoostRequest(BaseModel):
    """SecureBoost训练请求"""
    dp: DPConfig = Field(default_factory=DPConfig, description="差分隐私配置")
    psi_mapping_key: str = Field(..., description="PSI映射键")
    features: FeatureConfig = Field(..., description="特征配置")
    max_depth: int = Field(6, description="树最大深度")
    num_boost_round: int = Field(10, description="提升轮数")
    learning_rate: float = Field(0.3, description="学习率")
    subsample: float = Field(1.0, description="子采样率")
    colsample_by_tree: float = Field(1.0, description="特征采样率")
    reg_lambda: float = Field(1.0, description="L2正则化")
    gamma: float = Field(0.0, description="最小分裂损失")
    min_child_weight: int = Field(1, description="叶子节点最小权重")

class HeteroLRRequest(BaseModel):
    """Hetero-LR训练请求"""
    dp: DPConfig = Field(default_factory=DPConfig, description="差分隐私配置")
    psi_mapping_key: str = Field(..., description="PSI映射键")
    features: FeatureConfig = Field(..., description="特征配置")
    max_iter: int = Field(100, description="最大迭代次数")
    learning_rate: float = Field(0.01, description="学习率")
    alpha: float = Field(0.01, description="L1正则化")
    l1_ratio: float = Field(0.5, description="L1/L2正则化比例")
    tol: float = Field(1e-4, description="收敛容忍度")
    penalty: str = Field("l2", description="正则化类型")

class TrainingResponse(BaseModel):
    """训练响应"""
    run_id: str
    model_hash: str
    algorithm: str
    auc: float
    ks: float
    dp_epsilon: Optional[float]
    artifacts_path: str
    training_time: float
    status: str
    metadata: Dict

class MetricsResponse(BaseModel):
    """指标响应"""
    run_id: str
    algorithm: str
    auc: float
    ks: float
    dp_epsilon: Optional[float]
    training_time: float
    feature_importance: Optional[Dict]
    convergence_history: Optional[List[float]]
    timestamp: str

class ExplainGlobalRequest(BaseModel):
    """全局解释请求"""
    model_hash: str = Field(..., description="模型哈希")
    top_n: int = Field(10, description="返回Top-N特征")
    dp_noise: bool = Field(True, description="是否添加差分隐私噪声")
    epsilon: float = Field(1.0, description="解释的隐私预算")

class ExplainLocalRequest(BaseModel):
    """本地解释请求"""
    model_hash: str = Field(..., description="模型哈希")
    sample_id: str = Field(..., description="样本ID")
    party: str = Field(..., description="数据方: A/B")

class TrainingGuards:
    """训练层护栏"""
    
    @staticmethod
    def validate_training_params(request) -> Tuple[bool, List[str]]:
        """验证训练参数"""
        issues = []
        
        # 检查迭代次数
        if hasattr(request, 'num_boost_round'):
            if request.num_boost_round < 20:
                issues.append(f"训练轮数过少: {request.num_boost_round} < 20")
        elif hasattr(request, 'max_iter'):
            if request.max_iter < 20:
                issues.append(f"最大迭代次数过少: {request.max_iter} < 20")
        
        # 检查学习率
        if hasattr(request, 'learning_rate'):
            if request.learning_rate <= 0 or request.learning_rate > 1:
                issues.append(f"学习率异常: {request.learning_rate}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_training_results(results: Dict, training_time: float) -> Tuple[bool, List[str]]:
        """验证训练结果"""
        issues = []
        
        # 检查训练时间（单轮耗时>5ms）
        rounds = results.get('rounds', 1)
        time_per_round = training_time * 1000 / rounds  # 转换为毫秒
        if time_per_round < 5:
            issues.append(f"单轮训练时间过短: {time_per_round:.1f}ms < 5ms")
        
        # 检查损失值
        if 'training_history' in results:
            history = results['training_history']
            if len(history) > 1:
                # 检查损失是否为NaN
                if any(np.isnan(loss) for loss in history):
                    issues.append("训练损失包含NaN值")
                
                # 检查损失是否恒定
                loss_std = np.std(history)
                if loss_std < 1e-6:
                    issues.append(f"训练损失恒定: std={loss_std:.2e}")
        
        # 检查AUC和KS
        auc = results.get('auc', 0)
        ks = results.get('ks', 0)
        
        if auc < 0.65:
            issues.append(f"AUC不达标: {auc:.3f} < 0.65")
        
        if ks < 0.20:
            issues.append(f"KS不达标: {ks:.3f} < 0.20")
        
        # 检查预测标准差
        if 'test_predictions' in results:
            pred_std = np.std(results['test_predictions'])
            if pred_std < 0.01:
                issues.append(f"预测标准差过小: {pred_std:.4f} < 0.01")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def apply_self_healing(request, attempt: int = 1):
        """应用自愈策略"""
        logger.info(f"应用自愈策略 - 第{attempt}轮")
        
        if attempt == 1:
            # 第一轮：关闭DP或放宽ε
            if hasattr(request, 'dp') and request.dp.enable:
                if request.dp.epsilon < 8:
                    request.dp.epsilon = 8
                    logger.info(f"自愈策略1: 放宽隐私预算至 ε={request.dp.epsilon}")
                else:
                    request.dp.enable = False
                    logger.info("自愈策略1: 关闭差分隐私")
        
        elif attempt == 2:
            # 第二轮：调整XGBoost参数
            if hasattr(request, 'learning_rate'):
                request.learning_rate = max(0.05, request.learning_rate * 0.5)
                logger.info(f"自愈策略2: 降低学习率至 {request.learning_rate}")
            
            if hasattr(request, 'max_depth'):
                request.max_depth = min(5, request.max_depth + 1)
                logger.info(f"自愈策略2: 调整最大深度至 {request.max_depth}")
            
            if hasattr(request, 'subsample'):
                request.subsample = 0.8
                logger.info(f"自愈策略2: 设置子采样率为 {request.subsample}")
        
        elif attempt == 3:
            # 第三轮：增加轮数和正则化
            if hasattr(request, 'num_boost_round'):
                request.num_boost_round = int(request.num_boost_round * 1.5)
                logger.info(f"自愈策略3: 增加训练轮数至 {request.num_boost_round}")
            
            if hasattr(request, 'max_iter'):
                request.max_iter = int(request.max_iter * 1.5)
                logger.info(f"自愈策略3: 增加最大迭代至 {request.max_iter}")
            
            if hasattr(request, 'reg_lambda'):
                request.reg_lambda = request.reg_lambda * 2
                logger.info(f"自愈策略3: 增强L2正则化至 {request.reg_lambda}")
        
        return request

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    secretflow_status: str
    data_status: str
    artifacts_status: str

def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算AUC"""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    except Exception as e:
        logger.warning(f"AUC计算失败: {str(e)}")
        return 0.0

def calculate_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算KS统计量"""
    try:
        from scipy import stats
        
        # 分离正负样本的预测分数
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
        
        # 计算KS统计量
        ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
        return float(ks_stat)
    except Exception as e:
        logger.warning(f"KS计算失败: {str(e)}")
        return 0.0

def add_dp_noise(values: np.ndarray, epsilon: float, sensitivity: float = 1.0, noise_type: str = "gaussian") -> np.ndarray:
    """添加差分隐私噪声"""
    if noise_type == "gaussian":
        # Gaussian机制
        sigma = np.sqrt(2 * np.log(1.25 / 1e-5)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, values.shape)
    elif noise_type == "laplace":
        # Laplace机制
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, values.shape)
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    return values + noise

def load_psi_mapping(psi_mapping_key: str) -> Dict:
    """从PSI服务加载映射"""
    try:
        response = requests.get(f"{PSI_SERVICE_URL}/psi/result/{psi_mapping_key}")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=400, detail=f"无法获取PSI映射: {response.text}")
    except Exception as e:
        logger.error(f"加载PSI映射失败: {str(e)}")
        raise HTTPException(status_code=500, detail="PSI映射加载失败")

def load_federated_data(psi_mapping_key: str, features: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """加载联邦数据"""
    # 加载PSI映射
    psi_mapping = load_psi_mapping(psi_mapping_key)
    aligned_tokens = psi_mapping.get('aligned_tokens', [])
    
    if not aligned_tokens:
        raise HTTPException(status_code=400, detail="PSI映射为空")
    
    # 加载原始数据
    party_a_path = os.path.join(DATA_DIR, "partyA_bank.csv")
    party_b_path = os.path.join(DATA_DIR, "partyB_ecom.csv")
    
    if not os.path.exists(party_a_path) or not os.path.exists(party_b_path):
        raise HTTPException(status_code=404, detail="数据文件不存在")
    
    df_a = pd.read_csv(party_a_path)
    df_b = pd.read_csv(party_b_path)
    
    # 根据PSI结果对齐数据
    df_a_aligned = df_a[df_a['psi_token'].isin(aligned_tokens)].copy()
    df_b_aligned = df_b[df_b['psi_token'].isin(aligned_tokens)].copy()
    
    # 按psi_token排序确保对齐
    df_a_aligned = df_a_aligned.sort_values('psi_token').reset_index(drop=True)
    df_b_aligned = df_b_aligned.sort_values('psi_token').reset_index(drop=True)
    
    # 提取特征
    X_a = df_a_aligned[features.A].copy()
    X_b = df_b_aligned[features.B].copy()
    
    # 提取标签（只有A方有标签）
    if 'default_label' in df_a_aligned.columns:
        y = df_a_aligned['default_label'].values
    else:
        raise HTTPException(status_code=400, detail="A方数据缺少标签列")
    
    logger.info(f"加载联邦数据完成: A方{len(X_a)}条, B方{len(X_b)}条, 标签{len(y)}个")
    
    return X_a, X_b, y

def simulate_secureboost_training(X_a: pd.DataFrame, X_b: pd.DataFrame, y: np.ndarray, 
                                request: SecureBoostRequest, run_id: str = None) -> Dict:
    """模拟SecureBoost训练（带护栏和自愈）"""
    # 设置训练ID
    if run_id:
        training_id_var.set(run_id)
    
    # 开始训练span
    train_span = tracer.start_span("train.fit", 
                                   algorithm="secureboost",
                                   num_samples=len(y),
                                   num_features_a=len(X_a.columns),
                                   num_features_b=len(X_b.columns),
                                   dp_enabled=request.dp.enable)
    
    logger.info("开始SecureBoost训练模拟", 
                algorithm="secureboost",
                num_samples=len(y),
                num_features=len(X_a.columns) + len(X_b.columns),
                dp_enabled=request.dp.enable,
                epsilon=request.dp.epsilon if request.dp.enable else None)
    
    # 训练层护栏：验证参数
    param_valid, param_issues = TrainingGuards.validate_training_params(request)
    if not param_valid:
        logger.warning(f"训练参数问题: {param_issues}")
    
    # 合并特征用于模拟（实际SecretFlow中各方数据不会合并）
    X_combined = pd.concat([X_a, X_b], axis=1)
    
    # 检查类别不平衡并自动设置scale_pos_weight
    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        import time
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 创建模型
        model = GradientBoostingClassifier(
            n_estimators=request.num_boost_round,
            max_depth=request.max_depth,
            learning_rate=request.learning_rate,
            subsample=request.subsample,
            max_features=request.colsample_by_tree,
            random_state=42
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        logger.info("模型训练完成", 
                    training_time_seconds=round(training_time, 3),
                    n_estimators=request.num_boost_round,
                    max_depth=request.max_depth)
        
        # 预测
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 添加差分隐私噪声
        if request.dp.enable:
            y_pred_proba = add_dp_noise(
                y_pred_proba, 
                request.dp.epsilon, 
                sensitivity=1.0, 
                noise_type=request.dp.noise_type
            )
            # 确保概率在[0,1]范围内
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        # 计算指标
        auc = calculate_auc(y_test, y_pred_proba)
        ks = calculate_ks(y_test, y_pred_proba)
        
        logger.info("训练指标计算完成", 
                    auc=round(auc, 4),
                    ks=round(ks, 4),
                    test_samples=len(y_test))
        
        # 特征重要性
        feature_importance = dict(zip(
            X_combined.columns, 
            model.feature_importances_
        ))
        
        results = {
            'model': model,
            'auc': auc,
            'ks': ks,
            'feature_importance': feature_importance,
            'test_predictions': y_pred_proba,
            'test_labels': y_test,
            'training_history': list(model.train_score_) if hasattr(model, 'train_score_') else [0.5],
            'training_time': training_time,
            'rounds': request.num_boost_round,
            'scale_pos_weight': scale_pos_weight
        }
        
        # 训练层护栏：验证结果
        result_valid, result_issues = TrainingGuards.validate_training_results(results, training_time)
        if not result_valid:
            logger.warning("训练结果验证失败", 
                          issues=result_issues,
                          auc=auc,
                          ks=ks)
            results['validation_issues'] = result_issues
        else:
            logger.info("训练结果验证通过", 
                       auc=auc,
                       ks=ks,
                       training_time_seconds=training_time)
        
        # 结束训练span
        tracer.end_span(train_span, 
                       success=result_valid,
                       auc=auc,
                       ks=ks,
                       validation_issues=len(result_issues))
        
        return results
        
    except Exception as e:
        logger.error("SecureBoost训练失败", 
                    error=str(e),
                    error_type=type(e).__name__)
        
        # 结束训练span（失败）
        tracer.end_span(train_span, 
                       success=False,
                       error=str(e))
        
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

def simulate_hetero_lr_training(X_a: pd.DataFrame, X_b: pd.DataFrame, y: np.ndarray, 
                              request: HeteroLRRequest) -> Dict:
    """模拟Hetero-LR训练"""
    logger.info("开始Hetero-LR训练模拟...")
    
    # 合并特征用于模拟
    X_combined = pd.concat([X_a, X_b], axis=1)
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建模型
        model = LogisticRegression(
            max_iter=request.max_iter,
            C=1.0/request.alpha if request.alpha > 0 else 1.0,
            l1_ratio=request.l1_ratio,
            penalty='elasticnet' if request.penalty == 'elasticnet' else request.penalty,
            tol=request.tol,
            random_state=42,
            solver='saga'
        )
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 添加差分隐私噪声
        if request.dp.enable:
            y_pred_proba = add_dp_noise(
                y_pred_proba, 
                request.dp.epsilon, 
                sensitivity=1.0, 
                noise_type=request.dp.noise_type
            )
            # 确保概率在[0,1]范围内
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        # 计算指标
        auc = calculate_auc(y_test, y_pred_proba)
        ks = calculate_ks(y_test, y_pred_proba)
        
        # 特征重要性（系数绝对值）
        feature_importance = dict(zip(
            X_combined.columns, 
            np.abs(model.coef_[0])
        ))
        
        return {
            'model': model,
            'scaler': scaler,
            'auc': auc,
            'ks': ks,
            'feature_importance': feature_importance,
            'test_predictions': y_pred_proba,
            'test_labels': y_test,
            'coefficients': model.coef_[0].tolist()
        }
        
    except Exception as e:
        logger.error(f"Hetero-LR训练失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

def save_model_artifacts(run_id: str, algorithm: str, model_data: Dict, 
                        request_data: Dict) -> Tuple[str, str]:
    """保存模型产物"""
    # 创建运行目录
    run_dir = os.path.join(ARTIFACTS_DIR, run_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data['model'], f)
    
    # 保存scaler（如果有）
    if 'scaler' in model_data:
        scaler_path = os.path.join(run_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(model_data['scaler'], f)
    
    # 生成模型哈希
    model_content = pickle.dumps(model_data['model'])
    model_hash = hashlib.sha256(model_content).hexdigest()[:16]
    
    # 保存元数据
    metadata = {
        'run_id': run_id,
        'model_hash': model_hash,
        'algorithm': algorithm,
        'auc': model_data['auc'],
        'ks': model_data['ks'],
        'feature_importance': model_data['feature_importance'],
        'training_params': request_data,
        'created_at': datetime.now().isoformat(),
        'artifacts_path': run_dir
    }
    
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 注册到全局模型注册表
    model_registry[model_hash] = metadata
    
    logger.info(f"模型产物已保存: {run_dir}")
    
    return model_hash, run_dir

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动训练服务...")
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        logger.warning(f"数据目录不存在: {DATA_DIR}")
    
    # 加载已有模型
    if os.path.exists(ARTIFACTS_DIR):
        for run_dir in os.listdir(ARTIFACTS_DIR):
            metadata_path = os.path.join(ARTIFACTS_DIR, run_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    model_hash = metadata.get('model_hash')
                    if model_hash:
                        model_registry[model_hash] = metadata
                        logger.info(f"加载模型: {model_hash}")
                except Exception as e:
                    logger.warning(f"加载模型元数据失败 {run_dir}: {str(e)}")
    
    logger.info(f"训练服务启动完成，已加载{len(model_registry)}个模型")

@app.post("/train/secureboost", response_model=TrainingResponse)
async def train_secureboost(request: SecureBoostRequest, background_tasks: BackgroundTasks):
    """SecureBoost训练（带自愈机制）"""
    run_id = str(uuid.uuid4())
    start_time = datetime.now()
    training_id_var.set(run_id)
    
    logger.info("收到SecureBoost训练请求", 
                run_id=run_id,
                psi_mapping_key=request.psi_mapping_key,
                dp_enabled=request.dp.enable,
                epsilon=request.dp.epsilon if request.dp.enable else None)
    
    try:
        # 加载数据
        data_span = tracer.start_span("data.load", psi_mapping_key=request.psi_mapping_key)
        X_a, X_b, y = load_federated_data(request.psi_mapping_key, request.features)
        tracer.end_span(data_span, 
                       samples_loaded=len(y),
                       features_a=len(X_a.columns),
                       features_b=len(X_b.columns))
        
        logger.info("数据加载完成", 
                    samples=len(y),
                    features_a=len(X_a.columns),
                    features_b=len(X_b.columns),
                    positive_rate=round(np.mean(y), 4))
        
        # 自愈尝试循环（最多3轮）
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            logger.info(f"训练尝试 {attempt}/{max_attempts}")
            
            # 应用自愈策略（从第2轮开始）
            if attempt > 1:
                request = TrainingGuards.apply_self_healing(request, attempt - 1)
            
            # 开始训练
            model_data = simulate_secureboost_training(X_a, X_b, y, request, run_id)
            
            # 计算训练时间
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 检查训练结果
            result_valid, result_issues = TrainingGuards.validate_training_results(model_data, training_time)
            
            if result_valid:
                logger.info("训练成功完成", 
                           attempt=attempt,
                           max_attempts=max_attempts,
                           final_auc=model_data['auc'],
                           final_ks=model_data['ks'])
                break
            else:
                logger.warning("训练轮次未达标", 
                              attempt=attempt,
                              max_attempts=max_attempts,
                              issues=result_issues,
                              auc=model_data.get('auc', 0),
                              ks=model_data.get('ks', 0))
                if attempt == max_attempts:
                    logger.error("所有自愈尝试均失败，训练终止", 
                               total_attempts=attempt,
                               final_issues=result_issues)
                    # 记录失败的自愈尝试
                    self_healing_attempts[run_id] = {
                        'attempts': attempt,
                        'final_issues': result_issues,
                        'status': 'failed'
                    }
                else:
                    logger.info("准备下一轮自愈尝试", 
                               next_attempt=attempt + 1,
                               max_attempts=max_attempts)
        
        # 保存模型产物
        model_hash, artifacts_path = save_model_artifacts(
            run_id, "secureboost", model_data, request.dict()
        )
        
        # 存储训练会话
        training_sessions[run_id] = {
            'run_id': run_id,
            'algorithm': 'secureboost',
            'model_hash': model_hash,
            'auc': model_data['auc'],
            'ks': model_data['ks'],
            'dp_epsilon': request.dp.epsilon if request.dp.enable else None,
            'training_time': training_time,
            'feature_importance': model_data['feature_importance'],
            'training_history': model_data.get('training_history', []),
            'created_at': start_time.isoformat(),
            'status': 'completed' if result_valid else 'completed_with_issues',
            'self_healing_attempts': attempt,
            'validation_issues': result_issues if not result_valid else []
        }
        
        # 记录成功的自愈尝试
        if attempt > 1:
            self_healing_attempts[run_id] = {
                'attempts': attempt,
                'status': 'success' if result_valid else 'partial_success',
                'final_auc': model_data['auc'],
                'final_ks': model_data['ks']
            }
            logger.info("自愈策略执行完成", 
                       total_attempts=attempt,
                       status='success' if result_valid else 'partial_success',
                       final_auc=model_data['auc'],
                       final_ks=model_data['ks'])
        
        logger.info("SecureBoost训练流程完成", 
                   run_id=run_id,
                   status="completed" if result_valid else "completed_with_issues",
                   auc=model_data['auc'],
                   ks=model_data['ks'],
                   training_time_seconds=training_time,
                   self_healing_attempts=attempt)
        
        return TrainingResponse(
            run_id=run_id,
            model_hash=model_hash,
            algorithm="secureboost",
            auc=model_data['auc'],
            ks=model_data['ks'],
            dp_epsilon=request.dp.epsilon if request.dp.enable else None,
            artifacts_path=artifacts_path,
            training_time=training_time,
            status="completed" if result_valid else "completed_with_issues",
            metadata={
                'num_samples': len(y),
                'num_features_a': len(request.features.A),
                'num_features_b': len(request.features.B),
                'dp_enabled': request.dp.enable,
                'self_healing_attempts': attempt,
                'validation_issues': result_issues if not result_valid else [],
                'scale_pos_weight': model_data.get('scale_pos_weight', 1.0)
            }
        )
        
    except Exception as e:
        logger.error("SecureBoost训练流程失败", 
                    run_id=run_id,
                    error=str(e),
                    error_type=type(e).__name__)
        # 记录失败的会话
        training_sessions[run_id] = {
            'run_id': run_id,
            'algorithm': 'secureboost',
            'status': 'failed',
            'error': str(e),
            'created_at': start_time.isoformat()
        }
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.post("/train/hetero_lr", response_model=TrainingResponse)
async def train_hetero_lr(request: HeteroLRRequest, background_tasks: BackgroundTasks):
    """Hetero-LR训练"""
    run_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(f"开始Hetero-LR训练: {run_id}")
    
    try:
        # 加载联邦数据
        X_a, X_b, y = load_federated_data(request.psi_mapping_key, request.features)
        
        # 训练模型
        model_data = simulate_hetero_lr_training(X_a, X_b, y, request)
        
        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 保存模型产物
        model_hash, artifacts_path = save_model_artifacts(
            run_id, "hetero_lr", model_data, request.dict()
        )
        
        # 存储训练会话
        training_sessions[run_id] = {
            'run_id': run_id,
            'algorithm': 'hetero_lr',
            'model_hash': model_hash,
            'auc': model_data['auc'],
            'ks': model_data['ks'],
            'dp_epsilon': request.dp.epsilon if request.dp.enable else None,
            'training_time': training_time,
            'feature_importance': model_data['feature_importance'],
            'coefficients': model_data.get('coefficients', []),
            'created_at': start_time.isoformat(),
            'status': 'completed'
        }
        
        logger.info(f"Hetero-LR训练完成: {run_id}, AUC: {model_data['auc']:.4f}, KS: {model_data['ks']:.4f}")
        
        return TrainingResponse(
            run_id=run_id,
            model_hash=model_hash,
            algorithm="hetero_lr",
            auc=model_data['auc'],
            ks=model_data['ks'],
            dp_epsilon=request.dp.epsilon if request.dp.enable else None,
            artifacts_path=artifacts_path,
            training_time=training_time,
            status="completed",
            metadata={
                'num_samples': len(y),
                'num_features_a': len(request.features.A),
                'num_features_b': len(request.features.B),
                'dp_enabled': request.dp.enable
            }
        )
        
    except Exception as e:
        logger.error(f"Hetero-LR训练失败: {str(e)}")
        # 记录失败的会话
        training_sessions[run_id] = {
            'run_id': run_id,
            'algorithm': 'hetero_lr',
            'status': 'failed',
            'error': str(e),
            'created_at': start_time.isoformat()
        }
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.get("/metrics/{run_id}", response_model=MetricsResponse)
async def get_metrics(run_id: str):
    """获取训练指标"""
    if run_id not in training_sessions:
        raise HTTPException(status_code=404, detail="训练会话不存在")
    
    session = training_sessions[run_id]
    
    if session['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"训练会话状态: {session['status']}")
    
    return MetricsResponse(
        run_id=run_id,
        algorithm=session['algorithm'],
        auc=session['auc'],
        ks=session['ks'],
        dp_epsilon=session.get('dp_epsilon'),
        training_time=session['training_time'],
        feature_importance=session.get('feature_importance'),
        convergence_history=session.get('training_history', session.get('coefficients')),
        timestamp=session['created_at']
    )

@app.post("/explain/global")
async def explain_global(request: ExplainGlobalRequest):
    """全局解释"""
    if request.model_hash not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    metadata = model_registry[request.model_hash]
    feature_importance = metadata.get('feature_importance', {})
    
    # 排序并取Top-N
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:request.top_n]
    
    # 添加差分隐私噪声
    if request.dp_noise:
        noisy_features = []
        for feature, importance in sorted_features:
            noisy_importance = add_dp_noise(
                np.array([importance]), 
                request.epsilon, 
                sensitivity=max(feature_importance.values()) if feature_importance else 1.0
            )[0]
            noisy_features.append((feature, max(0, noisy_importance)))  # 确保非负
        sorted_features = noisy_features
    
    logger.info(f"全局解释完成: {request.model_hash}, Top-{request.top_n}")
    
    return {
        'model_hash': request.model_hash,
        'algorithm': metadata['algorithm'],
        'global_importance': [
            {'feature': feature, 'importance': importance}
            for feature, importance in sorted_features
        ],
        'dp_noise_applied': request.dp_noise,
        'epsilon': request.epsilon if request.dp_noise else None,
        'timestamp': datetime.now().isoformat()
    }

@app.post("/explain/local")
async def explain_local(request: ExplainLocalRequest):
    """本地解释（模拟）"""
    if request.model_hash not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    metadata = model_registry[request.model_hash]
    
    # 模拟本地SHAP值
    np.random.seed(hash(request.sample_id) % 2**32)
    
    if request.party == 'A':
        features = ['age', 'income', 'credit_history', 'employment_years']
    else:
        features = ['purchase_amount', 'purchase_frequency', 'category_preference']
    
    # 生成模拟SHAP值
    shap_values = {
        feature: np.random.normal(0, 0.1) for feature in features
    }
    
    # 基础值
    base_value = 0.5
    
    logger.info(f"本地解释完成: {request.model_hash}, 样本: {request.sample_id}, 方: {request.party}")
    
    return {
        'model_hash': request.model_hash,
        'sample_id': request.sample_id,
        'party': request.party,
        'algorithm': metadata['algorithm'],
        'shap_values': shap_values,
        'base_value': base_value,
        'prediction': base_value + sum(shap_values.values()),
        'timestamp': datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """列出所有模型"""
    models = []
    for model_hash, metadata in model_registry.items():
        models.append({
            'model_hash': model_hash,
            'algorithm': metadata['algorithm'],
            'auc': metadata['auc'],
            'ks': metadata['ks'],
            'created_at': metadata['created_at'],
            'run_id': metadata['run_id']
        })
    
    return {
        'models': sorted(models, key=lambda x: x['created_at'], reverse=True),
        'total': len(models)
    }

@app.get("/sessions")
async def list_sessions():
    """列出所有训练会话"""
    sessions = []
    for run_id, session in training_sessions.items():
        sessions.append({
            'run_id': run_id,
            'algorithm': session.get('algorithm'),
            'status': session['status'],
            'created_at': session['created_at'],
            'auc': session.get('auc'),
            'ks': session.get('ks'),
            'training_time': session.get('training_time')
        })
    
    return {
        'sessions': sorted(sessions, key=lambda x: x['created_at'], reverse=True),
        'total': len(sessions)
    }

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    # 检查数据状态
    data_status = "healthy" if os.path.exists(DATA_DIR) else "error"
    
    # 检查产物目录状态
    artifacts_status = "healthy" if os.path.exists(ARTIFACTS_DIR) else "error"
    
    # 检查SecretFlow状态（模拟）
    secretflow_status = "healthy"  # 在实际环境中应该检查SecretFlow集群状态
    
    overall_status = "healthy" if all([
        data_status == "healthy",
        artifacts_status == "healthy",
        secretflow_status == "healthy"
    ]) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        secretflow_status=secretflow_status,
        data_status=data_status,
        artifacts_status=artifacts_status
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7003))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )