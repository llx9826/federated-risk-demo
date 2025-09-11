from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import uuid
import hashlib
import time
import os
import pickle
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Federated Serving Service",
    description="联邦学习推理服务 - 在线打分、模型注册、审计回执",
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

# 配置
CONSENT_SERVICE_URL = os.getenv("CONSENT_SERVICE_URL", "http://consent-service:7002")
DATA_DIR = "/app/data"
MODELS_DIR = "/app/models"
ARTIFACTS_DIR = "/app/artifacts"

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 全局变量
model_registry = {}
feature_store = {}
production_model = None

# Pydantic模型
class ScoreRequest(BaseModel):
    subject: str = Field(..., description="用户标识")
    psi_token: str = Field(..., description="PSI对齐令牌")
    features: Optional[Dict[str, Any]] = Field(None, description="可选特征（Demo用）")
    consent_jwt: str = Field(..., description="同意票据JWT")

class ScoreResponse(BaseModel):
    score: float = Field(..., description="风险评分")
    decision: str = Field(..., description="决策结果")
    model_hash: str = Field(..., description="模型哈希")
    threshold: float = Field(..., description="决策阈值")
    request_id: str = Field(..., description="请求ID")
    timestamp: str = Field(..., description="时间戳")

class ModelPromoteRequest(BaseModel):
    model_hash: str = Field(..., description="模型哈希")
    threshold: float = Field(0.5, description="决策阈值")
    description: Optional[str] = Field(None, description="模型描述")

class ModelInfo(BaseModel):
    model_hash: str
    model_type: str
    auc: float
    ks: float
    dp_epsilon: Optional[float]
    created_at: str
    artifacts_path: str
    is_production: bool = False
    threshold: float = 0.5
    description: Optional[str] = None

class FeatureView(BaseModel):
    psi_token: str
    features: Dict[str, Any]
    timestamp: str

class AuditRecord(BaseModel):
    request_id: str
    subject: str
    consent_fingerprint: str
    model_hash: str
    threshold: float
    policy_version: str
    timestamp: str
    decision: str
    score: float
    features_used: List[str]

# 模型注册表管理
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.production_model = None
        self.load_registry()
    
    def register_model(self, model_hash: str, model_info: Dict):
        """注册模型"""
        self.models[model_hash] = {
            "model_hash": model_hash,
            "model_type": model_info.get("model_type", "unknown"),
            "auc": model_info.get("auc", 0.0),
            "ks": model_info.get("ks", 0.0),
            "dp_epsilon": model_info.get("dp_epsilon"),
            "created_at": datetime.now().isoformat(),
            "artifacts_path": model_info.get("artifacts_path", ""),
            "is_production": False,
            "threshold": 0.5,
            "description": model_info.get("description")
        }
        self.save_registry()
        logger.info(f"Model registered: {model_hash}")
    
    def promote_to_production(self, model_hash: str, threshold: float = 0.5, description: str = None):
        """将模型提升为生产版本"""
        if model_hash not in self.models:
            raise ValueError(f"Model {model_hash} not found")
        
        # 取消之前的生产模型
        if self.production_model:
            self.models[self.production_model]["is_production"] = False
        
        # 设置新的生产模型
        self.models[model_hash]["is_production"] = True
        self.models[model_hash]["threshold"] = threshold
        if description:
            self.models[model_hash]["description"] = description
        
        self.production_model = model_hash
        self.save_registry()
        logger.info(f"Model promoted to production: {model_hash}")
    
    def get_production_model(self):
        """获取生产模型"""
        if not self.production_model or self.production_model not in self.models:
            return None
        return self.models[self.production_model]
    
    def list_models(self):
        """列出所有模型"""
        return list(self.models.values())
    
    def save_registry(self):
        """保存注册表"""
        registry_file = os.path.join(MODELS_DIR, "registry.json")
        with open(registry_file, "w") as f:
            json.dump({
                "models": self.models,
                "production_model": self.production_model
            }, f, indent=2)
    
    def load_registry(self):
        """加载注册表"""
        registry_file = os.path.join(MODELS_DIR, "registry.json")
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                    self.models = data.get("models", {})
                    self.production_model = data.get("production_model")
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

# 特征视图管理
class FeatureStore:
    def __init__(self):
        self.features = {}
        self.load_features()
    
    def load_features(self):
        """从合成数据加载特征"""
        try:
            # 加载银行数据
            bank_file = "/app/data/partyA_bank.csv"
            if os.path.exists(bank_file):
                bank_df = pd.read_csv(bank_file)
                for _, row in bank_df.iterrows():
                    psi_token = row["psi_token"]
                    features = {
                        "age": row.get("age", 30),
                        "income": row.get("income", 50000),
                        "credit_history": row.get("credit_history", 5),
                        "loan_amount": row.get("loan_amount", 10000),
                        "employment_years": row.get("employment_years", 3)
                    }
                    self.features[psi_token] = {
                        "psi_token": psi_token,
                        "features": features,
                        "timestamp": datetime.now().isoformat(),
                        "source": "bank"
                    }
            
            # 加载电商数据
            ecom_file = "/app/data/partyB_ecom.csv"
            if os.path.exists(ecom_file):
                ecom_df = pd.read_csv(ecom_file)
                for _, row in ecom_df.iterrows():
                    psi_token = row["psi_token"]
                    if psi_token in self.features:
                        # 合并特征
                        self.features[psi_token]["features"].update({
                            "purchase_frequency": row.get("purchase_frequency", 2),
                            "avg_order_value": row.get("avg_order_value", 100),
                            "category_preference": row.get("category_preference", "electronics"),
                            "return_rate": row.get("return_rate", 0.1),
                            "account_age_days": row.get("account_age_days", 365)
                        })
                    else:
                        # 新建特征记录
                        features = {
                            "purchase_frequency": row.get("purchase_frequency", 2),
                            "avg_order_value": row.get("avg_order_value", 100),
                            "category_preference": row.get("category_preference", "electronics"),
                            "return_rate": row.get("return_rate", 0.1),
                            "account_age_days": row.get("account_age_days", 365)
                        }
                        self.features[psi_token] = {
                            "psi_token": psi_token,
                            "features": features,
                            "timestamp": datetime.now().isoformat(),
                            "source": "ecom"
                        }
            
            logger.info(f"Loaded {len(self.features)} feature records")
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
    
    def get_features(self, psi_token: str) -> Optional[Dict]:
        """获取特征"""
        return self.features.get(psi_token)
    
    def update_features(self, psi_token: str, features: Dict):
        """更新特征"""
        self.features[psi_token] = {
            "psi_token": psi_token,
            "features": features,
            "timestamp": datetime.now().isoformat(),
            "source": "manual"
        }

# 同意客户端
class ConsentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def verify_consent(self, consent_jwt: str, subject: str, purpose: str = "credit_scoring") -> Dict:
        """验证同意票据"""
        try:
            response = requests.post(
                f"{self.base_url}/consent/verify",
                json={
                    "jwt_token": consent_jwt,
                    "subject": subject,
                    "purpose": purpose
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"valid": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Consent verification failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def record_audit(self, audit_record: Dict) -> bool:
        """记录审计"""
        try:
            response = requests.post(
                f"{self.base_url}/audit/record",
                json=audit_record,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Audit recording failed: {e}")
            return False

# 模型推理
class ModelInference:
    def __init__(self):
        self.loaded_models = {}
    
    def load_model(self, model_hash: str, artifacts_path: str):
        """加载模型"""
        if model_hash in self.loaded_models:
            return self.loaded_models[model_hash]
        
        try:
            model_file = os.path.join(artifacts_path, "model.pkl")
            if os.path.exists(model_file):
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                self.loaded_models[model_hash] = model
                logger.info(f"Model loaded: {model_hash}")
                return model
            else:
                # 模拟模型（Demo用）
                logger.warning(f"Model file not found, using mock model: {model_hash}")
                mock_model = self.create_mock_model()
                self.loaded_models[model_hash] = mock_model
                return mock_model
        except Exception as e:
            logger.error(f"Failed to load model {model_hash}: {e}")
            # 返回模拟模型
            mock_model = self.create_mock_model()
            self.loaded_models[model_hash] = mock_model
            return mock_model
    
    def create_mock_model(self):
        """创建模拟模型"""
        class MockModel:
            def predict_proba(self, features):
                # 基于特征计算简单评分
                if isinstance(features, dict):
                    score = 0.0
                    if "age" in features:
                        score += min(features["age"] / 100, 0.3)
                    if "income" in features:
                        score += min(features["income"] / 200000, 0.2)
                    if "credit_history" in features:
                        score += min(features["credit_history"] / 20, 0.2)
                    if "purchase_frequency" in features:
                        score += min(features["purchase_frequency"] / 10, 0.15)
                    if "avg_order_value" in features:
                        score += min(features["avg_order_value"] / 1000, 0.15)
                    return [[1-score, score]]
                return [[0.7, 0.3]]  # 默认评分
        
        return MockModel()
    
    def predict(self, model_hash: str, artifacts_path: str, features: Dict) -> float:
        """模型预测"""
        model = self.load_model(model_hash, artifacts_path)
        try:
            # 预测概率
            proba = model.predict_proba(features)
            if hasattr(proba, 'shape') and len(proba.shape) > 1:
                return float(proba[0][1])  # 返回正类概率
            else:
                return float(proba[0]) if hasattr(proba, '__iter__') else float(proba)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.3  # 默认评分

# 初始化组件
model_registry = ModelRegistry()
feature_store = FeatureStore()
consent_client = ConsentClient(CONSENT_SERVICE_URL)
model_inference = ModelInference()

# API路由
@app.post("/score", response_model=ScoreResponse)
async def score_request(request: ScoreRequest):
    """在线打分"""
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        # 1. 验证同意票据
        consent_result = consent_client.verify_consent(
            request.consent_jwt, 
            request.subject
        )
        
        if not consent_result.get("valid", False):
            raise HTTPException(
                status_code=403, 
                detail=f"Consent verification failed: {consent_result.get('error', 'Unknown error')}"
            )
        
        # 2. 获取生产模型
        prod_model = model_registry.get_production_model()
        if not prod_model:
            raise HTTPException(status_code=404, detail="No production model available")
        
        # 3. 获取特征
        features = request.features
        if not features:
            feature_record = feature_store.get_features(request.psi_token)
            if not feature_record:
                raise HTTPException(status_code=404, detail="Features not found for psi_token")
            features = feature_record["features"]
        
        # 4. 模型推理
        score = model_inference.predict(
            prod_model["model_hash"],
            prod_model["artifacts_path"],
            features
        )
        
        # 5. 决策
        threshold = prod_model["threshold"]
        decision = "approve" if score < threshold else "reject"
        
        # 6. 记录审计
        audit_record = {
            "request_id": request_id,
            "subject": request.subject,
            "consent_fingerprint": hashlib.sha256(request.consent_jwt.encode()).hexdigest()[:16],
            "model_hash": prod_model["model_hash"],
            "threshold": threshold,
            "policy_version": consent_result.get("policy_version", "v1.0"),
            "timestamp": timestamp,
            "decision": decision,
            "score": score,
            "features_used": list(features.keys())
        }
        
        consent_client.record_audit(audit_record)
        
        return ScoreResponse(
            score=score,
            decision=decision,
            model_hash=prod_model["model_hash"],
            threshold=threshold,
            request_id=request_id,
            timestamp=timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/promote")
async def promote_model(request: ModelPromoteRequest):
    """提升模型为生产版本"""
    try:
        model_registry.promote_to_production(
            request.model_hash,
            request.threshold,
            request.description
        )
        return {
            "message": "Model promoted to production",
            "model_hash": request.model_hash,
            "threshold": request.threshold
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """列出所有模型"""
    models = model_registry.list_models()
    return [ModelInfo(**model) for model in models]

@app.get("/models/production")
async def get_production_model():
    """获取生产模型信息"""
    prod_model = model_registry.get_production_model()
    if not prod_model:
        raise HTTPException(status_code=404, detail="No production model")
    return ModelInfo(**prod_model)

@app.post("/models/register")
async def register_model(model_hash: str, model_info: Dict):
    """注册模型（通常由训练服务调用）"""
    try:
        model_registry.register_model(model_hash, model_info)
        return {"message": "Model registered", "model_hash": model_hash}
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/features/{psi_token}")
async def get_features(psi_token: str):
    """获取特征"""
    feature_record = feature_store.get_features(psi_token)
    if not feature_record:
        raise HTTPException(status_code=404, detail="Features not found")
    return FeatureView(**feature_record)

@app.post("/features/{psi_token}")
async def update_features(psi_token: str, features: Dict[str, Any]):
    """更新特征"""
    feature_store.update_features(psi_token, features)
    return {"message": "Features updated", "psi_token": psi_token}

@app.get("/healthz")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_count": len(model_registry.models),
        "features_count": len(feature_store.features),
        "production_model": model_registry.production_model
    }

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    return {
        "models": {
            "total": len(model_registry.models),
            "production": 1 if model_registry.production_model else 0
        },
        "features": {
            "total": len(feature_store.features)
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7004)