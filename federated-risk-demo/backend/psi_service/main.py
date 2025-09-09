from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import hashlib
import time
import uuid
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PSI服务",
    description="隐私集合求交服务",
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
class PSIRequest(BaseModel):
    party_id: str
    data_set: List[str]
    hash_function: str = "sha256"
    privacy_budget: float = 1.0

class PSIResponse(BaseModel):
    request_id: str
    status: str
    intersection_size: Optional[int] = None
    intersection_hash: Optional[str] = None
    privacy_cost: Optional[float] = None
    created_at: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float

# 全局状态
start_time = time.time()
psi_requests: Dict[str, PSIResponse] = {}
privacy_budgets: Dict[str, float] = {}

# PSI协议实现
class PSIProtocol:
    @staticmethod
    def hash_data(data: List[str], hash_function: str = "sha256") -> Set[str]:
        """对数据进行哈希处理"""
        if hash_function == "sha256":
            hasher = hashlib.sha256
        elif hash_function == "md5":
            hasher = hashlib.md5
        else:
            raise ValueError(f"不支持的哈希函数: {hash_function}")
        
        return {hasher(item.encode()).hexdigest() for item in data}
    
    @staticmethod
    def compute_intersection(set1: Set[str], set2: Set[str]) -> Set[str]:
        """计算两个集合的交集"""
        return set1.intersection(set2)
    
    @staticmethod
    def calculate_privacy_cost(data_size: int, intersection_size: int) -> float:
        """计算隐私预算消耗"""
        if data_size == 0:
            return 0.0
        return min(1.0, intersection_size / data_size * 0.1)

# API端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=time.time() - start_time
    )

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    return {
        "total_requests": len(psi_requests),
        "active_parties": len(privacy_budgets),
        "uptime": time.time() - start_time,
        "memory_usage": "N/A",
        "cpu_usage": "N/A"
    }

@app.post("/psi/compute", response_model=PSIResponse)
async def compute_psi(request: PSIRequest):
    """计算PSI"""
    try:
        # 验证请求
        if not request.data_set:
            raise HTTPException(status_code=400, detail="数据集不能为空")
        
        if request.privacy_budget <= 0:
            raise HTTPException(status_code=400, detail="隐私预算必须大于0")
        
        # 检查隐私预算
        current_budget = privacy_budgets.get(request.party_id, 10.0)
        if current_budget < request.privacy_budget:
            raise HTTPException(status_code=400, detail="隐私预算不足")
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 对数据进行哈希处理
        hashed_data = PSIProtocol.hash_data(request.data_set, request.hash_function)
        
        # 模拟与其他方的交集计算
        # 在实际实现中，这里会与其他参与方进行安全多方计算
        mock_other_data = {hashlib.sha256(f"item_{i}".encode()).hexdigest() for i in range(50)}
        intersection = PSIProtocol.compute_intersection(hashed_data, mock_other_data)
        
        # 计算隐私成本
        privacy_cost = PSIProtocol.calculate_privacy_cost(len(request.data_set), len(intersection))
        
        # 更新隐私预算
        privacy_budgets[request.party_id] = current_budget - privacy_cost
        
        # 创建响应
        response = PSIResponse(
            request_id=request_id,
            status="completed",
            intersection_size=len(intersection),
            intersection_hash=hashlib.sha256(str(sorted(intersection)).encode()).hexdigest(),
            privacy_cost=privacy_cost,
            created_at=datetime.now().isoformat()
        )
        
        # 存储请求记录
        psi_requests[request_id] = response
        
        logger.info(f"PSI计算完成: {request_id}, 交集大小: {len(intersection)}")
        
        return response
        
    except Exception as e:
        logger.error(f"PSI计算失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PSI计算失败: {str(e)}")

@app.get("/psi/status/{request_id}", response_model=PSIResponse)
async def get_psi_status(request_id: str):
    """获取PSI计算状态"""
    if request_id not in psi_requests:
        raise HTTPException(status_code=404, detail="请求不存在")
    
    return psi_requests[request_id]

@app.get("/psi/requests")
async def list_psi_requests(party_id: Optional[str] = None):
    """列出PSI请求"""
    requests = list(psi_requests.values())
    
    if party_id:
        # 在实际实现中，需要根据party_id过滤
        pass
    
    return {
        "requests": requests,
        "total": len(requests)
    }

@app.get("/privacy/budget/{party_id}")
async def get_privacy_budget(party_id: str):
    """获取隐私预算"""
    budget = privacy_budgets.get(party_id, 10.0)
    return {
        "party_id": party_id,
        "remaining_budget": budget,
        "total_budget": 10.0
    }

@app.post("/privacy/budget/{party_id}/reset")
async def reset_privacy_budget(party_id: str):
    """重置隐私预算"""
    privacy_budgets[party_id] = 10.0
    return {
        "party_id": party_id,
        "remaining_budget": 10.0,
        "message": "隐私预算已重置"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)