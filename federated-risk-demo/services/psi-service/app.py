#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSI (Private Set Intersection) 服务
模拟ECDH-PSI流程，使用哈希token进行安全交集计算
"""

import os
import uuid
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union
from collections import defaultdict

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局存储 (生产环境应使用Redis等)
PSI_SESSIONS: Dict[str, Dict] = {}
PSI_RESULTS: Dict[str, Dict] = {}
CLEANUP_INTERVAL = 3600  # 1小时清理一次

app = FastAPI(
    title="PSI Service",
    description="隐私集合交集服务 - 模拟ECDH-PSI流程",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic模型
class PSIStartRequest(BaseModel):
    """PSI会话启动请求"""
    session_name: Optional[str] = Field(None, description="会话名称")
    description: Optional[str] = Field(None, description="会话描述")
    ttl_minutes: Optional[int] = Field(60, description="会话存活时间(分钟)")

class PSIStartResponse(BaseModel):
    """PSI会话启动响应"""
    session_id: str
    created_at: str
    expires_at: str
    status: str

class PSIUploadRequest(BaseModel):
    """PSI数据上传请求"""
    session_id: str
    side: str  # 'A' or 'B'
    psi_tokens: List[str]
    metadata: Optional[Dict] = Field(default_factory=dict)

class PSIUploadResponse(BaseModel):
    """PSI数据上传响应"""
    session_id: str
    side: str
    token_count: int
    unique_count: int
    upload_time: str

class PSIComputeRequest(BaseModel):
    """PSI计算请求"""
    session_id: str
    compute_preview: bool = Field(True, description="是否计算预览样本")
    preview_limit: int = Field(10, description="预览样本数量")

class PSIComputeResponse(BaseModel):
    """PSI计算响应"""
    session_id: str
    intersect_size: int
    side_a_size: int
    side_b_size: int
    intersection_rate_a: float
    intersection_rate_b: float
    sample_preview: List[str]
    mapping_store_key: str
    compute_time: str
    metrics: Dict

class PSIResultResponse(BaseModel):
    """PSI结果响应"""
    session_id: str
    mapping_store_key: str
    intersection_tokens: List[str]
    metadata: Dict

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    active_sessions: int
    total_results: int

def cleanup_expired_sessions():
    """清理过期会话"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in PSI_SESSIONS.items():
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if current_time > expires_at:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del PSI_SESSIONS[session_id]
        # 同时清理相关结果
        keys_to_remove = [k for k in PSI_RESULTS.keys() if k.startswith(session_id)]
        for key in keys_to_remove:
            del PSI_RESULTS[key]
    
    if expired_sessions:
        logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

def compute_psi_intersection(tokens_a: Set[str], tokens_b: Set[str]) -> Dict:
    """计算PSI交集"""
    start_time = datetime.now()
    
    # 计算交集
    intersection = tokens_a & tokens_b
    
    # 计算指标
    metrics = {
        'side_a_size': len(tokens_a),
        'side_b_size': len(tokens_b),
        'intersection_size': len(intersection),
        'intersection_rate_a': len(intersection) / len(tokens_a) if tokens_a else 0,
        'intersection_rate_b': len(intersection) / len(tokens_b) if tokens_b else 0,
        'union_size': len(tokens_a | tokens_b),
        'jaccard_similarity': len(intersection) / len(tokens_a | tokens_b) if (tokens_a | tokens_b) else 0,
        'compute_time_ms': (datetime.now() - start_time).total_seconds() * 1000
    }
    
    return {
        'intersection': intersection,
        'metrics': metrics
    }

@app.post("/psi/start", response_model=PSIStartResponse)
async def start_psi_session(request: PSIStartRequest):
    """启动PSI会话"""
    session_id = str(uuid.uuid4())
    created_at = datetime.now()
    expires_at = created_at + timedelta(minutes=request.ttl_minutes)
    
    session_data = {
        'session_id': session_id,
        'session_name': request.session_name or f"PSI-{session_id[:8]}",
        'description': request.description,
        'created_at': created_at.isoformat(),
        'expires_at': expires_at.isoformat(),
        'status': 'created',
        'sides': {},
        'computed': False
    }
    
    PSI_SESSIONS[session_id] = session_data
    
    logger.info(f"创建PSI会话: {session_id}")
    
    return PSIStartResponse(
        session_id=session_id,
        created_at=created_at.isoformat(),
        expires_at=expires_at.isoformat(),
        status='created'
    )

@app.post("/psi/upload/{side}", response_model=PSIUploadResponse)
async def upload_psi_data(side: str, request: PSIUploadRequest):
    """上传PSI数据"""
    if side not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="side必须是'A'或'B'")
    
    if request.session_id not in PSI_SESSIONS:
        raise HTTPException(status_code=404, detail="PSI会话不存在")
    
    session = PSI_SESSIONS[request.session_id]
    
    # 检查会话是否过期
    if datetime.now() > datetime.fromisoformat(session['expires_at']):
        raise HTTPException(status_code=410, detail="PSI会话已过期")
    
    # 验证和处理tokens
    if not request.psi_tokens:
        raise HTTPException(status_code=400, detail="psi_tokens不能为空")
    
    # 去重并验证token格式
    unique_tokens = set()
    for token in request.psi_tokens:
        if not isinstance(token, str) or len(token) != 64:  # SHA256长度
            raise HTTPException(status_code=400, detail=f"无效的PSI token格式: {token[:10]}...")
        unique_tokens.add(token)
    
    upload_time = datetime.now().isoformat()
    
    # 存储数据
    session['sides'][side] = {
        'tokens': unique_tokens,
        'token_count': len(request.psi_tokens),
        'unique_count': len(unique_tokens),
        'upload_time': upload_time,
        'metadata': request.metadata
    }
    
    # 更新会话状态
    if len(session['sides']) == 2:
        session['status'] = 'ready_to_compute'
    else:
        session['status'] = 'partial_upload'
    
    logger.info(f"会话 {request.session_id} 上传方 {side}: {len(unique_tokens)} 个唯一tokens")
    
    return PSIUploadResponse(
        session_id=request.session_id,
        side=side,
        token_count=len(request.psi_tokens),
        unique_count=len(unique_tokens),
        upload_time=upload_time
    )

@app.post("/psi/upload/{side}/file")
async def upload_psi_file(side: str, session_id: str, file: UploadFile = File(...)):
    """通过文件上传PSI数据"""
    if side not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="side必须是'A'或'B'")
    
    if session_id not in PSI_SESSIONS:
        raise HTTPException(status_code=404, detail="PSI会话不存在")
    
    try:
        # 读取文件内容
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            # CSV文件处理
            import io
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            if 'psi_token' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV文件必须包含'psi_token'列")
            psi_tokens = df['psi_token'].dropna().astype(str).tolist()
        elif file.filename.endswith('.json'):
            # JSON文件处理
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                psi_tokens = data
            elif isinstance(data, dict) and 'psi_tokens' in data:
                psi_tokens = data['psi_tokens']
            else:
                raise HTTPException(status_code=400, detail="JSON格式不正确")
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式，请使用CSV或JSON")
        
        # 调用上传接口
        upload_request = PSIUploadRequest(
            session_id=session_id,
            side=side,
            psi_tokens=psi_tokens,
            metadata={'filename': file.filename, 'file_size': len(content)}
        )
        
        return await upload_psi_data(side, upload_request)
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

@app.post("/psi/compute", response_model=PSIComputeResponse)
async def compute_psi(request: PSIComputeRequest):
    """计算PSI交集"""
    if request.session_id not in PSI_SESSIONS:
        raise HTTPException(status_code=404, detail="PSI会话不存在")
    
    session = PSI_SESSIONS[request.session_id]
    
    # 检查会话状态
    if len(session['sides']) != 2:
        raise HTTPException(status_code=400, detail="需要双方都上传数据才能计算")
    
    if datetime.now() > datetime.fromisoformat(session['expires_at']):
        raise HTTPException(status_code=410, detail="PSI会话已过期")
    
    # 获取双方tokens
    tokens_a = session['sides']['A']['tokens']
    tokens_b = session['sides']['B']['tokens']
    
    # 计算交集
    result = compute_psi_intersection(tokens_a, tokens_b)
    intersection = result['intersection']
    metrics = result['metrics']
    
    # 生成预览样本
    sample_preview = []
    if request.compute_preview and intersection:
        sample_list = list(intersection)
        sample_count = min(request.preview_limit, len(sample_list))
        sample_preview = sample_list[:sample_count]
    
    # 生成存储key
    mapping_store_key = f"{request.session_id}_{uuid.uuid4().hex[:8]}"
    
    # 存储结果
    compute_time = datetime.now().isoformat()
    PSI_RESULTS[mapping_store_key] = {
        'session_id': request.session_id,
        'intersection_tokens': list(intersection),
        'compute_time': compute_time,
        'metrics': metrics,
        'metadata': {
            'side_a_metadata': session['sides']['A']['metadata'],
            'side_b_metadata': session['sides']['B']['metadata']
        }
    }
    
    # 更新会话状态
    session['computed'] = True
    session['status'] = 'computed'
    session['last_compute_time'] = compute_time
    
    logger.info(f"会话 {request.session_id} PSI计算完成: 交集大小 {len(intersection)}")
    
    return PSIComputeResponse(
        session_id=request.session_id,
        intersect_size=len(intersection),
        side_a_size=metrics['side_a_size'],
        side_b_size=metrics['side_b_size'],
        intersection_rate_a=metrics['intersection_rate_a'],
        intersection_rate_b=metrics['intersection_rate_b'],
        sample_preview=sample_preview,
        mapping_store_key=mapping_store_key,
        compute_time=compute_time,
        metrics=metrics
    )

@app.get("/psi/result/{mapping_store_key}", response_model=PSIResultResponse)
async def get_psi_result(mapping_store_key: str):
    """获取PSI计算结果"""
    if mapping_store_key not in PSI_RESULTS:
        raise HTTPException(status_code=404, detail="PSI结果不存在")
    
    result = PSI_RESULTS[mapping_store_key]
    
    return PSIResultResponse(
        session_id=result['session_id'],
        mapping_store_key=mapping_store_key,
        intersection_tokens=result['intersection_tokens'],
        metadata=result['metadata']
    )

@app.get("/psi/sessions")
async def list_psi_sessions():
    """列出所有PSI会话"""
    sessions = []
    for session_id, session_data in PSI_SESSIONS.items():
        sessions.append({
            'session_id': session_id,
            'session_name': session_data['session_name'],
            'status': session_data['status'],
            'created_at': session_data['created_at'],
            'expires_at': session_data['expires_at'],
            'sides_count': len(session_data['sides']),
            'computed': session_data['computed']
        })
    
    return {'sessions': sessions, 'total': len(sessions)}

@app.delete("/psi/sessions/{session_id}")
async def delete_psi_session(session_id: str):
    """删除PSI会话"""
    if session_id not in PSI_SESSIONS:
        raise HTTPException(status_code=404, detail="PSI会话不存在")
    
    # 删除会话
    del PSI_SESSIONS[session_id]
    
    # 删除相关结果
    keys_to_remove = [k for k in PSI_RESULTS.keys() if k.startswith(session_id)]
    for key in keys_to_remove:
        del PSI_RESULTS[key]
    
    logger.info(f"删除PSI会话: {session_id}")
    
    return {'message': f'会话 {session_id} 已删除'}

@app.post("/psi/cleanup")
async def manual_cleanup(background_tasks: BackgroundTasks):
    """手动清理过期会话"""
    background_tasks.add_task(cleanup_expired_sessions)
    return {'message': '清理任务已启动'}

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        active_sessions=len(PSI_SESSIONS),
        total_results=len(PSI_RESULTS)
    )

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    total_sessions = len(PSI_SESSIONS)
    computed_sessions = sum(1 for s in PSI_SESSIONS.values() if s['computed'])
    
    return {
        'total_sessions': total_sessions,
        'computed_sessions': computed_sessions,
        'pending_sessions': total_sessions - computed_sessions,
        'total_results': len(PSI_RESULTS),
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7001))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )