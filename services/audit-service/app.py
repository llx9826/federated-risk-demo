#!/usr/bin/env python3
"""
审计服务 - 联邦风控合规审计和记录

实现功能：
1. 审计记录生成和存储
2. 合规性检查和评分
3. 审计报告生成
4. 数据血缘追踪
5. 隐私影响评估
6. 监管报告导出
"""

import os
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
import jinja2
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# 环境配置
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/federated_risk")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CONSENT_GATEWAY_URL = os.getenv("CONSENT_GATEWAY_URL", "http://localhost:8001")
REPORT_STORAGE_PATH = os.getenv("REPORT_STORAGE_PATH", "./data/reports")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "./templates")

# 创建必要目录
Path("./logs").mkdir(exist_ok=True)
Path(REPORT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(TEMPLATE_PATH).mkdir(parents=True, exist_ok=True)
Path("./data/lineage").mkdir(parents=True, exist_ok=True)

# 全局变量
db_pool = None
redis_client = None
jinja_env = None

# Prometheus指标
audit_records_total = Counter('audit_records_total', 'Total audit records created', ['audit_type', 'compliance_status'])
audit_processing_duration = Histogram('audit_processing_duration_seconds', 'Audit processing duration', ['audit_type'])
compliance_score_gauge = Gauge('compliance_score', 'Compliance score', ['workflow_id', 'standard'])
reports_generated_total = Counter('reports_generated_total', 'Total reports generated', ['report_type', 'format'])
data_lineage_events_total = Counter('data_lineage_events_total', 'Total data lineage events', ['event_type'])

# 枚举定义
from enum import Enum

class AuditType(str, Enum):
    WORKFLOW_AUDIT = "workflow_audit"
    DATA_AUDIT = "data_audit"
    MODEL_AUDIT = "model_audit"
    PRIVACY_AUDIT = "privacy_audit"
    COMPLIANCE_AUDIT = "compliance_audit"

class ComplianceStandard(str, Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"

class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"

class LineageEventType(str, Enum):
    DATA_INGESTION = "data_ingestion"
    DATA_TRANSFORMATION = "data_transformation"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"

# FastAPI应用初始化
app = FastAPI(
    title="联邦风控审计服务",
    description="提供合规审计、数据血缘追踪和监管报告功能",
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
class AuditRecordRequest(BaseModel):
    """审计记录请求"""
    workflow_id: str = Field(..., description="工作流ID")
    audit_type: AuditType = Field(..., description="审计类型")
    audit_scope: str = Field(default="full", description="审计范围")
    compliance_standards: List[ComplianceStandard] = Field(default_factory=list, description="合规标准")
    step_results: Dict[str, Any] = Field(default_factory=dict, description="步骤结果")
    parties: List[str] = Field(default_factory=list, description="参与方")
    data_purpose: str = Field(default="risk_assessment", description="数据使用目的")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class ComplianceCheckRequest(BaseModel):
    """合规性检查请求"""
    audit_id: str = Field(..., description="审计ID")
    standards: List[ComplianceStandard] = Field(..., description="检查标准")
    detailed_check: bool = Field(default=True, description="详细检查")

class ReportGenerationRequest(BaseModel):
    """报告生成请求"""
    audit_id: str = Field(..., description="审计ID")
    report_type: str = Field(default="compliance_report", description="报告类型")
    format: ReportFormat = Field(default=ReportFormat.PDF, description="报告格式")
    include_recommendations: bool = Field(default=True, description="包含建议")
    template_name: Optional[str] = Field(None, description="模板名称")

class LineageEventRequest(BaseModel):
    """数据血缘事件请求"""
    workflow_id: str = Field(..., description="工作流ID")
    event_type: LineageEventType = Field(..., description="事件类型")
    source_entity: str = Field(..., description="源实体")
    target_entity: str = Field(..., description="目标实体")
    transformation_details: Dict[str, Any] = Field(default_factory=dict, description="转换详情")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class PrivacyImpactAssessmentRequest(BaseModel):
    """隐私影响评估请求"""
    workflow_id: str = Field(..., description="工作流ID")
    data_types: List[str] = Field(..., description="数据类型")
    processing_purposes: List[str] = Field(..., description="处理目的")
    data_subjects: List[str] = Field(..., description="数据主体")
    risk_factors: Dict[str, Any] = Field(default_factory=dict, description="风险因素")

class AuditRecordResponse(BaseModel):
    """审计记录响应"""
    audit_id: str
    workflow_id: str
    audit_type: AuditType
    compliance_score: float
    compliance_status: ComplianceStatus
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime
    report_url: Optional[str]

class ComplianceCheckResponse(BaseModel):
    """合规性检查响应"""
    audit_id: str
    overall_score: float
    standard_scores: Dict[str, float]
    compliance_status: ComplianceStatus
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    next_review_date: datetime

class ReportResponse(BaseModel):
    """报告响应"""
    report_id: str
    audit_id: str
    report_type: str
    format: ReportFormat
    file_path: str
    download_url: str
    generated_at: datetime
    expires_at: datetime

class LineageResponse(BaseModel):
    """数据血缘响应"""
    workflow_id: str
    lineage_graph: Dict[str, Any]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    impact_analysis: Dict[str, Any]

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    redis_status: str
    active_audits: int

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
                CREATE TABLE IF NOT EXISTS audit_records (
                    id SERIAL PRIMARY KEY,
                    audit_id VARCHAR(128) UNIQUE NOT NULL,
                    workflow_id VARCHAR(128) NOT NULL,
                    audit_type VARCHAR(64) NOT NULL,
                    audit_scope VARCHAR(128) NOT NULL,
                    compliance_standards TEXT[] NOT NULL,
                    compliance_score FLOAT NOT NULL DEFAULT 0.0,
                    compliance_status VARCHAR(32) NOT NULL,
                    step_results JSONB NOT NULL,
                    parties TEXT[] NOT NULL,
                    data_purpose VARCHAR(128) NOT NULL,
                    findings JSONB NOT NULL,
                    recommendations TEXT[] NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS compliance_checks (
                    id SERIAL PRIMARY KEY,
                    audit_id VARCHAR(128) NOT NULL,
                    standard VARCHAR(64) NOT NULL,
                    score FLOAT NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    violations JSONB NOT NULL,
                    recommendations TEXT[] NOT NULL,
                    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    next_review_date TIMESTAMPTZ,
                    UNIQUE(audit_id, standard)
                );
                
                CREATE TABLE IF NOT EXISTS audit_reports (
                    id SERIAL PRIMARY KEY,
                    report_id VARCHAR(128) UNIQUE NOT NULL,
                    audit_id VARCHAR(128) NOT NULL,
                    report_type VARCHAR(64) NOT NULL,
                    format VARCHAR(16) NOT NULL,
                    file_path VARCHAR(512) NOT NULL,
                    file_size BIGINT,
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL,
                    download_count INTEGER NOT NULL DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(128) NOT NULL,
                    event_type VARCHAR(64) NOT NULL,
                    source_entity VARCHAR(256) NOT NULL,
                    target_entity VARCHAR(256) NOT NULL,
                    transformation_details JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS privacy_assessments (
                    id SERIAL PRIMARY KEY,
                    assessment_id VARCHAR(128) UNIQUE NOT NULL,
                    workflow_id VARCHAR(128) NOT NULL,
                    data_types TEXT[] NOT NULL,
                    processing_purposes TEXT[] NOT NULL,
                    data_subjects TEXT[] NOT NULL,
                    risk_score FLOAT NOT NULL,
                    risk_level VARCHAR(32) NOT NULL,
                    mitigation_measures JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_audit_records_workflow ON audit_records (workflow_id);
                CREATE INDEX IF NOT EXISTS idx_audit_records_type ON audit_records (audit_type);
                CREATE INDEX IF NOT EXISTS idx_compliance_checks_audit ON compliance_checks (audit_id);
                CREATE INDEX IF NOT EXISTS idx_audit_reports_audit ON audit_reports (audit_id);
                CREATE INDEX IF NOT EXISTS idx_data_lineage_workflow ON data_lineage (workflow_id);
                CREATE INDEX IF NOT EXISTS idx_privacy_assessments_workflow ON privacy_assessments (workflow_id);
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

# 模板引擎初始化
async def init_templates():
    """初始化Jinja2模板引擎"""
    global jinja_env
    try:
        jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(TEMPLATE_PATH),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # 创建默认模板
        await create_default_templates()
        
        logger.info("模板引擎初始化成功")
    except Exception as e:
        logger.error(f"模板引擎初始化失败: {e}")
        raise

async def create_default_templates():
    """创建默认报告模板"""
    # 合规报告HTML模板
    compliance_template = """
<!DOCTYPE html>
<html>
<head>
    <title>联邦风控合规审计报告</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .score { font-size: 24px; font-weight: bold; color: {{ 'green' if compliance_score >= 0.8 else 'orange' if compliance_score >= 0.6 else 'red' }}; }
        .violation { background-color: #ffe6e6; padding: 10px; margin: 10px 0; border-left: 4px solid #ff0000; }
        .recommendation { background-color: #e6f3ff; padding: 10px; margin: 10px 0; border-left: 4px solid #0066cc; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>联邦风控合规审计报告</h1>
        <p>工作流ID: {{ workflow_id }}</p>
        <p>审计ID: {{ audit_id }}</p>
        <p>生成时间: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h2>合规性评分</h2>
        <div class="score">总体评分: {{ "%.1f" | format(compliance_score * 100) }}%</div>
        <p>状态: {{ compliance_status }}</p>
    </div>
    
    <div class="section">
        <h2>标准评分详情</h2>
        <table>
            <tr><th>合规标准</th><th>评分</th><th>状态</th></tr>
            {% for standard, score in standard_scores.items() %}
            <tr>
                <td>{{ standard.upper() }}</td>
                <td>{{ "%.1f" | format(score * 100) }}%</td>
                <td>{{ 'PASS' if score >= 0.8 else 'FAIL' }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    {% if violations %}
    <div class="section">
        <h2>合规性违规</h2>
        {% for violation in violations %}
        <div class="violation">
            <strong>{{ violation.standard }}</strong>: {{ violation.description }}
            <br><small>严重程度: {{ violation.severity }}</small>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if recommendations %}
    <div class="section">
        <h2>改进建议</h2>
        {% for recommendation in recommendations %}
        <div class="recommendation">
            {{ recommendation }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <h2>审计详情</h2>
        <p><strong>审计类型:</strong> {{ audit_type }}</p>
        <p><strong>审计范围:</strong> {{ audit_scope }}</p>
        <p><strong>参与方:</strong> {{ parties | join(', ') }}</p>
        <p><strong>数据用途:</strong> {{ data_purpose }}</p>
    </div>
</body>
</html>
    """
    
    template_file = Path(TEMPLATE_PATH) / "compliance_report.html"
    template_file.write_text(compliance_template, encoding='utf-8')

# 工具函数
async def generate_audit_id() -> str:
    """生成审计ID"""
    return f"audit_{uuid.uuid4().hex[:16]}"

async def calculate_compliance_score(step_results: Dict[str, Any], 
                                   standards: List[ComplianceStandard]) -> float:
    """计算合规性评分"""
    total_score = 0.0
    weight_sum = 0.0
    
    # 基于步骤结果计算评分
    step_weights = {
        "consent": 0.25,      # 同意收集权重
        "alignment": 0.15,    # 数据对齐权重
        "training": 0.20,     # 联邦训练权重
        "explanation": 0.15,  # 模型解释权重
        "deployment": 0.15,   # 模型部署权重
        "audit": 0.10        # 审计记录权重
    }
    
    for step, result in step_results.items():
        if step in step_weights:
            weight = step_weights[step]
            
            # 根据步骤类型计算评分
            if step == "consent":
                # 同意收集评分：检查是否有有效的同意令牌
                score = 1.0 if result.get("consent_tokens") else 0.0
            elif step == "alignment":
                # 数据对齐评分：基于对齐质量
                quality = result.get("alignment_quality", "low")
                score = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(quality, 0.0)
            elif step == "training":
                # 训练评分：基于模型性能和隐私保护
                performance = result.get("model_performance", {})
                score = min(performance.get("accuracy", 0.0), 1.0)
            elif step == "explanation":
                # 解释评分：基于解释质量
                quality = result.get("explanation_quality", 0.0)
                score = min(quality, 1.0)
            elif step == "deployment":
                # 部署评分：检查部署状态
                status = result.get("deployment_status", "failed")
                score = 1.0 if status == "deployed" else 0.0
            elif step == "audit":
                # 审计评分：基于合规状态
                status = result.get("compliance_status", "non_compliant")
                score = 1.0 if status == "compliant" else 0.5
            else:
                score = 0.5  # 默认评分
            
            total_score += score * weight
            weight_sum += weight
    
    # 标准化评分
    if weight_sum > 0:
        base_score = total_score / weight_sum
    else:
        base_score = 0.0
    
    # 根据合规标准调整评分
    standard_penalty = 0.0
    for standard in standards:
        if standard == ComplianceStandard.GDPR:
            # GDPR要求更严格的隐私保护
            if "consent" not in step_results:
                standard_penalty += 0.2
        elif standard == ComplianceStandard.PCI_DSS:
            # PCI DSS要求更严格的数据安全
            if "encryption" not in str(step_results):
                standard_penalty += 0.1
    
    final_score = max(0.0, min(1.0, base_score - standard_penalty))
    return final_score

async def determine_compliance_status(score: float) -> ComplianceStatus:
    """确定合规状态"""
    if score >= 0.9:
        return ComplianceStatus.COMPLIANT
    elif score >= 0.7:
        return ComplianceStatus.PARTIALLY_COMPLIANT
    else:
        return ComplianceStatus.NON_COMPLIANT

async def generate_findings(step_results: Dict[str, Any], 
                          compliance_score: float) -> List[Dict[str, Any]]:
    """生成审计发现"""
    findings = []
    
    # 检查各个步骤的问题
    if "consent" not in step_results:
        findings.append({
            "category": "数据同意",
            "severity": "high",
            "description": "缺少用户数据使用同意记录",
            "impact": "可能违反GDPR等隐私法规"
        })
    
    if "alignment" in step_results:
        alignment = step_results["alignment"]
        if alignment.get("intersection_size", 0) < 100:
            findings.append({
                "category": "数据质量",
                "severity": "medium",
                "description": "数据对齐交集过小，可能影响模型质量",
                "impact": "模型准确性和泛化能力可能不足"
            })
    
    if "training" in step_results:
        training = step_results["training"]
        if training.get("model_performance", {}).get("accuracy", 0) < 0.7:
            findings.append({
                "category": "模型性能",
                "severity": "medium",
                "description": "模型准确率低于预期阈值",
                "impact": "可能需要调整模型参数或增加训练数据"
            })
    
    if "explanation" not in step_results:
        findings.append({
            "category": "模型可解释性",
            "severity": "medium",
            "description": "缺少模型解释性分析",
            "impact": "可能影响模型的可信度和监管合规性"
        })
    
    if compliance_score < 0.8:
        findings.append({
            "category": "整体合规性",
            "severity": "high" if compliance_score < 0.6 else "medium",
            "description": f"整体合规评分({compliance_score:.2f})低于建议阈值(0.8)",
            "impact": "可能面临监管风险和合规性问题"
        })
    
    return findings

async def generate_recommendations(findings: List[Dict[str, Any]], 
                                 standards: List[ComplianceStandard]) -> List[str]:
    """生成改进建议"""
    recommendations = []
    
    # 基于发现生成建议
    for finding in findings:
        category = finding["category"]
        severity = finding["severity"]
        
        if category == "数据同意":
            recommendations.append("建议实施完整的用户同意收集和管理流程，确保符合GDPR等法规要求")
        elif category == "数据质量":
            recommendations.append("建议优化数据预处理和特征工程，提高数据对齐质量")
        elif category == "模型性能":
            recommendations.append("建议调整模型超参数、增加训练轮数或优化特征选择")
        elif category == "模型可解释性":
            recommendations.append("建议集成SHAP或LIME等模型解释工具，提供模型决策的可解释性")
        elif category == "整体合规性":
            if severity == "high":
                recommendations.append("建议立即审查和改进整个工作流程，确保满足监管要求")
            else:
                recommendations.append("建议定期评估和优化工作流程，持续提升合规水平")
    
    # 基于合规标准生成建议
    for standard in standards:
        if standard == ComplianceStandard.GDPR:
            recommendations.append("建议实施数据最小化原则，仅收集和处理必要的个人数据")
            recommendations.append("建议建立数据主体权利响应机制，支持访问、更正和删除请求")
        elif standard == ComplianceStandard.PCI_DSS:
            recommendations.append("建议加强数据传输和存储的加密保护")
            recommendations.append("建议实施访问控制和审计日志记录")
        elif standard == ComplianceStandard.CCPA:
            recommendations.append("建议提供清晰的隐私政策和数据使用说明")
            recommendations.append("建议支持消费者选择退出数据销售的权利")
    
    # 去重并返回
    return list(set(recommendations))

# API路由实现
@app.post("/audit/records", response_model=AuditRecordResponse, summary="创建审计记录")
async def create_audit_record(request: AuditRecordRequest, background_tasks: BackgroundTasks):
    """创建新的审计记录"""
    start_time = time.time()
    
    try:
        audit_id = await generate_audit_id()
        
        # 计算合规性评分
        compliance_score = await calculate_compliance_score(
            request.step_results, request.compliance_standards
        )
        
        # 确定合规状态
        compliance_status = await determine_compliance_status(compliance_score)
        
        # 生成审计发现
        findings = await generate_findings(request.step_results, compliance_score)
        
        # 生成改进建议
        recommendations = await generate_recommendations(findings, request.compliance_standards)
        
        # 保存审计记录
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_records (
                    audit_id, workflow_id, audit_type, audit_scope, compliance_standards,
                    compliance_score, compliance_status, step_results, parties, data_purpose,
                    findings, recommendations, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, 
                audit_id, request.workflow_id, request.audit_type.value, request.audit_scope,
                [s.value for s in request.compliance_standards], compliance_score, compliance_status.value,
                json.dumps(request.step_results), request.parties, request.data_purpose,
                json.dumps(findings), recommendations, json.dumps(request.metadata)
            )
        
        # 后台生成报告
        background_tasks.add_task(
            generate_compliance_report, audit_id, ReportFormat.PDF
        )
        
        # 更新指标
        audit_records_total.labels(
            audit_type=request.audit_type.value, 
            compliance_status=compliance_status.value
        ).inc()
        
        audit_processing_duration.labels(
            audit_type=request.audit_type.value
        ).observe(time.time() - start_time)
        
        for standard in request.compliance_standards:
            compliance_score_gauge.labels(
                workflow_id=request.workflow_id,
                standard=standard.value
            ).set(compliance_score)
        
        logger.info(f"审计记录创建成功: {audit_id}")
        
        return AuditRecordResponse(
            audit_id=audit_id,
            workflow_id=request.workflow_id,
            audit_type=request.audit_type,
            compliance_score=compliance_score,
            compliance_status=compliance_status,
            findings=findings,
            recommendations=recommendations,
            created_at=datetime.utcnow(),
            report_url=f"/audit/reports/{audit_id}/download"
        )
        
    except Exception as e:
        logger.error(f"创建审计记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建审计记录失败: {str(e)}")

@app.get("/audit/records/{audit_id}", summary="获取审计记录")
async def get_audit_record(audit_id: str):
    """获取指定的审计记录"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM audit_records WHERE audit_id = $1",
                audit_id
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="审计记录不存在")
            
            return {
                "audit_id": row['audit_id'],
                "workflow_id": row['workflow_id'],
                "audit_type": row['audit_type'],
                "audit_scope": row['audit_scope'],
                "compliance_standards": row['compliance_standards'],
                "compliance_score": row['compliance_score'],
                "compliance_status": row['compliance_status'],
                "step_results": row['step_results'],
                "parties": row['parties'],
                "data_purpose": row['data_purpose'],
                "findings": row['findings'],
                "recommendations": row['recommendations'],
                "metadata": row['metadata'],
                "created_at": row['created_at'].isoformat(),
                "updated_at": row['updated_at'].isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审计记录失败: {str(e)}")

@app.post("/audit/compliance/check", response_model=ComplianceCheckResponse, summary="执行合规性检查")
async def check_compliance(request: ComplianceCheckRequest):
    """执行详细的合规性检查"""
    try:
        # 获取审计记录
        async with db_pool.acquire() as conn:
            audit_row = await conn.fetchrow(
                "SELECT * FROM audit_records WHERE audit_id = $1",
                request.audit_id
            )
            
            if not audit_row:
                raise HTTPException(status_code=404, detail="审计记录不存在")
        
        standard_scores = {}
        all_violations = []
        all_recommendations = []
        
        # 对每个标准进行详细检查
        for standard in request.standards:
            score, violations, recommendations = await check_standard_compliance(
                standard, audit_row, request.detailed_check
            )
            
            standard_scores[standard.value] = score
            all_violations.extend(violations)
            all_recommendations.extend(recommendations)
            
            # 保存检查结果
            next_review = datetime.utcnow() + timedelta(days=90)  # 90天后复查
            
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_checks (
                        audit_id, standard, score, status, violations, recommendations, next_review_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (audit_id, standard)
                    DO UPDATE SET score = $3, status = $4, violations = $5, 
                                  recommendations = $6, checked_at = NOW(), next_review_date = $7
                """, 
                    request.audit_id, standard.value, score,
                    "pass" if score >= 0.8 else "fail",
                    json.dumps(violations), recommendations, next_review
                )
        
        # 计算总体评分
        overall_score = sum(standard_scores.values()) / len(standard_scores) if standard_scores else 0.0
        compliance_status = await determine_compliance_status(overall_score)
        
        return ComplianceCheckResponse(
            audit_id=request.audit_id,
            overall_score=overall_score,
            standard_scores=standard_scores,
            compliance_status=compliance_status,
            violations=all_violations,
            recommendations=list(set(all_recommendations)),
            next_review_date=datetime.utcnow() + timedelta(days=90)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合规性检查失败: {str(e)}")

async def check_standard_compliance(standard: ComplianceStandard, audit_row, detailed: bool):
    """检查特定标准的合规性"""
    violations = []
    recommendations = []
    score = 1.0
    
    step_results = audit_row['step_results']
    
    if standard == ComplianceStandard.GDPR:
        # GDPR合规性检查
        if "consent" not in step_results:
            violations.append({
                "rule": "Article 6 - Lawfulness of processing",
                "description": "缺少合法的数据处理依据",
                "severity": "high"
            })
            score -= 0.3
            recommendations.append("实施明确的用户同意机制")
        
        if "explanation" not in step_results:
            violations.append({
                "rule": "Article 22 - Automated decision-making",
                "description": "自动化决策缺少解释性",
                "severity": "medium"
            })
            score -= 0.2
            recommendations.append("提供自动化决策的解释和申诉机制")
    
    elif standard == ComplianceStandard.PCI_DSS:
        # PCI DSS合规性检查
        if "encryption" not in str(step_results).lower():
            violations.append({
                "rule": "Requirement 3 - Protect stored cardholder data",
                "description": "数据传输和存储缺少加密保护",
                "severity": "high"
            })
            score -= 0.4
            recommendations.append("实施端到端数据加密")
        
        if "audit" not in step_results:
            violations.append({
                "rule": "Requirement 10 - Log and monitor all access",
                "description": "缺少完整的访问日志记录",
                "severity": "medium"
            })
            score -= 0.2
            recommendations.append("建立完整的审计日志系统")
    
    elif standard == ComplianceStandard.CCPA:
        # CCPA合规性检查
        if "consent" not in step_results:
            violations.append({
                "rule": "Section 1798.100 - Consumer Rights",
                "description": "缺少消费者数据权利保护",
                "severity": "high"
            })
            score -= 0.3
            recommendations.append("实施消费者数据权利管理")
    
    return max(0.0, score), violations, recommendations

async def generate_compliance_report(audit_id: str, format: ReportFormat):
    """生成合规报告"""
    try:
        # 获取审计数据
        async with db_pool.acquire() as conn:
            audit_row = await conn.fetchrow(
                "SELECT * FROM audit_records WHERE audit_id = $1",
                audit_id
            )
            
            if not audit_row:
                logger.error(f"审计记录不存在: {audit_id}")
                return
            
            # 获取合规检查结果
            compliance_rows = await conn.fetch(
                "SELECT * FROM compliance_checks WHERE audit_id = $1",
                audit_id
            )
        
        # 准备报告数据
        report_data = {
            "audit_id": audit_id,
            "workflow_id": audit_row['workflow_id'],
            "audit_type": audit_row['audit_type'],
            "audit_scope": audit_row['audit_scope'],
            "compliance_score": audit_row['compliance_score'],
            "compliance_status": audit_row['compliance_status'],
            "parties": audit_row['parties'],
            "data_purpose": audit_row['data_purpose'],
            "findings": audit_row['findings'],
            "recommendations": audit_row['recommendations'],
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "standard_scores": {row['standard']: row['score'] for row in compliance_rows},
            "violations": []
        }
        
        # 收集所有违规
        for row in compliance_rows:
            if row['violations']:
                for violation in row['violations']:
                    violation['standard'] = row['standard']
                    report_data['violations'].append(violation)
        
        # 生成报告文件
        report_id = f"report_{uuid.uuid4().hex[:16]}"
        
        if format == ReportFormat.PDF:
            file_path = await generate_pdf_report(report_id, report_data)
        elif format == ReportFormat.HTML:
            file_path = await generate_html_report(report_id, report_data)
        elif format == ReportFormat.JSON:
            file_path = await generate_json_report(report_id, report_data)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
        
        # 保存报告记录
        file_size = Path(file_path).stat().st_size
        expires_at = datetime.utcnow() + timedelta(days=30)  # 30天后过期
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_reports (
                    report_id, audit_id, report_type, format, file_path, file_size, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                report_id, audit_id, "compliance_report", format.value, 
                file_path, file_size, expires_at
            )
        
        # 更新指标
        reports_generated_total.labels(
            report_type="compliance_report", 
            format=format.value
        ).inc()
        
        logger.info(f"合规报告生成成功: {report_id}")
        
    except Exception as e:
        logger.error(f"生成合规报告失败 {audit_id}: {e}")

async def generate_pdf_report(report_id: str, data: Dict[str, Any]) -> str:
    """生成PDF报告"""
    file_path = Path(REPORT_STORAGE_PATH) / f"{report_id}.pdf"
    
    # 创建PDF文档
    doc = SimpleDocTemplate(str(file_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # 标题
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # 居中
    )
    story.append(Paragraph("联邦风控合规审计报告", title_style))
    story.append(Spacer(1, 12))
    
    # 基本信息
    info_data = [
        ["审计ID", data['audit_id']],
        ["工作流ID", data['workflow_id']],
        ["审计类型", data['audit_type']],
        ["合规评分", f"{data['compliance_score']:.2%}"],
        ["合规状态", data['compliance_status']],
        ["生成时间", data['generated_at']]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # 标准评分
    if data['standard_scores']:
        story.append(Paragraph("标准评分详情", styles['Heading2']))
        
        score_data = [["合规标准", "评分", "状态"]]
        for standard, score in data['standard_scores'].items():
            status = "PASS" if score >= 0.8 else "FAIL"
            score_data.append([standard.upper(), f"{score:.2%}", status])
        
        score_table = Table(score_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(score_table)
        story.append(Spacer(1, 20))
    
    # 违规情况
    if data['violations']:
        story.append(Paragraph("合规性违规", styles['Heading2']))
        
        for violation in data['violations']:
            story.append(Paragraph(
                f"<b>{violation['standard']}</b>: {violation['description']}",
                styles['Normal']
            ))
            story.append(Paragraph(
                f"严重程度: {violation['severity']}",
                styles['Normal']
            ))
            story.append(Spacer(1, 10))
    
    # 改进建议
    if data['recommendations']:
        story.append(Paragraph("改进建议", styles['Heading2']))
        
        for i, recommendation in enumerate(data['recommendations'], 1):
            story.append(Paragraph(
                f"{i}. {recommendation}",
                styles['Normal']
            ))
            story.append(Spacer(1, 8))
    
    # 构建PDF
    doc.build(story)
    
    return str(file_path)

async def generate_html_report(report_id: str, data: Dict[str, Any]) -> str:
    """生成HTML报告"""
    file_path = Path(REPORT_STORAGE_PATH) / f"{report_id}.html"
    
    # 使用Jinja2模板
    template = jinja_env.get_template("compliance_report.html")
    html_content = template.render(**data)
    
    file_path.write_text(html_content, encoding='utf-8')
    
    return str(file_path)

async def generate_json_report(report_id: str, data: Dict[str, Any]) -> str:
    """生成JSON报告"""
    file_path = Path(REPORT_STORAGE_PATH) / f"{report_id}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return str(file_path)

@app.post("/audit/reports/generate", response_model=ReportResponse, summary="生成审计报告")
async def generate_report(request: ReportGenerationRequest, background_tasks: BackgroundTasks):
    """生成指定格式的审计报告"""
    try:
        # 检查审计记录是否存在
        async with db_pool.acquire() as conn:
            audit_row = await conn.fetchrow(
                "SELECT audit_id FROM audit_records WHERE audit_id = $1",
                request.audit_id
            )
            
            if not audit_row:
                raise HTTPException(status_code=404, detail="审计记录不存在")
        
        # 后台生成报告
        background_tasks.add_task(
            generate_compliance_report, request.audit_id, request.format
        )
        
        # 生成报告ID
        report_id = f"report_{uuid.uuid4().hex[:16]}"
        
        return ReportResponse(
            report_id=report_id,
            audit_id=request.audit_id,
            report_type=request.report_type,
            format=request.format,
            file_path="",  # 后台生成，暂时为空
            download_url=f"/audit/reports/{request.audit_id}/download",
            generated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")

@app.get("/audit/reports/{audit_id}/download", summary="下载审计报告")
async def download_report(audit_id: str, format: str = "pdf"):
    """下载审计报告文件"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM audit_reports 
                WHERE audit_id = $1 AND format = $2 
                ORDER BY generated_at DESC LIMIT 1
            """, audit_id, format)
            
            if not row:
                raise HTTPException(status_code=404, detail="报告文件不存在")
            
            if datetime.utcnow() > row['expires_at']:
                raise HTTPException(status_code=410, detail="报告文件已过期")
            
            file_path = Path(row['file_path'])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="报告文件不存在")
            
            # 更新下载次数
            await conn.execute(
                "UPDATE audit_reports SET download_count = download_count + 1 WHERE report_id = $1",
                row['report_id']
            )
            
            # 返回文件
            media_type = {
                "pdf": "application/pdf",
                "html": "text/html",
                "json": "application/json"
            }.get(format, "application/octet-stream")
            
            return FileResponse(
                path=str(file_path),
                media_type=media_type,
                filename=f"audit_report_{audit_id}.{format}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载报告失败: {str(e)}")

@app.post("/audit/lineage/events", summary="记录数据血缘事件")
async def record_lineage_event(request: LineageEventRequest):
    """记录数据血缘事件"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO data_lineage (
                    workflow_id, event_type, source_entity, target_entity, 
                    transformation_details, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                request.workflow_id, request.event_type.value, request.source_entity,
                request.target_entity, json.dumps(request.transformation_details),
                json.dumps(request.metadata)
            )
        
        # 更新指标
        data_lineage_events_total.labels(event_type=request.event_type.value).inc()
        
        return {
            "status": "success",
            "message": "数据血缘事件记录成功",
            "event_id": f"lineage_{uuid.uuid4().hex[:16]}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录数据血缘事件失败: {str(e)}")

@app.get("/audit/lineage/{workflow_id}", response_model=LineageResponse, summary="获取数据血缘")
async def get_data_lineage(workflow_id: str):
    """获取工作流的数据血缘图"""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM data_lineage WHERE workflow_id = $1 ORDER BY created_at",
                workflow_id
            )
            
            if not rows:
                raise HTTPException(status_code=404, detail="数据血缘记录不存在")
        
        # 构建血缘图
        entities = set()
        relationships = []
        
        for row in rows:
            entities.add(row['source_entity'])
            entities.add(row['target_entity'])
            
            relationships.append({
                "source": row['source_entity'],
                "target": row['target_entity'],
                "event_type": row['event_type'],
                "transformation_details": row['transformation_details'],
                "created_at": row['created_at'].isoformat()
            })
        
        # 构建图结构
        lineage_graph = {
            "nodes": [{
                "id": entity,
                "label": entity,
                "type": "data_entity"
            } for entity in entities],
            "edges": [{
                "source": rel['source'],
                "target": rel['target'],
                "label": rel['event_type'],
                "metadata": rel['transformation_details']
            } for rel in relationships]
        }
        
        # 影响分析
        impact_analysis = {
            "total_entities": len(entities),
            "total_transformations": len(relationships),
            "data_flow_complexity": len(relationships) / len(entities) if entities else 0,
            "critical_paths": []  # 可以进一步分析关键路径
        }
        
        return LineageResponse(
            workflow_id=workflow_id,
            lineage_graph=lineage_graph,
            entities=[{"id": e, "type": "data_entity"} for e in entities],
            relationships=relationships,
            impact_analysis=impact_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据血缘失败: {str(e)}")

@app.post("/audit/privacy/assessment", summary="隐私影响评估")
async def privacy_impact_assessment(request: PrivacyImpactAssessmentRequest):
    """执行隐私影响评估"""
    try:
        assessment_id = f"pia_{uuid.uuid4().hex[:16]}"
        
        # 计算隐私风险评分
        risk_score = await calculate_privacy_risk(
            request.data_types, request.processing_purposes, 
            request.data_subjects, request.risk_factors
        )
        
        # 确定风险等级
        if risk_score >= 0.8:
            risk_level = "high"
        elif risk_score >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # 生成缓解措施
        mitigation_measures = await generate_mitigation_measures(
            risk_score, request.data_types, request.processing_purposes
        )
        
        # 保存评估结果
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO privacy_assessments (
                    assessment_id, workflow_id, data_types, processing_purposes,
                    data_subjects, risk_score, risk_level, mitigation_measures
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 
                assessment_id, request.workflow_id, request.data_types,
                request.processing_purposes, request.data_subjects,
                risk_score, risk_level, json.dumps(mitigation_measures)
            )
        
        return {
            "assessment_id": assessment_id,
            "workflow_id": request.workflow_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "mitigation_measures": mitigation_measures,
            "recommendations": await generate_privacy_recommendations(risk_level),
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"隐私影响评估失败: {str(e)}")

async def calculate_privacy_risk(data_types: List[str], purposes: List[str], 
                               subjects: List[str], risk_factors: Dict[str, Any]) -> float:
    """计算隐私风险评分"""
    base_risk = 0.0
    
    # 数据类型风险
    sensitive_data_types = ["personal_id", "financial", "health", "biometric", "location"]
    for data_type in data_types:
        if data_type.lower() in sensitive_data_types:
            base_risk += 0.2
    
    # 处理目的风险
    high_risk_purposes = ["profiling", "automated_decision", "marketing", "sharing"]
    for purpose in purposes:
        if purpose.lower() in high_risk_purposes:
            base_risk += 0.15
    
    # 数据主体风险
    vulnerable_subjects = ["children", "elderly", "patients", "employees"]
    for subject in subjects:
        if subject.lower() in vulnerable_subjects:
            base_risk += 0.1
    
    # 额外风险因素
    if risk_factors.get("cross_border_transfer", False):
        base_risk += 0.2
    if risk_factors.get("third_party_sharing", False):
        base_risk += 0.15
    if risk_factors.get("automated_processing", False):
        base_risk += 0.1
    
    return min(1.0, base_risk)

async def generate_mitigation_measures(risk_score: float, data_types: List[str], 
                                     purposes: List[str]) -> Dict[str, Any]:
    """生成隐私风险缓解措施"""
    measures = {
        "technical_measures": [],
        "organizational_measures": [],
        "legal_measures": []
    }
    
    if risk_score >= 0.8:
        measures["technical_measures"].extend([
            "实施端到端加密",
            "使用差分隐私技术",
            "实施数据匿名化",
            "建立访问控制机制"
        ])
        measures["organizational_measures"].extend([
            "指定数据保护官",
            "建立隐私管理体系",
            "定期隐私培训",
            "建立事件响应机制"
        ])
        measures["legal_measures"].extend([
            "更新隐私政策",
            "建立合规监控",
            "实施数据保护影响评估"
        ])
    elif risk_score >= 0.5:
        measures["technical_measures"].extend([
            "数据传输加密",
            "访问日志记录",
            "数据备份保护"
        ])
        measures["organizational_measures"].extend([
            "隐私意识培训",
            "定期安全审计"
        ])
        measures["legal_measures"].extend([
            "隐私政策更新",
            "合规性检查"
        ])
    
    return measures

async def generate_privacy_recommendations(risk_level: str) -> List[str]:
    """生成隐私保护建议"""
    if risk_level == "high":
        return [
            "立即实施全面的隐私保护措施",
            "考虑进行正式的数据保护影响评估",
            "建立专门的隐私管理团队",
            "定期审查和更新隐私政策",
            "实施持续的隐私监控"
        ]
    elif risk_level == "medium":
        return [
            "加强现有的隐私保护措施",
            "定期进行隐私风险评估",
            "提供隐私培训",
            "建立隐私事件响应流程"
        ]
    else:
        return [
            "维持当前的隐私保护水平",
            "定期监控隐私合规状态",
            "保持隐私政策的时效性"
        ]

@app.get("/audit/workflows/{workflow_id}/summary", summary="获取工作流审计摘要")
async def get_workflow_audit_summary(workflow_id: str):
    """获取工作流的审计摘要"""
    try:
        async with db_pool.acquire() as conn:
            # 获取审计记录
            audit_rows = await conn.fetch(
                "SELECT * FROM audit_records WHERE workflow_id = $1 ORDER BY created_at DESC",
                workflow_id
            )
            
            if not audit_rows:
                raise HTTPException(status_code=404, detail="工作流审计记录不存在")
            
            # 获取合规检查
            compliance_rows = await conn.fetch("""
                SELECT cc.* FROM compliance_checks cc
                JOIN audit_records ar ON cc.audit_id = ar.audit_id
                WHERE ar.workflow_id = $1
            """, workflow_id)
            
            # 获取隐私评估
            privacy_rows = await conn.fetch(
                "SELECT * FROM privacy_assessments WHERE workflow_id = $1",
                workflow_id
            )
            
            # 获取数据血缘
            lineage_rows = await conn.fetch(
                "SELECT * FROM data_lineage WHERE workflow_id = $1",
                workflow_id
            )
        
        # 构建摘要
        latest_audit = audit_rows[0] if audit_rows else None
        
        summary = {
            "workflow_id": workflow_id,
            "total_audits": len(audit_rows),
            "latest_audit": {
                "audit_id": latest_audit['audit_id'],
                "compliance_score": latest_audit['compliance_score'],
                "compliance_status": latest_audit['compliance_status'],
                "created_at": latest_audit['created_at'].isoformat()
            } if latest_audit else None,
            "compliance_history": [
                {
                    "audit_id": row['audit_id'],
                    "score": row['compliance_score'],
                    "status": row['compliance_status'],
                    "date": row['created_at'].isoformat()
                }
                for row in audit_rows[:10]  # 最近10次
            ],
            "compliance_standards": list(set(
                standard for row in compliance_rows 
                for standard in [row['standard']]
            )),
            "privacy_assessments": len(privacy_rows),
            "data_lineage_events": len(lineage_rows),
            "risk_indicators": {
                "low_compliance_audits": len([r for r in audit_rows if r['compliance_score'] < 0.7]),
                "high_risk_privacy_assessments": len([r for r in privacy_rows if r['risk_level'] == 'high']),
                "recent_violations": len([r for r in compliance_rows if r['status'] == 'fail'])
            }
        }
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取审计摘要失败: {str(e)}")

@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """服务健康检查"""
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
        
        # 获取活跃审计数量
        active_audits = 0
        try:
            async with db_pool.acquire() as conn:
                active_audits = await conn.fetchval(
                    "SELECT COUNT(*) FROM audit_records WHERE created_at > NOW() - INTERVAL '24 hours'"
                )
        except Exception:
            pass
        
        overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status=db_status,
            redis_status=redis_status,
            active_audits=active_audits or 0
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_status="unknown",
            redis_status="unknown",
            active_audits=0
        )

@app.get("/metrics", summary="获取Prometheus指标")
async def get_metrics():
    """获取Prometheus监控指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("审计服务启动中...")
    
    try:
        await init_database()
        await init_redis()
        await init_templates()
        
        logger.info("审计服务启动成功")
    except Exception as e:
        logger.error(f"审计服务启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("审计服务关闭中...")
    
    try:
        if db_pool:
            await db_pool.close()
        if redis_client:
            await redis_client.close()
        
        logger.info("审计服务关闭完成")
    except Exception as e:
        logger.error(f"审计服务关闭失败: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info"
    )