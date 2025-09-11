from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
import jwt
import hashlib
import time
import uuid
from datetime import datetime, timedelta
import logging
from passlib.context import CryptContext
import casbin

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="同意管理服务",
    description="用户同意和权限管理服务",
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

# 安全配置
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# 数据模型
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str
    is_active: bool = True

class ConsentRequest(BaseModel):
    user_id: str
    purpose: str
    data_types: List[str]
    retention_period: int  # 天数
    third_parties: List[str] = []
    description: str

class ConsentResponse(BaseModel):
    id: str
    user_id: str
    purpose: str
    data_types: List[str]
    retention_period: int
    third_parties: List[str]
    description: str
    status: str  # pending, approved, rejected, revoked
    created_at: str
    updated_at: str

class ConsentPolicy(BaseModel):
    id: str
    name: str
    description: str
    required_data_types: List[str]
    max_retention_period: int
    allowed_purposes: List[str]
    created_at: str
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float

# 全局状态
start_time = time.time()
users_db: Dict[str, User] = {}
consent_requests_db: Dict[str, ConsentResponse] = {}
consent_policies_db: Dict[str, ConsentPolicy] = {}
user_credentials: Dict[str, str] = {}  # username -> hashed_password

# 认证相关函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
        "total_users": len(users_db),
        "total_consent_requests": len(consent_requests_db),
        "total_policies": len(consent_policies_db),
        "uptime": time.time() - start_time
    }

@app.post("/auth/register", response_model=User)
async def register_user(user_data: UserCreate):
    """用户注册"""
    # 检查用户名是否已存在
    if user_data.username in user_credentials:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 创建用户
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    user = User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        role=user_data.role,
        created_at=datetime.now().isoformat()
    )
    
    users_db[user_id] = user
    user_credentials[user_data.username] = hashed_password
    
    logger.info(f"用户注册成功: {user_data.username}")
    return user

@app.post("/auth/login", response_model=Token)
async def login_user(login_data: UserLogin):
    """用户登录"""
    # 验证用户凭据
    if login_data.username not in user_credentials:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    if not verify_password(login_data.password, user_credentials[login_data.username]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": login_data.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"用户登录成功: {login_data.username}")
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/auth/me", response_model=User)
async def get_current_user(current_user: str = Depends(verify_token)):
    """获取当前用户信息"""
    for user in users_db.values():
        if user.username == current_user:
            return user
    raise HTTPException(status_code=404, detail="用户不存在")

@app.post("/consent/request", response_model=ConsentResponse)
async def create_consent_request(request: ConsentRequest, current_user: str = Depends(verify_token)):
    """创建同意请求"""
    request_id = str(uuid.uuid4())
    
    consent_response = ConsentResponse(
        id=request_id,
        user_id=request.user_id,
        purpose=request.purpose,
        data_types=request.data_types,
        retention_period=request.retention_period,
        third_parties=request.third_parties,
        description=request.description,
        status="pending",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    consent_requests_db[request_id] = consent_response
    
    logger.info(f"同意请求创建成功: {request_id}")
    return consent_response

@app.get("/consent/requests", response_model=List[ConsentResponse])
async def list_consent_requests(user_id: Optional[str] = None, current_user: str = Depends(verify_token)):
    """列出同意请求"""
    requests = list(consent_requests_db.values())
    
    if user_id:
        requests = [r for r in requests if r.user_id == user_id]
    
    return requests

@app.get("/consent/requests/{request_id}", response_model=ConsentResponse)
async def get_consent_request(request_id: str, current_user: str = Depends(verify_token)):
    """获取同意请求详情"""
    if request_id not in consent_requests_db:
        raise HTTPException(status_code=404, detail="同意请求不存在")
    
    return consent_requests_db[request_id]

@app.post("/consent/requests/{request_id}/approve")
async def approve_consent_request(request_id: str, current_user: str = Depends(verify_token)):
    """批准同意请求"""
    if request_id not in consent_requests_db:
        raise HTTPException(status_code=404, detail="同意请求不存在")
    
    consent_requests_db[request_id].status = "approved"
    consent_requests_db[request_id].updated_at = datetime.now().isoformat()
    
    logger.info(f"同意请求已批准: {request_id}")
    return {"message": "同意请求已批准"}

@app.post("/consent/requests/{request_id}/reject")
async def reject_consent_request(request_id: str, current_user: str = Depends(verify_token)):
    """拒绝同意请求"""
    if request_id not in consent_requests_db:
        raise HTTPException(status_code=404, detail="同意请求不存在")
    
    consent_requests_db[request_id].status = "rejected"
    consent_requests_db[request_id].updated_at = datetime.now().isoformat()
    
    logger.info(f"同意请求已拒绝: {request_id}")
    return {"message": "同意请求已拒绝"}

@app.post("/consent/requests/{request_id}/revoke")
async def revoke_consent_request(request_id: str, current_user: str = Depends(verify_token)):
    """撤销同意请求"""
    if request_id not in consent_requests_db:
        raise HTTPException(status_code=404, detail="同意请求不存在")
    
    consent_requests_db[request_id].status = "revoked"
    consent_requests_db[request_id].updated_at = datetime.now().isoformat()
    
    logger.info(f"同意请求已撤销: {request_id}")
    return {"message": "同意请求已撤销"}

@app.post("/policies", response_model=ConsentPolicy)
async def create_consent_policy(policy_data: dict, current_user: str = Depends(verify_token)):
    """创建同意政策"""
    policy_id = str(uuid.uuid4())
    
    policy = ConsentPolicy(
        id=policy_id,
        name=policy_data["name"],
        description=policy_data["description"],
        required_data_types=policy_data["required_data_types"],
        max_retention_period=policy_data["max_retention_period"],
        allowed_purposes=policy_data["allowed_purposes"],
        created_at=datetime.now().isoformat()
    )
    
    consent_policies_db[policy_id] = policy
    
    logger.info(f"同意政策创建成功: {policy_id}")
    return policy

@app.get("/policies", response_model=List[ConsentPolicy])
async def list_consent_policies(current_user: str = Depends(verify_token)):
    """列出同意政策"""
    return list(consent_policies_db.values())

@app.get("/policies/{policy_id}", response_model=ConsentPolicy)
async def get_consent_policy(policy_id: str, current_user: str = Depends(verify_token)):
    """获取同意政策详情"""
    if policy_id not in consent_policies_db:
        raise HTTPException(status_code=404, detail="同意政策不存在")
    
    return consent_policies_db[policy_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)