import requests
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
import hashlib
import time

logger = logging.getLogger(__name__)

class ConsentClient:
    """同意管理服务客户端"""
    
    def __init__(self, base_url: str, timeout: int = 10, retry_count: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_count = retry_count
        self.session = requests.Session()
        
        # 设置默认headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'FederatedServing/1.0'
        })
    
    def verify_consent(self, consent_jwt: str, subject: str, 
                      purpose: str = "credit_scoring", 
                      required_features: Optional[List[str]] = None) -> Dict:
        """验证同意票据
        
        Args:
            consent_jwt: JWT同意票据
            subject: 用户标识
            purpose: 使用目的
            required_features: 需要的特征列表
        
        Returns:
            Dict: 验证结果
                - valid: bool, 是否有效
                - consent_info: Dict, 同意信息
                - policy_version: str, 策略版本
                - error: str, 错误信息（如果有）
        """
        try:
            payload = {
                "jwt_token": consent_jwt,
                "subject": subject,
                "purpose": purpose
            }
            
            if required_features:
                payload["required_features"] = required_features
            
            response = self._make_request(
                "POST", 
                "/consent/verify", 
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Consent verified successfully for subject: {subject}")
                return {
                    "valid": True,
                    "consent_info": result.get("consent_info", {}),
                    "policy_version": result.get("policy_version", "v1.0"),
                    "allowed_features": result.get("allowed_features", []),
                    "expires_at": result.get("expires_at"),
                    "fingerprint": self._generate_consent_fingerprint(consent_jwt)
                }
            elif response.status_code == 403:
                error_detail = response.json().get("detail", "Consent verification failed")
                logger.warning(f"Consent verification failed for {subject}: {error_detail}")
                return {
                    "valid": False,
                    "error": error_detail,
                    "error_code": "CONSENT_DENIED"
                }
            elif response.status_code == 401:
                logger.warning(f"Invalid consent token for {subject}")
                return {
                    "valid": False,
                    "error": "Invalid or expired consent token",
                    "error_code": "INVALID_TOKEN"
                }
            else:
                error_msg = f"Unexpected response: {response.status_code}"
                logger.error(f"Consent verification error for {subject}: {error_msg}")
                return {
                    "valid": False,
                    "error": error_msg,
                    "error_code": "SERVICE_ERROR"
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Consent verification timeout for {subject}")
            return {
                "valid": False,
                "error": "Consent service timeout",
                "error_code": "TIMEOUT"
            }
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to consent service for {subject}")
            return {
                "valid": False,
                "error": "Cannot connect to consent service",
                "error_code": "CONNECTION_ERROR"
            }
        except Exception as e:
            logger.error(f"Consent verification failed for {subject}: {e}")
            return {
                "valid": False,
                "error": f"Verification error: {str(e)}",
                "error_code": "UNKNOWN_ERROR"
            }
    
    def record_audit(self, audit_record: Dict) -> bool:
        """记录审计日志
        
        Args:
            audit_record: 审计记录
                - request_id: str, 请求ID
                - subject: str, 用户标识
                - consent_fingerprint: str, 同意票据指纹
                - model_hash: str, 模型哈希
                - threshold: float, 决策阈值
                - policy_version: str, 策略版本
                - timestamp: str, 时间戳
                - decision: str, 决策结果
                - score: float, 评分
                - features_used: List[str], 使用的特征
        
        Returns:
            bool: 记录是否成功
        """
        try:
            # 验证必需字段
            required_fields = [
                "request_id", "subject", "consent_fingerprint", 
                "model_hash", "decision", "score"
            ]
            
            for field in required_fields:
                if field not in audit_record:
                    logger.error(f"Missing required audit field: {field}")
                    return False
            
            # 添加默认值
            audit_record.setdefault("timestamp", datetime.now().isoformat())
            audit_record.setdefault("policy_version", "v1.0")
            audit_record.setdefault("threshold", 0.5)
            audit_record.setdefault("features_used", [])
            
            response = self._make_request(
                "POST", 
                "/audit/record", 
                json=audit_record
            )
            
            if response.status_code == 200:
                logger.info(f"Audit recorded successfully: {audit_record['request_id']}")
                return True
            else:
                logger.error(f"Failed to record audit: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Audit recording failed: {e}")
            return False
    
    def query_audit(self, request_id: Optional[str] = None, 
                   subject: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 100) -> Optional[List[Dict]]:
        """查询审计记录
        
        Args:
            request_id: 请求ID
            subject: 用户标识
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
        
        Returns:
            List[Dict]: 审计记录列表，失败时返回None
        """
        try:
            params = {"limit": limit}
            
            if request_id:
                params["request_id"] = request_id
            if subject:
                params["subject"] = subject
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            
            response = self._make_request(
                "GET", 
                "/audit/query", 
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("records", [])
            else:
                logger.error(f"Failed to query audit: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Audit query failed: {e}")
            return None
    
    def get_consent_policies(self) -> Optional[Dict]:
        """获取同意策略
        
        Returns:
            Dict: 策略信息，失败时返回None
        """
        try:
            response = self._make_request("GET", "/consent/policies")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get policies: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Get policies failed: {e}")
            return None
    
    def revoke_consent(self, subject: str, reason: str = "user_request") -> bool:
        """撤销同意
        
        Args:
            subject: 用户标识
            reason: 撤销原因
        
        Returns:
            bool: 撤销是否成功
        """
        try:
            payload = {
                "subject": subject,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self._make_request(
                "POST", 
                "/consent/revoke", 
                json=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Consent revoked successfully for {subject}")
                return True
            else:
                logger.error(f"Failed to revoke consent: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Consent revocation failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool: 服务是否健康
        """
        try:
            response = self._make_request("GET", "/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _make_request(self, method: str, endpoint: str, 
                     timeout: Optional[int] = None, 
                     retry_count: Optional[int] = None,
                     **kwargs) -> requests.Response:
        """发起HTTP请求（带重试）
        
        Args:
            method: HTTP方法
            endpoint: API端点
            timeout: 超时时间
            retry_count: 重试次数
            **kwargs: 其他请求参数
        
        Returns:
            requests.Response: 响应对象
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout
        retry_count = retry_count or self.retry_count
        
        last_exception = None
        
        for attempt in range(retry_count + 1):
            try:
                response = self.session.request(
                    method, url, timeout=timeout, **kwargs
                )
                return response
                
            except (requests.exceptions.Timeout, 
                   requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < retry_count:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {retry_count + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected request error: {e}")
                raise
        
        # 这行代码理论上不会执行到
        raise last_exception
    
    def _generate_consent_fingerprint(self, consent_jwt: str) -> str:
        """生成同意票据指纹
        
        Args:
            consent_jwt: JWT同意票据
        
        Returns:
            str: 指纹字符串
        """
        return hashlib.sha256(consent_jwt.encode()).hexdigest()[:16]
    
    def get_service_info(self) -> Dict:
        """获取服务信息
        
        Returns:
            Dict: 服务信息
        """
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "is_healthy": self.health_check()
        }

class ConsentCache:
    """同意验证结果缓存"""
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache = {}
        self.cache_ttl = cache_ttl_seconds
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """获取缓存的验证结果
        
        Args:
            cache_key: 缓存键
        
        Returns:
            Dict: 缓存的结果，如果过期或不存在则返回None
        """
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        
        # 检查是否过期
        if time.time() - cached_item["timestamp"] > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        return cached_item["result"]
    
    def set(self, cache_key: str, result: Dict):
        """设置缓存
        
        Args:
            cache_key: 缓存键
            result: 验证结果
        """
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def generate_cache_key(self, consent_jwt: str, subject: str, purpose: str) -> str:
        """生成缓存键
        
        Args:
            consent_jwt: JWT同意票据
            subject: 用户标识
            purpose: 使用目的
        
        Returns:
            str: 缓存键
        """
        key_data = f"{consent_jwt}:{subject}:{purpose}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def clear_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict:
        """获取缓存统计
        
        Returns:
            Dict: 缓存统计信息
        """
        return {
            "total_entries": len(self.cache),
            "cache_ttl_seconds": self.cache_ttl
        }

class CachedConsentClient(ConsentClient):
    """带缓存的同意客户端"""
    
    def __init__(self, base_url: str, timeout: int = 10, retry_count: int = 3,
                 cache_ttl_seconds: int = 300):
        super().__init__(base_url, timeout, retry_count)
        self.cache = ConsentCache(cache_ttl_seconds)
    
    def verify_consent(self, consent_jwt: str, subject: str, 
                      purpose: str = "credit_scoring", 
                      required_features: Optional[List[str]] = None,
                      use_cache: bool = True) -> Dict:
        """验证同意票据（带缓存）
        
        Args:
            consent_jwt: JWT同意票据
            subject: 用户标识
            purpose: 使用目的
            required_features: 需要的特征列表
            use_cache: 是否使用缓存
        
        Returns:
            Dict: 验证结果
        """
        # 生成缓存键
        cache_key = self.cache.generate_cache_key(consent_jwt, subject, purpose)
        
        # 尝试从缓存获取
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Using cached consent verification for {subject}")
                return cached_result
        
        # 调用父类方法
        result = super().verify_consent(consent_jwt, subject, purpose, required_features)
        
        # 只缓存成功的验证结果
        if use_cache and result.get("valid", False):
            self.cache.set(cache_key, result)
        
        return result
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.cache.clear()
        logger.info("Consent verification cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计
        
        Returns:
            Dict: 缓存统计信息
        """
        return self.cache.get_stats()