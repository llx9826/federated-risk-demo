import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib

logger = logging.getLogger(__name__)

class FeatureView:
    """特征视图管理器 - 负责特征存储、检索和一致性保证"""
    
    def __init__(self, data_dir: str = "/app/data", cache_ttl_hours: int = 24):
        self.data_dir = data_dir
        self.cache_ttl_hours = cache_ttl_hours
        self.features_cache = {}
        self.feature_schemas = {}
        self.last_refresh = None
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化加载
        self.refresh_features()
    
    def refresh_features(self, force: bool = False) -> bool:
        """刷新特征缓存
        
        Args:
            force: 是否强制刷新
        
        Returns:
            bool: 刷新是否成功
        """
        try:
            # 检查是否需要刷新
            if not force and self.last_refresh:
                time_diff = datetime.now() - self.last_refresh
                if time_diff.total_seconds() < self.cache_ttl_hours * 3600:
                    return True
            
            logger.info("Refreshing feature cache...")
            
            # 清空缓存
            self.features_cache.clear()
            self.feature_schemas.clear()
            
            # 加载银行特征
            self._load_bank_features()
            
            # 加载电商特征
            self._load_ecom_features()
            
            # 合并特征
            self._merge_features()
            
            # 生成特征schema
            self._generate_feature_schemas()
            
            self.last_refresh = datetime.now()
            
            logger.info(f"Feature cache refreshed: {len(self.features_cache)} records loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh features: {e}")
            return False
    
    def _load_bank_features(self):
        """加载银行方特征"""
        bank_file = os.path.join(self.data_dir, "partyA_bank.csv")
        if not os.path.exists(bank_file):
            logger.warning(f"Bank features file not found: {bank_file}")
            return
        
        try:
            df = pd.read_csv(bank_file)
            logger.info(f"Loading {len(df)} bank feature records")
            
            for _, row in df.iterrows():
                psi_token = row["psi_token"]
                
                # 提取银行特征
                bank_features = {
                    "age": self._safe_convert(row.get("age"), int, 30),
                    "income": self._safe_convert(row.get("income"), float, 50000.0),
                    "credit_history": self._safe_convert(row.get("credit_history"), int, 5),
                    "loan_amount": self._safe_convert(row.get("loan_amount"), float, 10000.0),
                    "employment_years": self._safe_convert(row.get("employment_years"), int, 3),
                    "debt_to_income": self._safe_convert(row.get("debt_to_income"), float, 0.3),
                    "credit_score": self._safe_convert(row.get("credit_score"), int, 650),
                    "num_credit_cards": self._safe_convert(row.get("num_credit_cards"), int, 2),
                    "mortgage_balance": self._safe_convert(row.get("mortgage_balance"), float, 0.0),
                    "savings_balance": self._safe_convert(row.get("savings_balance"), float, 5000.0)
                }
                
                # 添加标签（如果存在）
                if "default_label" in row:
                    bank_features["default_label"] = self._safe_convert(row["default_label"], int, 0)
                
                # 初始化特征记录
                self.features_cache[psi_token] = {
                    "psi_token": psi_token,
                    "bank_features": bank_features,
                    "ecom_features": {},
                    "merged_features": {},
                    "last_updated": datetime.now().isoformat(),
                    "data_sources": ["bank"]
                }
                
        except Exception as e:
            logger.error(f"Failed to load bank features: {e}")
    
    def _load_ecom_features(self):
        """加载电商方特征"""
        ecom_file = os.path.join(self.data_dir, "partyB_ecom.csv")
        if not os.path.exists(ecom_file):
            logger.warning(f"Ecom features file not found: {ecom_file}")
            return
        
        try:
            df = pd.read_csv(ecom_file)
            logger.info(f"Loading {len(df)} ecom feature records")
            
            for _, row in df.iterrows():
                psi_token = row["psi_token"]
                
                # 提取电商特征
                ecom_features = {
                    "purchase_frequency": self._safe_convert(row.get("purchase_frequency"), float, 2.0),
                    "avg_order_value": self._safe_convert(row.get("avg_order_value"), float, 100.0),
                    "category_preference": str(row.get("category_preference", "electronics")),
                    "return_rate": self._safe_convert(row.get("return_rate"), float, 0.1),
                    "account_age_days": self._safe_convert(row.get("account_age_days"), int, 365),
                    "total_spent": self._safe_convert(row.get("total_spent"), float, 1000.0),
                    "num_orders": self._safe_convert(row.get("num_orders"), int, 10),
                    "avg_session_duration": self._safe_convert(row.get("avg_session_duration"), float, 15.0),
                    "mobile_usage_ratio": self._safe_convert(row.get("mobile_usage_ratio"), float, 0.6),
                    "review_score_avg": self._safe_convert(row.get("review_score_avg"), float, 4.0)
                }
                
                # 如果已存在记录，更新电商特征
                if psi_token in self.features_cache:
                    self.features_cache[psi_token]["ecom_features"] = ecom_features
                    self.features_cache[psi_token]["data_sources"].append("ecom")
                else:
                    # 创建新记录
                    self.features_cache[psi_token] = {
                        "psi_token": psi_token,
                        "bank_features": {},
                        "ecom_features": ecom_features,
                        "merged_features": {},
                        "last_updated": datetime.now().isoformat(),
                        "data_sources": ["ecom"]
                    }
                
        except Exception as e:
            logger.error(f"Failed to load ecom features: {e}")
    
    def _merge_features(self):
        """合并银行和电商特征"""
        for psi_token, record in self.features_cache.items():
            merged = {}
            
            # 合并银行特征
            merged.update(record["bank_features"])
            
            # 合并电商特征
            merged.update(record["ecom_features"])
            
            # 计算衍生特征
            derived_features = self._calculate_derived_features(merged)
            merged.update(derived_features)
            
            record["merged_features"] = merged
    
    def _calculate_derived_features(self, features: Dict) -> Dict:
        """计算衍生特征
        
        Args:
            features: 原始特征字典
        
        Returns:
            Dict: 衍生特征字典
        """
        derived = {}
        
        try:
            # 收入相关衍生特征
            if "income" in features and "loan_amount" in features:
                income = features["income"]
                loan_amount = features["loan_amount"]
                if income > 0:
                    derived["loan_to_income_ratio"] = loan_amount / income
            
            # 消费相关衍生特征
            if "avg_order_value" in features and "purchase_frequency" in features:
                derived["monthly_spending_estimate"] = (
                    features["avg_order_value"] * features["purchase_frequency"]
                )
            
            # 风险评分
            if "credit_score" in features and "debt_to_income" in features:
                credit_score = features["credit_score"]
                debt_ratio = features["debt_to_income"]
                # 简单风险评分计算
                risk_score = max(0, min(1, (800 - credit_score) / 200 + debt_ratio))
                derived["risk_score"] = risk_score
            
            # 客户价值评分
            if "total_spent" in features and "account_age_days" in features:
                total_spent = features["total_spent"]
                account_age = max(1, features["account_age_days"])
                derived["customer_value_score"] = total_spent / (account_age / 365)
            
            # 活跃度评分
            if "purchase_frequency" in features and "avg_session_duration" in features:
                freq = features["purchase_frequency"]
                duration = features["avg_session_duration"]
                derived["engagement_score"] = min(1, (freq * duration) / 100)
            
        except Exception as e:
            logger.warning(f"Failed to calculate some derived features: {e}")
        
        return derived
    
    def _generate_feature_schemas(self):
        """生成特征schema"""
        if not self.features_cache:
            return
        
        # 获取一个样本记录
        sample_record = next(iter(self.features_cache.values()))
        merged_features = sample_record["merged_features"]
        
        # 生成schema
        self.feature_schemas = {
            "bank_features": self._infer_schema(sample_record["bank_features"]),
            "ecom_features": self._infer_schema(sample_record["ecom_features"]),
            "merged_features": self._infer_schema(merged_features)
        }
    
    def _infer_schema(self, features: Dict) -> Dict:
        """推断特征schema
        
        Args:
            features: 特征字典
        
        Returns:
            Dict: schema字典
        """
        schema = {}
        for key, value in features.items():
            if isinstance(value, int):
                schema[key] = "integer"
            elif isinstance(value, float):
                schema[key] = "float"
            elif isinstance(value, str):
                schema[key] = "string"
            elif isinstance(value, bool):
                schema[key] = "boolean"
            else:
                schema[key] = "unknown"
        return schema
    
    def _safe_convert(self, value: Any, target_type: type, default: Any) -> Any:
        """安全类型转换
        
        Args:
            value: 要转换的值
            target_type: 目标类型
            default: 默认值
        
        Returns:
            转换后的值或默认值
        """
        try:
            if pd.isna(value) or value is None:
                return default
            return target_type(value)
        except (ValueError, TypeError):
            return default
    
    def get_features(self, psi_token: str, feature_type: str = "merged") -> Optional[Dict]:
        """获取特征
        
        Args:
            psi_token: PSI令牌
            feature_type: 特征类型 (bank/ecom/merged)
        
        Returns:
            Dict: 特征字典，如果不存在则返回None
        """
        if psi_token not in self.features_cache:
            return None
        
        record = self.features_cache[psi_token]
        
        if feature_type == "bank":
            return record["bank_features"].copy()
        elif feature_type == "ecom":
            return record["ecom_features"].copy()
        elif feature_type == "merged":
            return record["merged_features"].copy()
        else:
            return record["merged_features"].copy()
    
    def get_feature_record(self, psi_token: str) -> Optional[Dict]:
        """获取完整特征记录
        
        Args:
            psi_token: PSI令牌
        
        Returns:
            Dict: 完整特征记录，如果不存在则返回None
        """
        if psi_token not in self.features_cache:
            return None
        
        return self.features_cache[psi_token].copy()
    
    def update_features(self, psi_token: str, features: Dict, feature_type: str = "manual") -> bool:
        """更新特征
        
        Args:
            psi_token: PSI令牌
            features: 新特征字典
            feature_type: 特征来源类型
        
        Returns:
            bool: 更新是否成功
        """
        try:
            timestamp = datetime.now().isoformat()
            
            if psi_token in self.features_cache:
                # 更新现有记录
                record = self.features_cache[psi_token]
                record["merged_features"].update(features)
                record["last_updated"] = timestamp
                
                if feature_type not in record["data_sources"]:
                    record["data_sources"].append(feature_type)
            else:
                # 创建新记录
                self.features_cache[psi_token] = {
                    "psi_token": psi_token,
                    "bank_features": {},
                    "ecom_features": {},
                    "merged_features": features.copy(),
                    "last_updated": timestamp,
                    "data_sources": [feature_type]
                }
            
            logger.info(f"Features updated for {psi_token}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update features for {psi_token}: {e}")
            return False
    
    def list_psi_tokens(self, limit: int = 100, offset: int = 0) -> List[str]:
        """列出PSI令牌
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            List[str]: PSI令牌列表
        """
        tokens = list(self.features_cache.keys())
        return tokens[offset:offset + limit]
    
    def get_feature_statistics(self) -> Dict:
        """获取特征统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.features_cache:
            return {}
        
        stats = {
            "total_records": len(self.features_cache),
            "data_sources": {},
            "feature_coverage": {},
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None
        }
        
        # 统计数据源
        for record in self.features_cache.values():
            for source in record["data_sources"]:
                stats["data_sources"][source] = stats["data_sources"].get(source, 0) + 1
        
        # 统计特征覆盖率
        if self.features_cache:
            sample_features = next(iter(self.features_cache.values()))["merged_features"]
            total_records = len(self.features_cache)
            
            for feature_name in sample_features.keys():
                count = sum(1 for record in self.features_cache.values() 
                          if feature_name in record["merged_features"] 
                          and record["merged_features"][feature_name] is not None)
                stats["feature_coverage"][feature_name] = count / total_records
        
        return stats
    
    def get_feature_schemas(self) -> Dict:
        """获取特征schemas
        
        Returns:
            Dict: 特征schemas
        """
        return self.feature_schemas.copy()
    
    def validate_features(self, features: Dict, feature_type: str = "merged") -> Dict:
        """验证特征格式
        
        Args:
            features: 要验证的特征
            feature_type: 特征类型
        
        Returns:
            Dict: 验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if feature_type not in self.feature_schemas:
            result["warnings"].append(f"No schema found for feature type: {feature_type}")
            return result
        
        schema = self.feature_schemas[feature_type]
        
        # 检查必需特征
        for required_feature in schema.keys():
            if required_feature not in features:
                result["warnings"].append(f"Missing feature: {required_feature}")
        
        # 检查特征类型
        for feature_name, value in features.items():
            if feature_name in schema:
                expected_type = schema[feature_name]
                actual_type = type(value).__name__
                
                type_mapping = {
                    "int": "integer",
                    "float": "float",
                    "str": "string",
                    "bool": "boolean"
                }
                
                if type_mapping.get(actual_type) != expected_type:
                    result["errors"].append(
                        f"Type mismatch for {feature_name}: expected {expected_type}, got {actual_type}"
                    )
                    result["valid"] = False
        
        return result