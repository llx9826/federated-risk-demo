import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    """模型注册表管理器"""
    
    def __init__(self, registry_dir: str = "/app/models"):
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self.models = {}
        self.production_model = None
        
        # 确保目录存在
        os.makedirs(registry_dir, exist_ok=True)
        
        # 加载现有注册表
        self.load_registry()
    
    def register_model(self, model_hash: str, model_info: Dict) -> bool:
        """注册新模型
        
        Args:
            model_hash: 模型唯一标识
            model_info: 模型信息字典
                - model_type: 模型类型 (SecureBoost/Hetero-LR)
                - auc: AUC指标
                - ks: KS指标
                - dp_epsilon: 差分隐私参数
                - artifacts_path: 模型文件路径
                - description: 模型描述
        
        Returns:
            bool: 注册是否成功
        """
        try:
            self.models[model_hash] = {
                "model_hash": model_hash,
                "model_type": model_info.get("model_type", "unknown"),
                "auc": float(model_info.get("auc", 0.0)),
                "ks": float(model_info.get("ks", 0.0)),
                "dp_epsilon": model_info.get("dp_epsilon"),
                "created_at": datetime.now().isoformat(),
                "artifacts_path": model_info.get("artifacts_path", ""),
                "is_production": False,
                "threshold": 0.5,
                "description": model_info.get("description", ""),
                "training_session_id": model_info.get("training_session_id", ""),
                "feature_importance": model_info.get("feature_importance", {}),
                "validation_metrics": model_info.get("validation_metrics", {})
            }
            
            self.save_registry()
            logger.info(f"Model registered successfully: {model_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_hash}: {e}")
            return False
    
    def promote_to_production(self, model_hash: str, threshold: float = 0.5, 
                            description: Optional[str] = None) -> bool:
        """将模型提升为生产版本
        
        Args:
            model_hash: 要提升的模型哈希
            threshold: 决策阈值
            description: 提升说明
        
        Returns:
            bool: 提升是否成功
        """
        try:
            if model_hash not in self.models:
                raise ValueError(f"Model {model_hash} not found in registry")
            
            # 取消之前的生产模型
            if self.production_model and self.production_model in self.models:
                self.models[self.production_model]["is_production"] = False
                self.models[self.production_model]["demoted_at"] = datetime.now().isoformat()
            
            # 设置新的生产模型
            self.models[model_hash]["is_production"] = True
            self.models[model_hash]["threshold"] = threshold
            self.models[model_hash]["promoted_at"] = datetime.now().isoformat()
            
            if description:
                self.models[model_hash]["promotion_description"] = description
            
            self.production_model = model_hash
            self.save_registry()
            
            logger.info(f"Model promoted to production: {model_hash} with threshold {threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_hash}: {e}")
            return False
    
    def get_production_model(self) -> Optional[Dict]:
        """获取当前生产模型信息
        
        Returns:
            Dict: 生产模型信息，如果没有则返回None
        """
        if not self.production_model or self.production_model not in self.models:
            return None
        
        return self.models[self.production_model].copy()
    
    def get_model(self, model_hash: str) -> Optional[Dict]:
        """获取指定模型信息
        
        Args:
            model_hash: 模型哈希
        
        Returns:
            Dict: 模型信息，如果不存在则返回None
        """
        return self.models.get(model_hash, {}).copy() if model_hash in self.models else None
    
    def list_models(self, include_production_only: bool = False) -> List[Dict]:
        """列出所有模型
        
        Args:
            include_production_only: 是否只返回生产模型
        
        Returns:
            List[Dict]: 模型列表
        """
        models = list(self.models.values())
        
        if include_production_only:
            models = [m for m in models if m.get("is_production", False)]
        
        # 按创建时间倒序排列
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return models
    
    def delete_model(self, model_hash: str) -> bool:
        """删除模型
        
        Args:
            model_hash: 要删除的模型哈希
        
        Returns:
            bool: 删除是否成功
        """
        try:
            if model_hash not in self.models:
                logger.warning(f"Model {model_hash} not found for deletion")
                return False
            
            # 不能删除生产模型
            if self.models[model_hash].get("is_production", False):
                raise ValueError("Cannot delete production model")
            
            del self.models[model_hash]
            self.save_registry()
            
            logger.info(f"Model deleted: {model_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_hash}: {e}")
            return False
    
    def update_model_metadata(self, model_hash: str, metadata: Dict) -> bool:
        """更新模型元数据
        
        Args:
            model_hash: 模型哈希
            metadata: 要更新的元数据
        
        Returns:
            bool: 更新是否成功
        """
        try:
            if model_hash not in self.models:
                raise ValueError(f"Model {model_hash} not found")
            
            # 更新允许的字段
            allowed_fields = {
                "description", "threshold", "feature_importance", 
                "validation_metrics", "tags"
            }
            
            for key, value in metadata.items():
                if key in allowed_fields:
                    self.models[model_hash][key] = value
            
            self.models[model_hash]["updated_at"] = datetime.now().isoformat()
            self.save_registry()
            
            logger.info(f"Model metadata updated: {model_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model metadata {model_hash}: {e}")
            return False
    
    def get_model_performance_history(self, model_hash: str) -> List[Dict]:
        """获取模型性能历史
        
        Args:
            model_hash: 模型哈希
        
        Returns:
            List[Dict]: 性能历史记录
        """
        model = self.get_model(model_hash)
        if not model:
            return []
        
        # 返回基本性能指标
        return [{
            "timestamp": model.get("created_at"),
            "auc": model.get("auc"),
            "ks": model.get("ks"),
            "dp_epsilon": model.get("dp_epsilon"),
            "validation_metrics": model.get("validation_metrics", {})
        }]
    
    def save_registry(self):
        """保存注册表到文件"""
        try:
            registry_data = {
                "models": self.models,
                "production_model": self.production_model,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # 原子写入
            temp_file = self.registry_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            # 原子替换
            if os.path.exists(self.registry_file):
                os.replace(temp_file, self.registry_file)
            else:
                os.rename(temp_file, self.registry_file)
            
            logger.debug("Registry saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # 清理临时文件
            temp_file = self.registry_file + ".tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def load_registry(self):
        """从文件加载注册表"""
        if not os.path.exists(self.registry_file):
            logger.info("Registry file not found, starting with empty registry")
            return
        
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.models = data.get("models", {})
            self.production_model = data.get("production_model")
            
            # 验证数据完整性
            if self.production_model and self.production_model not in self.models:
                logger.warning(f"Production model {self.production_model} not found in models, resetting")
                self.production_model = None
            
            logger.info(f"Registry loaded: {len(self.models)} models, production: {self.production_model}")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            # 备份损坏的文件
            backup_file = f"{self.registry_file}.backup.{int(datetime.now().timestamp())}"
            try:
                os.rename(self.registry_file, backup_file)
                logger.info(f"Corrupted registry backed up to {backup_file}")
            except:
                pass
            
            # 重置为空注册表
            self.models = {}
            self.production_model = None
    
    def get_registry_stats(self) -> Dict:
        """获取注册表统计信息
        
        Returns:
            Dict: 统计信息
        """
        total_models = len(self.models)
        production_models = sum(1 for m in self.models.values() if m.get("is_production", False))
        
        model_types = {}
        for model in self.models.values():
            model_type = model.get("model_type", "unknown")
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            "total_models": total_models,
            "production_models": production_models,
            "model_types": model_types,
            "current_production": self.production_model,
            "registry_file_size": os.path.getsize(self.registry_file) if os.path.exists(self.registry_file) else 0
        }