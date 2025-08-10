"""
LLM Configuration Manager
Centralized management of LLM providers and model mappings from config
"""

import logging
from typing import Dict, Any, Optional, Tuple

from src.core.config import config, LLMProvider

logger = logging.getLogger(__name__)


class LLMConfigManager:
    """
    Centralized LLM configuration manager that properly maps config enums to actual model names and API keys
    """
    
    def __init__(self):
        self._model_mappings = self._build_model_mappings()
        self._api_key_mappings = self._build_api_key_mappings()
    
    def _build_model_mappings(self) -> Dict[str, str]:
        """Build mapping from LLMProvider enum to actual model names"""
        return {
            LLMProvider.GEMINI: "gemini-2.5-flash-lite",
            LLMProvider.OPENAI: "gpt-4o-mini"
        }
    
    def _build_api_key_mappings(self) -> Dict[str, str]:
        """Build mapping from provider to API key config field"""
        return {
            "gemini": "gemini_api_key", 
            "openai": "openai_api_key"
        }
    
    def get_model_config(self, llm_provider: LLMProvider) -> Tuple[str, str, Optional[str]]:
        """
        Get model configuration for a given LLM provider
        
        Returns:
            Tuple of (provider, model_name, api_key)
        """
        try:
            # Get the actual model name
            model_name = self._model_mappings.get(llm_provider)
            if not model_name:
                logger.warning(f"Unknown LLM provider: {llm_provider}")
                return "unknown", "unknown", None
            
            # Extract provider name
            if llm_provider.startswith("gemini"):
                provider = "gemini"
            elif llm_provider.startswith("gpt") or "openai" in llm_provider.lower():
                provider = "openai"
            else:
                provider = "unknown"
            
            # Get API key
            api_key_field = self._api_key_mappings.get(provider)
            # Use proper config access instead of getattr
            api_key = getattr(config, api_key_field) if api_key_field and hasattr(config, api_key_field) else None
            
            return provider, model_name, api_key
            
        except Exception as e:
            logger.error(f"Failed to get model config for {llm_provider}: {str(e)}")
            return "unknown", "unknown", None
    
    def get_query_llm_config(self) -> Tuple[str, str, Optional[str]]:
        """Get configuration for query LLM from config"""
        return self.get_model_config(config.query_llm)
    
    def get_response_llm_config(self) -> Tuple[str, str, Optional[str]]:
        """Get configuration for response LLM from config"""
        return self.get_model_config(config.response_llm)
    
    def get_embedding_config(self) -> Tuple[str, int, int, int]:
        """Get embedding configuration from config"""
        return (
            str(config.embedding_model),
            config.embedding_dimension,
            config.embedding_max_length,
            config.batch_size
        )
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        return {
            "type": str(config.vector_db_type),
            "faiss_index_type": config.faiss_index_type,
            "hnsw_m": config.hnsw_m,
            "hnsw_ef_construction": config.hnsw_ef_construction,
            "hnsw_ef_search": config.hnsw_ef_search
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return {
            "max_workers": config.max_workers,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "batch_size": config.batch_size,
            "max_batch_size": config.max_batch_size,
            "gpu_memory_fraction": config.gpu_memory_fraction,
            "enable_mixed_precision": config.enable_mixed_precision
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate that all required configuration is present"""
        issues = []
        warnings = []
        
        # Check query LLM
        query_provider, query_model, query_key = self.get_query_llm_config()
        if query_provider == "unknown":
            issues.append(f"Invalid query LLM: {config.query_llm}")
        elif not query_key:
            warnings.append(f"No API key for query LLM provider: {query_provider}")
        
        # Check response LLM  
        response_provider, response_model, response_key = self.get_response_llm_config()
        if response_provider == "unknown":
            issues.append(f"Invalid response LLM: {config.response_llm}")
        elif not response_key:
            warnings.append(f"No API key for response LLM provider: {response_provider}")
        
        # Check embedding model
        if not hasattr(config.embedding_model, 'value') and not isinstance(config.embedding_model, str):
            issues.append(f"Invalid embedding model: {config.embedding_model}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "query_llm": f"{query_provider}/{query_model}",
            "response_llm": f"{response_provider}/{response_model}",
            "embedding_model": str(config.embedding_model)
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all configuration"""
        query_provider, query_model, query_key = self.get_query_llm_config()
        response_provider, response_model, response_key = self.get_response_llm_config()
        
        return {
            "llms": {
                "query": {
                    "provider": query_provider,
                    "model": query_model,
                    "has_api_key": query_key is not None
                },
                "response": {
                    "provider": response_provider,
                    "model": response_model,
                    "has_api_key": response_key is not None
                }
            },
            "embedding": {
                "model": str(config.embedding_model),
                "dimension": config.embedding_dimension,
                "max_length": config.embedding_max_length
            },
            "vector_db": {
                "type": str(config.vector_db_type),
                "index_type": config.faiss_index_type
            },
            "processing": {
                "batch_size": config.batch_size,
                "max_workers": config.max_workers,
                "gpu_provider": str(config.gpu_provider)
            }
        }


# Global instance
llm_config_manager = LLMConfigManager()