"""
Redis Connection Manager with Centralized Configuration Support
Provides connection pooling, retry logic, health checks, and caching operations
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import timedelta

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
except ImportError:
    raise ImportError("redis package is required. Install with: pip install redis")

from src.core.config import config

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Redis connection manager with pooling, retry logic, and health monitoring
    Integrated with centralized configuration system
    """
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0
        
        # Configuration from central config
        self.host = config.redis_host
        self.port = config.redis_port
        self.db = config.redis_db
        self.password = config.redis_password
        self.max_connections = config.redis_max_connections
        self.timeout = config.redis_timeout
        
        logger.info(f"RedisConnectionManager initialized for {self.host}:{self.port}")

    async def initialize(self) -> bool:
        """Initialize Redis connection pool and test connectivity"""
        try:
            logger.info("🔄 Initializing Redis connection pool...")
            
            # Create connection pool
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                health_check_interval=30,
                retry_on_timeout=True,
                decode_responses=True
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            logger.info("✅ Redis connection pool initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis connection: {str(e)}")
            self.is_connected = False
            return False

    async def _test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self._last_health_check < self._health_check_interval:
            return {"status": "healthy" if self.is_connected else "unhealthy"}
            
        try:
            if not self.redis_client:
                raise Exception("Redis client not initialized")
                
            # Test basic operations
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=10)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if value != "test_value":
                raise Exception("Redis read/write test failed")
            
            # Get Redis info
            info = await self.redis_client.info()
            
            self.is_connected = True
            self._last_health_check = current_time
            
            return {
                "status": "healthy",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "version": info.get("redis_version", "unknown"),
                "connection_pool_size": self.max_connections
            }
            
        except Exception as e:
            self.is_connected = False
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis with retry logic"""
        if not self.is_connected:
            logger.warning("Redis not connected, skipping get operation")
            return None
            
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Redis get operation failed for key '{key}': {str(e)}")
            return None

    async def set(
        self, 
        key: str, 
        value: Union[str, int, float], 
        ex: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """Set value in Redis with optional expiration"""
        if not self.is_connected:
            logger.warning("Redis not connected, skipping set operation")
            return False
            
        try:
            # Use default TTL from config if not specified
            if ex is None:
                ex = config.cache_ttl_hours * 3600
                
            result = await self.redis_client.set(key, value, ex=ex, nx=nx)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis set operation failed for key '{key}': {str(e)}")
            return False

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from Redis"""
        try:
            value = await self.get(key)
            if value is None:
                return None
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for key '{key}': {str(e)}")
            return None

    async def set_json(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ex: Optional[int] = None
    ) -> bool:
        """Set JSON value in Redis"""
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return await self.set(key, json_str, ex=ex)
        except Exception as e:
            logger.error(f"Failed to set JSON for key '{key}': {str(e)}")
            return False

    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis"""
        if not self.is_connected:
            logger.warning("Redis not connected, skipping delete operation")
            return 0
            
        try:
            return await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete operation failed: {str(e)}")
            return 0

    async def exists(self, *keys: str) -> int:
        """Check if keys exist in Redis"""
        if not self.is_connected:
            return 0
            
        try:
            return await self.redis_client.exists(*keys)
        except Exception as e:
            logger.error(f"Redis exists operation failed: {str(e)}")
            return 0

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment value in Redis"""
        if not self.is_connected:
            return None
            
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment operation failed for key '{key}': {str(e)}")
            return None

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for key"""
        if not self.is_connected:
            return False
            
        try:
            return await self.redis_client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redis expire operation failed for key '{key}': {str(e)}")
            return False

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (use carefully in production)"""
        if not self.is_connected:
            return []
            
        try:
            return await self.redis_client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis keys operation failed: {str(e)}")
            return []

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        try:
            keys = await self.keys(pattern)
            if keys:
                return await self.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.redis_client.info()
            
            return {
                "connected": self.is_connected,
                "total_connections_received": info.get("total_connections_received", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), 
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"connected": False, "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100

    async def close(self):
        """Close Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.pool:
                await self.pool.disconnect()
            self.is_connected = False
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")


# Global Redis manager instance
redis_manager = RedisConnectionManager()


# Convenience functions for easy access
async def get_cache(key: str) -> Optional[str]:
    """Convenience function for getting cached values"""
    return await redis_manager.get(key)


async def set_cache(key: str, value: Union[str, int, float], ex: Optional[int] = None) -> bool:
    """Convenience function for setting cached values"""
    return await redis_manager.set(key, value, ex=ex)


async def get_json_cache(key: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting JSON cached values"""
    return await redis_manager.get_json(key)


async def set_json_cache(key: str, value: Dict[str, Any], ex: Optional[int] = None) -> bool:
    """Convenience function for setting JSON cached values"""
    return await redis_manager.set_json(key, value, ex=ex)


async def delete_cache(*keys: str) -> int:
    """Convenience function for deleting cached values"""
    return await redis_manager.delete(*keys)