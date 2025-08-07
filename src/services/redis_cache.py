"""
Redis Connection Manager and Caching Service for BajajFinsev
Provides Redis connectivity, caching operations, and connection pooling
"""

import redis
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
import redis.asyncio as redis_async
from redis.exceptions import ConnectionError, TimeoutError, RedisError

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Manages Redis connections with connection pooling, retry logic, and health monitoring
    Optimized for the hybrid RAG system caching needs
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        retry_on_timeout: bool = True,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        
        # Connection pools
        self._sync_pool: Optional[redis.ConnectionPool] = None
        self._async_pool: Optional[redis_async.ConnectionPool] = None
        
        # Client instances
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[redis_async.Redis] = None
        
        # Health monitoring
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        self._is_healthy = False
        
        logger.info(f"Redis connection manager initialized for {host}:{port}")
    
    def _create_sync_pool(self) -> redis.ConnectionPool:
        """Create synchronous Redis connection pool"""
        return redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            retry_on_timeout=self.retry_on_timeout,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
            health_check_interval=30
        )
    
    def _create_async_pool(self) -> redis_async.ConnectionPool:
        """Create asynchronous Redis connection pool"""
        return redis_async.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            retry_on_timeout=self.retry_on_timeout,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout
        )
    
    @property
    def sync_client(self) -> redis.Redis:
        """Get synchronous Redis client with connection pooling"""
        if not self._sync_client:
            if not self._sync_pool:
                self._sync_pool = self._create_sync_pool()
            self._sync_client = redis.Redis(connection_pool=self._sync_pool)
        return self._sync_client
    
    @property
    def async_client(self) -> redis_async.Redis:
        """Get asynchronous Redis client with connection pooling"""
        if not self._async_client:
            if not self._async_pool:
                self._async_pool = self._create_async_pool()
            self._async_client = redis_async.Redis(connection_pool=self._async_pool)
        return self._async_client
    
    async def health_check(self, force: bool = False) -> bool:
        """
        Perform Redis health check with caching to avoid excessive checks
        """
        current_time = time.time()
        
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return self._is_healthy
        
        try:
            # Ping Redis server
            result = await self.async_client.ping()
            self._is_healthy = result is True
            self._last_health_check = current_time
            
            if self._is_healthy:
                logger.debug("Redis health check passed")
            else:
                logger.warning("Redis health check failed: ping returned False")
                
        except Exception as e:
            self._is_healthy = False
            self._last_health_check = current_time
            logger.error(f"Redis health check failed: {str(e)}")
        
        return self._is_healthy
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get Redis connection pool statistics"""
        stats = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
            "is_healthy": await self.health_check(),
            "sync_pool_stats": {},
            "async_pool_stats": {}
        }
        
        try:
            if self._sync_pool:
                stats["sync_pool_stats"] = {
                    "created_connections": self._sync_pool.created_connections,
                    "available_connections": len(self._sync_pool._available_connections),
                    "in_use_connections": len(self._sync_pool._in_use_connections)
                }
            
            if self._async_pool:
                stats["async_pool_stats"] = {
                    "created_connections": self._async_pool.created_connections,
                    "available_connections": len(self._async_pool._available_connections),
                    "in_use_connections": len(self._async_pool._in_use_connections)
                }
                
        except Exception as e:
            logger.error(f"Error getting connection stats: {str(e)}")
        
        return stats
    
    async def close(self):
        """Close all Redis connections and clean up resources"""
        try:
            if self._async_client:
                await self._async_client.close()
                self._async_client = None
            
            if self._async_pool:
                await self._async_pool.disconnect()
                self._async_pool = None
            
            if self._sync_pool:
                self._sync_pool.disconnect()
                self._sync_pool = None
                
            self._sync_client = None
            logger.info("Redis connection manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")


class RedisCache:
    """
    High-level Redis caching service for the hybrid RAG system
    Provides caching for embeddings, query results, and document chunks
    """
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.redis_manager = connection_manager
        self._default_ttl = 3600  # 1 hour default TTL
        
        # Cache key prefixes for different data types
        self.PREFIXES = {
            "embedding": "emb:",
            "query_result": "qr:",
            "document_chunk": "doc:",
            "session": "sess:",
            "performance": "perf:",
            "response": "resp:"
        }
        
        logger.info("Redis cache service initialized")
    
    def _make_key(self, prefix: str, identifier: str) -> str:
        """Create properly formatted cache key"""
        return f"{self.PREFIXES.get(prefix, prefix)}{identifier}"
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        prefix: str = ""
    ) -> bool:
        """
        Set a value in Redis cache with optional TTL and prefix
        """
        try:
            cache_key = self._make_key(prefix, key) if prefix else key
            ttl = ttl or self._default_ttl
            
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            result = await self.redis_manager.async_client.setex(
                cache_key, ttl, serialized_value
            )
            
            logger.debug(f"Cache SET: {cache_key} (TTL: {ttl}s)")
            return result
            
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {str(e)}")
            return False
    
    async def get(self, key: str, prefix: str = "") -> Optional[Any]:
        """
        Get a value from Redis cache with optional prefix
        """
        try:
            cache_key = self._make_key(prefix, key) if prefix else key
            
            value = await self.redis_manager.async_client.get(cache_key)
            
            if value is None:
                logger.debug(f"Cache MISS: {cache_key}")
                return None
            
            logger.debug(f"Cache HIT: {cache_key}")
            
            # Try to deserialize JSON, fallback to string
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return value.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {str(e)}")
            return None
    
    async def delete(self, key: str, prefix: str = "") -> bool:
        """
        Delete a key from Redis cache
        """
        try:
            cache_key = self._make_key(prefix, key) if prefix else key
            result = await self.redis_manager.async_client.delete(cache_key)
            
            logger.debug(f"Cache DELETE: {cache_key}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str, prefix: str = "") -> bool:
        """
        Check if a key exists in Redis cache
        """
        try:
            cache_key = self._make_key(prefix, key) if prefix else key
            result = await self.redis_manager.async_client.exists(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}: {str(e)}")
            return False
    
    async def increment(self, key: str, prefix: str = "", amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in Redis cache
        """
        try:
            cache_key = self._make_key(prefix, key) if prefix else key
            result = await self.redis_manager.async_client.incrby(cache_key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Redis INCREMENT failed for key {key}: {str(e)}")
            return None
    
    async def set_hash(self, hash_key: str, field_values: Dict[str, Any], prefix: str = "") -> bool:
        """
        Set multiple fields in a Redis hash
        """
        try:
            cache_key = self._make_key(prefix, hash_key) if prefix else hash_key
            
            # Serialize complex values
            serialized_values = {}
            for field, value in field_values.items():
                if isinstance(value, (dict, list, tuple)):
                    serialized_values[field] = json.dumps(value, default=str)
                else:
                    serialized_values[field] = str(value)
            
            result = await self.redis_manager.async_client.hmset(cache_key, serialized_values)
            logger.debug(f"Cache HSET: {cache_key} with {len(field_values)} fields")
            return result
            
        except Exception as e:
            logger.error(f"Redis HSET failed for hash {hash_key}: {str(e)}")
            return False
    
    async def get_hash(self, hash_key: str, prefix: str = "") -> Optional[Dict[str, Any]]:
        """
        Get all fields from a Redis hash
        """
        try:
            cache_key = self._make_key(prefix, hash_key) if prefix else hash_key
            result = await self.redis_manager.async_client.hgetall(cache_key)
            
            if not result:
                return None
            
            # Deserialize values
            deserialized_result = {}
            for field, value in result.items():
                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                
                try:
                    deserialized_result[field_str] = json.loads(value_str)
                except json.JSONDecodeError:
                    deserialized_result[field_str] = value_str
            
            logger.debug(f"Cache HGET: {cache_key} with {len(deserialized_result)} fields")
            return deserialized_result
            
        except Exception as e:
            logger.error(f"Redis HGET failed for hash {hash_key}: {str(e)}")
            return None
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern (use with caution)
        """
        try:
            keys = await self.redis_manager.async_client.keys(pattern)
            if keys:
                result = await self.redis_manager.async_client.delete(*keys)
                logger.info(f"Cache CLEAR: Deleted {result} keys matching pattern '{pattern}'")
                return result
            return 0
            
        except Exception as e:
            logger.error(f"Redis CLEAR failed for pattern {pattern}: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive Redis cache statistics
        """
        try:
            info = await self.redis_manager.async_client.info()
            
            stats = {
                "connection_stats": await self.redis_manager.get_connection_stats(),
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total = hits + misses
            stats["cache_hit_rate"] = (hits / total * 100) if total > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e)}


# Global Redis instances (will be initialized in main application)
redis_manager: Optional[RedisConnectionManager] = None
redis_cache: Optional[RedisCache] = None


async def initialize_redis(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None
) -> tuple[RedisConnectionManager, RedisCache]:
    """
    Initialize global Redis connection manager and cache service
    """
    global redis_manager, redis_cache
    
    try:
        # Create connection manager
        redis_manager = RedisConnectionManager(
            host=host,
            port=port,
            password=password
        )
        
        # Test connection
        health_ok = await redis_manager.health_check(force=True)
        if not health_ok:
            raise ConnectionError("Failed to connect to Redis server")
        
        # Create cache service
        redis_cache = RedisCache(redis_manager)
        
        logger.info(f"Redis initialized successfully at {host}:{port}")
        return redis_manager, redis_cache
        
    except Exception as e:
        logger.error(f"Redis initialization failed: {str(e)}")
        raise


async def shutdown_redis():
    """
    Shutdown global Redis connections
    """
    global redis_manager, redis_cache
    
    try:
        if redis_manager:
            await redis_manager.close()
            redis_manager = None
            
        redis_cache = None
        logger.info("Redis shutdown completed")
        
    except Exception as e:
        logger.error(f"Redis shutdown error: {str(e)}")


# Async context manager for Redis operations
@asynccontextmanager
async def redis_operation():
    """
    Async context manager for Redis operations with automatic error handling
    """
    try:
        if not redis_manager or not redis_cache:
            raise RuntimeError("Redis not initialized. Call initialize_redis() first.")
            
        # Check health before operations
        if not await redis_manager.health_check():
            logger.warning("Redis health check failed, operations may fail")
        
        yield redis_cache
        
    except Exception as e:
        logger.error(f"Redis operation error: {str(e)}")
        raise