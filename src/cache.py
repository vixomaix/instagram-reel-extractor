"""
Caching Module for Instagram Reel Extractor

Supports disk-based caching, Redis caching, and hybrid mode.
Provides content-addressable storage for analysis results.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from diskcache import Cache as DiskCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)


class CacheInterface:
    """Abstract interface for caching backends"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    def clear(self) -> bool:
        raise NotImplementedError


class DiskCacheBackend(CacheInterface):
    """Disk-based caching using diskcache library"""
    
    def __init__(self, cache_dir: Path = config.CACHE_DIR, max_size_mb: int = config.CACHE_MAX_SIZE_MB):
        self.cache_dir = cache_dir / "disk"
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self._cache = None
        
        if DISKCACHE_AVAILABLE:
            try:
                self._cache = DiskCache(str(self.cache_dir), size_limit=self.max_size)
                logger.info(f"Initialized disk cache at {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize disk cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        if not self._cache:
            return None
        try:
            return self._cache.get(key)
        except Exception as e:
            logger.warning(f"Disk cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self._cache:
            return False
        try:
            if ttl:
                self._cache.set(key, value, expire=ttl)
            else:
                self._cache.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self._cache:
            return False
        try:
            self._cache.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Disk cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self._cache:
            return False
        try:
            return key in self._cache
        except Exception:
            return False
    
    def clear(self) -> bool:
        if not self._cache:
            return False
        try:
            self._cache.clear()
            return True
        except Exception as e:
            logger.warning(f"Disk cache clear error: {e}")
            return False


class RedisCacheBackend(CacheInterface):
    """Redis-based caching"""
    
    def __init__(
        self,
        host: str = config.REDIS_HOST,
        port: int = config.REDIS_PORT,
        db: int = config.REDIS_DB,
        password: Optional[str] = config.REDIS_PASSWORD
    ):
        self._redis = None
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                # Test connection
                self._redis.ping()
                logger.info(f"Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self._redis = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes"""
        return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize bytes to value"""
        return pickle.loads(value)
    
    def get(self, key: str) -> Optional[Any]:
        if not self._redis:
            return None
        try:
            value = self._redis.get(key)
            return self._deserialize(value) if value else None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self._redis:
            return False
        try:
            serialized = self._serialize(value)
            if ttl:
                self._redis.setex(key, ttl, serialized)
            else:
                self._redis.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self._redis:
            return False
        try:
            self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self._redis:
            return False
        try:
            return self._redis.exists(key) > 0
        except Exception:
            return False
    
    def clear(self) -> bool:
        if not self._redis:
            return False
        try:
            self._redis.flushdb()
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False


class HybridCache(CacheInterface):
    """Hybrid cache that uses both Redis and disk cache"""
    
    def __init__(
        self,
        redis_host: str = config.REDIS_HOST,
        redis_port: int = config.REDIS_PORT,
        cache_dir: Path = config.CACHE_DIR
    ):
        self.redis = RedisCacheBackend(redis_host, redis_port)
        self.disk = DiskCacheBackend(cache_dir)
    
    def get(self, key: str) -> Optional[Any]:
        # Try Redis first (fast)
        value = self.redis.get(key)
        if value is not None:
            return value
        
        # Fall back to disk
        value = self.disk.get(key)
        if value is not None:
            # Promote to Redis
            self.redis.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Set in both caches
        redis_ok = self.redis.set(key, value, ttl)
        disk_ok = self.disk.set(key, value, ttl)
        return redis_ok or disk_ok
    
    def delete(self, key: str) -> bool:
        redis_ok = self.redis.delete(key)
        disk_ok = self.disk.delete(key)
        return redis_ok or disk_ok
    
    def exists(self, key: str) -> bool:
        return self.redis.exists(key) or self.disk.exists(key)
    
    def clear(self) -> bool:
        redis_ok = self.redis.clear()
        disk_ok = self.disk.clear()
        return redis_ok or disk_ok


class CacheManager:
    """Main cache manager that provides a unified interface"""
    
    def __init__(self, cache_type: Optional[str] = None):
        self.cache_type = cache_type or config.CACHE_TYPE
        self._cache: Optional[CacheInterface] = None
        self._init_cache()
    
    def _init_cache(self):
        """Initialize the appropriate cache backend"""
        if not config.CACHE_ENABLED:
            logger.info("Caching is disabled")
            return
        
        if self.cache_type == "redis":
            self._cache = RedisCacheBackend()
        elif self.cache_type == "disk":
            self._cache = DiskCacheBackend()
        elif self.cache_type == "hybrid":
            self._cache = HybridCache()
        else:
            logger.warning(f"Unknown cache type: {self.cache_type}, using disk")
            self._cache = DiskCacheBackend()
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data"""
        if isinstance(data, (str, bytes)):
            key_data = data if isinstance(data, bytes) else data.encode()
        else:
            key_data = json.dumps(data, sort_keys=True).encode()
        
        hash_value = hashlib.sha256(key_data).hexdigest()[:32]
        return f"{prefix}:{hash_value}"
    
    def get_video_cache_key(self, video_path: Path, analysis_type: str) -> str:
        """Generate cache key for video analysis"""
        # Use file modification time and size for cache key
        stat = video_path.stat()
        key_data = f"{video_path.name}:{stat.st_size}:{stat.st_mtime}:{analysis_type}"
        return self._generate_key("video", key_data)
    
    def get_frame_cache_key(self, frame_path: Path, analysis_type: str) -> str:
        """Generate cache key for frame analysis"""
        # Use file hash for content-addressable caching
        stat = frame_path.stat()
        key_data = f"{frame_path.name}:{stat.st_size}:{stat.st_mtime}:{analysis_type}"
        return self._generate_key("frame", key_data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._cache:
            return None
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self._cache:
            return False
        ttl = ttl or config.CACHE_TTL
        return self._cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self._cache:
            return False
        return self._cache.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._cache:
            return False
        return self._cache.exists(key)
    
    def clear(self) -> bool:
        """Clear all cached data"""
        if not self._cache:
            return False
        return self._cache.clear()
    
    def cached(self, prefix: str, ttl: Optional[int] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_data = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                cache_key = self._generate_key(prefix, key_data)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
cache_manager = CacheManager()


def get_cache() -> CacheManager:
    """Get the global cache manager instance"""
    return cache_manager
