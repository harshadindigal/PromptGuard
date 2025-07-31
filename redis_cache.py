# filename: promptshield/redis_cache.py
import redis
import json
import time
import hashlib
from typing import Dict, Any, Optional

class RedisCache:
    """
    Redis-based cache implementation for PromptShield.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, ttl: int = 600):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            ttl: Time to live in seconds (default: 10 minutes)
        """
        self.redis = redis.Redis(host=host, port=port, db=db, password=password)
        self.ttl = ttl
    
    def _hash_key(self, key: str) -> str:
        """
        Hash a key to use as a Redis key.
        
        Args:
            key: The key to hash
            
        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from the cache.
        
        Args:
            key: The key to get
            
        Returns:
            The value if found, None otherwise
        """
        hashed_key = self._hash_key(key)
        value = self.redis.get(f"promptshield:{hashed_key}")
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        
        return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The key to set
            value: The value to set
        """
        hashed_key = self._hash_key(key)
        self.redis.setex(
            f"promptshield:{hashed_key}",
            self.ttl,
            json.dumps(value)
        )
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The key to delete
        """
        hashed_key = self._hash_key(key)
        self.redis.delete(f"promptshield:{hashed_key}")
    
    def flush(self) -> None:
        """
        Flush all values from the cache.
        """
        # Only flush keys with our prefix
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match="promptshield:*")
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
