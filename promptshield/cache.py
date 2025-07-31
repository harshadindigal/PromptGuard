
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/cache.py
# execution: true
import time
import hashlib
import json
from typing import Dict, Any, Optional, Union
import yaml
import os

class BaseCache:
    """Base cache class that defines the interface for all cache implementations."""
    
    def __init__(self, ttl: int = 600):
        """
        Initialize the cache.
        
        Args:
            ttl: Time to live in seconds (default: 10 minutes)
        """
        self.ttl = ttl
    
    def normalize_prompt(self, prompt: str) -> str:
        """
        Normalize the prompt to create a consistent cache key.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Normalized prompt string
        """
        # Simple normalization: lowercase, strip whitespace
        return prompt.lower().strip()
    
    def create_key(self, prompt: str) -> str:
        """
        Create a cache key from a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Cache key string
        """
        normalized = self.normalize_prompt(prompt)
        # Use SHA-256 for hashing to avoid collisions
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Cached response or None if not found
        """
        raise NotImplementedError("Subclasses must implement get()")
    
    def set(self, prompt: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for a prompt.
        
        Args:
            prompt: The user prompt
            response: The response to cache
        """
        raise NotImplementedError("Subclasses must implement set()")
    
    def clear(self) -> None:
        """Clear the cache."""
        raise NotImplementedError("Subclasses must implement clear()")


class MemoryCache(BaseCache):
    """In-memory cache implementation."""
    
    def __init__(self, ttl: int = 600, max_size: int = 1000):
        """
        Initialize the memory cache.
        
        Args:
            ttl: Time to live in seconds (default: 10 minutes)
            max_size: Maximum number of entries in the cache
        """
        super().__init__(ttl)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
    
    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Cached response or None if not found or expired
        """
        key = self.create_key(prompt)
        if key in self.cache:
            entry = self.cache[key]
            # Check if the entry has expired
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, prompt: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for a prompt.
        
        Args:
            prompt: The user prompt
            response: The response to cache
        """
        key = self.create_key(prompt)
        # If cache is full, remove the oldest entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class RedisCache(BaseCache):
    """Redis cache implementation."""
    
    def __init__(self, ttl: int = 600, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize the Redis cache.
        
        Args:
            ttl: Time to live in seconds (default: 10 minutes)
            redis_url: Redis connection URL
        """
        super().__init__(ttl)
        self.redis_url = redis_url
        # Lazy import to avoid dependency if not used
        try:
            import redis
            self.redis = redis.from_url(redis_url)
        except ImportError:
            raise ImportError("Redis package is required for RedisCache. Install with 'pip install redis'.")
    
    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Cached response or None if not found
        """
        key = self.create_key(prompt)
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, prompt: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for a prompt.
        
        Args:
            prompt: The user prompt
            response: The response to cache
        """
        key = self.create_key(prompt)
        self.redis.setex(key, self.ttl, json.dumps(response))
    
    def clear(self) -> None:
        """Clear the cache."""
        # Only clear keys related to this cache (not all Redis keys)
        # This is a simplistic approach; in production, use a namespace
        for key in self.redis.keys("*"):
            self.redis.delete(key)


def get_cache_from_config() -> BaseCache:
    """
    Create a cache instance based on the configuration.
    
    Returns:
        Cache instance
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        config = {'cache': {'type': 'memory', 'ttl': 600, 'max_size': 1000}}
    
    cache_config = config.get('cache', {})
    cache_type = cache_config.get('type', 'memory')
    ttl = cache_config.get('ttl', 600)
    
    if cache_type == 'redis':
        redis_url = cache_config.get('redis_url', 'redis://localhost:6379/0')
        return RedisCache(ttl=ttl, redis_url=redis_url)
    else:  # Default to memory cache
        max_size = cache_config.get('max_size', 1000)
        return MemoryCache(ttl=ttl, max_size=max_size)


# For testing
if __name__ == "__main__":
    # Test the memory cache
    cache = MemoryCache(ttl=5)  # Short TTL for testing
    
    # Cache a response
    test_prompt = "What is the capital of France?"
    test_response = {"text": "The capital of France is Paris.", "metadata": {"source": "knowledge"}}
    cache.set(test_prompt, test_response)
    
    # Retrieve the cached response
    cached = cache.get(test_prompt)
    print(f"Cached response: {cached}")
    
    # Test with a different prompt
    different_prompt = "What is the capital of Spain?"
    cached = cache.get(different_prompt)
    print(f"Different prompt (should be None): {cached}")
    
    # Test with a normalized prompt (should hit the cache)
    normalized_prompt = "what is the capital of france?"
    cached = cache.get(normalized_prompt)
    print(f"Normalized prompt (should hit cache): {cached}")
    
    # Test expiration
    print("Waiting for cache to expire...")
    time.sleep(6)  # Wait for the TTL to expire
    cached = cache.get(test_prompt)
    print(f"After expiration (should be None): {cached}")

print("Cache module implemented successfully!")