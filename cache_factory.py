# filename: promptshield/cache_factory.py
import os
import yaml
from typing import Dict, Any, Optional

def get_cache_from_config() -> Any:
    """
    Get a cache instance based on the configuration.
    
    Returns:
        Cache instance
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        config = {}
    
    # Get cache configuration
    cache_config = config.get("cache", {})
    cache_type = cache_config.get("type", "in_memory")
    
    if cache_type == "redis":
        try:
            from promptshield.redis_cache import RedisCache
            
            # Get Redis configuration
            redis_config = cache_config.get("redis", {})
            host = redis_config.get("host", "localhost")
            port = redis_config.get("port", 6379)
            db = redis_config.get("db", 0)
            password = redis_config.get("password")
            ttl = redis_config.get("ttl", 600)
            
            return RedisCache(host=host, port=port, db=db, password=password, ttl=ttl)
        except ImportError:
            print("Redis is not installed. Falling back to in-memory cache.")
            from promptshield.cache import InMemoryCache
            return InMemoryCache()
    else:
        from promptshield.cache import InMemoryCache
        return InMemoryCache()
