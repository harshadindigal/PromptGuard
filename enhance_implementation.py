
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: enhance_implementation.py
# execution: true
import os
import time
from typing import Dict, Any, List

def create_redis_cache_implementation():
    """Create a Redis cache implementation."""
    redis_cache_code = """# filename: promptshield/redis_cache.py
import redis
import json
import time
import hashlib
from typing import Dict, Any, Optional

class RedisCache:
    \"\"\"
    Redis-based cache implementation for PromptShield.
    \"\"\"
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, ttl: int = 600):
        \"\"\"
        Initialize the Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            ttl: Time to live in seconds (default: 10 minutes)
        \"\"\"
        self.redis = redis.Redis(host=host, port=port, db=db, password=password)
        self.ttl = ttl
    
    def _hash_key(self, key: str) -> str:
        \"\"\"
        Hash a key to use as a Redis key.
        
        Args:
            key: The key to hash
            
        Returns:
            Hashed key
        \"\"\"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        \"\"\"
        Get a value from the cache.
        
        Args:
            key: The key to get
            
        Returns:
            The value if found, None otherwise
        \"\"\"
        hashed_key = self._hash_key(key)
        value = self.redis.get(f"promptshield:{hashed_key}")
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        
        return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        \"\"\"
        Set a value in the cache.
        
        Args:
            key: The key to set
            value: The value to set
        \"\"\"
        hashed_key = self._hash_key(key)
        self.redis.setex(
            f"promptshield:{hashed_key}",
            self.ttl,
            json.dumps(value)
        )
    
    def delete(self, key: str) -> None:
        \"\"\"
        Delete a value from the cache.
        
        Args:
            key: The key to delete
        \"\"\"
        hashed_key = self._hash_key(key)
        self.redis.delete(f"promptshield:{hashed_key}")
    
    def flush(self) -> None:
        \"\"\"
        Flush all values from the cache.
        \"\"\"
        # Only flush keys with our prefix
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match="promptshield:*")
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
"""
    
    # Write the Redis cache implementation
    os.makedirs("promptshield", exist_ok=True)
    with open("promptshield/redis_cache.py", "w") as f:
        f.write(redis_cache_code)
    
    return "promptshield/redis_cache.py"

def create_enhanced_cache_factory():
    """Create an enhanced cache factory that supports both in-memory and Redis caches."""
    cache_factory_code = """# filename: promptshield/cache_factory.py
import os
import yaml
from typing import Dict, Any, Optional

def get_cache_from_config() -> Any:
    \"\"\"
    Get a cache instance based on the configuration.
    
    Returns:
        Cache instance
    \"\"\"
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
"""
    
    # Write the cache factory implementation
    os.makedirs("promptshield", exist_ok=True)
    with open("promptshield/cache_factory.py", "w") as f:
        f.write(cache_factory_code)
    
    return "promptshield/cache_factory.py"

def create_enhanced_config_yaml():
    """Create an enhanced config.yaml with Redis cache configuration."""
    config_yaml = """# PromptShield Configuration

# Routing rules
routing:
  rules:
    - if: "label == 'nonsense' or label == 'spam'"
      action: "block"
    - if: "label == 'repeat'"
      action: "cache"
    - if: "label == 'low_cost'"
      model: "cheap_model"
    - if: "label == 'valuable'"
      model: "default_model"

# Model configurations
models:
  openai:
    default_model: "gpt-4"
    cheap_model: "gpt-3.5-turbo"
  ollama:
    default_model: "llama3-70b"
    cheap_model: "mistral-instruct"
  anthropic:
    default_model: "claude-v1"
    cheap_model: "claude-haiku"
  vllm:
    default_model: "llama3-70b"
    cheap_model: "mistral-instruct"

# Cache configuration
cache:
  type: "in_memory"  # Options: "in_memory", "redis"
  ttl: 600  # Time to live in seconds (10 minutes)
  
  # Redis configuration (used when type is "redis")
  redis:
    host: "localhost"
    port: 6379
    db: 0
    password: null
    ttl: 600  # Time to live in seconds (10 minutes)

# Classifier configuration
classifier:
  use_ml_model: true
  ml_model_path: "models/classifier.pkl"
  min_words: 4
  profanity_check: true
  repeat_detection: true
  
# Logging configuration
logging:
  level: "INFO"
  file: "logs/promptshield.log"
  json_log: "logs/prompt_logs.jsonl"
  stdout: true
"""
    
    # Write the enhanced config.yaml
    with open("config.yaml", "w") as f:
        f.write(config_yaml)
    
    return "config.yaml"

def create_comprehensive_tests():
    """Create more comprehensive tests for the core components."""
    test_classifier_code = """# filename: tests/test_classifier.py
import unittest
from promptshield.classifier import PromptClassifier

class TestPromptClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = PromptClassifier()
    
    def test_nonsense_classification(self):
        result = self.classifier.classify("asdjklasdjkl", "test_session")
        self.assertEqual(result["label"], "nonsense")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_spam_classification(self):
        result = self.classifier.classify("You are stupid", "test_session")
        self.assertEqual(result["label"], "spam")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_low_cost_classification(self):
        result = self.classifier.classify("What is 2 + 2?", "test_session")
        self.assertEqual(result["label"], "low_cost")
        self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_valuable_classification(self):
        result = self.classifier.classify("Write a detailed analysis of the economic impact of artificial intelligence on global labor markets over the next decade", "test_session")
        self.assertEqual(result["label"], "valuable")
        self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_repeat_classification(self):
        # First query
        self.classifier.classify("What is the capital of France?", "test_session")
        
        # Repeat the query
        result = self.classifier.classify("What is the capital of France?", "test_session")
        self.assertEqual(result["label"], "repeat")
        self.assertEqual(result["confidence"], 1.0)
    
    def test_short_prompt(self):
        result = self.classifier.classify("Hi", "test_session")
        self.assertEqual(result["label"], "nonsense")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_different_sessions(self):
        # First query in session 1
        self.classifier.classify("What is the capital of France?", "session_1")
        
        # Same query in session 2 (should not be classified as repeat)
        result = self.classifier.classify("What is the capital of France?", "session_2")
        self.assertNotEqual(result["label"], "repeat")

if __name__ == "__main__":
    unittest.main()
"""
    
    test_router_code = """# filename: tests/test_router.py
import unittest
from promptshield.router import PromptRouter

class TestPromptRouter(unittest.TestCase):
    def setUp(self):
        self.router = PromptRouter()
    
    def test_block_nonsense(self):
        classification = {"label": "nonsense", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "block")
        self.assertEqual(decision["reason"], "Prompt classified as nonsense")
    
    def test_block_spam(self):
        classification = {"label": "spam", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "block")
        self.assertEqual(decision["reason"], "Prompt classified as spam")
    
    def test_cache_repeat(self):
        classification = {"label": "repeat", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "cache")
    
    def test_route_low_cost(self):
        classification = {"label": "low_cost", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-3.5-turbo")
    
    def test_route_valuable(self):
        classification = {"label": "valuable", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-4")
    
    def test_different_providers(self):
        # OpenAI
        classification = {"label": "valuable", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-4")
        
        # Anthropic
        decision = self.router.route(classification, "anthropic", "claude-v1", "claude-haiku")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "claude-v1")
        
        # Ollama
        decision = self.router.route(classification, "ollama", "llama3-70b", "mistral-instruct")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "llama3-70b")

if __name__ == "__main__":
    unittest.main()
"""
    
    test_cache_code = """# filename: tests/test_cache.py
import unittest
import time
from promptshield.cache import InMemoryCache

class TestInMemoryCache(unittest.TestCase):
    def setUp(self):
        self.cache = InMemoryCache(ttl=1)  # 1 second TTL for testing
    
    def test_set_get(self):
        # Set a value
        self.cache.set("test_key", {"value": "test_value"})
        
        # Get the value
        value = self.cache.get("test_key")
        self.assertEqual(value["value"], "test_value")
    
    def test_ttl(self):
        # Set a value
        self.cache.set("test_key", {"value": "test_value"})
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Get the value (should be None)
        value = self.cache.get("test_key")
        self.assertIsNone(value)
    
    def test_nonexistent_key(self):
        # Get a nonexistent key
        value = self.cache.get("nonexistent_key")
        self.assertIsNone(value)

try:
    from promptshield.redis_cache import RedisCache
    
    class TestRedisCache(unittest.TestCase):
        def setUp(self):
            try:
                self.cache = RedisCache(ttl=1)  # 1 second TTL for testing
                self.cache.flush()  # Clear the cache before testing
            except:
                self.skipTest("Redis is not available")
        
        def test_set_get(self):
            # Set a value
            self.cache.set("test_key", {"value": "test_value"})
            
            # Get the value
            value = self.cache.get("test_key")
            self.assertEqual(value["value"], "test_value")
        
        def test_ttl(self):
            # Set a value
            self.cache.set("test_key", {"value": "test_value"})
            
            # Wait for TTL to expire
            time.sleep(1.1)
            
            # Get the value (should be None)
            value = self.cache.get("test_key")
            self.assertIsNone(value)
        
        def test_nonexistent_key(self):
            # Get a nonexistent key
            value = self.cache.get("nonexistent_key")
            self.assertIsNone(value)
        
        def tearDown(self):
            try:
                self.cache.flush()
            except:
                pass
except ImportError:
    pass  # Redis is not available

if __name__ == "__main__":
    unittest.main()
"""
    
    # Create the test directory
    os.makedirs("tests", exist_ok=True)
    
    # Write the test files
    with open("tests/test_classifier.py", "w") as f:
        f.write(test_classifier_code)
    
    with open("tests/test_router.py", "w") as f:
        f.write(test_router_code)
    
    with open("tests/test_cache.py", "w") as f:
        f.write(test_cache_code)
    
    return ["tests/test_classifier.py", "tests/test_router.py", "tests/test_cache.py"]

def main():
    """Main function to enhance the implementation."""
    print("Enhancing PromptShield Implementation")
    print("===================================")
    
    # Create Redis cache implementation
    print("\n1. Creating Redis Cache Implementation")
    redis_cache_file = create_redis_cache_implementation()
    print(f"  Created {redis_cache_file}")
    
    # Create enhanced cache factory
    print("\n2. Creating Enhanced Cache Factory")
    cache_factory_file = create_enhanced_cache_factory()
    print(f"  Created {cache_factory_file}")
    
    # Create enhanced config.yaml
    print("\n3. Creating Enhanced Config YAML")
    config_file = create_enhanced_config_yaml()
    print(f"  Created {config_file}")
    
    # Create comprehensive tests
    print("\n4. Creating Comprehensive Tests")
    test_files = create_comprehensive_tests()
    for test_file in test_files:
        print(f"  Created {test_file}")
    
    print("\nEnhancement Summary")
    print("===================")
    print("1. Added Redis cache implementation for improved scalability")
    print("2. Created a cache factory to support multiple cache backends")
    print("3. Enhanced configuration with more options")
    print("4. Added comprehensive tests for core components")
    print("\nThese enhancements address the improvement areas identified in the verification:")
    print("- Redis caching option has been added")
    print("- Testing coverage has been significantly enhanced")
    print("- Configuration options have been expanded for better flexibility")

if __name__ == "__main__":
    main()