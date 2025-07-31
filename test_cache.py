# filename: tests/test_cache.py
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
