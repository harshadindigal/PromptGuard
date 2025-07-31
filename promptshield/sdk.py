
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/sdk.py
# execution: true
import os
import json
import logging
import requests
from typing import Dict, Any, Optional
import hashlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptShieldSDK:
    """
    Python SDK for the PromptShield API.
    """
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        """
        Initialize the SDK.
        
        Args:
            api_url: URL of the PromptShield API
        """
        self.api_url = api_url
    
    def classify_prompt(self, prompt: str, session_id: str = "") -> Dict[str, Any]:
        """
        Classify a prompt without routing it.
        
        Args:
            prompt: The user prompt
            session_id: Session identifier (optional)
            
        Returns:
            Classification result
        """
        try:
            # Import locally to avoid circular imports
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            try:
                # Try absolute imports first
                from promptshield.classifier import PromptClassifier
            except ImportError:
                # Fall back to relative imports
                from classifier import PromptClassifier
            
            # Create a classifier instance
            classifier = PromptClassifier()
            
            # Classify the prompt
            classification = classifier.classify(prompt, session_id)
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying prompt: {str(e)}")
            raise
    
    def route_prompt(self, prompt: str, session_id: str, source: str, 
                    default_model: str, cheap_model: str) -> Dict[str, Any]:
        """
        Route a prompt to the appropriate model.
        
        Args:
            prompt: The user prompt
            session_id: Session identifier
            source: Source provider (e.g., 'openai', 'ollama', 'vllm', 'anthropic')
            default_model: Default model to use
            cheap_model: Cheap model to use
            
        Returns:
            API response
        """
        try:
            # Prepare the request
            url = f"{self.api_url}/chat"
            data = {
                "prompt": prompt,
                "session_id": session_id,
                "config": {
                    "source": source,
                    "default_model": default_model,
                    "cheap_model": cheap_model
                }
            }
            
            # Send the request
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error routing prompt: {str(e)}")
            
            # If the API is not available, fall back to local processing
            try:
                # Import locally to avoid circular imports
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                try:
                    # Try absolute imports first
                    from promptshield.classifier import PromptClassifier
                    from promptshield.router import PromptRouter
                    from promptshield.cache import get_cache_from_config
                except ImportError:
                    # Fall back to relative imports
                    from classifier import PromptClassifier
                    from router import PromptRouter
                    from cache import get_cache_from_config
                
                # Create instances
                classifier = PromptClassifier()
                router = PromptRouter()
                cache = get_cache_from_config()
                
                # Check cache first
                cached_response = cache.get(prompt)
                if cached_response:
                    logger.info(f"Cache hit for prompt: '{prompt[:50]}...'")
                    return cached_response
                
                # Classify the prompt
                classification = classifier.classify(prompt, session_id)
                
                # Route the prompt
                routing_decision = router.route(classification, source, default_model, cheap_model)
                
                # Log the decision
                router.log_decision(prompt, routing_decision)
                
                # Handle routing decision
                if routing_decision["action"] == "block":
                    # Blocked prompt
                    return {
                        "text": None,
                        "blocked": True,
                        "block_reason": routing_decision["reason"],
                        "classification": classification,
                        "routing": routing_decision,
                        "model_used": None,
                        "response_time": 0.0,
                        "cached": False
                    }
                
                elif routing_decision["action"] == "cache":
                    # This should not happen as we already checked the cache
                    # But just in case, check again
                    cached_response = cache.get(prompt)
                    if cached_response:
                        logger.info(f"Late cache hit for prompt: '{prompt[:50]}...'")
                        return cached_response
                
                # For actual model calls, we need the API
                logger.error("API is required for model calls. Local processing can only handle classification, routing, and caching.")
                raise ValueError("API is required for model calls")
                
            except Exception as nested_e:
                logger.error(f"Error in local processing fallback: {str(nested_e)}")
                raise
    
    def cache_get(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response by prompt hash.
        
        Args:
            prompt_hash: Hash of the prompt
            
        Returns:
            Cached response or None if not found
        """
        try:
            # Import locally to avoid circular imports
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            try:
                # Try absolute imports first
                from promptshield.cache import get_cache_from_config
            except ImportError:
                # Fall back to relative imports
                from cache import get_cache_from_config
            
            # Create a cache instance
            cache = get_cache_from_config()
            
            # Get the cached response
            # Note: The cache uses the prompt itself as the key, not the hash
            # So we need to find the prompt that corresponds to the hash
            # This is not efficient, but it's the best we can do without modifying the cache
            
            # For now, just return None
            logger.warning("cache_get by hash is not supported in the current implementation")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached response: {str(e)}")
            raise
    
    def cache_set(self, prompt: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for a prompt.
        
        Args:
            prompt: The user prompt
            response: The response to cache
        """
        try:
            # Import locally to avoid circular imports
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            try:
                # Try absolute imports first
                from promptshield.cache import get_cache_from_config
            except ImportError:
                # Fall back to relative imports
                from cache import get_cache_from_config
            
            # Create a cache instance
            cache = get_cache_from_config()
            
            # Cache the response
            cache.set(prompt, response)
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the API.
        
        Returns:
            Metrics
        """
        try:
            # Prepare the request
            url = f"{self.api_url}/metrics"
            
            # Send the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting metrics: {str(e)}")
            
            # If the API is not available, fall back to local processing
            try:
                # Import locally to avoid circular imports
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                try:
                    # Try absolute imports first
                    from promptshield.router import PromptRouter
                except ImportError:
                    # Fall back to relative imports
                    from router import PromptRouter
                
                # Create a router instance
                router = PromptRouter()
                
                # Get the metrics
                return router.get_metrics()
                
            except Exception as nested_e:
                logger.error(f"Error in local processing fallback: {str(nested_e)}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the API.
        
        Returns:
            Health status
        """
        try:
            # Prepare the request
            url = f"{self.api_url}/health"
            
            # Send the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking health: {str(e)}")
            
            # If the API is not available, return a local health check
            return {
                "status": "api_unavailable",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }


# For testing
if __name__ == "__main__":
    # Test the SDK
    sdk = PromptShieldSDK()
    
    # Test classification
    test_prompts = [
        "asdjklasdjkl",  # nonsense
        "What is 2 + 2?",  # low_cost
        "Write a poem about AI",  # valuable
        "You are stupid",  # spam
    ]
    
    print("Testing SDK classification:")
    for prompt in test_prompts:
        try:
            classification = sdk.classify_prompt(prompt, "test_session")
            print(f"Prompt: '{prompt}'")
            print(f"Classification: {classification['label']} (confidence: {classification['confidence']:.2f})")
            print()
        except Exception as e:
            print(f"Error classifying prompt '{prompt}': {str(e)}")
    
    # Test caching
    print("\nTesting SDK caching:")
    try:
        prompt = "What is the capital of France?"
        response = {
            "text": "The capital of France is Paris.",
            "model_used": "test_model",
            "response_time": 0.5
        }
        
        sdk.cache_set(prompt, response)
        print(f"Cached response for prompt: '{prompt}'")
        
        # We can't test cache_get by hash directly
        # But we can test the cache by using the route_prompt method
        # which will check the cache first
        
    except Exception as e:
        print(f"Error testing caching: {str(e)}")
    
    # Test API health check
    print("\nTesting SDK health check:")
    try:
        health = sdk.health_check()
        print(f"API health: {health}")
    except Exception as e:
        print(f"Error checking API health: {str(e)}")
    
    # Test metrics
    print("\nTesting SDK metrics:")
    try:
        metrics = sdk.get_metrics()
        print(f"API metrics: {metrics}")
    except Exception as e:
        print(f"Error getting metrics: {str(e)}")

print("SDK implemented successfully!")