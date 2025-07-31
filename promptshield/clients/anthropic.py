
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/clients/anthropic.py
# execution: true
import os
import logging
from typing import Dict, Any, Optional
import time

# Set up logging
logger = logging.getLogger(__name__)

class AnthropicClient:
    """
    Client for the Anthropic Claude API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set the ANTHROPIC_API_KEY environment variable or pass it to the constructor.")
        
        # Initialize the Anthropic client lazily
        self.client = None
    
    def _get_client(self):
        """
        Get the Anthropic client, initializing it if necessary.
        
        Returns:
            Anthropic client
        """
        if self.client is None:
            try:
                import anthropic
                # Updated initialization based on the latest Anthropic Python SDK
                self.client = anthropic.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package is required. Install with 'pip install anthropic'.")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
                raise
        
        return self.client
    
    def send_prompt(self, prompt: str, model: str = "claude-haiku", max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Send a prompt to the Anthropic API.
        
        Args:
            prompt: The user prompt
            model: The model to use (default: claude-haiku)
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response
        """
        client = self._get_client()
        
        # Default parameters
        params = {
            "model": model,
            "max_tokens": 1000,
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Try to send the request with retries
        retries = 0
        while retries <= max_retries:
            try:
                start_time = time.time()
                response = client.completions.create(**params)
                end_time = time.time()
                
                # Extract the response text
                response_text = response.completion
                
                # Create a response object
                result = {
                    "text": response_text,
                    "model": model,
                    "response_time": end_time - start_time,
                    "token_usage": {
                        "prompt_tokens": getattr(response, "prompt_tokens", 0),
                        "completion_tokens": getattr(response, "completion_tokens", 0),
                        "total_tokens": getattr(response, "prompt_tokens", 0) + getattr(response, "completion_tokens", 0)
                    },
                    "raw_response": response
                }
                
                return result
                
            except Exception as e:
                retries += 1
                logger.warning(f"Error sending prompt to Anthropic (attempt {retries}/{max_retries}): {str(e)}")
                
                if retries <= max_retries:
                    # Exponential backoff
                    time.sleep(2 ** retries)
                else:
                    logger.error(f"Failed to send prompt to Anthropic after {max_retries} attempts")
                    raise
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Estimate the number of tokens in a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: ~4 characters per token
        return len(prompt) // 4 + 1
    
    def estimate_cost(self, prompt: str, model: str = "claude-haiku") -> float:
        """
        Estimate the cost of a prompt.
        
        Args:
            prompt: The user prompt
            model: The model to use
            
        Returns:
            Estimated cost in USD
        """
        # Estimate tokens
        tokens = self.estimate_tokens(prompt)
        
        # Cost per 1M tokens (as of July 2025)
        # These are approximate and may need to be updated
        costs = {
            "claude-v1": 8.0,  # $8.00 per 1M tokens
            "claude-haiku": 0.25  # $0.25 per 1M tokens
        }
        
        # Get the cost for the model or use a default
        cost_per_1m = costs.get(model, 1.0)
        
        # Calculate the cost
        # Assuming response is about the same length as the prompt
        total_tokens = tokens * 2
        return (total_tokens / 1000000) * cost_per_1m


# For testing
if __name__ == "__main__":
    # Test the Anthropic client
    # Note: This requires an Anthropic API key to be set in the environment
    client = AnthropicClient()
    
    # Check if API key is available
    if not client.api_key:
        print("No Anthropic API key found. Skipping API test.")
    else:
        try:
            # Test sending a prompt
            prompt = "What is the capital of France?"
            print(f"Sending prompt to Anthropic: '{prompt}'")
            
            response = client.send_prompt(prompt, model="claude-haiku")
            
            print(f"Response: {response['text']}")
            print(f"Model: {response['model']}")
            print(f"Response time: {response['response_time']:.2f} seconds")
            print(f"Token usage: {response['token_usage']}")
            
            # Test cost estimation
            estimated_cost = client.estimate_cost(prompt, model="claude-haiku")
            print(f"Estimated cost: ${estimated_cost:.8f}")
            
        except Exception as e:
            print(f"Error testing Anthropic client: {str(e)}")
    
    # Test token estimation (doesn't require API key)
    test_prompts = [
        "Hello, world!",
        "What is the capital of France?",
        "Write a short story about a robot learning to feel emotions."
    ]
    
    print("\nTesting token estimation:")
    for prompt in test_prompts:
        tokens = client.estimate_tokens(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Estimated tokens: {tokens}")
        print(f"Estimated cost (claude-haiku): ${client.estimate_cost(prompt, 'claude-haiku'):.8f}")
        print(f"Estimated cost (claude-v1): ${client.estimate_cost(prompt, 'claude-v1'):.8f}")
        print()

print("Anthropic client implemented successfully!")