
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/clients/openai.py
# execution: true
import os
import json
import logging
from typing import Dict, Any, Optional, List
import time

# Set up logging
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for the OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set the OPENAI_API_KEY environment variable or pass it to the constructor.")
        
        # Initialize the OpenAI client lazily
        self.client = None
    
    def _get_client(self):
        """
        Get the OpenAI client, initializing it if necessary.
        
        Returns:
            OpenAI client
        """
        if self.client is None:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package is required. Install with 'pip install openai'.")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                raise
        
        return self.client
    
    def send_prompt(self, prompt: str, model: str = "gpt-3.5-turbo", max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Send a prompt to the OpenAI API.
        
        Args:
            prompt: The user prompt
            model: The model to use (default: gpt-3.5-turbo)
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response
        """
        client = self._get_client()
        
        # Default parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Try to send the request with retries
        retries = 0
        while retries <= max_retries:
            try:
                start_time = time.time()
                response = client.chat.completions.create(**params)
                end_time = time.time()
                
                # Extract the response text
                response_text = response.choices[0].message.content
                
                # Create a response object
                result = {
                    "text": response_text,
                    "model": model,
                    "response_time": end_time - start_time,
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "raw_response": response.model_dump()
                }
                
                return result
                
            except Exception as e:
                retries += 1
                logger.warning(f"Error sending prompt to OpenAI (attempt {retries}/{max_retries}): {str(e)}")
                
                if retries <= max_retries:
                    # Exponential backoff
                    time.sleep(2 ** retries)
                else:
                    logger.error(f"Failed to send prompt to OpenAI after {max_retries} attempts")
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
    
    def estimate_cost(self, prompt: str, model: str = "gpt-3.5-turbo") -> float:
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
        
        # Cost per 1K tokens (as of July 2025)
        # These are approximate and may need to be updated
        costs = {
            "gpt-4": 0.03,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.002  # $0.002 per 1K tokens
        }
        
        # Get the cost for the model or use a default
        cost_per_1k = costs.get(model, 0.01)
        
        # Calculate the cost
        # Assuming response is about the same length as the prompt
        total_tokens = tokens * 2
        return (total_tokens / 1000) * cost_per_1k


# For testing
if __name__ == "__main__":
    # Test the OpenAI client
    # Note: This requires an OpenAI API key to be set in the environment
    client = OpenAIClient()
    
    # Check if API key is available
    if not client.api_key:
        print("No OpenAI API key found. Skipping API test.")
    else:
        try:
            # Test sending a prompt
            prompt = "What is the capital of France?"
            print(f"Sending prompt to OpenAI: '{prompt}'")
            
            response = client.send_prompt(prompt, model="gpt-3.5-turbo")
            
            print(f"Response: {response['text']}")
            print(f"Model: {response['model']}")
            print(f"Response time: {response['response_time']:.2f} seconds")
            print(f"Token usage: {response['token_usage']}")
            
            # Test cost estimation
            estimated_cost = client.estimate_cost(prompt, model="gpt-3.5-turbo")
            print(f"Estimated cost: ${estimated_cost:.6f}")
            
        except Exception as e:
            print(f"Error testing OpenAI client: {str(e)}")
    
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
        print(f"Estimated cost (gpt-3.5-turbo): ${client.estimate_cost(prompt, 'gpt-3.5-turbo'):.6f}")
        print(f"Estimated cost (gpt-4): ${client.estimate_cost(prompt, 'gpt-4'):.6f}")
        print()

print("OpenAI client implemented successfully!")