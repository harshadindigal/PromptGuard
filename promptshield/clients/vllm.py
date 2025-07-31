
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/clients/vllm.py
# execution: true
import os
import json
import logging
import requests
from typing import Dict, Any
import time

# Set up logging
logger = logging.getLogger(__name__)

class VLLMClient:
    """
    Client for the vLLM API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: Base URL for the vLLM API (default: http://localhost:8000)
        """
        self.base_url = base_url
    
    def send_prompt(self, prompt: str, model: str = "llama3-70b", max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Send a prompt to the vLLM API.
        
        Args:
            prompt: The user prompt
            model: The model to use (default: llama3-70b)
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response
        """
        # API endpoint
        url = f"{self.base_url}/generate"
        
        # Default parameters
        params = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": False
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Try to send the request with retries
        retries = 0
        while retries <= max_retries:
            try:
                start_time = time.time()
                response = requests.post(url, json=params)
                response.raise_for_status()  # Raise exception for HTTP errors
                end_time = time.time()
                
                # Parse the response
                response_data = response.json()
                
                # Extract the response text
                # The structure might vary depending on the vLLM server configuration
                if "text" in response_data:
                    response_text = response_data["text"]
                elif "outputs" in response_data and len(response_data["outputs"]) > 0:
                    response_text = response_data["outputs"][0].get("text", "")
                else:
                    response_text = str(response_data)  # Fallback
                
                # Create a response object
                result = {
                    "text": response_text,
                    "model": model,
                    "response_time": end_time - start_time,
                    "token_usage": {
                        # vLLM might not provide token counts in the response
                        "prompt_tokens": response_data.get("prompt_tokens", self.estimate_tokens(prompt)),
                        "completion_tokens": response_data.get("completion_tokens", self.estimate_tokens(response_text)),
                        "total_tokens": response_data.get("total_tokens", 
                                                         self.estimate_tokens(prompt) + self.estimate_tokens(response_text))
                    },
                    "raw_response": response_data
                }
                
                return result
                
            except requests.exceptions.RequestException as e:
                retries += 1
                logger.warning(f"Error sending prompt to vLLM (attempt {retries}/{max_retries}): {str(e)}")
                
                if retries <= max_retries:
                    # Exponential backoff
                    time.sleep(2 ** retries)
                else:
                    logger.error(f"Failed to send prompt to vLLM after {max_retries} attempts")
                    raise
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Model information
        """
        url = f"{self.base_url}/models"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting models: {str(e)}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4 + 1
    
    def estimate_cost(self, prompt: str, model: str = "llama3-70b") -> float:
        """
        Estimate the cost of a prompt.
        
        Args:
            prompt: The user prompt
            model: The model to use
            
        Returns:
            Estimated cost in USD (always 0 for vLLM as it's self-hosted)
        """
        # vLLM is self-hosted, so the cost is 0
        return 0.0


# For testing
if __name__ == "__main__":
    # Test the vLLM client
    client = VLLMClient()
    
    try:
        # Get model information
        print("Getting model information...")
        models = client.get_models()
        print(f"Model information: {json.dumps(models, indent=2)}")
        
        # Test sending a prompt
        prompt = "What is the capital of France?"
        print(f"\nSending prompt to vLLM: '{prompt}'")
        
        response = client.send_prompt(prompt)
        
        print(f"Response: {response['text']}")
        print(f"Model: {response['model']}")
        print(f"Response time: {response['response_time']:.2f} seconds")
        print(f"Token usage: {response['token_usage']}")
        
    except Exception as e:
        print(f"Error testing vLLM client: {str(e)}")
        print("\nvLLM may not be running. You can install and run it from https://github.com/vllm-project/vllm")
    
    # Test token estimation (doesn't require vLLM to be running)
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
        print(f"Estimated cost (always 0 for self-hosted vLLM): ${client.estimate_cost(prompt):.2f}")
        print()

print("vLLM client implemented successfully!")