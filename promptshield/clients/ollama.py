
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/clients/ollama.py
# execution: true
import os
import json
import logging
import requests
from typing import Dict, Any, Optional
import time

# Set up logging
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for the Ollama API.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
    
    def send_prompt(self, prompt: str, model: str = "llama3-70b", max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Send a prompt to the Ollama API.
        
        Args:
            prompt: The user prompt
            model: The model to use (default: llama3-70b)
            max_retries: Maximum number of retries on failure
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response
        """
        # API endpoint
        url = f"{self.base_url}/api/generate"
        
        # Default parameters
        params = {
            "model": model,
            "prompt": prompt,
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
                response_text = response_data.get("response", "")
                
                # Create a response object
                result = {
                    "text": response_text,
                    "model": model,
                    "response_time": end_time - start_time,
                    "token_usage": {
                        "prompt_tokens": response_data.get("prompt_eval_count", 0),
                        "completion_tokens": response_data.get("eval_count", 0),
                        "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
                    },
                    "raw_response": response_data
                }
                
                return result
                
            except requests.exceptions.RequestException as e:
                retries += 1
                logger.warning(f"Error sending prompt to Ollama (attempt {retries}/{max_retries}): {str(e)}")
                
                if retries <= max_retries:
                    # Exponential backoff
                    time.sleep(2 ** retries)
                else:
                    logger.error(f"Failed to send prompt to Ollama after {max_retries} attempts")
                    raise
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            List of available models
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {str(e)}")
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
    
    def estimate_cost(self, prompt: str, model: str = "llama3-70b") -> float:
        """
        Estimate the cost of a prompt.
        
        Args:
            prompt: The user prompt
            model: The model to use
            
        Returns:
            Estimated cost in USD (always 0 for Ollama as it's self-hosted)
        """
        # Ollama is self-hosted, so the cost is 0
        return 0.0


# For testing
if __name__ == "__main__":
    # Test the Ollama client
    client = OllamaClient()
    
    try:
        # List available models
        print("Listing available models...")
        models = client.list_models()
        print(f"Available models: {json.dumps(models, indent=2)}")
        
        # Test sending a prompt
        prompt = "What is the capital of France?"
        print(f"\nSending prompt to Ollama: '{prompt}'")
        
        # Check if llama3-70b is available, otherwise use the first available model
        available_models = models.get("models", [])
        model_names = [model.get("name") for model in available_models]
        
        if model_names:
            if "llama3-70b" in model_names:
                model = "llama3-70b"
            else:
                model = model_names[0]
                print(f"Using available model: {model}")
            
            response = client.send_prompt(prompt, model=model)
            
            print(f"Response: {response['text']}")
            print(f"Model: {response['model']}")
            print(f"Response time: {response['response_time']:.2f} seconds")
            print(f"Token usage: {response['token_usage']}")
        else:
            print("No models available. Make sure Ollama is running and has models installed.")
        
    except Exception as e:
        print(f"Error testing Ollama client: {str(e)}")
        print("\nOllama may not be running. You can install and run it from https://ollama.ai/")
    
    # Test token estimation (doesn't require Ollama to be running)
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
        print(f"Estimated cost (always 0 for self-hosted Ollama): ${client.estimate_cost(prompt):.2f}")
        print()

print("Ollama client implemented successfully!")