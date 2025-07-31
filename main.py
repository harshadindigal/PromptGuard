
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/main.py
# execution: true
import os
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import time
from datetime import datetime

# Import our modules
from promptshield.classifier import PromptClassifier
from promptshield.router import PromptRouter
from promptshield.cache import get_cache_from_config

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define request and response models
class Config(BaseModel):
    source: str = Field(..., description="Source provider (e.g., 'openai', 'ollama', 'vllm', 'anthropic')")
    default_model: str = Field(..., description="Default model to use")
    cheap_model: str = Field(..., description="Cheap model to use")

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    session_id: str = Field(..., description="Session identifier")
    config: Config = Field(..., description="Configuration")

class ChatResponse(BaseModel):
    text: Optional[str] = Field(None, description="Response text")
    blocked: bool = Field(False, description="Whether the prompt was blocked")
    block_reason: Optional[str] = Field(None, description="Reason for blocking")
    classification: Dict[str, Any] = Field({}, description="Classification result")
    routing: Dict[str, Any] = Field({}, description="Routing decision")
    model_used: Optional[str] = Field(None, description="Model used for response")
    response_time: float = Field(0.0, description="Response time in seconds")
    cached: bool = Field(False, description="Whether the response was cached")

# Initialize the app
app = FastAPI(
    title="PromptShield API",
    description="A middleware system that intercepts user prompts before they reach LLMs and intelligently filters, classifies, and routes queries.",
    version="1.0.0"
)

# Initialize components
classifier = PromptClassifier()
router = PromptRouter()
cache = get_cache_from_config()

# Client factory
def get_client(source: str):
    """
    Get a client for the specified source.
    
    Args:
        source: Source provider (e.g., 'openai', 'ollama', 'vllm', 'anthropic')
        
    Returns:
        Client instance
    """
    if source == "openai":
        from promptshield.clients.openai import OpenAIClient
        return OpenAIClient()
    elif source == "anthropic":
        from promptshield.clients.anthropic import AnthropicClient
        return AnthropicClient()
    elif source == "ollama":
        from promptshield.clients.ollama import OllamaClient
        return OllamaClient()
    elif source == "vllm":
        from promptshield.clients.vllm import VLLMClient
        return VLLMClient()
    else:
        raise ValueError(f"Unsupported source: {source}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat request.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response
    """
    start_time = time.time()
    
    try:
        # Extract request data
        prompt = request.prompt
        session_id = request.session_id
        source = request.config.source
        default_model = request.config.default_model
        cheap_model = request.config.cheap_model
        
        logger.info(f"Received request - Session: {session_id}, Source: {source}, Prompt: '{prompt[:50]}...'")
        
        # Check cache first
        cached_response = cache.get(prompt)
        if cached_response:
            logger.info(f"Cache hit for prompt: '{prompt[:50]}...'")
            end_time = time.time()
            
            # Update cached response with current request info
            cached_response["response_time"] = end_time - start_time
            cached_response["cached"] = True
            
            return JSONResponse(cached_response)
        
        # Classify the prompt
        classification = classifier.classify(prompt, session_id)
        logger.info(f"Classification: {classification['label']} (confidence: {classification['confidence']:.2f})")
        
        # Route the prompt
        routing_decision = router.route(classification, source, default_model, cheap_model)
        logger.info(f"Routing decision: {routing_decision['action']}")
        
        # Log the decision
        router.log_decision(prompt, routing_decision)
        
        # Handle routing decision
        if routing_decision["action"] == "block":
            # Blocked prompt
            response = {
                "text": None,
                "blocked": True,
                "block_reason": routing_decision["reason"],
                "classification": classification,
                "routing": routing_decision,
                "model_used": None,
                "response_time": time.time() - start_time,
                "cached": False
            }
            return JSONResponse(response)
        
        elif routing_decision["action"] == "cache":
            # This should not happen as we already checked the cache
            # But just in case, check again
            cached_response = cache.get(prompt)
            if cached_response:
                logger.info(f"Late cache hit for prompt: '{prompt[:50]}...'")
                end_time = time.time()
                
                # Update cached response with current request info
                cached_response["response_time"] = end_time - start_time
                cached_response["cached"] = True
                
                return JSONResponse(cached_response)
            
            # If we get here, the prompt was classified as a repeat but not found in the cache
            # This could happen if the cache expired between classification and routing
            # Fall through to normal processing
        
        # Get the client for the source
        try:
            client = get_client(source)
        except Exception as e:
            logger.error(f"Error getting client for source '{source}': {str(e)}")
            raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")
        
        # Get the model to use
        model = routing_decision.get("model", default_model)
        
        # Send the prompt to the model
        try:
            model_response = client.send_prompt(prompt, model=model)
        except Exception as e:
            logger.error(f"Error sending prompt to {source} model '{model}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error from {source} API: {str(e)}")
        
        # Create the response
        response = {
            "text": model_response["text"],
            "blocked": False,
            "block_reason": None,
            "classification": classification,
            "routing": routing_decision,
            "model_used": model,
            "response_time": time.time() - start_time,
            "cached": False,
            "token_usage": model_response.get("token_usage", {})
        }
        
        # Cache the response
        cache.set(prompt, response)
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    """
    Get metrics.
    
    Returns:
        Metrics
    """
    return router.get_metrics()

# For testing
if __name__ == "__main__":
    # Create a simple test script that doesn't actually run the server
    print("PromptShield API server implementation:")
    print("- FastAPI app created with /chat endpoint")
    print("- Supports OpenAI, Anthropic, Ollama, and vLLM clients")
    print("- Includes classification, routing, and caching")
    print("- Provides health check and metrics endpoints")
    print("\nTo run the server, execute:")
    print("  uvicorn promptshield.main:app --host 0.0.0.0 --port 8080")
    print("\nAPI Documentation will be available at:")
    print("  http://localhost:8080/docs")

print("API server implemented successfully!")