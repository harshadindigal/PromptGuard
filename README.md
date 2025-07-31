# PromptShield

A modular Python middleware system that intercepts user prompts before they reach large language models (LLMs) and intelligently filters, classifies, and routes queries to reduce wasted compute resources and costs.

## Features

- **Prompt Classification**: Hybrid rules + ML model approach to classify prompts as nonsense, spam, repeat, low-cost, or valuable
- **Query Routing**: Intelligent routing to block, cache, or send to appropriate model based on classification
- **Cost and Usage Analytics**: Track metrics on blocked prompts, model usage, and estimated cost savings
- **Flexible Deployment**: Works with both closed API models and self-hosted open-weight models
- **Developer Integrations**: Python SDK and CLI tools for easy integration

## Supported Models

### Closed API Models
- OpenAI GPT-3.5/4
- Anthropic Claude

### Self-hosted Open-weight Models
- LLaMA via Ollama
- vLLM/TGI

## Installation

```bash
pip install promptshield
```

Or install from source:

```bash
git clone https://github.com/harshadindigal/PromptGuard.git
cd PromptGuard
pip install -e .
```

## Configuration

PromptShield uses a YAML configuration file. Here's a sample configuration:

```yaml
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
```

## Usage

### API Server

Start the API server:

```bash
uvicorn promptshield.main:app --host 0.0.0.0 --port 8080
```

Send a request to the API:

```python
import requests

response = requests.post(
    "http://localhost:8080/chat",
    json={
        "prompt": "What is the capital of France?",
        "session_id": "user123",
        "config": {
            "source": "openai",
            "default_model": "gpt-4",
            "cheap_model": "gpt-3.5-turbo"
        }
    }
)

print(response.json())
```

### Python SDK

```python
from promptshield.sdk import PromptShieldSDK

# Initialize the SDK
sdk = PromptShieldSDK(api_url="http://localhost:8080")

# Classify a prompt
classification = sdk.classify_prompt("What is the capital of France?")
print(classification)

# Route a prompt
response = sdk.route_prompt(
    prompt="What is the capital of France?",
    session_id="user123",
    source="openai",
    default_model="gpt-4",
    cheap_model="gpt-3.5-turbo"
)
print(response)
```

### CLI

```bash
# Classify a prompt
promptshield classify "What is the capital of France?"

# Route a prompt
promptshield route "What is the capital of France?" --source openai --default-model gpt-4 --cheap-model gpt-3.5-turbo

# Process a file of prompts
promptshield process prompts.json --output results.json
```

## License

MIT
