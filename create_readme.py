
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: create_readme.py
# execution: true
# Create a comprehensive README.md file

with open("README.md", "w") as f:
    f.write("# PromptShield\n\n")
    f.write("A modular Python middleware system that intercepts user prompts before they reach large language models (LLMs) and intelligently filters, classifies, and routes queries to reduce wasted compute resources and costs.\n\n")
    
    f.write("## Features\n\n")
    f.write("- **Prompt Classification**: Hybrid rules + ML model approach to classify prompts as nonsense, spam, repeat, low-cost, or valuable\n")
    f.write("- **Query Routing**: Intelligent routing to block, cache, or send to appropriate model based on classification\n")
    f.write("- **Cost and Usage Analytics**: Track metrics on blocked prompts, model usage, and estimated cost savings\n")
    f.write("- **Flexible Deployment**: Works with both closed API models and self-hosted open-weight models\n")
    f.write("- **Developer Integrations**: Python SDK and CLI tools for easy integration\n\n")
    
    f.write("## Supported Models\n\n")
    f.write("### Closed API Models\n")
    f.write("- OpenAI GPT-3.5/4\n")
    f.write("- Anthropic Claude\n\n")
    
    f.write("### Self-hosted Open-weight Models\n")
    f.write("- LLaMA via Ollama\n")
    f.write("- vLLM/TGI\n\n")
    
    f.write("## Installation\n\n")
    f.write("```bash\n")
    f.write("pip install promptshield\n")
    f.write("```\n\n")
    
    f.write("Or install from source:\n\n")
    f.write("```bash\n")
    f.write("git clone https://github.com/harshadindigal/PromptGuard.git\n")
    f.write("cd PromptGuard\n")
    f.write("pip install -e .\n")
    f.write("```\n\n")
    
    f.write("## Configuration\n\n")
    f.write("PromptShield uses a YAML configuration file. Here's a sample configuration:\n\n")
    f.write("```yaml\n")
    f.write("routing:\n")
    f.write("  rules:\n")
    f.write("    - if: \"label == 'nonsense' or label == 'spam'\"\n")
    f.write("      action: \"block\"\n")
    f.write("    - if: \"label == 'repeat'\"\n")
    f.write("      action: \"cache\"\n")
    f.write("    - if: \"label == 'low_cost'\"\n")
    f.write("      model: \"cheap_model\"\n")
    f.write("    - if: \"label == 'valuable'\"\n")
    f.write("      model: \"default_model\"\n\n")
    
    f.write("models:\n")
    f.write("  openai:\n")
    f.write("    default_model: \"gpt-4\"\n")
    f.write("    cheap_model: \"gpt-3.5-turbo\"\n")
    f.write("  ollama:\n")
    f.write("    default_model: \"llama3-70b\"\n")
    f.write("    cheap_model: \"mistral-instruct\"\n")
    f.write("  anthropic:\n")
    f.write("    default_model: \"claude-v1\"\n")
    f.write("    cheap_model: \"claude-haiku\"\n")
    f.write("```\n\n")
    
    f.write("## Usage\n\n")
    f.write("### API Server\n\n")
    f.write("Start the API server:\n\n")
    f.write("```bash\n")
    f.write("uvicorn promptshield.main:app --host 0.0.0.0 --port 8080\n")
    f.write("```\n\n")
    
    f.write("Send a request to the API:\n\n")
    f.write("```python\n")
    f.write("import requests\n\n")
    f.write("response = requests.post(\n")
    f.write("    \"http://localhost:8080/chat\",\n")
    f.write("    json={\n")
    f.write("        \"prompt\": \"What is the capital of France?\",\n")
    f.write("        \"session_id\": \"user123\",\n")
    f.write("        \"config\": {\n")
    f.write("            \"source\": \"openai\",\n")
    f.write("            \"default_model\": \"gpt-4\",\n")
    f.write("            \"cheap_model\": \"gpt-3.5-turbo\"\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write(")\n\n")
    f.write("print(response.json())\n")
    f.write("```\n\n")
    
    f.write("### Python SDK\n\n")
    f.write("```python\n")
    f.write("from promptshield.sdk import PromptShieldSDK\n\n")
    f.write("# Initialize the SDK\n")
    f.write("sdk = PromptShieldSDK(api_url=\"http://localhost:8080\")\n\n")
    f.write("# Classify a prompt\n")
    f.write("classification = sdk.classify_prompt(\"What is the capital of France?\")\n")
    f.write("print(classification)\n\n")
    f.write("# Route a prompt\n")
    f.write("response = sdk.route_prompt(\n")
    f.write("    prompt=\"What is the capital of France?\",\n")
    f.write("    session_id=\"user123\",\n")
    f.write("    source=\"openai\",\n")
    f.write("    default_model=\"gpt-4\",\n")
    f.write("    cheap_model=\"gpt-3.5-turbo\"\n")
    f.write(")\n")
    f.write("print(response)\n")
    f.write("```\n\n")
    
    f.write("### CLI\n\n")
    f.write("```bash\n")
    f.write("# Classify a prompt\n")
    f.write("promptshield classify \"What is the capital of France?\"\n\n")
    f.write("# Route a prompt\n")
    f.write("promptshield route \"What is the capital of France?\" --source openai --default-model gpt-4 --cheap-model gpt-3.5-turbo\n\n")
    f.write("# Process a file of prompts\n")
    f.write("promptshield process prompts.json --output results.json\n")
    f.write("```\n\n")
    
    f.write("## License\n\n")
    f.write("MIT\n")

print("README.md created successfully!")

# Read and display the first few lines of the README
with open("README.md", "r") as f:
    content = f.read()
    lines = content.split("\n")
    preview = "\n".join(lines[:20])
    print("\nPreview of README.md:")
    print(preview)