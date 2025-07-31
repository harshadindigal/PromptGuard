
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: setup_project_structure.py
# execution: true
import os

# Create the directory structure
directories = [
    "promptshield",
    "promptshield/clients",
    "promptshield/logs",
    "promptshield/dashboard",
    "tests"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Create empty files for the main modules
files = [
    "promptshield/__init__.py",
    "promptshield/main.py",
    "promptshield/router.py",
    "promptshield/classifier.py",
    "promptshield/cache.py",
    "promptshield/sdk.py",
    "promptshield/cli.py",
    "promptshield/clients/__init__.py",
    "promptshield/clients/openai.py",
    "promptshield/clients/anthropic.py",
    "promptshield/clients/ollama.py",
    "promptshield/clients/vllm.py",
    "promptshield/dashboard/__init__.py",
    "tests/__init__.py",
    "tests/test_classifier.py",
    "tests/test_router.py",
    "tests/test_cache.py"
]

for file_path in files:
    with open(file_path, 'w') as f:
        pass  # Create empty file
    print(f"Created file: {file_path}")

# Create a basic README.md
with open("README.md", 'w') as f:
    f.write("# PromptShield\n\n")
    f.write("A modular Python middleware system that intercepts user prompts before they reach large language models (LLMs) and intelligently filters, classifies, and routes queries to reduce wasted compute resources and costs.\n\n")
    f.write("## Features\n\n")
    f.write("- Prompt classification (nonsense/spam/low-value vs valuable)\n")
    f.write("- Query routing (block, cache, cheap model, main model)\n")
    f.write("- Cost and usage analytics\n")
    f.write("- Flexible deployment and developer integrations\n")
    f.write("- Support for both closed API models and self-hosted open-weight models\n")
print("Created README.md")

# Create requirements.txt
with open("requirements.txt", 'w') as f:
    f.write("fastapi>=0.95.0\n")
    f.write("uvicorn>=0.22.0\n")
    f.write("pydantic>=2.0.0\n")
    f.write("redis>=4.5.0\n")
    f.write("PyYAML>=6.0\n")
    f.write("transformers>=4.30.0\n")
    f.write("openai>=0.27.0\n")
    f.write("anthropic>=0.3.0\n")
    f.write("requests>=2.28.0\n")
    f.write("python-dotenv>=1.0.0\n")
    f.write("click>=8.1.0\n")
    f.write("streamlit>=1.22.0\n")
print("Created requirements.txt")

# Create a basic config.yaml
with open("config.yaml", 'w') as f:
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
    f.write("  vllm:\n")
    f.write("    default_model: \"llama3-70b\"\n")
    f.write("    cheap_model: \"mistral-instruct\"\n\n")
    f.write("cache:\n")
    f.write("  type: \"memory\"  # Options: memory, redis\n")
    f.write("  ttl: 600  # Time to live in seconds (10 minutes)\n")
    f.write("  max_size: 1000  # Maximum number of entries in memory cache\n\n")
    f.write("classifier:\n")
    f.write("  use_ml_model: true\n")
    f.write("  model_name: \"distilbert-base-uncased\"\n")
    f.write("  min_words: 4  # Block if prompt length < min_words\n")
    f.write("  confidence_threshold: 0.7  # Minimum confidence for classification\n")
print("Created config.yaml")

print("Project structure setup complete!")