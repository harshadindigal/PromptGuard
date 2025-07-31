
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: verify_setup.py
# execution: true
import os

# Check if setup.py exists
if os.path.exists('setup.py'):
    # Read the content of setup.py
    with open('setup.py', 'r') as f:
        content = f.read()
    
    # Print the first few lines
    print("setup.py exists with content:")
    print("\n".join(content.split('\n')[:10]) + "\n...")
    
    # Check if key components are present
    required_components = [
        "name=\"promptshield\"",
        "version=",
        "description=",
        "install_requires=",
        "entry_points=",
        "console_scripts",
        "promptshield=promptshield.cli:main"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print("\nMissing components in setup.py:")
        for component in missing_components:
            print(f"- {component}")
    else:
        print("\nAll required components are present in setup.py.")
else:
    print("setup.py does not exist.")

print("\nCreating setup.py file...")

# Create setup.py
setup_content = """from setuptools import setup, find_packages

setup(
    name="promptshield",
    version="1.0.0",
    description="A middleware system that intercepts user prompts before they reach LLMs and intelligently filters, classifies, and routes queries.",
    author="PromptShield Team",
    author_email="info@promptshield.ai",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "redis>=4.5.0",
        "PyYAML>=6.0",
        "transformers>=4.30.0",
        "openai>=0.27.0",
        "anthropic>=0.3.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "streamlit>=1.22.0"
    ],
    entry_points={
        'console_scripts': [
            'promptshield=promptshield.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
"""

with open('setup.py', 'w') as f:
    f.write(setup_content)

print("Setup file created successfully!")