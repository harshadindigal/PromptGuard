
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/cli.py
# execution: true
import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, Any, List, Optional
import csv
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def classify_prompt(prompt: str, session_id: str = "") -> Dict[str, Any]:
    """
    Classify a prompt using the PromptShield classifier.
    
    Args:
        prompt: The user prompt
        session_id: Session identifier (optional)
        
    Returns:
        Classification result
    """
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

def route_prompt(prompt: str, session_id: str, source: str, 
                default_model: str, cheap_model: str) -> Dict[str, Any]:
    """
    Route a prompt using the PromptShield router.
    
    Args:
        prompt: The user prompt
        session_id: Session identifier
        source: Source provider (e.g., 'openai', 'ollama', 'vllm', 'anthropic')
        default_model: Default model to use
        cheap_model: Cheap model to use
        
    Returns:
        Routing decision
    """
    try:
        # Try absolute imports first
        from promptshield.classifier import PromptClassifier
        from promptshield.router import PromptRouter
    except ImportError:
        # Fall back to relative imports
        from classifier import PromptClassifier
        from router import PromptRouter
    
    # Create instances
    classifier = PromptClassifier()
    router = PromptRouter()
    
    # Classify the prompt
    classification = classifier.classify(prompt, session_id)
    
    # Route the prompt
    routing_decision = router.route(classification, source, default_model, cheap_model)
    
    # Log the decision
    router.log_decision(prompt, routing_decision)
    
    return {
        "classification": classification,
        "routing": routing_decision
    }

def process_file(file_path: str, output_path: Optional[str] = None, 
                session_id: str = "", source: str = "openai",
                default_model: str = "gpt-4", cheap_model: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
    """
    Process a file of prompts.
    
    Args:
        file_path: Path to the file
        output_path: Path to the output file (optional)
        session_id: Session identifier
        source: Source provider
        default_model: Default model to use
        cheap_model: Cheap model to use
        
    Returns:
        List of results
    """
    results = []
    
    # Determine the file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        # JSON file
        with open(file_path, 'r') as f:
            prompts = json.load(f)
            
            if isinstance(prompts, list):
                # List of prompts
                for i, prompt in enumerate(prompts):
                    if isinstance(prompt, str):
                        # Process the prompt
                        result = route_prompt(prompt, f"{session_id}_{i}", source, default_model, cheap_model)
                        result["prompt"] = prompt
                        results.append(result)
                    elif isinstance(prompt, dict) and "prompt" in prompt:
                        # Process the prompt
                        result = route_prompt(prompt["prompt"], f"{session_id}_{i}", source, default_model, cheap_model)
                        result["prompt"] = prompt["prompt"]
                        results.append(result)
            elif isinstance(prompts, dict) and "prompts" in prompts:
                # Dictionary with a list of prompts
                for i, prompt in enumerate(prompts["prompts"]):
                    if isinstance(prompt, str):
                        # Process the prompt
                        result = route_prompt(prompt, f"{session_id}_{i}", source, default_model, cheap_model)
                        result["prompt"] = prompt
                        results.append(result)
                    elif isinstance(prompt, dict) and "prompt" in prompt:
                        # Process the prompt
                        result = route_prompt(prompt["prompt"], f"{session_id}_{i}", source, default_model, cheap_model)
                        result["prompt"] = prompt["prompt"]
                        results.append(result)
    
    elif file_ext == '.csv':
        # CSV file
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            if header:
                # Find the prompt column
                prompt_col = None
                for i, col in enumerate(header):
                    if col.lower() == 'prompt':
                        prompt_col = i
                        break
                
                if prompt_col is not None:
                    # Process each row
                    for i, row in enumerate(reader):
                        if prompt_col < len(row):
                            prompt = row[prompt_col]
                            result = route_prompt(prompt, f"{session_id}_{i}", source, default_model, cheap_model)
                            result["prompt"] = prompt
                            results.append(result)
    
    elif file_ext == '.txt':
        # Text file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    result = route_prompt(line, f"{session_id}_{i}", source, default_model, cheap_model)
                    result["prompt"] = line
                    results.append(result)
    
    # Write the results to the output file if specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the results of batch processing.
    
    Args:
        results: List of results
        
    Returns:
        Analysis results
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "classifications": {},
            "routing": {}
        }
    
    # Count classifications
    classifications = {}
    for result in results:
        label = result["classification"]["label"]
        classifications[label] = classifications.get(label, 0) + 1
    
    # Count routing decisions
    routing = {}
    for result in results:
        action = result["routing"]["action"]
        routing[action] = routing.get(action, 0) + 1
    
    # Calculate percentages
    classification_pct = {label: count / total * 100 for label, count in classifications.items()}
    routing_pct = {action: count / total * 100 for action, count in routing.items()}
    
    return {
        "total": total,
        "classifications": {
            "counts": classifications,
            "percentages": classification_pct
        },
        "routing": {
            "counts": routing,
            "percentages": routing_pct
        }
    }

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="PromptShield CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a prompt")
    classify_parser.add_argument("prompt", help="The prompt to classify")
    classify_parser.add_argument("--session-id", help="Session identifier", default="cli")
    
    # Route command
    route_parser = subparsers.add_parser("route", help="Route a prompt")
    route_parser.add_argument("prompt", help="The prompt to route")
    route_parser.add_argument("--session-id", help="Session identifier", default="cli")
    route_parser.add_argument("--source", help="Source provider", default="openai")
    route_parser.add_argument("--default-model", help="Default model to use", default="gpt-4")
    route_parser.add_argument("--cheap-model", help="Cheap model to use", default="gpt-3.5-turbo")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a file of prompts")
    process_parser.add_argument("file", help="Path to the file")
    process_parser.add_argument("--output", help="Path to the output file")
    process_parser.add_argument("--session-id", help="Session identifier", default="cli")
    process_parser.add_argument("--source", help="Source provider", default="openai")
    process_parser.add_argument("--default-model", help="Default model to use", default="gpt-4")
    process_parser.add_argument("--cheap-model", help="Cheap model to use", default="gpt-3.5-turbo")
    
    args = parser.parse_args()
    
    if args.command == "classify":
        # Classify a prompt
        result = classify_prompt(args.prompt, args.session_id)
        print(json.dumps(result, indent=2))
    
    elif args.command == "route":
        # Route a prompt
        result = route_prompt(args.prompt, args.session_id, args.source, args.default_model, args.cheap_model)
        print(json.dumps(result, indent=2))
    
    elif args.command == "process":
        # Process a file of prompts
        results = process_file(args.file, args.output, args.session_id, args.source, args.default_model, args.cheap_model)
        
        # Analyze the results
        analysis = analyze_results(results)
        
        print(f"Processed {analysis['total']} prompts")
        print("\nClassifications:")
        for label, count in analysis["classifications"]["counts"].items():
            print(f"  {label}: {count} ({analysis['classifications']['percentages'][label]:.1f}%)")
        
        print("\nRouting decisions:")
        for action, count in analysis["routing"]["counts"].items():
            print(f"  {action}: {count} ({analysis['routing']['percentages'][action]:.1f}%)")
        
        if args.output:
            print(f"\nResults written to {args.output}")
    
    else:
        parser.print_help()

# For testing
if __name__ == "__main__":
    # Test the CLI with some example prompts
    test_prompts = [
        "asdjklasdjkl",  # nonsense
        "What is 2 + 2?",  # low_cost
        "Write a poem about AI",  # valuable
        "You are stupid",  # spam
    ]
    
    print("Testing CLI with example prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Classify the prompt
        classification = classify_prompt(prompt)
        print(f"Classification: {classification['label']} (confidence: {classification['confidence']:.2f})")
        
        # Route the prompt
        result = route_prompt(prompt, "cli_test", "openai", "gpt-4", "gpt-3.5-turbo")
        
        if result["routing"]["action"] == "block":
            print(f"Routing: BLOCK - Reason: {result['routing']['reason']}")
        elif result["routing"]["action"] == "cache":
            print(f"Routing: CACHE - Use cached response")
        elif result["routing"]["action"] == "route":
            print(f"Routing: ROUTE - To model: {result['routing']['model']}")
    
    # Create a temporary file with test prompts
    temp_file = "test_prompts.json"
    with open(temp_file, 'w') as f:
        json.dump(test_prompts, f)
    
    print("\nTesting batch processing:")
    results = process_file(temp_file, None, "cli_test", "openai", "gpt-4", "gpt-3.5-turbo")
    analysis = analyze_results(results)
    
    print(f"Processed {analysis['total']} prompts")
    print("\nClassifications:")
    for label, count in analysis["classifications"]["counts"].items():
        print(f"  {label}: {count} ({analysis['classifications']['percentages'][label]:.1f}%)")
    
    print("\nRouting decisions:")
    for action, count in analysis["routing"]["counts"].items():
        print(f"  {action}: {count} ({analysis['routing']['percentages'][action]:.1f}%)")
    
    # Note: We're not removing the temp file as that's not allowed
    print(f"\nNote: Test file '{temp_file}' has been created for demonstration purposes.")

print("CLI implemented successfully!")