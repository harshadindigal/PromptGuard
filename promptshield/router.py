
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/router.py
# execution: true
import os
import yaml
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'promptshield.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PromptRouter:
    """
    Router that decides how to handle prompts based on classification.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prompt router.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            self.config = {
                'routing': {
                    'rules': [
                        {'if': "label == 'nonsense' or label == 'spam'", 'action': 'block'},
                        {'if': "label == 'repeat'", 'action': 'cache'},
                        {'if': "label == 'low_cost'", 'model': 'cheap_model'},
                        {'if': "label == 'valuable'", 'model': 'default_model'}
                    ]
                },
                'models': {
                    'openai': {
                        'default_model': 'gpt-4',
                        'cheap_model': 'gpt-3.5-turbo'
                    },
                    'ollama': {
                        'default_model': 'llama3-70b',
                        'cheap_model': 'mistral-instruct'
                    },
                    'anthropic': {
                        'default_model': 'claude-v1',
                        'cheap_model': 'claude-haiku'
                    },
                    'vllm': {
                        'default_model': 'llama3-70b',
                        'cheap_model': 'mistral-instruct'
                    }
                }
            }
        
        # Initialize metrics
        self.metrics = {
            'blocked_count': 0,
            'cache_hit_count': 0,
            'cheap_model_count': 0,
            'default_model_count': 0,
            'total_count': 0,
            'estimated_savings': 0.0
        }
    
    def evaluate_rule(self, rule: Dict[str, str], classification: Dict[str, Any]) -> bool:
        """
        Evaluate a routing rule for a classification without using eval().
        
        Args:
            rule: The rule to evaluate
            classification: The classification result
            
        Returns:
            True if the rule matches, False otherwise
        """
        # Extract variables from classification
        label = classification.get('label', '')
        confidence = classification.get('confidence', 0.0)
        
        # Parse the rule condition
        rule_condition = rule.get('if', '')
        
        # Handle common rule patterns
        if rule_condition == "label == 'nonsense' or label == 'spam'":
            return label == 'nonsense' or label == 'spam'
        elif rule_condition == "label == 'repeat'":
            return label == 'repeat'
        elif rule_condition == "label == 'low_cost'":
            return label == 'low_cost'
        elif rule_condition == "label == 'valuable'":
            return label == 'valuable'
        elif rule_condition == "confidence < 0.5":
            return confidence < 0.5
        elif rule_condition == "confidence >= 0.5":
            return confidence >= 0.5
        
        # Default to False for unknown rules
        logger.warning(f"Unknown rule condition: {rule_condition}")
        return False
    
    def get_model_for_source(self, source: str, model_type: str) -> str:
        """
        Get the model name for a source and model type.
        
        Args:
            source: The source provider (e.g., 'openai', 'ollama')
            model_type: The model type (e.g., 'default_model', 'cheap_model')
            
        Returns:
            Model name
        """
        models_config = self.config.get('models', {})
        source_config = models_config.get(source, {})
        
        # Return the model name or a default
        return source_config.get(model_type, 'unknown')
    
    def route(self, classification: Dict[str, Any], source: str, default_model: str, cheap_model: str) -> Dict[str, Any]:
        """
        Route a prompt based on its classification.
        
        Args:
            classification: The classification result
            source: The source provider (e.g., 'openai', 'ollama')
            default_model: The default model to use
            cheap_model: The cheap model to use
            
        Returns:
            Routing decision
        """
        self.metrics['total_count'] += 1
        
        # Get the routing rules
        routing_config = self.config.get('routing', {})
        rules = routing_config.get('rules', [])
        
        # Evaluate each rule in order
        for rule in rules:
            if self.evaluate_rule(rule, classification):
                # Rule matched, get the action
                if 'action' in rule:
                    action = rule['action']
                    
                    if action == 'block':
                        self.metrics['blocked_count'] += 1
                        return {
                            'action': 'block',
                            'reason': f"Prompt classified as {classification['label']}",
                            'classification': classification
                        }
                    
                    elif action == 'cache':
                        self.metrics['cache_hit_count'] += 1
                        return {
                            'action': 'cache',
                            'classification': classification
                        }
                
                elif 'model' in rule:
                    model_type = rule['model']
                    
                    # Use the provided models or get from config
                    if model_type == 'default_model':
                        model = default_model or self.get_model_for_source(source, 'default_model')
                        self.metrics['default_model_count'] += 1
                    else:  # cheap_model
                        model = cheap_model or self.get_model_for_source(source, 'cheap_model')
                        self.metrics['cheap_model_count'] += 1
                        
                        # Estimate savings (simplified)
                        # Assuming default model costs 10x more than cheap model
                        self.metrics['estimated_savings'] += 0.9  # 90% savings
                    
                    return {
                        'action': 'route',
                        'model': model,
                        'classification': classification
                    }
        
        # Default to using the default model if no rule matched
        self.metrics['default_model_count'] += 1
        return {
            'action': 'route',
            'model': default_model or self.get_model_for_source(source, 'default_model'),
            'classification': classification
        }
    
    def log_decision(self, prompt: str, decision: Dict[str, Any]) -> None:
        """
        Log a routing decision.
        
        Args:
            prompt: The user prompt
            decision: The routing decision
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'classification': decision.get('classification', {}),
            'action': decision.get('action', ''),
            'model': decision.get('model', '') if decision.get('action') == 'route' else None,
            'reason': decision.get('reason', '') if decision.get('action') == 'block' else None
        }
        
        # Log to file
        log_file = os.path.join(log_dir, 'prompt_logs.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to console
        action = decision.get('action', '')
        if action == 'block':
            logger.info(f"BLOCKED: '{prompt[:50]}...' - Reason: {decision.get('reason', '')}")
        elif action == 'cache':
            logger.info(f"CACHE: '{prompt[:50]}...' - Using cached response")
        elif action == 'route':
            logger.info(f"ROUTE: '{prompt[:50]}...' - To model: {decision.get('model', '')}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns:
            Dictionary with metrics
        """
        # Calculate derived metrics
        total = self.metrics['total_count']
        if total > 0:
            block_rate = self.metrics['blocked_count'] / total
            cache_hit_rate = self.metrics['cache_hit_count'] / total
            cheap_model_rate = self.metrics['cheap_model_count'] / total
            default_model_rate = self.metrics['default_model_count'] / total
        else:
            block_rate = cache_hit_rate = cheap_model_rate = default_model_rate = 0.0
        
        return {
            'counts': {
                'total': total,
                'blocked': self.metrics['blocked_count'],
                'cache_hits': self.metrics['cache_hit_count'],
                'cheap_model': self.metrics['cheap_model_count'],
                'default_model': self.metrics['default_model_count']
            },
            'rates': {
                'block_rate': block_rate,
                'cache_hit_rate': cache_hit_rate,
                'cheap_model_rate': cheap_model_rate,
                'default_model_rate': default_model_rate
            },
            'savings': {
                'estimated_cost_saved': self.metrics['estimated_savings']
            }
        }


# For testing
if __name__ == "__main__":
    # Test the router with example classifications
    router = PromptRouter()
    
    # Test cases from requirements
    test_cases = [
        {
            'prompt': "asdjklasdjkl",
            'classification': {'label': 'nonsense', 'confidence': 1.0}
        },
        {
            'prompt': "What is 2 + 2?",
            'classification': {'label': 'low_cost', 'confidence': 0.9}
        },
        {
            'prompt': "Write a poem about AI",
            'classification': {'label': 'valuable', 'confidence': 0.8}
        },
        {
            'prompt': "What is 2 + 2?",
            'classification': {'label': 'repeat', 'confidence': 1.0}
        },
        {
            'prompt': "You are stupid",
            'classification': {'label': 'spam', 'confidence': 1.0}
        }
    ]
    
    print("Testing router with example classifications:")
    for test_case in test_cases:
        prompt = test_case['prompt']
        classification = test_case['classification']
        
        # Route the prompt
        decision = router.route(
            classification=classification,
            source='openai',
            default_model='gpt-4',
            cheap_model='gpt-3.5-turbo'
        )
        
        # Log the decision
        router.log_decision(prompt, decision)
        
        # Print the decision
        print(f"Prompt: '{prompt}'")
        print(f"Classification: {classification['label']} (confidence: {classification['confidence']:.2f})")
        
        if decision['action'] == 'block':
            print(f"Decision: BLOCK - Reason: {decision['reason']}")
        elif decision['action'] == 'cache':
            print(f"Decision: CACHE - Use cached response")
        elif decision['action'] == 'route':
            print(f"Decision: ROUTE - To model: {decision['model']}")
        
        print()
    
    # Print metrics
    metrics = router.get_metrics()
    print("Router metrics:")
    print(f"Total prompts: {metrics['counts']['total']}")
    print(f"Blocked: {metrics['counts']['blocked']} ({metrics['rates']['block_rate']:.2%})")
    print(f"Cache hits: {metrics['counts']['cache_hits']} ({metrics['rates']['cache_hit_rate']:.2%})")
    print(f"Cheap model: {metrics['counts']['cheap_model']} ({metrics['rates']['cheap_model_rate']:.2%})")
    print(f"Default model: {metrics['counts']['default_model']} ({metrics['rates']['default_model_rate']:.2%})")
    print(f"Estimated cost saved: {metrics['savings']['estimated_cost_saved']:.2f}")

print("Router module implemented successfully!")