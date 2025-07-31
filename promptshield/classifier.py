
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: promptshield/classifier.py
# execution: true
import re
import os
import yaml
from typing import Dict, Any, Tuple, List, Optional
import hashlib

class RuleBasedClassifier:
    """
    Rule-based classifier for prompt classification.
    Uses simple rules to classify prompts without ML.
    """
    
    def __init__(self, min_words: int = 4):
        """
        Initialize the rule-based classifier.
        
        Args:
            min_words: Minimum number of words required (default: 4)
        """
        self.min_words = min_words
        # Common spam/nonsense patterns
        self.spam_patterns = [
            r'\b(stupid|idiot|dumb|fool)\b',  # Simple insults
            r'^\s*[a-z0-9]{1,3}\s*$',  # Very short inputs like "a", "123"
            r'^\s*[a-zA-Z0-9]{20,}\s*$',  # Long string with no spaces (likely gibberish)
            r'^\s*(.)\1{5,}\s*$',  # Repeated characters like "aaaaaa"
        ]
        
        # Simple patterns for low-cost queries
        self.low_cost_patterns = [
            r'^\s*what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\s*$',  # Simple math like "what is 2 + 2"
            r'^\s*define\s+[a-zA-Z]+\s*$',  # Simple definition queries
            r'^\s*who\s+is\s+[a-zA-Z\s]+\s*$',  # Simple who-is queries
            r'^\s*when\s+was\s+[a-zA-Z\s]+\s+(born|invented|created|founded)\s*$',  # Simple date queries
        ]
    
    def count_words(self, text: str) -> int:
        """Count the number of words in a text."""
        return len(text.split())
    
    def is_nonsense(self, prompt: str) -> bool:
        """Check if the prompt is nonsense."""
        # Check for very short prompts
        if self.count_words(prompt) < self.min_words:
            return True
        
        # Check for gibberish (high ratio of non-alphabetic characters)
        alpha_count = sum(c.isalpha() for c in prompt)
        if len(prompt) > 0 and alpha_count / len(prompt) < 0.5:
            return True
        
        return False
    
    def is_spam(self, prompt: str) -> bool:
        """Check if the prompt is spam."""
        for pattern in self.spam_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False
    
    def is_low_cost(self, prompt: str) -> bool:
        """Check if the prompt can be handled by a cheaper model."""
        for pattern in self.low_cost_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False
    
    def classify(self, prompt: str) -> Tuple[str, float]:
        """
        Classify a prompt using rules.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Tuple of (classification label, confidence score)
        """
        if self.is_nonsense(prompt):
            return "nonsense", 1.0
        
        if self.is_spam(prompt):
            return "spam", 1.0
        
        if self.is_low_cost(prompt):
            return "low_cost", 0.9
        
        # Default to valuable with medium confidence
        return "valuable", 0.8


class MLClassifier:
    """
    ML-based classifier for prompt classification.
    Uses a pre-trained model to classify prompts.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", confidence_threshold: float = 0.7):
        """
        Initialize the ML classifier.
        
        Args:
            model_name: Name of the pre-trained model to use
            confidence_threshold: Minimum confidence for classification
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Load the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=5)
            
            # Define the label mapping
            self.id2label = {
                0: "nonsense",
                1: "spam",
                2: "repeat",
                3: "low_cost",
                4: "valuable"
            }
            self.label2id = {v: k for k, v in self.id2label.items()}
            
            print(f"ML model '{self.model_name}' loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading ML model: {str(e)}")
            return False
    
    def classify(self, prompt: str) -> Tuple[str, float]:
        """
        Classify a prompt using the ML model.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Tuple of (classification label, confidence score)
        """
        # Lazy load the model when needed
        if self.model is None:
            success = self.load_model()
            if not success:
                # Fall back to rule-based classification
                rule_classifier = RuleBasedClassifier()
                return rule_classifier.classify(prompt)
        
        try:
            import torch
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            
            # Get the model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Get the predicted class and confidence
                predicted_class_id = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_id].item()
                
                # Map to label
                label = self.id2label[predicted_class_id]
                
                # If confidence is below threshold, fall back to rule-based
                if confidence < self.confidence_threshold:
                    rule_classifier = RuleBasedClassifier()
                    return rule_classifier.classify(prompt)
                
                return label, confidence
        except Exception as e:
            print(f"Error during ML classification: {str(e)}")
            # Fall back to rule-based classification
            rule_classifier = RuleBasedClassifier()
            return rule_classifier.classify(prompt)


class RepeatDetector:
    """
    Detector for repeated prompts within a session.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the repeat detector.
        
        Args:
            max_history: Maximum number of prompts to remember per session
        """
        self.max_history = max_history
        self.session_history: Dict[str, List[str]] = {}
    
    def normalize_prompt(self, prompt: str) -> str:
        """Normalize the prompt for comparison."""
        return prompt.lower().strip()
    
    def hash_prompt(self, prompt: str) -> str:
        """Create a hash of the normalized prompt."""
        normalized = self.normalize_prompt(prompt)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def is_repeat(self, prompt: str, session_id: str) -> bool:
        """
        Check if a prompt is a repeat within a session.
        
        Args:
            prompt: The user prompt
            session_id: The session identifier
            
        Returns:
            True if the prompt is a repeat, False otherwise
        """
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        
        prompt_hash = self.hash_prompt(prompt)
        
        # Check if the prompt is in the session history
        if prompt_hash in self.session_history[session_id]:
            return True
        
        # Add the prompt to the session history
        self.session_history[session_id].append(prompt_hash)
        
        # Limit the history size
        if len(self.session_history[session_id]) > self.max_history:
            self.session_history[session_id].pop(0)
        
        return False


class PromptClassifier:
    """
    Main classifier that combines rule-based, ML-based, and repeat detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the prompt classifier.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            config = {'classifier': {'use_ml_model': True, 'model_name': 'distilbert-base-uncased', 'min_words': 4, 'confidence_threshold': 0.7}}
        
        classifier_config = config.get('classifier', {})
        
        # Initialize components
        self.use_ml_model = classifier_config.get('use_ml_model', True)
        self.min_words = classifier_config.get('min_words', 4)
        self.confidence_threshold = classifier_config.get('confidence_threshold', 0.7)
        self.model_name = classifier_config.get('model_name', 'distilbert-base-uncased')
        
        self.rule_classifier = RuleBasedClassifier(min_words=self.min_words)
        self.ml_classifier = None
        if self.use_ml_model:
            self.ml_classifier = MLClassifier(
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold
            )
        
        self.repeat_detector = RepeatDetector()
    
    def classify(self, prompt: str, session_id: str = "") -> Dict[str, Any]:
        """
        Classify a prompt.
        
        Args:
            prompt: The user prompt
            session_id: The session identifier (optional)
            
        Returns:
            Dictionary with classification results
        """
        # Check for repeat first
        if session_id and self.repeat_detector.is_repeat(prompt, session_id):
            return {
                "label": "repeat",
                "confidence": 1.0,
                "session_id": session_id
            }
        
        # Use ML classifier if available and enabled
        if self.use_ml_model and self.ml_classifier:
            try:
                label, confidence = self.ml_classifier.classify(prompt)
            except Exception as e:
                print(f"ML classification failed: {str(e)}")
                # Fall back to rule-based classification
                label, confidence = self.rule_classifier.classify(prompt)
        else:
            # Use rule-based classification
            label, confidence = self.rule_classifier.classify(prompt)
        
        return {
            "label": label,
            "confidence": confidence,
            "session_id": session_id
        }


# For testing
if __name__ == "__main__":
    # Test the classifier with example prompts
    classifier = PromptClassifier()
    
    test_prompts = [
        "asdjklasdjkl",  # nonsense
        "What is 2 + 2?",  # low_cost
        "Write a poem about AI",  # valuable
        "You are stupid",  # spam
    ]
    
    session_id = "test_session"
    
    print("Testing classifier with example prompts:")
    for prompt in test_prompts:
        result = classifier.classify(prompt, session_id)
        print(f"Prompt: '{prompt}'")
        print(f"Classification: {result['label']} (confidence: {result['confidence']:.2f})")
        print()
    
    # Test repeat detection
    repeat_prompt = "What is 2 + 2?"
    print(f"Testing repeat detection with prompt: '{repeat_prompt}'")
    result = classifier.classify(repeat_prompt, session_id)
    print(f"Classification: {result['label']} (confidence: {result['confidence']:.2f})")

print("Classifier module implemented successfully!")