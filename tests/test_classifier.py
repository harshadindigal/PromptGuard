# filename: tests/test_classifier.py
import unittest
from promptshield.classifier import PromptClassifier

class TestPromptClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = PromptClassifier()
    
    def test_nonsense_classification(self):
        result = self.classifier.classify("asdjklasdjkl", "test_session")
        self.assertEqual(result["label"], "nonsense")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_spam_classification(self):
        result = self.classifier.classify("You are stupid", "test_session")
        self.assertEqual(result["label"], "spam")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_low_cost_classification(self):
        result = self.classifier.classify("What is 2 + 2?", "test_session")
        self.assertEqual(result["label"], "low_cost")
        self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_valuable_classification(self):
        result = self.classifier.classify("Write a detailed analysis of the economic impact of artificial intelligence on global labor markets over the next decade", "test_session")
        self.assertEqual(result["label"], "valuable")
        self.assertGreaterEqual(result["confidence"], 0.7)
    
    def test_repeat_classification(self):
        # First query
        self.classifier.classify("What is the capital of France?", "test_session")
        
        # Repeat the query
        result = self.classifier.classify("What is the capital of France?", "test_session")
        self.assertEqual(result["label"], "repeat")
        self.assertEqual(result["confidence"], 1.0)
    
    def test_short_prompt(self):
        result = self.classifier.classify("Hi", "test_session")
        self.assertEqual(result["label"], "nonsense")
        self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_different_sessions(self):
        # First query in session 1
        self.classifier.classify("What is the capital of France?", "session_1")
        
        # Same query in session 2 (should not be classified as repeat)
        result = self.classifier.classify("What is the capital of France?", "session_2")
        self.assertNotEqual(result["label"], "repeat")

if __name__ == "__main__":
    unittest.main()
