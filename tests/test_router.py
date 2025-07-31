# filename: tests/test_router.py
import unittest
from promptshield.router import PromptRouter

class TestPromptRouter(unittest.TestCase):
    def setUp(self):
        self.router = PromptRouter()
    
    def test_block_nonsense(self):
        classification = {"label": "nonsense", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "block")
        self.assertEqual(decision["reason"], "Prompt classified as nonsense")
    
    def test_block_spam(self):
        classification = {"label": "spam", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "block")
        self.assertEqual(decision["reason"], "Prompt classified as spam")
    
    def test_cache_repeat(self):
        classification = {"label": "repeat", "confidence": 1.0}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "cache")
    
    def test_route_low_cost(self):
        classification = {"label": "low_cost", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-3.5-turbo")
    
    def test_route_valuable(self):
        classification = {"label": "valuable", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-4")
    
    def test_different_providers(self):
        # OpenAI
        classification = {"label": "valuable", "confidence": 0.8}
        decision = self.router.route(classification, "openai", "gpt-4", "gpt-3.5-turbo")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "gpt-4")
        
        # Anthropic
        decision = self.router.route(classification, "anthropic", "claude-v1", "claude-haiku")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "claude-v1")
        
        # Ollama
        decision = self.router.route(classification, "ollama", "llama3-70b", "mistral-instruct")
        self.assertEqual(decision["action"], "route")
        self.assertEqual(decision["model"], "llama3-70b")

if __name__ == "__main__":
    unittest.main()
