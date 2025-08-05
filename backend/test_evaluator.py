"""
Lightweight test evaluator for the Ethical AI Testbed.
This version skips heavy ML model loading and provides mock responses for testing.
"""
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
import numpy as np

@dataclass
class EthicalParameters:
    """Simplified parameters for testing"""
    virtue_threshold: float = 0.15
    deontological_threshold: float = 0.15
    consequentialist_threshold: float = 0.15
    enable_graph_attention: bool = False
    enable_intent_hierarchy: bool = False
    enable_causal_analysis: bool = False
    enable_uncertainty_analysis: bool = False
    enable_purpose_alignment: bool = False

@dataclass
class EthicalSpan:
    """Simplified ethical span for testing"""
    start: int
    end: int
    text: str
    virtue_score: float
    deontological_score: float
    consequentialist_score: float
    virtue_violation: bool = False
    deontological_violation: bool = False
    consequentialist_violation: bool = False
    is_minimal: bool = True
    intent_scores: Dict[str, float] = field(default_factory=dict)
    dominant_intent: str = "neutral"
    intent_confidence: float = 0.0

class TestEthicalEvaluator:
    """Lightweight test evaluator that provides mock responses"""
    
    def __init__(self, parameters: EthicalParameters = None):
        self.parameters = parameters or EthicalParameters()
        self.model_loaded = True  # Pretend we loaded a model
        
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """Mock evaluation that returns a simple response"""
        # Simple mock response
        return {
            "input_text": text,
            "tokens": text.split(),
            "spans": [],
            "minimal_spans": [],
            "overall_ethical": True,
            "processing_time": 0.1,
            "evaluation_id": f"test_eval_{int(time.time() * 1000)}",
            "test_mode": True,
            "message": "This is a test response from the lightweight evaluator"
        }

def get_test_evaluator() -> TestEthicalEvaluator:
    """Get a test evaluator instance"""
    return TestEthicalEvaluator()
