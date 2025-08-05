#!/usr/bin/env python3
"""
🧪 OPERATIONAL TESTS: ETHICAL VECTOR ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════

This module contains operational tests to analyze how ethical inputs are vectorized
and processed by the ethical engine.
"""

import pytest
import numpy as np
import time
import traceback
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Import the ethical engine components
from backend.ethical_engine import EthicalEvaluator, EthicalParameters, EthicalSpan

@dataclass
class EthicalTestCase:
    """Represents a test case for ethical evaluation."""
    name: str
    text: str
    expected_violations: List[str]  # List of expected violation types
    description: str = ""

# Define test cases with varying ethical considerations
ETHICAL_TEST_CASES = [
    EthicalTestCase(
        name="harmful_threat",
        text="I'm going to hurt someone if they don't do what I say.",
        expected_violations=["violence"],
        description="Direct threat of violence"
    ),
    EthicalTestCase(
        name="deceptive_statement",
        text="This investment is completely risk-free with guaranteed high returns.",
        expected_violations=["deception"],
        description="Financially deceptive statement"
    ),
    EthicalTestCase(
        name="discriminatory_content",
        text="People from that country are all lazy and untrustworthy.",
        expected_violations=["discrimination"],
        description="Stereotypical and discriminatory statement"
    ),
    EthicalTestCase(
        name="manipulative_language",
        text="If you were a good friend, you would lend me money without asking questions.",
        expected_violations=["manipulation"],
        description="Emotionally manipulative language"
    ),
    EthicalTestCase(
        name="ethical_statement",
        text="Let's work together to find a solution that benefits everyone involved.",
        expected_violations=[],
        description="Positive, collaborative statement"
    ),
    EthicalTestCase(
        name="neutral_statement",
        text="The weather is nice today and the sky is blue.",
        expected_violations=[],
        description="Completely neutral statement"
    )
]

def interpret_score(score: float) -> str:
    """Convert numerical score to human-readable interpretation."""
    abs_score = abs(score)
    if abs_score < 0.1:
        return "neutral"
    elif abs_score < 0.3:
        return "slightly" + (" unethical" if score > 0 else " ethical")
    elif abs_score < 0.5:
        return "moderately" + (" unethical" if score > 0 else " ethical")
    else:
        return "strongly" + (" unethical" if score > 0 else " ethical")

def analyze_ethical_vectors(text: str, evaluation) -> None:
    """Analyze and print detailed ethical vector information."""
    print(f"\n{'='*80}")
    print(f"ETHICAL VECTOR ANALYSIS")
    print("="*80)
    print(f"\nTEXT: {text}")
    
    # Print overall ethical assessment if available
    if hasattr(evaluation, 'overall_ethical'):
        print(f"\nOVERALL ETHICAL ASSESSMENT: {evaluation.overall_ethical}")
    
    # Print vector scores for each span
    if hasattr(evaluation, 'spans') and evaluation.spans:
        print("\nSPAN ANALYSIS:")
        for i, span in enumerate(evaluation.spans):
            if not hasattr(span, 'virtue_score'):
                continue
                
            print(f"\n  Span {i+1}: '{span.text}'")
            
            # Print vector scores with interpretations
            print(f"    Virtue (Ethical Character):       {span.virtue_score:+.3f}  →  {interpret_score(span.virtue_score)}")
            print(f"    Deontological (Rules/Duties):     {span.deontological_score:+.3f}  →  {interpret_score(span.deontological_score)}")
            print(f"    Consequentialist (Outcomes):      {span.consequentialist_score:+.3f}  →  {interpret_score(span.consequentialist_score)}")
            
            # Calculate and print vector magnitude (overall ethical strength)
            vector_mag = (span.virtue_score**2 + 
                         span.deontological_score**2 + 
                         span.consequentialist_score**2)**0.5
            print(f"    \n    Vector Magnitude (Ethical Strength): {vector_mag:.3f}")
            
            # Print intent analysis if available
            if hasattr(span, 'dominant_intent') and span.dominant_intent != 'neutral':
                print(f"    \n    Detected Intent: {span.dominant_intent} (confidence: {getattr(span, 'intent_confidence', 0.0):.3f})")
    
    print("\n" + "="*80 + "\n")

class TestEthicalVectors:
    """Test suite for analyzing ethical input vectorization."""
    
    @pytest.fixture
    def evaluator(self):
        """Fixture to provide a configured EthicalEvaluator instance with simplified settings."""
        params = EthicalParameters(
            virtue_threshold=0.15,
            deontological_threshold=0.15,
            consequentialist_threshold=0.15,
            enable_dynamic_scaling=False,
            enable_cascade_filtering=False,
            enable_learning_mode=False,
            exponential_scaling=True,
            enable_graph_attention=False,
            enable_intent_hierarchy=False,
            enable_contrastive_learning=False,
            enable_causal_analysis=False,
            enable_uncertainty_analysis=False,
            enable_purpose_alignment=False,
        )
        return EthicalEvaluator(parameters=params)
    
    def test_ethical_vectors(self, evaluator):
        """Test and analyze the ethical vectors for various text inputs."""
        for test_case in ETHICAL_TEST_CASES:
            try:
                print(f"\n{'='*80}")
                print(f"TEST CASE: {test_case.name.upper()}")
                print(f"DESCRIPTION: {test_case.description}")
                
                # Perform ethical evaluation
                start_time = time.time()
                evaluation = evaluator.evaluate_text(test_case.text)
                processing_time = time.time() - start_time
                
                # Analyze and print vector information
                analyze_ethical_vectors(test_case.text, evaluation)
                print(f"Evaluation completed in {processing_time:.3f}s")
                
                # Basic sanity check - ensure we have spans
                assert hasattr(evaluation, 'spans') and evaluation.spans, \
                    f"No spans were generated for test case: {test_case.name}"
                
                # Check that at least one span has vector scores
                has_scores = any(hasattr(span, 'virtue_score') for span in evaluation.spans)
                assert has_scores, f"No vector scores found in spans for test case: {test_case.name}"
                
            except Exception as e:
                print(f"\n!!! ERROR during evaluation: {str(e)}")
                print(f"Test case: {test_case.name}")
                print(f"Text: {test_case.text}")
                print("Traceback:", traceback.format_exc())
                raise
    
    def test_evaluation_consistency(self, evaluator):
        """Test that the evaluator produces consistent results for the same input."""
        test_text = ETHICAL_TEST_CASES[0].text  # Use the first test case
        
        # Run multiple evaluations
        results = []
        for _ in range(3):
            evaluation = evaluator.evaluate_text(test_text)
            results.append({
                'overall_ethical': evaluation.overall_ethical,
                'violation_count': len(evaluation.violations) if hasattr(evaluation, 'violations') else 0,
                'processing_time': evaluation.processing_time
            })
        
        # Check that results are consistent
        first_result = results[0]
        for result in results[1:]:
            assert result['overall_ethical'] == first_result['overall_ethical'], \
                "Inconsistent ethical evaluation results"
            assert result['violation_count'] == first_result['violation_count'], \
                "Inconsistent violation counts"
            
        print("\n=== EVALUATION CONSISTENCY ===")
        print(f"Text: {test_text}")
        for i, result in enumerate(results, 1):
            print(f"Run {i}: Ethical: {result['overall_ethical']}, "
                  f"Violations: {result['violation_count']}, "
                  f"Time: {result['processing_time']:.4f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
