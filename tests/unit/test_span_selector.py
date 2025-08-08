#!/usr/bin/env python3
"""
🧪 UNIT TESTS: SPAN SELECTOR
═══════════════════════════════════════════════════════════════════════════════════

This module contains unit tests for the DifferenceSetSpanSelector and its
integration with the EthicalEvaluator for optimal span analysis.
"""

import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Tuple, Set

# Import the components we'll be testing
from backend.span_selector import DifferenceSetSpanSelector
from backend.ethical_engine import (
    EthicalEvaluator,
    EthicalParameters,
    EthicalSpan,
    EthicalEvaluation
)

# Test data with deliberate ethical violations
ETHICAL_TEXT = """
This is a completely ethical text that promotes positive values, 
respect for autonomy, and truthful information sharing.
It supports cognitive freedom and social cooperation.
"""

UNETHICAL_TEXT = """
This text contains some misleading information and promotes harmful behaviors.
It encourages manipulation of others and disrespects individual autonomy.
There are also statements that could lead to negative outcomes for vulnerable groups.
"""

MIXED_TEXT = """
This is mostly ethical content that promotes good values.
However, there is a misleading claim here that could harm people.
The rest of the text returns to promoting positive social norms.
"""

class TestDifferenceSetSpanSelector:
    """Test suite for the DifferenceSetSpanSelector and its integration."""
    
    def test_span_selector_initialization(self):
        """Test that the span selector initializes correctly with various parameters."""
        # Default initialization
        selector = DifferenceSetSpanSelector()
        assert selector.window_size > 0
        assert isinstance(selector.difference_set, list)
        
        # Custom window size (gets closest preset, which is 21 for requested 20)
        selector = DifferenceSetSpanSelector(window_size=20)
        assert selector.window_size == 21
        assert len(selector.difference_set) > 0
        
        # Asymmetric configuration
        selector = DifferenceSetSpanSelector(symmetric=False)
        assert not selector.symmetric
    
    def test_span_selection_coverage(self):
        """Test that span selection provides good coverage with minimal spans."""
        # Test with different document lengths
        for doc_length in [10, 30, 100]:
            selector = DifferenceSetSpanSelector(window_size=min(31, doc_length))
            spans = selector.generate_spans(doc_length)
            
            # Verify we have a reasonable number of spans (should be subquadratic)
            assert len(spans) > 0
            assert len(spans) < doc_length * doc_length
            
            # For longer documents, verify we're getting good coverage
            if doc_length >= 30:
                # Coverage ratio should be significantly less than n²
                coverage_ratio = len(spans) / (doc_length * doc_length)
                assert coverage_ratio < 0.5, f"Coverage ratio too high: {coverage_ratio}"
            
            # Verify each span is valid
            for start, end in spans:
                assert 0 <= start < doc_length
                assert 0 < end <= doc_length
                assert start < end
    
    def test_span_pairwise_coverage(self):
        """Test that the difference set approach covers all important token pairs."""
        doc_length = 30
        selector = DifferenceSetSpanSelector(window_size=doc_length)
        spans = selector.generate_spans(doc_length)
        
        # Track which token pairs are covered by spans
        covered_pairs = set()
        for start, end in spans:
            for i in range(start, end):
                for j in range(i+1, end):
                    covered_pairs.add((i, j))
        
        # Calculate coverage percentage
        total_possible_pairs = doc_length * (doc_length - 1) // 2
        coverage_percentage = len(covered_pairs) / total_possible_pairs
        
        # We should have good coverage of pairs (typically >80% for a well-designed difference set)
        assert coverage_percentage > 0.7, f"Pair coverage too low: {coverage_percentage}"
        print(f"Pair coverage: {coverage_percentage:.2%}")


class TestSpanSelectorIntegration:
    """Test the integration of DifferenceSetSpanSelector with EthicalEvaluator."""
    
    @pytest.fixture(autouse=True)
    def setup_evaluator(self):
        """Set up a test instance of the ethical evaluator with default parameters."""
        # Use test-friendly parameters
        params = EthicalParameters(
            virtue_threshold=0.1,
            deontological_threshold=0.1,
            consequentialist_threshold=0.1,
            max_span_length=10,
            enable_dynamic_scaling=True,
            enable_cascade_filtering=True,
            enable_graph_attention=False,  # Disable for unit tests
            enable_intent_hierarchy=False,  # Disable for unit tests
            enable_contrastive_learning=False,
            enable_causal_analysis=False,
            enable_uncertainty_analysis=False,
            enable_purpose_alignment=False,
            exponential_scaling=True,
        )
        # Create evaluator with test parameters
        self.evaluator = EthicalEvaluator(parameters=params)
        
        # Ensure span selector is initialized with reasonable values for testing
        self.evaluator.span_selector = DifferenceSetSpanSelector(
            window_size=10,
            symmetric=True
        )
    
    @patch('backend.ethical_engine.SentenceTransformer')
    def test_ethical_detection(self, mock_transformer):
        """Test that the span selector correctly identifies ethical content."""
        # Setup mock transformer to return appropriate embeddings for ethical text
        # In the new mathematical framework, positive values indicate ethical content
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.5, 0.5]])  # Positive values -> ethical
        mock_transformer.return_value = mock_model
        
        # Apply the mock
        self.evaluator.model = mock_model
        
        # Create mock ethical vectors
        # With positive embeddings and positive vectors, dot product will be positive (ethical)
        self.evaluator.p_v = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_d = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_c = np.array([1.0, 1.0, 1.0])
        
        # Mock the generate_spans method to return simplified spans
        self.evaluator.span_selector.generate_spans = MagicMock(return_value=[(0, 10), (5, 15)])
        
        # Evaluate text
        evaluation = self.evaluator.evaluate_text(ETHICAL_TEXT)
        
        # Check results - should detect no violations
        assert evaluation.overall_ethical
        assert len(evaluation.minimal_spans) == 0
    
    @patch('backend.ethical_engine.SentenceTransformer')
    def test_unethical_detection(self, mock_transformer):
        """Test that the span selector correctly identifies unethical content."""
        # Setup mock transformer to return appropriate embeddings for unethical text
        # In the new mathematical framework, negative values indicate unethical content
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[-0.2, -0.2, -0.2]])  # Negative values -> unethical
        mock_transformer.return_value = mock_model
        
        # Apply the mock
        self.evaluator.model = mock_model
        
        # Create mock ethical vectors
        # With negative embeddings and positive vectors, dot product will be negative (unethical)
        self.evaluator.p_v = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_d = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_c = np.array([1.0, 1.0, 1.0])
        
        # Mock the generate_spans method to return simplified spans
        self.evaluator.span_selector.generate_spans = MagicMock(return_value=[(0, 10), (5, 15)])
        
        # Evaluate text
        evaluation = self.evaluator.evaluate_text(UNETHICAL_TEXT)
        
        # Verify that spans with negative values are correctly identified as violations
        assert any(span.is_violation for span in evaluation.spans), "Should have at least one violation"
    
    @patch('backend.ethical_engine.SentenceTransformer')
    def test_mixed_content_detection(self, mock_transformer):
        """Test that the span selector correctly identifies mixed ethical/unethical content."""
        # Setup mock model to simulate mixed ethical/unethical content
        mock_model = MagicMock()
        
        # Create a list to store generated spans
        test_spans = []
        
        # Store original method references
        _original_evaluate_span = self.evaluator.evaluate_span
        _original_cascade = self.evaluator.parameters.enable_cascade_filtering 
        _original_generate_spans = self.evaluator.span_selector.generate_spans
        
        # Disable cascade filtering to ensure detailed analysis
        self.evaluator.parameters.enable_cascade_filtering = False
        
        # Mock span generation to cover different sections of text
        self.evaluator.span_selector.generate_spans = MagicMock(return_value=[
            (0, 10),  # Should contain "misleading" -> unethical
            (10, 20), # Should be ethical
            (20, 30)  # Should be ethical
        ])

        # Create mock vectors for deterministic scoring
        self.evaluator.p_v = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_d = np.array([1.0, 1.0, 1.0])
        self.evaluator.p_c = np.array([1.0, 1.0, 1.0])
        
        # Define our mock span evaluation function with mixed ethical/unethical results
        def _mock_evaluate_span(tokens, start, end, adjusted_thresholds=None):
            span_text = ' '.join(tokens[start:end])
            
            if "misleading claim" in span_text.lower():
                # Unethical span with negative scores
                span = EthicalSpan(
                    start=start,
                    end=end,
                    text=span_text,
                    virtue_score=-0.2,      # Negative = unethical
                    deontological_score=-0.2,
                    consequentialist_score=-0.2,
                    combined_score=-0.2,
                    is_violation=True,
                    violation_type="deceptive_content",
                    virtue_violation=True,
                    deontological_violation=True,
                    consequentialist_violation=True
                )
            else:
                # Ethical span with positive scores
                span = EthicalSpan(
                    start=start,
                    end=end,
                    text=span_text,
                    virtue_score=0.2,      # Positive = ethical
                    deontological_score=0.2,
                    consequentialist_score=0.2,
                    combined_score=0.2,
                    is_violation=False
                )
            
            # Track for verification
            test_spans.append(span)
            return span
        
        # Apply our mock
        self.evaluator.evaluate_span = _mock_evaluate_span
        
        try:
            # Run the evaluation
            evaluation = self.evaluator.evaluate_text(MIXED_TEXT)
            
            # Verify our mock created a mix of ethical/unethical spans
            has_violation = any(span.is_violation for span in test_spans)
            has_non_violation = any(not span.is_violation for span in test_spans)
            assert has_violation and has_non_violation, "Mock should generate both ethical and unethical spans"
            
            # Verify the evaluation results
            assert evaluation.overall_ethical is False, "Mixed text with violations should be marked as unethical"
            assert any(span.is_violation for span in evaluation.spans), "Should have at least one span with violation"
            assert not all(span.is_violation for span in evaluation.spans), "Should not have all spans as violations"
        finally:
            # Restore original methods
            self.evaluator.evaluate_span = _original_evaluate_span
            self.evaluator.parameters.enable_cascade_filtering = _original_cascade
            self.evaluator.span_selector.generate_spans = _original_generate_spans
    
    def test_coverage_ratio_logging(self, caplog):
        """Test that coverage ratio is properly logged."""
        # Setup
        caplog.set_level(logging.INFO)
        
        # Disable cascade filtering to ensure detailed analysis is not skipped
        original_cascade = self.evaluator.parameters.enable_cascade_filtering
        self.evaluator.parameters.enable_cascade_filtering = False
        
        # Save the real generate_spans
        original_generate_spans = self.evaluator.span_selector.generate_spans
        
        # Mock the generate_spans method to return predictable spans and log the expected message
        def mock_generate_spans(token_count, min_span_length=3, max_span_length=None):
            # Call the original function to ensure it logs the message
            spans = original_generate_spans(token_count, min_span_length, max_span_length)
            # But return our predictable spans
            return [(0, 5), (3, 8), (5, 10)]
        
        self.evaluator.span_selector.generate_spans = mock_generate_spans
        
        # Mock evaluate_span to avoid actual analysis
        # Use positive scores to represent ethical content in the new math framework
        original_evaluate_span = self.evaluator.evaluate_span
        self.evaluator.evaluate_span = MagicMock(return_value=EthicalSpan(
            start=0,
            end=5,
            text="test",
            virtue_score=0.2,  # Positive score means ethical in new framework
            deontological_score=0.2,
            consequentialist_score=0.2,
            combined_score=0.2,
            is_violation=False
        ))
        
        try:
            # Execute
            self.evaluator.evaluate_text("Test text for coverage ratio logging.")
            
            # Verify
            # Check for the log message from DifferenceSetSpanSelector
            assert any("Generated" in record.message and "unique spans for" in record.message and "using difference set" in record.message 
                      for record in caplog.records), "Expected log message about generated spans not found"
        finally:
            # Restore original methods and parameters
            self.evaluator.evaluate_span = original_evaluate_span
            self.evaluator.span_selector.generate_spans = original_generate_spans
            self.evaluator.parameters.enable_cascade_filtering = original_cascade


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
