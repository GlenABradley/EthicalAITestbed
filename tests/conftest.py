"""
🧪 TEST CONFIGURATION AND FIXTURES
═══════════════════════════════════════════════════════════════════════════════════

This module contains pytest configuration and common fixtures used across test modules.
"""

import pytest
import numpy as np
from typing import Generator, Dict, Any

# Add test-specific configurations here

@pytest.fixture(scope="session")
def sample_texts() -> Dict[str, str]:
    """Provide sample texts for testing different ethical scenarios."""
    return {
        "harmful": "I'm going to hurt someone if they don't do what I say.",
        "neutral": "The weather is nice today.",
        "deceptive": "This is a completely risk-free investment with guaranteed returns.",
        "discriminatory": "People from that country are all lazy and untrustworthy.",
        "manipulative": "If you really cared about me, you would do this for me.",
        "ethical": "Let's work together to find a solution that benefits everyone."
    }

@pytest.fixture(scope="session")
def test_parameters():
    """Provide a standard set of test parameters."""
    from backend.ethical_engine import EthicalParameters
    
    return EthicalParameters(
        virtue_threshold=0.15,
        deontological_threshold=0.15,
        consequentialist_threshold=0.15,
        max_span_length=5,
        min_span_length=1,
        enable_dynamic_scaling=True,
        enable_cascade_filtering=True,
        enable_graph_attention=False,
        enable_intent_hierarchy=False
    )

@pytest.fixture(scope="module")
def ethical_evaluator(test_parameters):
    """Provide a configured EthicalEvaluator instance for testing."""
    from backend.ethical_engine import EthicalEvaluator
    
    # Initialize with test parameters
    return EthicalEvaluator(parameters=test_parameters)

# Add any additional fixtures here as needed
