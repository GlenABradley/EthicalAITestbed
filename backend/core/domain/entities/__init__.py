"""
Domain Entities - Core Business Objects

This package contains the core domain entities that represent the fundamental
business objects in the Ethical AI evaluation system.

Entities:
- EthicalSpan: Represents an evaluated span of text with ethical scores
- EthicalEvaluation: Represents a complete ethical evaluation of text

Author: AI Developer Testbed Team
Version: 1.1.0 - Clean Architecture Implementation
"""

from backend.core.domain.entities.ethical_span import EthicalSpan
from backend.core.domain.entities.ethical_evaluation import EthicalEvaluation

__all__ = [
    'EthicalSpan',
    'EthicalEvaluation'
]
