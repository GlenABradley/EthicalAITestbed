"""
Domain Value Objects - Immutable Business Values

This package contains the immutable value objects that represent configuration
and parameter values in the Ethical AI evaluation system.

Value Objects:
- EthicalParameters: Configuration parameters for ethical evaluation

Author: AI Developer Testbed Team
Version: 1.1.0 - Clean Architecture Implementation
"""

from backend.core.domain.value_objects.ethical_parameters import EthicalParameters

__all__ = [
    'EthicalParameters'
]
