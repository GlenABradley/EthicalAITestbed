"""
Domain Layer - Core Business Logic

This package contains the pure domain entities, value objects, aggregates, and domain services
that represent the core business logic of the Ethical AI evaluation system.

Components:
- entities: Core domain entities (EthicalSpan, EthicalEvaluation)
- value_objects: Immutable value objects (EthicalParameters)
- aggregates: Domain aggregates that combine multiple entities
- services: Domain services for complex business logic

Author: AI Developer Testbed Team
Version: 1.1.0 - Clean Architecture Implementation
"""

# Import core domain components for easy access
from core.domain.entities import *
from core.domain.value_objects import *
from core.domain.aggregates import *
from core.domain.services import *
