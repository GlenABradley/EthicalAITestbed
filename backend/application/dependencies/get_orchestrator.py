"""
Dependency for injecting the unified orchestrator.

This module provides a dependency function for FastAPI to inject
the unified orchestrator instance into route handlers.
"""

import logging
from fastapi import Depends

# Import the global orchestrator instance
from unified_ethical_orchestrator import get_unified_orchestrator

logger = logging.getLogger(__name__)

async def get_orchestrator():
    """
    Dependency function to get the unified orchestrator instance.
    
    This function is used with FastAPI's dependency injection system
    to provide the orchestrator to route handlers.
    
    Returns:
        The unified orchestrator instance
    """
    logger.debug("Injecting unified orchestrator")
    return get_unified_orchestrator()
