"""
Dependency for injecting the database connection.

This module provides a dependency function for FastAPI to inject
the database connection into route handlers.
"""

import logging
from fastapi import Request

logger = logging.getLogger(__name__)

async def get_database(request: Request):
    """
    Dependency function to get the database connection.
    
    This function is used with FastAPI's dependency injection system
    to provide the database connection to route handlers.
    
    Args:
        request: The FastAPI request object
        
    Returns:
        The database connection from app.state.db
    """
    logger.debug("Injecting database connection from app state")
    return request.app.state.db
