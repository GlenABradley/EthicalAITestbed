"""
Learning Controller for the Ethical AI Testbed.

This controller handles all learning-related API endpoints and routes.
It implements proper dependency injection and separation of concerns.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

# Import use cases
from application.use_cases.get_learning_stats_use_case import GetLearningStatsUseCase

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator
from application.dependencies.get_database import get_database

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["learning"],
    responses={404: {"description": "Not found"}},
)

@router.get("/learning-stats")
async def get_learning_stats(
    orchestrator=Depends(get_orchestrator),
    db=Depends(get_database)
):
    """
    Get learning system statistics.
    
    This endpoint returns statistics about the learning system,
    including the number of entries, recent feedback, and performance metrics.
    
    Returns:
        Dict[str, Any]: Learning system statistics
    """
    try:
        # Create use case with injected dependencies
        use_case = GetLearningStatsUseCase(orchestrator, db)
        
        # Execute use case
        stats = await use_case.execute()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_learning_stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning stats retrieval error: {str(e)}"
        )
