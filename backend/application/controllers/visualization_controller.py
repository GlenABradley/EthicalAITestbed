"""
Visualization Controller for the Ethical AI Testbed.

This controller handles all visualization-related API endpoints and routes.
It implements proper dependency injection and separation of concerns.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body

# Import use cases
from application.use_cases.get_heat_map_data_use_case import GetHeatMapDataUseCase

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["visualization"],
    responses={404: {"description": "Not found"}},
)

@router.post("/heat-map-mock")
async def get_heat_map_mock(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Generate heat-map data from ethical analysis for UI visualization.
    
    This endpoint generates heat-map data based on ethical analysis
    for visualization in the UI. It provides a detailed breakdown of
    ethical considerations across different perspectives.
    
    Args:
        request: The heat map request
        
    Returns:
        Dict[str, Any]: Heat map data
    """
    try:
        # Create use case with injected dependencies
        use_case = GetHeatMapDataUseCase(orchestrator)
        
        # Extract text from request
        text = request.get("text", "")
        if not text:
            raise ValueError("Text is required for heat map generation")
            
        # Execute use case
        heat_map_data = await use_case.execute(text)
        
        return heat_map_data
        
    except Exception as e:
        logger.error(f"Error in get_heat_map_mock: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heat map generation error: {str(e)}"
        )
