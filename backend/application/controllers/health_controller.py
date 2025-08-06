"""
Health Controller for the Ethical AI Testbed.

This controller handles all health-related API endpoints and routes.
It implements proper dependency injection and separation of concerns.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

# Import use cases
from application.use_cases.perform_health_check_use_case import PerformHealthCheckUseCase

# Import response models
from application.dtos.system_health_response import SystemHealthResponse

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator
from application.dependencies.get_database import get_database

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=SystemHealthResponse)
async def health_check(
    orchestrator=Depends(get_orchestrator),
    db=Depends(get_database)
):
    """
    🏥 **System Health Check**
    
    Provides comprehensive health and status information for the entire
    Ethical AI system, including all components and performance metrics.
    
    This endpoint is used for:
    - Load balancer health checks
    - Monitoring system alerts
    - System administration
    - Debugging and troubleshooting
    
    Returns detailed information about:
    - Overall system status (healthy, degraded, error)
    - Component health status (orchestrator, database, config)
    - Performance metrics (if available)
    - Feature availability
    - Configuration validity
    - Uptime and system information
    
    Response Codes:
    - 200: System is healthy or degraded but operational
    - 503: System is in an error state and may not be fully functional
    """
    try:
        # Create use case with injected dependencies
        use_case = PerformHealthCheckUseCase(orchestrator, db)
        
        # Execute use case
        health_status = await use_case.execute()
        
        # Return response
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check error: {str(e)}"
        )
