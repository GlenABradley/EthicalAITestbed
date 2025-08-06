"""
Evaluation Controller for the Ethical AI Testbed.

This controller handles all evaluation-related API endpoints and routes.
It implements proper dependency injection and separation of concerns.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status

# Import domain models
from core.domain.entities.ethical_evaluation import EthicalEvaluation
from core.domain.value_objects.ethical_parameters import EthicalParameters

# Import use cases
from application.use_cases.evaluate_text_use_case import EvaluateTextUseCase
from application.use_cases.store_evaluation_result_use_case import StoreEvaluationResultUseCase

# Import request/response models
from application.dtos.evaluation_request import EvaluationRequest
from application.dtos.evaluation_response import EvaluationResponse

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator
from application.dependencies.get_database import get_database

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_text(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    db=Depends(get_database)
):
    """
    Evaluate text for ethical considerations across multiple perspectives.
    
    This endpoint performs comprehensive ethical evaluation using the
    v3.0 semantic embedding framework with orthogonal vector projections.
    
    Args:
        request: The evaluation request containing text and parameters
        background_tasks: FastAPI background tasks for async operations
        orchestrator: The unified ethical orchestrator (injected)
        db: Database connection (injected)
        
    Returns:
        EvaluationResponse: Comprehensive ethical evaluation results
    """
    try:
        # Create use case with injected dependencies
        use_case = EvaluateTextUseCase(orchestrator)
        
        # Execute use case
        evaluation_result = await use_case.execute(
            text=request.text,
            context=request.context,
            parameters=request.parameters,
            tau_slider=request.tau_slider
        )
        
        # Create response
        response = EvaluationResponse(
            evaluation=evaluation_result,
            clean_text=orchestrator.generate_clean_text(evaluation_result),
            delta_summary=orchestrator.generate_delta_summary(evaluation_result)
        )
        
        # Store result in background
        request_id = f"req_{evaluation_result.evaluation_id}"
        background_tasks.add_task(
            StoreEvaluationResultUseCase(db).execute,
            request=request,
            response=response,
            request_id=request_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in evaluate_text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )
