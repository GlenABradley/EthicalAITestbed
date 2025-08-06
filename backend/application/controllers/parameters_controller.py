"""
Parameters Controller for the Ethical AI Testbed.

This controller handles all parameter-related API endpoints and routes.
It implements proper dependency injection and separation of concerns.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body

# Import use cases
from application.use_cases.get_parameters_use_case import GetParametersUseCase
from application.use_cases.update_parameters_use_case import UpdateParametersUseCase
from application.use_cases.update_threshold_scaling_use_case import UpdateThresholdScalingUseCase

# Import request/response models
from application.dtos.threshold_scaling_request import ThresholdScalingRequest
from application.dtos.threshold_scaling_response import ThresholdScalingResponse
from application.dtos.update_all_thresholds_request import UpdateAllThresholdsRequest

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["parameters"],
    responses={404: {"description": "Not found"}},
)

@router.get("/parameters")
async def get_parameters(
    orchestrator=Depends(get_orchestrator)
):
    """
    Get current evaluation parameters.
    
    This endpoint returns the current ethical evaluation parameters,
    including thresholds, weights, and configuration settings.
    
    Returns:
        Dict[str, Any]: Current evaluation parameters
    """
    try:
        # Create use case with injected dependencies
        use_case = GetParametersUseCase(orchestrator)
        
        # Execute use case
        parameters = await use_case.execute()
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error in get_parameters: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parameter retrieval error: {str(e)}"
        )

@router.post("/parameters")
async def update_parameters(
    params: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Update evaluation parameters.
    
    This endpoint allows updating the ethical evaluation parameters,
    including thresholds, weights, and configuration settings.
    
    Args:
        params: The parameters to update
        
    Returns:
        Dict[str, Any]: Updated parameters
    """
    try:
        # Create use case with injected dependencies
        use_case = UpdateParametersUseCase(orchestrator)
        
        # Execute use case
        updated_params = await use_case.execute(params)
        
        return updated_params
        
    except Exception as e:
        logger.error(f"Error in update_parameters: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parameter update error: {str(e)}"
        )

@router.post("/threshold-scaling", response_model=ThresholdScalingResponse)
async def update_threshold_scaling(
    request: ThresholdScalingRequest,
    orchestrator=Depends(get_orchestrator)
):
    """
    Update a specific evaluation threshold based on a normalized slider value.
    
    This endpoint allows dynamic adjustment of ethical evaluation thresholds
    using a normalized slider value (0.0 to 1.0). It supports both exponential
    and linear scaling.
    
    Args:
        request: The threshold scaling request
        
    Returns:
        ThresholdScalingResponse: The scaling operation results
    """
    try:
        # Create use case with injected dependencies
        use_case = UpdateThresholdScalingUseCase(orchestrator)
        
        # Execute use case
        response = await use_case.execute(
            threshold_type=request.threshold_type,
            slider_value=request.slider_value,
            use_exponential=request.use_exponential
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in update_threshold_scaling: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threshold scaling error: {str(e)}"
        )

@router.post("/update-all-thresholds")
async def update_all_thresholds(
    request: UpdateAllThresholdsRequest,
    orchestrator=Depends(get_orchestrator)
):
    """
    Update all three ethical evaluation thresholds at once.
    
    This endpoint allows updating all three ethical evaluation thresholds
    (virtue, deontological, consequentialist) in a single request.
    
    Args:
        request: The update all thresholds request
        
    Returns:
        Dict[str, Any]: The updated thresholds
    """
    try:
        # Create use case with injected dependencies
        use_case = UpdateThresholdScalingUseCase(orchestrator)
        
        # Execute use cases for each threshold
        virtue_response = await use_case.execute(
            threshold_type="virtue",
            slider_value=request.virtue_threshold,
            use_exponential=request.use_exponential
        )
        
        deontological_response = await use_case.execute(
            threshold_type="deontological",
            slider_value=request.deontological_threshold,
            use_exponential=request.use_exponential
        )
        
        consequentialist_response = await use_case.execute(
            threshold_type="consequentialist",
            slider_value=request.consequentialist_threshold,
            use_exponential=request.use_exponential
        )
        
        # Combine responses
        return {
            "virtue": virtue_response,
            "deontological": deontological_response,
            "consequentialist": consequentialist_response,
            "status": "success",
            "message": "All thresholds updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in update_all_thresholds: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threshold update error: {str(e)}"
        )
