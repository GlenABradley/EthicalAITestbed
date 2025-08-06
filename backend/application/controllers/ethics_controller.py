"""Ethics Controller for the Ethical AI Testbed.

This controller handles all ML ethics assistant API endpoints and routes.
It implements proper dependency injection and separation of concerns following
the Clean Architecture pattern.

Endpoints:
- /api/ethics/comprehensive-analysis: Comprehensive multi-framework ethical analysis
- /api/ethics/meta-analysis: Meta-ethical analysis of philosophical foundations
- /api/ethics/normative-analysis: Normative ethical analysis across major frameworks
- /api/ethics/applied-analysis: Applied ethical analysis for practical implementation
- /api/ethics/ml-training-guidance: ML-specific training guidance and recommendations

Each endpoint delegates to its corresponding use case, maintaining a clear
separation between HTTP request handling and business logic implementation.

Author: AI Developer Testbed Team
Version: 1.2.1 - Clean Architecture Implementation
Last Updated: 2025-08-06
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body

# Import use cases
from application.use_cases.comprehensive_ethics_analysis_use_case import ComprehensiveEthicsAnalysisUseCase
from application.use_cases.meta_ethics_analysis_use_case import MetaEthicsAnalysisUseCase
from application.use_cases.normative_ethics_analysis_use_case import NormativeEthicsAnalysisUseCase
from application.use_cases.applied_ethics_analysis_use_case import AppliedEthicsAnalysisUseCase
from application.use_cases.ml_training_guidance_use_case import MLTrainingGuidanceUseCase

# Import dependencies
from application.dependencies.get_orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/ethics",
    tags=["ethics"],
    responses={404: {"description": "Not found"}},
)

@router.post("/comprehensive-analysis")
async def comprehensive_ethics_analysis(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Comprehensive multi-framework ethical analysis for ML development.
    
    This endpoint provides a comprehensive ethical analysis across
    multiple ethical frameworks, specifically tailored for ML development.
    
    Args:
        request: The analysis request
        
    Returns:
        Dict[str, Any]: Comprehensive ethical analysis
    """
    try:
        # Create use case with injected dependencies
        use_case = ComprehensiveEthicsAnalysisUseCase(orchestrator)
        
        # Execute use case
        analysis = await use_case.execute(request)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in comprehensive_ethics_analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ethics analysis error: {str(e)}"
        )

@router.post("/meta-analysis")
async def meta_ethics_analysis(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Meta-ethical analysis focusing on philosophical foundations.

    This endpoint provides a meta-ethical analysis focusing on
    philosophical foundations and ethical theory.

    Args:
        request: The analysis request

    Returns:
        Dict[str, Any]: Meta-ethical analysis
    """
    try:
        # Create use case with injected dependencies
        use_case = MetaEthicsAnalysisUseCase(orchestrator)

        # Create controller with injected dependencies
        controller = EthicsController(
            use_case,
            NormativeEthicsAnalysisUseCase(orchestrator),
            AppliedEthicsAnalysisUseCase(orchestrator),
            MLTrainingGuidanceUseCase(orchestrator)
        )

        # Execute use case
        analysis = await controller.meta_ethics_analysis(request)

        return analysis

    except Exception as e:
        logger.error(f"Error in meta_ethics_analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meta-ethics analysis error: {str(e)}"
        )

@router.post("/normative-analysis")
async def normative_ethics_analysis(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Normative ethical analysis across major moral frameworks.

    This endpoint provides a normative ethical analysis across
    major moral frameworks, including virtue ethics, deontology,
    and consequentialism.

    Args:
        request: The analysis request

    Returns:
        Dict[str, Any]: Normative ethical analysis
    """
    try:
        # Create use case with injected dependencies
        use_case = NormativeEthicsAnalysisUseCase(orchestrator)

        # Create controller with injected dependencies
        controller = EthicsController(
            MetaEthicsAnalysisUseCase(orchestrator),
            use_case,
            AppliedEthicsAnalysisUseCase(orchestrator),
            MLTrainingGuidanceUseCase(orchestrator)
        )

        # Execute use case
        analysis = await controller.normative_ethics_analysis(request)

        return analysis

    except Exception as e:
        logger.error(f"Error in normative_ethics_analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Normative ethics analysis error: {str(e)}"
        )

@router.post("/applied-analysis")
async def applied_ethics_analysis(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    Applied ethical analysis for practical implementation.
    
    This endpoint provides an applied ethical analysis for
    practical implementation in ML systems.
    
    Args:
        request: The analysis request
        
    Returns:
        Dict[str, Any]: Applied ethical analysis
    """
    try:
        # Create use case with injected dependencies
        use_case = AppliedEthicsAnalysisUseCase(orchestrator)
        
        # Execute use case
        analysis = await use_case.execute(request)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in applied_ethics_analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Applied ethics analysis error: {str(e)}"
        )

@router.post("/ml-training-guidance")
async def ml_training_guidance(
    request: Dict[str, Any] = Body(...),
    orchestrator=Depends(get_orchestrator)
):
    """
    ML-specific training guidance and ethical recommendations.
    
    This endpoint provides ML-specific training guidance and
    ethical recommendations for model development.
    
    Args:
        request: The guidance request
        
    Returns:
        Dict[str, Any]: ML training guidance
    """
    try:
        # Create use case with injected dependencies
        use_case = MLTrainingGuidanceUseCase(orchestrator)
        
        # Execute use case
        guidance = await use_case.execute(request)
        
        return guidance
        
    except Exception as e:
        logger.error(f"Error in ml_training_guidance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML training guidance error: {str(e)}"
        )
