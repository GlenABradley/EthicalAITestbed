"""
Evaluate Text Use Case for the Ethical AI Testbed.

This module defines the use case for evaluating text for ethical considerations.
"""

import logging
from typing import Dict, Any, Optional

from core.domain.entities.ethical_evaluation import EthicalEvaluation

logger = logging.getLogger(__name__)

class EvaluateTextUseCase:
    """
    Use case for evaluating text for ethical considerations.
    
    This class implements the use case for evaluating text using the
    unified ethical orchestrator. It follows the Clean Architecture
    pattern for use cases.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the use case with dependencies.
        
        Args:
            orchestrator: The unified ethical orchestrator
        """
        self.orchestrator = orchestrator
        
    async def execute(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tau_slider: Optional[float] = None
    ) -> EthicalEvaluation:
        """
        Execute the use case to evaluate text.
        
        Args:
            text: The text to evaluate
            context: Additional context for the evaluation
            parameters: Evaluation parameters and preferences
            tau_slider: Threshold scaling slider value (0.0 to 1.0)
            
        Returns:
            EthicalEvaluation: The evaluation results
        """
        logger.info(f"Evaluating text (length: {len(text)})")
        
        # Default values
        if context is None:
            context = {}
        if parameters is None:
            parameters = {}
            
        # Add tau_slider to context if provided
        if tau_slider is not None:
            context["tau_slider"] = tau_slider
            
        # Delegate to orchestrator
        evaluation_result = await self.orchestrator.evaluate_content(
            text=text,
            context=context,
            parameters=parameters
        )
        
        return evaluation_result
