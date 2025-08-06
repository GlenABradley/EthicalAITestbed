"""
Get Parameters Use Case for the Ethical AI Testbed.

This module defines the use case for retrieving evaluation parameters.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GetParametersUseCase:
    """
    Use case for retrieving evaluation parameters.
    
    This class implements the use case for retrieving the current
    ethical evaluation parameters. It follows the Clean Architecture
    pattern for use cases.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the use case with dependencies.
        
        Args:
            orchestrator: The unified ethical orchestrator
        """
        self.orchestrator = orchestrator
        
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the use case to retrieve parameters.
        
        Returns:
            Dict[str, Any]: The current evaluation parameters
        """
        logger.info("Retrieving evaluation parameters")
        
        try:
            # Get evaluator from orchestrator
            evaluator = self.orchestrator.evaluator
            
            # Get parameters
            parameters = {
                "virtue_threshold": evaluator.virtue_threshold,
                "deontological_threshold": evaluator.deontological_threshold,
                "consequentialist_threshold": evaluator.consequentialist_threshold,
                "span_overlap_threshold": evaluator.span_overlap_threshold,
                "confidence_threshold": evaluator.confidence_threshold,
                "max_spans_to_check": evaluator.max_spans_to_check,
                "min_span_length": evaluator.min_span_length,
                "max_span_length": evaluator.max_span_length,
                "use_graph_attention": evaluator.use_graph_attention,
                "use_intent_hierarchy": evaluator.use_intent_hierarchy,
                "use_causal_counterfactual": evaluator.use_causal_counterfactual,
                "use_uncertainty_analyzer": evaluator.use_uncertainty_analyzer,
                "use_irl_purpose_alignment": evaluator.use_irl_purpose_alignment
            }
            
            # Add configuration parameters if available
            if hasattr(self.orchestrator, 'config'):
                parameters["config"] = self.orchestrator.config
                
            return parameters
            
        except Exception as e:
            logger.error(f"Error retrieving parameters: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "status": "error",
                "message": "Failed to retrieve parameters"
            }
