"""
Update Parameters Use Case for the Ethical AI Testbed.

This module defines the use case for updating evaluation parameters.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class UpdateParametersUseCase:
    """
    Use case for updating evaluation parameters.
    
    This class implements the use case for updating the ethical
    evaluation parameters. It follows the Clean Architecture
    pattern for use cases.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the use case with dependencies.
        
        Args:
            orchestrator: The unified ethical orchestrator
        """
        self.orchestrator = orchestrator
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the use case to update parameters.
        
        Args:
            params: The parameters to update
            
        Returns:
            Dict[str, Any]: The updated parameters
        """
        logger.info(f"Updating evaluation parameters: {params}")
        
        try:
            # Get evaluator from orchestrator
            evaluator = self.orchestrator.evaluator
            
            # Update parameters
            updated = {}
            
            # Handle threshold parameters
            if "virtue_threshold" in params:
                evaluator.virtue_threshold = float(params["virtue_threshold"])
                updated["virtue_threshold"] = evaluator.virtue_threshold
                
            if "deontological_threshold" in params:
                evaluator.deontological_threshold = float(params["deontological_threshold"])
                updated["deontological_threshold"] = evaluator.deontological_threshold
                
            if "consequentialist_threshold" in params:
                evaluator.consequentialist_threshold = float(params["consequentialist_threshold"])
                updated["consequentialist_threshold"] = evaluator.consequentialist_threshold
                
            # Handle other numeric parameters
            for param in ["span_overlap_threshold", "confidence_threshold", 
                         "max_spans_to_check", "min_span_length", "max_span_length"]:
                if param in params:
                    setattr(evaluator, param, float(params[param]))
                    updated[param] = getattr(evaluator, param)
                    
            # Handle boolean parameters
            for param in ["use_graph_attention", "use_intent_hierarchy", 
                         "use_causal_counterfactual", "use_uncertainty_analyzer", 
                         "use_irl_purpose_alignment"]:
                if param in params:
                    setattr(evaluator, param, bool(params[param]))
                    updated[param] = getattr(evaluator, param)
                    
            # Handle configuration parameters
            if "config" in params and hasattr(self.orchestrator, 'config'):
                for key, value in params["config"].items():
                    self.orchestrator.config[key] = value
                    updated[f"config.{key}"] = value
                    
            return {
                "status": "success",
                "message": "Parameters updated successfully",
                "updated": updated
            }
            
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to update parameters: {str(e)}",
                "error": str(e)
            }
