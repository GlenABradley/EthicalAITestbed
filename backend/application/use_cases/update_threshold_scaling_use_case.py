"""
Update Threshold Scaling Use Case for the Ethical AI Testbed.

This module defines the use case for updating threshold scaling.
"""

import logging
import math
from typing import Dict, Any

from application.dtos.threshold_scaling_response import ThresholdScalingResponse

logger = logging.getLogger(__name__)

class UpdateThresholdScalingUseCase:
    """
    Use case for updating threshold scaling.
    
    This class implements the use case for updating threshold scaling
    based on a normalized slider value. It follows the Clean Architecture
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
        threshold_type: str,
        slider_value: float,
        use_exponential: bool = True
    ) -> ThresholdScalingResponse:
        """
        Execute the use case to update threshold scaling.
        
        Args:
            threshold_type: Type of threshold to update ('virtue', 'deontological', or 'consequentialist')
            slider_value: Slider value between 0.0 and 1.0 to scale the threshold
            use_exponential: Whether to use exponential scaling
            
        Returns:
            ThresholdScalingResponse: The scaling operation results
        """
        logger.info(f"Updating {threshold_type} threshold scaling: slider_value={slider_value}, use_exponential={use_exponential}")
        
        try:
            # Validate threshold type
            valid_types = ["virtue", "deontological", "consequentialist"]
            if threshold_type not in valid_types:
                raise ValueError(f"Invalid threshold type: {threshold_type}. Must be one of: {', '.join(valid_types)}")
                
            # Validate slider value
            if not 0.0 <= slider_value <= 1.0:
                raise ValueError(f"Invalid slider value: {slider_value}. Must be between 0.0 and 1.0")
                
            # Get evaluator from orchestrator
            evaluator = self.orchestrator.evaluator
            
            # Calculate scaled threshold
            if use_exponential:
                # Exponential scaling (more granularity at lower values)
                # Formula: threshold = 0.5 * e^(-5 * slider_value)
                scaled_threshold = 0.5 * math.exp(-5 * slider_value)
                formula = "0.5 * e^(-5 * slider_value)"
                scaling_type = "exponential"
            else:
                # Linear scaling
                # Formula: threshold = 0.5 * (1 - slider_value)
                scaled_threshold = 0.5 * (1 - slider_value)
                formula = "0.5 * (1 - slider_value)"
                scaling_type = "linear"
                
            # Apply the threshold
            if threshold_type == "virtue":
                evaluator.virtue_threshold = scaled_threshold
                attribute_name = "virtue_threshold"
            elif threshold_type == "deontological":
                evaluator.deontological_threshold = scaled_threshold
                attribute_name = "deontological_threshold"
            else:  # consequentialist
                evaluator.consequentialist_threshold = scaled_threshold
                attribute_name = "consequentialist_threshold"
                
            # Create response
            updated_parameters = {attribute_name: scaled_threshold}
            
            return ThresholdScalingResponse(
                status="success",
                slider_value=slider_value,
                scaled_threshold=scaled_threshold,
                scaling_type=scaling_type,
                formula=formula,
                updated_parameters=updated_parameters
            )
            
        except Exception as e:
            logger.error(f"Error updating threshold scaling: {str(e)}", exc_info=True)
            raise
