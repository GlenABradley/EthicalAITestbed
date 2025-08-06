"""
Threshold Scaling Response DTO for the Ethical AI Testbed.

This module defines the data transfer object for threshold scaling responses.
"""

from typing import Dict
from pydantic import BaseModel, Field

class ThresholdScalingResponse(BaseModel):
    """
    🎓 THRESHOLD SCALING RESPONSE MODEL:
    ════════════════════════════════════
    
    This model defines the response structure for threshold scaling operations.
    It provides detailed information about the applied scaling.
    """
    status: str = Field(..., description="Operation status (success/error)")
    slider_value: float = Field(..., description="Original slider value (0.0 to 1.0)")
    scaled_threshold: float = Field(..., description="Resulting threshold value (0.0 to 0.5)")
    scaling_type: str = Field(..., description="Type of scaling applied (exponential/linear)")
    formula: str = Field(..., description="Mathematical formula used for scaling")
    updated_parameters: Dict[str, float] = Field(
        ...,
        description="Updated parameter values that were affected by this scaling"
    )
