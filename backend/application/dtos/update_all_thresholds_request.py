"""
Update All Thresholds Request DTO for the Ethical AI Testbed.

This module defines the data transfer object for updating all thresholds at once.
"""

from pydantic import BaseModel, Field

class UpdateAllThresholdsRequest(BaseModel):
    """
    Request model for updating all threshold values at once.
    """
    virtue_threshold: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Virtue threshold slider value (0.0 to 1.0)"
    )
    deontological_threshold: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Deontological threshold slider value (0.0 to 1.0)"
    )
    consequentialist_threshold: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Consequentialist threshold slider value (0.0 to 1.0)"
    )
    use_exponential: bool = Field(
        default=True, 
        description="Whether to use exponential scaling"
    )
