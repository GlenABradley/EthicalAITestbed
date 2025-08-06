"""
Threshold Scaling Request DTO for the Ethical AI Testbed.

This module defines the data transfer object for threshold scaling requests.
"""

from pydantic import BaseModel, Field

class ThresholdScalingRequest(BaseModel):
    """
    🎓 THRESHOLD SCALING REQUEST MODEL:
    ═══════════════════════════════════
    
    This model defines the structure for threshold scaling requests.
    It's used to dynamically adjust the sensitivity of the ethical evaluation.
    """
    threshold_type: str = Field(
        ...,
        description="Type of threshold to update ('virtue', 'deontological', or 'consequentialist')",
        example="virtue"
    )
    slider_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Slider value between 0.0 and 1.0 to scale the threshold",
        example=0.5
    )
    use_exponential: bool = Field(
        default=True,
        description="Whether to use exponential scaling (provides more granularity at lower values)",
        example=True
    )
