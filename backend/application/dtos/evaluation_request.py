"""
Evaluation Request DTO for the Ethical AI Testbed.

This module defines the data transfer object for evaluation requests.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class EvaluationRequest(BaseModel):
    """
    🎓 EVALUATION REQUEST MODEL:
    ════════════════════════════════
    
    This model defines the structure for ethical evaluation requests.
    It includes comprehensive validation and defaults to ensure
    robust API behavior.
    """
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=50000,
        description="Text content to evaluate for ethical compliance",
        example="This is a sample text for ethical evaluation."
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for the evaluation",
        example={"domain": "healthcare", "cultural_context": "western"}
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Evaluation parameters and preferences",
        example={"confidence_threshold": 0.8, "explanation_level": "detailed"}
    )
    mode: str = Field(
        default="production",
        description="Evaluation mode (development, production, research, educational)",
        example="production"
    )
    priority: str = Field(
        default="normal",
        description="Processing priority (critical, high, normal, background)",
        example="normal"
    )
    tau_slider: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Threshold scaling slider value from frontend (0.0 to 1.0)",
        example=0.5
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text content for basic safety."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")
        return v
        
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        """Validate evaluation mode."""
        valid_modes = ["development", "production", "research", "educational"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {', '.join(valid_modes)}")
        return v.lower()
        
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        """Validate processing priority."""
        valid_priorities = ["critical", "high", "normal", "background"]
        if v.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of: {', '.join(valid_priorities)}")
        return v.lower()
