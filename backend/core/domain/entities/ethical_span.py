"""
EthicalSpan Entity - Core Domain Entity

This module defines the EthicalSpan entity, which represents a span of text
with ethical evaluation scores and violation flags across multiple ethical perspectives.

Author: AI Developer Testbed Team
Version: 1.1.0 - Clean Architecture Implementation
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

class EthicalSpan(BaseModel):
    """Represents a span of text with ethical evaluation"""
    text: str = Field(..., description="The text content of the span")
    start: int = Field(..., ge=0, description="Start character offset of the span")
    end: int = Field(..., ge=0, description="End character offset of the span")
    virtue_score: float = Field(..., ge=0.0, le=1.0, description="Virtue ethics score (0-1)")
    deontological_score: float = Field(..., ge=0.0, le=1.0, description="Deontological ethics score (0-1)")
    consequentialist_score: float = Field(..., ge=0.0, le=1.0, description="Consequentialist ethics score (0-1)")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Combined ethical score (0-1)")
    is_violation: bool = Field(..., description="Whether this span represents an ethical violation")
    virtue_violation: bool = Field(False, description="Whether this span violates virtue ethics")
    deontological_violation: bool = Field(False, description="Whether this span violates deontological ethics")
    consequentialist_violation: bool = Field(False, description="Whether this span violates consequentialist ethics")
    
    # Optional fields with defaults
    violation_type: Optional[str] = Field(
        default=None,
        description="Type of ethical violation if is_violation is True"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of the ethical evaluation"
    )
    
    # v1.1 UPGRADE: Enhanced analysis fields
    dominant_intent: Optional[str] = Field(
        default=None,
        description="The dominant intent detected for this span"
    )
    intent_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for the dominant intent classification"
    )
    intent_category: Optional[str] = Field(
        default=None,
        description="Detected intent category for this span"
    )
    intent_scores: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Dictionary mapping intent categories to their confidence scores"
    )
    causal_impact: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Causal impact score (0-1)"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncertainty score (0-1)"
    )
    purpose_alignment: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Purpose alignment score (0-1)"
    )
    
    # Added is_minimal field to fix the reference in to_dict method
    is_minimal: bool = Field(
        default=False,
        description="Whether this span is a minimal unethical span"
    )
    
    @model_validator(mode='after')
    def validate_span_indices(self):
        if hasattr(self, 'start') and hasattr(self, 'end') and self.start > self.end:
            raise ValueError('start index must be less than or equal to end index')
        return self

    @model_validator(mode='after')
    def validate_span(self):
        """Validate span values"""
        # If text is provided, validate it matches the span
        if hasattr(self, 'text') and hasattr(self, 'start') and hasattr(self, 'end'):
            text_length = len(self.text)
            span_length = self.end - self.start
            if span_length <= 0:
                raise ValueError("end must be greater than start")
            if span_length != text_length:
                # Adjust the end index to match the text length
                self.end = self.start + text_length
                
        # If it's a violation, ensure violation_type is provided
        if getattr(self, 'is_violation', False) and not getattr(self, 'violation_type', None):
            raise ValueError("violation_type is required when is_violation is True")
            
        return self

    @property
    def any_violation(self) -> bool:
        """Check if any perspective flags this span as unethical"""
        return self.virtue_violation or self.deontological_violation or self.consequentialist_violation
    
    def has_violation(self) -> bool:
        """Compatibility method for optimized evaluation engine"""
        return self.any_violation
    
    @property
    def violation_perspectives(self) -> List[str]:
        """Return list of perspectives that flag this span"""
        perspectives = []
        if self.virtue_violation:
            perspectives.append("virtue")
        if self.deontological_violation:
            perspectives.append("deontological")
        if self.consequentialist_violation:
            perspectives.append("consequentialist")
        return perspectives
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for API responses"""
        return {
            'start': int(self.start),
            'end': int(self.end),
            'text': str(self.text),
            'virtue_score': float(self.virtue_score),
            'deontological_score': float(self.deontological_score),
            'consequentialist_score': float(self.consequentialist_score),
            'virtue_violation': bool(self.virtue_violation),
            'deontological_violation': bool(self.deontological_violation),
            'consequentialist_violation': bool(self.consequentialist_violation),
            'any_violation': bool(self.any_violation),
            'violation_perspectives': self.violation_perspectives,
            'is_minimal': bool(self.is_minimal)
        }
