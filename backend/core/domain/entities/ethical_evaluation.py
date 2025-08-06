"""
EthicalEvaluation Entity - Core Domain Entity

This module defines the EthicalEvaluation entity, which represents a complete
ethical evaluation of text with spans, scores, and analysis results. This is a
central domain entity in the Ethical AI Testbed system that encapsulates the
results of multi-perspective ethical analysis across virtue ethics, deontological
ethics, and consequentialist frameworks.

The EthicalEvaluation entity contains:
- The original input text and its tokenization
- All evaluated spans with their ethical scores
- Minimal unethical spans (the smallest units with ethical violations)
- Overall ethical assessment
- Performance metrics (processing time)
- Parameters used for the evaluation
- Advanced analysis results (causal, uncertainty, purpose alignment)

This entity is immutable and serves as a comprehensive record of an ethical
evaluation, suitable for persistence, API responses, and further analysis.

Author: AI Developer Testbed Team
Version: 1.2.1 - Clean Architecture Implementation
Last Updated: 2025-08-06
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from backend.core.domain.entities.ethical_span import EthicalSpan
from backend.core.domain.value_objects.ethical_parameters import EthicalParameters

class DynamicScalingResult(BaseModel):
    """Results of dynamic threshold scaling"""
    original_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Original thresholds before scaling"
    )
    adjusted_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Adjusted thresholds after scaling"
    )
    scaling_factor: float = Field(
        ...,
        description="Factor by which thresholds were scaled"
    )
    ambiguity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ambiguity score used for scaling (0-1)"
    )
    scaling_method: str = Field(
        default="exponential",
        description="Method used for scaling (exponential or linear)"
    )

class EthicalEvaluation(BaseModel):
    """Complete ethical evaluation result"""
    input_text: str = Field(..., description="The input text that was evaluated")
    tokens: List[str] = Field(default_factory=list, description="List of tokens from the input text")
    spans: List[EthicalSpan] = Field(default_factory=list, description="List of all evaluated spans")
    minimal_spans: List[EthicalSpan] = Field(default_factory=list, description="Minimal unethical spans")
    overall_ethical: bool = Field(..., description="Whether the overall text is considered ethical")
    processing_time: float = Field(..., ge=0, description="Time taken to process the evaluation in seconds")
    parameters: EthicalParameters = Field(..., description="Parameters used for this evaluation")
    dynamic_scaling_result: Optional[DynamicScalingResult] = Field(
        default=None, 
        description="Results of dynamic scaling if enabled"
    )
    evaluation_id: str = Field(
        default_factory=lambda: f"eval_{int(time.time() * 1000)}",
        description="Unique identifier for this evaluation"
    )
    causal_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of causal counterfactual analysis if enabled"
    )
    uncertainty_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of uncertainty analysis if enabled"
    )
    purpose_alignment_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of purpose alignment analysis if enabled"
    )
    
    @property
    def violation_count(self) -> int:
        """Count of spans with violations"""
        return len([s for s in self.spans if s.any_violation])
    
    @property
    def minimal_violation_count(self) -> int:
        """Count of minimal spans with violations"""
        return len([s for s in self.minimal_spans if s.any_violation])
    
    @property
    def all_spans_with_scores(self) -> List[Dict[str, Any]]:
        """All spans with their scores for detailed analysis"""
        return [s.to_dict() for s in self.spans]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary for API responses"""
        result = {
            'evaluation_id': str(self.evaluation_id),
            'input_text': str(self.input_text),
            'tokens': [str(token) for token in self.tokens],
            'spans': [span.to_dict() for span in self.spans],
            'minimal_spans': [span.to_dict() for span in self.minimal_spans],
            'all_spans_with_scores': self.all_spans_with_scores,
            'overall_ethical': bool(self.overall_ethical),
            'processing_time': float(self.processing_time),
            'violation_count': int(self.violation_count),
            'minimal_violation_count': int(self.minimal_violation_count),
            'parameters': self.parameters.to_dict()
        }
        
        # Add optional fields if they exist
        if self.dynamic_scaling_result:
            result['dynamic_scaling_result'] = self.dynamic_scaling_result.model_dump() if hasattr(self.dynamic_scaling_result, 'model_dump') else self.dynamic_scaling_result
            
        if self.causal_analysis:
            result['causal_analysis'] = self.causal_analysis
            
        if self.uncertainty_analysis:
            result['uncertainty_analysis'] = self.uncertainty_analysis
            
        if self.purpose_alignment_analysis:
            result['purpose_alignment_analysis'] = self.purpose_alignment_analysis
            
        return result
