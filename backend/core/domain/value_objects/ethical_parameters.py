"""
EthicalParameters Value Object - Core Domain Value Object

This module defines the EthicalParameters value object, which represents the
configuration parameters for ethical evaluation. As a value object in the domain model,
EthicalParameters is immutable and defined by its attributes rather than an identity.

The EthicalParameters value object encapsulates all configuration settings that control
the behavior of the ethical evaluation engine, including:

- Ethical thresholds for each perspective (virtue, deontological, consequentialist)
- Vector weights for balancing different ethical frameworks
- Span detection parameters for text analysis granularity
- Model configuration parameters for embedding and analysis
- Feature flags for enabling/disabling advanced analysis capabilities
- Performance optimization settings

This value object is used throughout the system to ensure consistent evaluation
parameters and to enable parameter adjustments without modifying core evaluation logic.

Author: AI Developer Testbed Team
Version: 1.2.1 - Clean Architecture Implementation
Last Updated: 2025-08-06
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator

class EthicalParameters(BaseModel):
    """Configuration parameters for ethical evaluation"""
    # Thresholds for each perspective (τ_P) - Optimized for granular sensitivity
    virtue_threshold: float = Field(default=0.15, description="Virtue ethics threshold (0-1)")
    deontological_threshold: float = Field(default=0.15, description="Deontological ethics threshold (0-1)")
    consequentialist_threshold: float = Field(default=0.15, description="Consequentialist ethics threshold (0-1)")
    
    # Vector magnitudes for ethical axes
    virtue_weight: float = Field(default=1.0, description="Weight for virtue ethics")
    deontological_weight: float = Field(default=1.0, description="Weight for deontological ethics")
    consequentialist_weight: float = Field(default=1.0, description="Weight for consequentialist ethics")
    
    # Span detection parameters (optimized for performance)
    max_span_length: int = Field(default=5, gt=0, description="Maximum token span length to evaluate")
    min_span_length: int = Field(default=1, gt=0, description="Minimum token span length to evaluate")
    
    # Model parameters - v1.1 UPGRADE: Keep proven MiniLM but add graph attention
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model to use for embeddings"
    )
    
    # Dynamic scaling parameters
    enable_dynamic_scaling: bool = Field(default=False, description="Enable dynamic threshold scaling")
    enable_cascade_filtering: bool = Field(default=False, description="Enable cascade filtering")
    enable_learning_mode: bool = Field(default=False, description="Enable learning from feedback")
    exponential_scaling: bool = Field(default=True, description="Use exponential scaling for thresholds")
    
    # Cascade filtering thresholds - fine-tuned for better accuracy
    cascade_high_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="High threshold for cascade filtering (0-1)"
    )
    cascade_low_threshold: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Low threshold for cascade filtering (0-1)"
    )
    
    # Learning parameters
    learning_weight: float = 0.3
    min_learning_samples: int = 5
    
    # v1.1 UPGRADE: Graph Attention Parameters for Distributed Pattern Detection
    enable_graph_attention: bool = Field(
        default=True,
        description="Enable Graph Attention Network for distributed patterns"
    )
    graph_decay_lambda: float = Field(
        default=5.0,
        description="Decay factor for graph attention distance weighting",
        gt=0.0
    )
    graph_similarity_threshold: float = Field(
        default=0.1,
        description="Minimum similarity for graph edges (0-1)",
        ge=0.0,
        le=1.0
    )
    graph_attention_heads: int = Field(
        default=4,
        description="Number of attention heads in graph attention layer",
        gt=0
    )
    
    # v1.1 UPGRADE: Intent Hierarchy parameters
    enable_intent_hierarchy: bool = Field(
        default=True,
        description="Enable intent hierarchy for harm classification"
    )
    intent_threshold: float = Field(
        default=0.6,
        description="Confidence threshold for intent classification (0-1)",
        ge=0.0,
        le=1.0
    )
    intent_categories: List[str] = Field(
        default_factory=lambda: [
            "fraud", "manipulation", "coercion", "deception", 
            "harassment", "discrimination", "violence", "exploitation"
        ],
        description="List of intent categories for harm classification"
    )
    enable_contrastive_learning: bool = Field(
        default=False,
        description="Enable contrastive learning for intent classification"
    )
    
    # v1.1 UPGRADE: Causal Analysis parameters
    enable_causal_analysis: bool = Field(
        default=True,
        description="Enable causal counterfactual analysis"
    )
    autonomy_delta_threshold: float = Field(
        default=0.1,
        description="Minimum autonomy delta for significant causal impact (0-1)",
        ge=0.0,
        le=1.0
    )
    causal_intervention_types: List[str] = Field(
        default_factory=lambda: ["removal", "masking", "neutralize", "soften"],
        description="Types of interventions to apply for causal analysis"
    )
    max_counterfactuals_per_span: int = Field(
        default=4,
        description="Maximum number of counterfactuals to generate per span",
        gt=0
    )
    
    # v1.1 UPGRADE: Uncertainty Analysis parameters
    enable_uncertainty_analysis: bool = Field(
        default=True,
        description="Enable uncertainty estimation and human routing"
    )
    uncertainty_threshold: float = Field(
        default=0.25,
        description="Uncertainty threshold for human routing (0-1)",
        ge=0.0,
        le=1.0
    )
    bootstrap_samples: int = Field(
        default=3,
        description="Number of bootstrap samples for uncertainty estimation",
        gt=0
    )
    uncertainty_dropout_rate: float = Field(
        default=0.15,
        description="Dropout rate for uncertainty estimation (0-1)",
        ge=0.0,
        le=1.0
    )
    auto_human_routing: bool = Field(
        default=True,
        description="Automatically route uncertain cases to human review"
    )
    
    # v1.1 UPGRADE: IRL Purpose Alignment parameters
    enable_purpose_alignment: bool = Field(
        default=True,
        description="Enable purpose alignment analysis"
    )
    purpose_alignment_threshold: float = Field(
        default=0.95,
        description="Minimum alignment score for purpose verification (0.5-1.0)",
        ge=0.5,
        le=1.0
    )
    auto_purpose_inference: bool = Field(
        default=True,
        description="Automatically infer purpose from context"
    )
    purpose_weight_adaptation: bool = Field(
        default=False,
        description="Dynamically adjust weights based on purpose alignment"
    )
    
    @field_validator('cascade_high_threshold', 'cascade_low_threshold', mode='after')
    @classmethod
    def validate_cascade_thresholds(cls, v, info):
        if info.field_name == 'cascade_low_threshold' and hasattr(info.data, 'cascade_high_threshold'):
            if v >= info.data.cascade_high_threshold:
                raise ValueError('cascade_low_threshold must be less than cascade_high_threshold')
        return v
    
    @field_validator('min_span_length', mode='after')
    @classmethod
    def validate_span_lengths(cls, v, info):
        if hasattr(info.data, 'max_span_length') and v > info.data.max_span_length:
            raise ValueError('min_span_length must be less than or equal to max_span_length')
        return v
    
    @model_validator(mode='after')
    def validate_parameters(self):
        """Validate parameters after initialization"""
        # Ensure thresholds are within valid range (0-1)
        for threshold_field in ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']:
            threshold_value = getattr(self, threshold_field, None)
            if threshold_value is not None and not 0 <= threshold_value <= 1:
                raise ValueError(f"{threshold_field} must be between 0 and 1")
                
        # Ensure span lengths are valid
        if hasattr(self, 'min_span_length') and hasattr(self, 'max_span_length'):
            if self.min_span_length <= 0 or self.max_span_length <= 0:
                raise ValueError("span lengths must be positive integers")
            if self.min_span_length > self.max_span_length:
                raise ValueError("min_span_length cannot be greater than max_span_length")
        
        # Graph attention validation
        if getattr(self, 'enable_graph_attention', False):
            if getattr(self, 'graph_decay_lambda', 0) <= 0:
                raise ValueError("graph_decay_lambda must be positive")
            if not 0 <= getattr(self, 'graph_similarity_threshold', 0) <= 1:
                raise ValueError("graph_similarity_threshold must be between 0 and 1")
            if getattr(self, 'graph_attention_heads', 0) <= 0:
                raise ValueError("graph_attention_heads must be positive")
                
        # Intent hierarchy validation
        if getattr(self, 'enable_intent_hierarchy', False) and not 0 <= getattr(self, 'intent_threshold', 0) <= 1:
            raise ValueError("intent_threshold must be between 0 and 1")
                
        # Causal analysis validation
        if getattr(self, 'enable_causal_analysis', False):
            if not 0 <= getattr(self, 'autonomy_delta_threshold', 0) <= 1:
                raise ValueError("autonomy_delta_threshold must be between 0 and 1")
            if getattr(self, 'max_counterfactuals_per_span', 0) <= 0:
                raise ValueError("max_counterfactuals_per_span must be positive")
                
        # Uncertainty analysis validation
        if getattr(self, 'enable_uncertainty_analysis', False):
            if not 0 <= getattr(self, 'uncertainty_threshold', 0) <= 1:
                raise ValueError("uncertainty_threshold must be between 0 and 1")
            if getattr(self, 'bootstrap_samples', 0) <= 0:
                raise ValueError("bootstrap_samples must be positive")
            if not 0 <= getattr(self, 'uncertainty_dropout_rate', 0) < 1:
                raise ValueError("uncertainty_dropout_rate must be between 0 and 1")
                
        # Purpose alignment validation
        if getattr(self, 'enable_purpose_alignment', False):
            if not 0 <= getattr(self, 'purpose_alignment_threshold', 0) <= 1:
                raise ValueError("purpose_alignment_threshold must be between 0 and 1")
                
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for API responses"""
        return self.model_dump()
