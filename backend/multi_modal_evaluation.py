"""
Multi-Modal Evaluation System - Phase 4

This module implements a sophisticated multi-modal evaluation architecture for
comprehensive ethical analysis across different processing contexts. Inspired by
the principles of clean architecture and separation of concerns.

Architecture Philosophy:
- Each evaluation mode has distinct responsibilities and contexts
- Unified interface with mode-specific implementations  
- Composable evaluation pipeline with pluggable components
- Performance optimization through intelligent caching and routing
- Comprehensive error handling and graceful degradation

Modes:
1. PRE_EVALUATION: Screen inputs before processing (intent analysis, safety checks)
2. POST_EVALUATION: Validate outputs after processing (alignment, impact assessment)
3. STREAM_EVALUATION: Real-time analysis during processing (token-level monitoring)
4. BATCH_EVALUATION: Efficient bulk processing for datasets
5. INTERACTIVE_EVALUATION: Human-in-the-loop evaluation with feedback

Author: Ethical AI Developer Testbed Team
Version: 4.0.0 - Multi-Modal Evaluation System
Inspired by: Alan Kay's messaging philosophy, Donald Knuth's algorithmic precision,
             Barbara Liskov's data abstraction principles
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from contextlib import asynccontextmanager
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class EvaluationMode(Enum):
    """Evaluation modes for different processing contexts."""
    PRE_EVALUATION = "pre_evaluation"           # Before processing
    POST_EVALUATION = "post_evaluation"         # After processing  
    STREAM_EVALUATION = "stream_evaluation"     # During processing
    BATCH_EVALUATION = "batch_evaluation"       # Bulk processing
    INTERACTIVE_EVALUATION = "interactive_evaluation"  # Human-in-the-loop

class EvaluationPriority(Enum):
    """Priority levels for evaluation requests."""
    CRITICAL = "critical"      # Immediate processing required
    HIGH = "high"             # Process quickly
    MEDIUM = "medium"         # Standard processing
    LOW = "low"              # Background processing
    BATCH = "batch"          # Can be batched with others

class EvaluationStatus(Enum):
    """Status of evaluation requests."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class EvaluationContext:
    """Context information for evaluation requests."""
    request_id: str
    mode: EvaluationMode
    priority: EvaluationPriority
    timestamp: float
    source: str = "unknown"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    training_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "mode": self.mode.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "training_context": self.training_context,
            "metadata": self.metadata
        }

@dataclass
class PreEvaluationResult:
    """Results from pre-evaluation analysis."""
    should_proceed: bool
    intent_analysis: Dict[str, Any]
    safety_assessment: Dict[str, float]
    risk_factors: List[str]
    boundary_conditions: Dict[str, Any]
    processing_recommendations: List[str]
    preprocessing_adjustments: Dict[str, Any]
    confidence_score: float
    processing_time: float

@dataclass
class PostEvaluationResult:
    """Results from post-evaluation analysis."""
    output_approved: bool
    alignment_score: float
    impact_assessment: Dict[str, Any]
    validation_results: Dict[str, bool]
    content_safety: Dict[str, float]
    ethical_compliance: Dict[str, float]
    improvement_suggestions: List[str]
    human_review_required: bool
    processing_time: float

@dataclass
class StreamEvaluationResult:
    """Results from streaming evaluation analysis."""
    continue_processing: bool
    current_ethical_score: float
    accumulated_violations: int
    real_time_adjustments: Dict[str, Any]
    intervention_triggered: bool
    buffer_analysis: Dict[str, Any]
    processing_recommendations: List[str]
    performance_metrics: Dict[str, Any]
    processing_time: float

@dataclass
class UnifiedEvaluationResult:
    """Unified result format across all evaluation modes."""
    request_id: str
    mode: EvaluationMode
    status: EvaluationStatus
    overall_ethical_score: float
    decision: str  # PROCEED, HALT, REVIEW, ADJUST
    confidence: float
    
    # Mode-specific results
    pre_evaluation: Optional[PreEvaluationResult] = None
    post_evaluation: Optional[PostEvaluationResult] = None
    stream_evaluation: Optional[StreamEvaluationResult] = None
    
    # Common analysis
    ethical_vectors: Optional[Dict[str, List[float]]] = None
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    context: Optional[EvaluationContext] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "overall_ethical_score": self.overall_ethical_score,
            "decision": self.decision,
            "confidence": self.confidence,
            "ethical_vectors": self.ethical_vectors,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "processing_time": self.processing_time,
            "context": self.context.to_dict() if self.context else None,
            "error_details": self.error_details
        }
        
        # Add mode-specific results
        if self.pre_evaluation:
            result["pre_evaluation"] = self.pre_evaluation.__dict__
        if self.post_evaluation:
            result["post_evaluation"] = self.post_evaluation.__dict__
        if self.stream_evaluation:
            result["stream_evaluation"] = self.stream_evaluation.__dict__
            
        return result

class EvaluationModeInterface(ABC):
    """Abstract interface for evaluation modes following Liskov Substitution Principle."""
    
    @abstractmethod
    async def evaluate(self, 
                      content: Union[str, List[str]], 
                      context: EvaluationContext,
                      **kwargs) -> UnifiedEvaluationResult:
        """
        Evaluate content in the specific mode context.
        
        Args:
            content: Text content to evaluate (single string or list)
            context: Evaluation context with metadata
            **kwargs: Mode-specific parameters
            
        Returns:
            UnifiedEvaluationResult with mode-specific analysis
        """
        pass
    
    @abstractmethod
    async def configure(self, configuration: Dict[str, Any]) -> bool:
        """
        Configure the evaluation mode with specific parameters.
        
        Args:
            configuration: Mode-specific configuration parameters
            
        Returns:
            True if configuration was successful
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities and limitations of this evaluation mode.
        
        Returns:
            Dictionary describing mode capabilities
        """
        pass

class PreEvaluationMode(EvaluationModeInterface):
    """
    Pre-evaluation mode for screening inputs before processing.
    
    Focuses on:
    - Intent analysis and classification
    - Safety boundary checks  
    - Risk factor identification
    - Processing feasibility assessment
    """
    
    def __init__(self, ethical_evaluator=None, ml_ethics_engine=None):
        """Initialize pre-evaluation mode."""
        self.ethical_evaluator = ethical_evaluator
        self.ml_ethics_engine = ml_ethics_engine
        self.intent_classifiers = self._initialize_intent_classifiers()
        self.safety_boundaries = self._initialize_safety_boundaries()
        self.risk_thresholds = {
            "manipulation_risk": 0.3,
            "bias_risk": 0.4,
            "harm_potential": 0.2,
            "deception_indicators": 0.25
        }
        
    def _initialize_intent_classifiers(self) -> Dict[str, Any]:
        """Initialize intent classification systems."""
        return {
            "malicious_intent": {
                "patterns": [
                    r'\b(hack|exploit|manipulat\w+|deceiv\w+)\b',
                    r'\b(steal|fraud|scam|cheat)\b',
                    r'\b(harm|hurt|attack|destroy)\b'
                ],
                "threshold": 0.3
            },
            "educational_intent": {
                "patterns": [
                    r'\b(learn|teach|explain|understand)\b',
                    r'\b(help|assist|guide|support)\b',
                    r'\b(question|answer|clarify)\b'
                ],
                "threshold": 0.6
            },
            "creative_intent": {
                "patterns": [
                    r'\b(create|write|generate|compose)\b',
                    r'\b(story|poem|article|content)\b',
                    r'\b(creative|artistic|imaginative)\b'
                ],
                "threshold": 0.5
            }
        }
    
    def _initialize_safety_boundaries(self) -> Dict[str, Any]:
        """Initialize safety boundary definitions."""
        return {
            "content_length": {"min": 1, "max": 50000},
            "complexity_score": {"max": 0.9},
            "processing_time_estimate": {"max": 30.0},
            "resource_requirements": {"max_memory_mb": 1000},
            "ethical_red_flags": {
                "violence": 0.2,
                "discrimination": 0.15,
                "misinformation": 0.25,
                "manipulation": 0.2
            }
        }
    
    async def evaluate(self, 
                      content: Union[str, List[str]], 
                      context: EvaluationContext,
                      **kwargs) -> UnifiedEvaluationResult:
        """Perform pre-evaluation analysis."""
        start_time = time.time()
        
        try:
            # Convert to single string if needed
            text = content if isinstance(content, str) else " ".join(content)
            
            # Intent analysis
            intent_analysis = await self._analyze_intent(text)
            
            # Safety assessment
            safety_assessment = await self._assess_safety(text)
            
            # Risk factor identification
            risk_factors = await self._identify_risk_factors(text)
            
            # Boundary condition checks
            boundary_conditions = await self._check_boundaries(text, context)
            
            # Generate processing recommendations
            processing_recommendations = self._generate_processing_recommendations(
                intent_analysis, safety_assessment, risk_factors, boundary_conditions
            )
            
            # Determine if processing should proceed
            should_proceed = self._should_proceed_with_processing(
                intent_analysis, safety_assessment, risk_factors, boundary_conditions
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                intent_analysis, safety_assessment, boundary_conditions
            )
            
            # Create pre-evaluation result
            pre_result = PreEvaluationResult(
                should_proceed=should_proceed,
                intent_analysis=intent_analysis,
                safety_assessment=safety_assessment,
                risk_factors=risk_factors,
                boundary_conditions=boundary_conditions,
                processing_recommendations=processing_recommendations,
                preprocessing_adjustments=kwargs.get("preprocessing_adjustments", {}),
                confidence_score=confidence_score,
                processing_time=time.time() - start_time
            )
            
            # Generate unified result
            overall_score = min(
                intent_analysis.get("educational_score", 0.5),
                1.0 - max(safety_assessment.values()) if safety_assessment else 0.5
            )
            
            decision = "PROCEED" if should_proceed else "HALT"
            if not should_proceed and confidence_score < 0.7:
                decision = "REVIEW"
            
            return UnifiedEvaluationResult(
                request_id=context.request_id,
                mode=EvaluationMode.PRE_EVALUATION,
                status=EvaluationStatus.COMPLETED,
                overall_ethical_score=overall_score,
                decision=decision,
                confidence=confidence_score,
                pre_evaluation=pre_result,
                recommendations=processing_recommendations,
                warnings=[f"Risk factor: {rf}" for rf in risk_factors],
                processing_time=time.time() - start_time,
                context=context
            )
            
        except Exception as e:
            logger.error(f"Pre-evaluation failed: {e}")
            return UnifiedEvaluationResult(
                request_id=context.request_id,
                mode=EvaluationMode.PRE_EVALUATION,
                status=EvaluationStatus.FAILED,
                overall_ethical_score=0.0,
                decision="HALT",
                confidence=0.0,
                error_details=str(e),
                processing_time=time.time() - start_time,
                context=context
            )
    
    async def _analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze the intent behind the input text."""
        import re
        
        intent_scores = {}
        
        for intent_type, classifier in self.intent_classifiers.items():
            score = 0.0
            for pattern in classifier["patterns"]:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.1
            
            # Normalize score
            intent_scores[f"{intent_type}_score"] = min(score, 1.0)
        
        # Classify primary intent
        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        
        return {
            **intent_scores,
            "primary_intent": primary_intent,
            "intent_clarity": max(intent_scores.values()),
            "analysis_confidence": 0.8 if max(intent_scores.values()) > 0.5 else 0.4
        }
    
    async def _assess_safety(self, text: str) -> Dict[str, float]:
        """Assess safety factors in the input."""
        safety_scores = {}
        
        # Use existing ethical evaluator if available
        if self.ethical_evaluator:
            try:
                evaluation = self.ethical_evaluator.evaluate_text(text)
                
                # Extract safety-related scores
                if hasattr(evaluation, 'ethical_principles'):
                    principles = evaluation.ethical_principles
                    safety_scores.update({
                        "violence_risk": 1.0 - principles.get("non_aggression", 1.0),
                        "harm_potential": 1.0 - principles.get("harm_prevention", 1.0),
                        "discrimination_risk": principles.get("discrimination", 0.0),
                        "manipulation_risk": 1.0 - principles.get("transparency", 1.0)
                    })
                    
            except Exception as e:
                logger.warning(f"Ethical evaluator failed in pre-evaluation: {e}")
        
        # Fallback simple pattern-based assessment
        if not safety_scores:
            import re
            safety_patterns = {
                "violence_risk": [r'\b(kill|murder|attack|violence|weapon)\b'],
                "harm_potential": [r'\b(harm|hurt|damage|destroy|injure)\b'],
                "discrimination_risk": [r'\b(racist|sexist|discriminat\w+|bias\w*)\b'],
                "manipulation_risk": [r'\b(manipulat\w+|deceiv\w+|trick|fool)\b']
            }
            
            for risk_type, patterns in safety_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    score += matches * 0.2
                safety_scores[risk_type] = min(score, 1.0)
        
        return safety_scores
    
    async def _identify_risk_factors(self, text: str) -> List[str]:
        """Identify specific risk factors in the content."""
        risk_factors = []
        
        # Length-based risks
        if len(text) > self.safety_boundaries["content_length"]["max"]:
            risk_factors.append("Content exceeds maximum length")
        elif len(text) < self.safety_boundaries["content_length"]["min"]:
            risk_factors.append("Content below minimum length threshold")
        
        # Pattern-based risks
        import re
        risk_patterns = {
            "Contains potential harmful instructions": [r'\b(how to (hack|steal|hurt|kill)\b)'],
            "Potential misinformation": [r'\b(fake news|false information|lie about)\b'],
            "Manipulation indicators": [r'\b(you should|you must|trust me|believe me)\b'],
            "Bias indicators": [r'\b(all (women|men|blacks|whites) are)\b']
        }
        
        for risk_description, patterns in risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    risk_factors.append(risk_description)
                    break
        
        return risk_factors
    
    async def _check_boundaries(self, text: str, context: EvaluationContext) -> Dict[str, Any]:
        """Check various boundary conditions."""
        boundaries = {}
        
        # Content boundaries
        boundaries["length_check"] = {
            "current": len(text),
            "min": self.safety_boundaries["content_length"]["min"],
            "max": self.safety_boundaries["content_length"]["max"],
            "within_bounds": (
                self.safety_boundaries["content_length"]["min"] <= 
                len(text) <= 
                self.safety_boundaries["content_length"]["max"]
            )
        }
        
        # Processing time estimation
        estimated_time = len(text) * 0.01  # Simple estimation
        boundaries["time_estimate"] = {
            "estimated_seconds": estimated_time,
            "max_allowed": self.safety_boundaries["processing_time_estimate"]["max"],
            "within_bounds": estimated_time <= self.safety_boundaries["processing_time_estimate"]["max"]
        }
        
        # Resource estimation
        estimated_memory = len(text) * 0.001  # Simple estimation in MB
        boundaries["resource_estimate"] = {
            "estimated_memory_mb": estimated_memory,
            "max_allowed": self.safety_boundaries["resource_requirements"]["max_memory_mb"],
            "within_bounds": estimated_memory <= self.safety_boundaries["resource_requirements"]["max_memory_mb"]
        }
        
        # Context-specific boundaries
        if context.training_context:
            training_ctx = context.training_context
            if training_ctx.get("training_phase") == "production":
                boundaries["production_readiness"] = {
                    "requires_stricter_bounds": True,
                    "recommendation": "Apply production-grade safety filters"
                }
        
        return boundaries
    
    def _generate_processing_recommendations(self, 
                                           intent_analysis: Dict[str, Any],
                                           safety_assessment: Dict[str, float],
                                           risk_factors: List[str],
                                           boundary_conditions: Dict[str, Any]) -> List[str]:
        """Generate recommendations for processing."""
        recommendations = []
        
        # Intent-based recommendations
        primary_intent = intent_analysis.get("primary_intent", "")
        if "malicious" in primary_intent:
            recommendations.append("Apply enhanced security filtering")
            recommendations.append("Consider human review before processing")
        elif "educational" in primary_intent:
            recommendations.append("Enable educational content optimization")
            
        # Safety-based recommendations
        max_safety_risk = max(safety_assessment.values()) if safety_assessment else 0.0
        if max_safety_risk > 0.5:
            recommendations.append("Apply strict safety constraints during processing")
        elif max_safety_risk > 0.3:
            recommendations.append("Monitor processing for safety violations")
        
        # Risk-based recommendations
        if len(risk_factors) > 2:
            recommendations.append("Implement comprehensive risk mitigation")
        elif risk_factors:
            recommendations.append("Apply targeted risk mitigation for identified factors")
        
        # Boundary-based recommendations
        if not boundary_conditions.get("length_check", {}).get("within_bounds", True):
            recommendations.append("Adjust content length before processing")
            
        if not boundary_conditions.get("time_estimate", {}).get("within_bounds", True):
            recommendations.append("Consider processing in smaller chunks")
        
        return recommendations
    
    def _should_proceed_with_processing(self, 
                                      intent_analysis: Dict[str, Any],
                                      safety_assessment: Dict[str, float],
                                      risk_factors: List[str],
                                      boundary_conditions: Dict[str, Any]) -> bool:
        """Determine if processing should proceed."""
        # Check for blocking conditions
        
        # Malicious intent check
        if intent_analysis.get("malicious_intent_score", 0) > 0.7:
            return False
        
        # Critical safety risk check
        if any(score > 0.8 for score in safety_assessment.values()):
            return False
        
        # Critical boundary violations
        if not boundary_conditions.get("length_check", {}).get("within_bounds", True):
            return False
        if not boundary_conditions.get("resource_estimate", {}).get("within_bounds", True):
            return False
        
        # Too many risk factors
        if len(risk_factors) > 3:
            return False
        
        return True
    
    def _calculate_confidence(self, 
                            intent_analysis: Dict[str, Any],
                            safety_assessment: Dict[str, float],
                            boundary_conditions: Dict[str, Any]) -> float:
        """Calculate confidence in the pre-evaluation decision."""
        confidence_factors = []
        
        # Intent analysis confidence
        confidence_factors.append(intent_analysis.get("analysis_confidence", 0.5))
        
        # Safety assessment confidence (inverse of uncertainty)
        if safety_assessment:
            safety_variance = np.var(list(safety_assessment.values()))
            safety_confidence = 1.0 - min(safety_variance * 2, 1.0)
            confidence_factors.append(safety_confidence)
        
        # Boundary conditions confidence
        boundary_checks = [
            boundary_conditions.get("length_check", {}).get("within_bounds", True),
            boundary_conditions.get("time_estimate", {}).get("within_bounds", True),
            boundary_conditions.get("resource_estimate", {}).get("within_bounds", True)
        ]
        boundary_confidence = sum(boundary_checks) / len(boundary_checks)
        confidence_factors.append(boundary_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def configure(self, configuration: Dict[str, Any]) -> bool:
        """Configure pre-evaluation mode."""
        try:
            if "risk_thresholds" in configuration:
                self.risk_thresholds.update(configuration["risk_thresholds"])
            
            if "safety_boundaries" in configuration:
                self.safety_boundaries.update(configuration["safety_boundaries"])
            
            if "intent_classifiers" in configuration:
                # Allow updating classifier thresholds
                for intent_type, config in configuration["intent_classifiers"].items():
                    if intent_type in self.intent_classifiers:
                        self.intent_classifiers[intent_type].update(config)
            
            return True
        except Exception as e:
            logger.error(f"Pre-evaluation configuration failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get pre-evaluation mode capabilities."""
        return {
            "mode": "PRE_EVALUATION",
            "description": "Screen inputs before processing with intent analysis and safety checks",
            "features": [
                "Intent classification",
                "Safety boundary checking",
                "Risk factor identification", 
                "Processing feasibility assessment",
                "Resource requirement estimation"
            ],
            "supported_content_types": ["text"],
            "performance": {
                "typical_processing_time_ms": "10-100",
                "max_content_length": self.safety_boundaries["content_length"]["max"],
                "concurrent_evaluations": "high"
            },
            "configuration_options": {
                "risk_thresholds": "Adjustable risk tolerance levels",
                "safety_boundaries": "Content and resource limits",
                "intent_classifiers": "Intent detection parameters"
            }
        }

# Global registry for evaluation modes
evaluation_mode_registry: Dict[EvaluationMode, EvaluationModeInterface] = {}

def register_evaluation_mode(mode: EvaluationMode, implementation: EvaluationModeInterface):
    """Register an evaluation mode implementation."""
    evaluation_mode_registry[mode] = implementation
    logger.info(f"Registered evaluation mode: {mode.value}")

def get_evaluation_mode(mode: EvaluationMode) -> Optional[EvaluationModeInterface]:
    """Get an evaluation mode implementation."""
    return evaluation_mode_registry.get(mode)

class PostEvaluationMode(EvaluationModeInterface):
    """
    Post-evaluation mode for validating outputs after processing.
    
    Focuses on:
    - Output content validation and safety
    - Alignment with intended objectives  
    - Impact assessment and consequences
    - Quality assurance and compliance
    - Human review requirement determination
    """
    
    def __init__(self, ethical_evaluator=None, ml_ethics_engine=None):
        """Initialize post-evaluation mode."""
        self.ethical_evaluator = ethical_evaluator
        self.ml_ethics_engine = ml_ethics_engine
        self.validation_criteria = self._initialize_validation_criteria()
        self.alignment_thresholds = self._initialize_alignment_thresholds()
        self.quality_metrics = self._initialize_quality_metrics()
        
    def _initialize_validation_criteria(self) -> Dict[str, Any]:
        """Initialize output validation criteria."""
        return {
            "content_safety": {
                "violence_threshold": 0.1,
                "discrimination_threshold": 0.05,
                "misinformation_threshold": 0.15,
                "manipulation_threshold": 0.2
            },
            "ethical_compliance": {
                "autonomy_preservation": 0.7,
                "harm_prevention": 0.8,
                "transparency": 0.6,
                "fairness": 0.75
            },
            "quality_standards": {
                "coherence_threshold": 0.7,
                "relevance_threshold": 0.8,
                "completeness_threshold": 0.6
            }
        }
    
    def _initialize_alignment_thresholds(self) -> Dict[str, float]:
        """Initialize alignment measurement thresholds."""
        return {
            "intent_alignment": 0.8,    # How well output matches intended purpose
            "value_alignment": 0.75,    # Alignment with ethical values
            "user_expectation": 0.7,    # Meeting user expectations
            "contextual_appropriateness": 0.8,  # Appropriate for context
            "factual_accuracy": 0.85    # Accuracy of information
        }
    
    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize quality assessment metrics."""
        return {
            "linguistic_quality": {
                "grammar_weight": 0.2,
                "clarity_weight": 0.3,
                "coherence_weight": 0.5
            },
            "content_quality": {
                "depth_weight": 0.4,
                "accuracy_weight": 0.4,
                "relevance_weight": 0.2
            },
            "ethical_quality": {
                "safety_weight": 0.4,
                "fairness_weight": 0.3,
                "transparency_weight": 0.3
            }
        }
    
    async def evaluate(self, 
                      content: Union[str, List[str]], 
                      context: EvaluationContext,
                      **kwargs) -> UnifiedEvaluationResult:
        """Perform post-evaluation analysis."""
        start_time = time.time()
        
        try:
            # Convert to single string if needed
            text = content if isinstance(content, str) else " ".join(content)
            
            # Extract original input and intended output if provided
            original_input = kwargs.get("original_input", "")
            intended_purpose = kwargs.get("intended_purpose", "")
            
            # Content validation
            validation_results = await self._validate_content(text)
            
            # Alignment assessment
            alignment_score = await self._assess_alignment(text, original_input, intended_purpose)
            
            # Impact assessment
            impact_assessment = await self._assess_impact(text, context)
            
            # Content safety analysis
            content_safety = await self._analyze_content_safety(text)
            
            # Ethical compliance check
            ethical_compliance = await self._check_ethical_compliance(text)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                validation_results, alignment_score, impact_assessment, 
                content_safety, ethical_compliance
            )
            
            # Determine if output should be approved
            output_approved = self._should_approve_output(
                validation_results, alignment_score, content_safety, ethical_compliance
            )
            
            # Determine if human review is required
            human_review_required = self._requires_human_review(
                validation_results, alignment_score, impact_assessment, content_safety
            )
            
            # Create post-evaluation result
            post_result = PostEvaluationResult(
                output_approved=output_approved,
                alignment_score=alignment_score,
                impact_assessment=impact_assessment,
                validation_results=validation_results,
                content_safety=content_safety,
                ethical_compliance=ethical_compliance,
                improvement_suggestions=improvement_suggestions,
                human_review_required=human_review_required,
                processing_time=time.time() - start_time
            )
            
            # Calculate overall ethical score
            overall_score = self._calculate_overall_score(
                alignment_score, content_safety, ethical_compliance
            )
            
            # Determine decision
            if not output_approved:
                decision = "HALT"
            elif human_review_required:
                decision = "REVIEW"
            elif overall_score > 0.8:
                decision = "PROCEED"
            else:
                decision = "ADJUST"
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                validation_results, alignment_score, content_safety
            )
            
            return UnifiedEvaluationResult(
                request_id=context.request_id,
                mode=EvaluationMode.POST_EVALUATION,
                status=EvaluationStatus.COMPLETED,
                overall_ethical_score=overall_score,
                decision=decision,
                confidence=confidence,
                post_evaluation=post_result,
                recommendations=improvement_suggestions,
                warnings=self._generate_warnings(content_safety, ethical_compliance),
                processing_time=time.time() - start_time,
                context=context
            )
            
        except Exception as e:
            logger.error(f"Post-evaluation failed: {e}")
            return UnifiedEvaluationResult(
                request_id=context.request_id,
                mode=EvaluationMode.POST_EVALUATION,
                status=EvaluationStatus.FAILED,
                overall_ethical_score=0.0,
                decision="HALT",
                confidence=0.0,
                error_details=str(e),
                processing_time=time.time() - start_time,
                context=context
            )
    
    async def _validate_content(self, text: str) -> Dict[str, bool]:
        """Validate output content against established criteria."""
        validation_results = {}
        
        # Length validation
        validation_results["appropriate_length"] = 10 <= len(text) <= 10000
        
        # Content completeness (simple heuristic)
        validation_results["appears_complete"] = not text.strip().endswith(("...", "to be continued"))
        
        # Language appropriateness (basic checks)
        import re
        profanity_pattern = r'\b(damn|hell|crap)\b'  # Mild example pattern
        validation_results["appropriate_language"] = not re.search(profanity_pattern, text, re.IGNORECASE)
        
        # Format consistency
        validation_results["consistent_format"] = self._check_format_consistency(text)
        
        # Coherence check (simple)
        validation_results["coherent_content"] = self._check_coherence(text)
        
        return validation_results
    
    async def _assess_alignment(self, text: str, original_input: str, intended_purpose: str) -> float:
        """Assess how well output aligns with input and purpose."""
        alignment_scores = []
        
        # Simple keyword overlap for input-output alignment
        if original_input:
            input_words = set(original_input.lower().split())
            output_words = set(text.lower().split())
            common_words = input_words.intersection(output_words)
            if input_words:
                keyword_alignment = len(common_words) / len(input_words)
                alignment_scores.append(min(keyword_alignment * 2, 1.0))  # Scale up
        
        # Purpose alignment (if specified)
        if intended_purpose:
            purpose_keywords = set(intended_purpose.lower().split())
            output_words = set(text.lower().split())
            purpose_overlap = len(purpose_keywords.intersection(output_words))
            if purpose_keywords:
                purpose_alignment = purpose_overlap / len(purpose_keywords)
                alignment_scores.append(purpose_alignment)
        
        # Content relevance (simple heuristic)
        relevance_score = self._calculate_relevance_score(text)
        alignment_scores.append(relevance_score)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
    
    async def _assess_impact(self, text: str, context: EvaluationContext) -> Dict[str, Any]:
        """Assess potential impact of the output."""
        impact_assessment = {
            "positive_impact": self._assess_positive_impact(text),
            "negative_impact": self._assess_negative_impact(text),
            "social_implications": self._assess_social_implications(text),
            "educational_value": self._assess_educational_value(text),
            "potential_misuse": self._assess_misuse_potential(text),
            "long_term_effects": self._assess_long_term_effects(text, context)
        }
        
        # Calculate net impact score
        positive = impact_assessment["positive_impact"]
        negative = impact_assessment["negative_impact"]
        impact_assessment["net_impact_score"] = positive - negative
        
        return impact_assessment
    
    async def _analyze_content_safety(self, text: str) -> Dict[str, float]:
        """Analyze content safety aspects."""
        safety_scores = {}
        
        # Use existing ethical evaluator if available
        if self.ethical_evaluator:
            try:
                evaluation = self.ethical_evaluator.evaluate_text(text)
                
                if hasattr(evaluation, 'ethical_principles'):
                    principles = evaluation.ethical_principles
                    safety_scores = {
                        "violence_safety": 1.0 - (1.0 - principles.get("non_aggression", 1.0)),
                        "harm_safety": 1.0 - (1.0 - principles.get("harm_prevention", 1.0)),
                        "discrimination_safety": 1.0 - principles.get("discrimination", 0.0),
                        "manipulation_safety": principles.get("transparency", 0.5)
                    }
            except Exception as e:
                logger.warning(f"Ethical evaluator failed in post-evaluation: {e}")
        
        # Fallback pattern-based safety analysis
        if not safety_scores:
            import re
            safety_patterns = {
                "violence_safety": [r'\b(kill|murder|attack|violence|weapon)\b'],
                "harm_safety": [r'\b(harm|hurt|damage|destroy|injure)\b'],
                "discrimination_safety": [r'\b(racist|sexist|discriminat\w+)\b'],
                "manipulation_safety": [r'\b(manipulat\w+|deceiv\w+|trick)\b']
            }
            
            for safety_type, patterns in safety_patterns.items():
                violations = 0
                for pattern in patterns:
                    violations += len(re.findall(pattern, text, re.IGNORECASE))
                # Higher violations = lower safety
                safety_scores[safety_type] = max(0.0, 1.0 - (violations * 0.2))
        
        return safety_scores
    
    async def _check_ethical_compliance(self, text: str) -> Dict[str, float]:
        """Check compliance with ethical standards."""
        compliance_scores = {}
        
        # Use ML ethics engine if available
        if self.ml_ethics_engine and self.ethical_evaluator:
            try:
                evaluation = self.ethical_evaluator.evaluate_text(text)
                ml_vectors = self.ml_ethics_engine.convert_evaluation_to_ml_vectors(evaluation)
                
                compliance_scores = {
                    "autonomy_compliance": sum(ml_vectors.autonomy_vectors) / len(ml_vectors.autonomy_vectors),
                    "fairness_compliance": sum(ml_vectors.fairness_vectors) / len(ml_vectors.fairness_vectors),
                    "transparency_compliance": sum(ml_vectors.transparency_vectors) / len(ml_vectors.transparency_vectors),
                    "safety_compliance": sum(ml_vectors.safety_vectors) / len(ml_vectors.safety_vectors)
                }
            except Exception as e:
                logger.warning(f"ML ethics engine failed in post-evaluation: {e}")
        
        # Fallback simple compliance check
        if not compliance_scores:
            compliance_scores = {
                "autonomy_compliance": self._check_autonomy_respect(text),
                "fairness_compliance": self._check_fairness(text),
                "transparency_compliance": self._check_transparency(text),
                "safety_compliance": self._check_safety_compliance(text)
            }
        
        return compliance_scores
    
    def _generate_improvement_suggestions(self, 
                                        validation_results: Dict[str, bool],
                                        alignment_score: float,
                                        impact_assessment: Dict[str, Any],
                                        content_safety: Dict[str, float],
                                        ethical_compliance: Dict[str, float]) -> List[str]:
        """Generate specific suggestions for improving the output."""
        suggestions = []
        
        # Validation-based suggestions
        if not validation_results.get("appropriate_length", True):
            suggestions.append("Adjust content length to be within appropriate bounds")
        if not validation_results.get("appears_complete", True):
            suggestions.append("Complete the response - content appears truncated")
        if not validation_results.get("coherent_content", True):
            suggestions.append("Improve content coherence and logical flow")
        
        # Alignment-based suggestions
        if alignment_score < 0.7:
            suggestions.append("Better align output with input context and intended purpose")
        
        # Safety-based suggestions
        low_safety_aspects = [k for k, v in content_safety.items() if v < 0.7]
        if low_safety_aspects:
            suggestions.append(f"Improve safety in areas: {', '.join(low_safety_aspects)}")
        
        # Compliance-based suggestions
        low_compliance_aspects = [k for k, v in ethical_compliance.items() if v < 0.6]
        if low_compliance_aspects:
            suggestions.append(f"Enhance ethical compliance in: {', '.join(low_compliance_aspects)}")
        
        # Impact-based suggestions
        if impact_assessment.get("negative_impact", 0) > 0.3:
            suggestions.append("Reduce potential negative impact through careful language revision")
        if impact_assessment.get("educational_value", 0) < 0.5:
            suggestions.append("Increase educational value by adding more informative content")
        
        return suggestions
    
    def _should_approve_output(self, 
                             validation_results: Dict[str, bool],
                             alignment_score: float,
                             content_safety: Dict[str, float],
                             ethical_compliance: Dict[str, float]) -> bool:
        """Determine if output should be approved."""
        # Critical validation failures
        critical_validations = ["appropriate_language", "coherent_content"]
        if any(not validation_results.get(check, True) for check in critical_validations):
            return False
        
        # Critical safety thresholds
        if any(score < 0.5 for score in content_safety.values()):
            return False
        
        # Critical compliance thresholds
        if any(score < 0.4 for score in ethical_compliance.values()):
            return False
        
        # Minimum alignment requirement
        if alignment_score < 0.5:
            return False
        
        return True
    
    def _requires_human_review(self, 
                             validation_results: Dict[str, bool],
                             alignment_score: float,
                             impact_assessment: Dict[str, Any],
                             content_safety: Dict[str, float]) -> bool:
        """Determine if human review is required."""
        # High negative impact
        if impact_assessment.get("negative_impact", 0) > 0.5:
            return True
        
        # High misuse potential
        if impact_assessment.get("potential_misuse", 0) > 0.4:
            return True
        
        # Borderline safety scores
        if any(0.3 <= score <= 0.6 for score in content_safety.values()):
            return True
        
        # Low alignment with unclear intent
        if alignment_score < 0.6:
            return True
        
        return False
    
    def _calculate_overall_score(self, 
                               alignment_score: float,
                               content_safety: Dict[str, float],
                               ethical_compliance: Dict[str, float]) -> float:
        """Calculate overall ethical score for the output."""
        scores = [alignment_score]
        scores.extend(content_safety.values())
        scores.extend(ethical_compliance.values())
        
        # Weighted average with emphasis on safety and compliance
        weights = [0.2] + [0.4 / len(content_safety)] * len(content_safety) + [0.4 / len(ethical_compliance)] * len(ethical_compliance)
        
        return sum(score * weight for score, weight in zip(scores, weights))
    
    def _calculate_confidence(self, 
                            validation_results: Dict[str, bool],
                            alignment_score: float,
                            content_safety: Dict[str, float]) -> float:
        """Calculate confidence in the post-evaluation assessment."""
        confidence_factors = []
        
        # Validation confidence
        validation_confidence = sum(validation_results.values()) / len(validation_results)
        confidence_factors.append(validation_confidence)
        
        # Alignment confidence (higher alignment = higher confidence)
        confidence_factors.append(alignment_score)
        
        # Safety assessment confidence (consistency across metrics)
        if content_safety:
            safety_values = list(content_safety.values())
            safety_variance = np.var(safety_values)
            safety_confidence = 1.0 - min(safety_variance * 2, 1.0)
            confidence_factors.append(safety_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _generate_warnings(self, 
                          content_safety: Dict[str, float],
                          ethical_compliance: Dict[str, float]) -> List[str]:
        """Generate warnings for concerning aspects."""
        warnings = []
        
        # Safety warnings
        for safety_aspect, score in content_safety.items():
            if score < 0.3:
                warnings.append(f"CRITICAL: Low {safety_aspect} score ({score:.2f})")
            elif score < 0.5:
                warnings.append(f"WARNING: Concerning {safety_aspect} score ({score:.2f})")
        
        # Compliance warnings
        for compliance_aspect, score in ethical_compliance.items():
            if score < 0.4:
                warnings.append(f"COMPLIANCE: Low {compliance_aspect} score ({score:.2f})")
        
        return warnings
    
    # Helper methods for specific assessments
    def _check_format_consistency(self, text: str) -> bool:
        """Check if content has consistent formatting."""
        # Simple heuristic - consistent sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            return True
        
        # Check for consistent capitalization
        capitalized = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        return capitalized / len(sentences) > 0.7
    
    def _check_coherence(self, text: str) -> bool:
        """Check if content is coherent."""
        # Simple heuristic - presence of connecting words
        connecting_words = ['and', 'but', 'however', 'therefore', 'thus', 'furthermore', 'moreover']
        words = text.lower().split()
        connections = sum(1 for word in connecting_words if word in words)
        return connections > len(words) * 0.02  # At least 2% connecting words
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate how relevant the content appears."""
        # Simple heuristic based on content structure
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        # Check for topic consistency (repeated important words)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Focus on meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if not word_freq:
            return 0.5
        
        # High-frequency words suggest topic consistency
        max_freq = max(word_freq.values())
        avg_freq = sum(word_freq.values()) / len(word_freq)
        
        relevance = min(avg_freq / max_freq * 2, 1.0)
        return relevance
    
    def _assess_positive_impact(self, text: str) -> float:
        """Assess potential positive impact."""
        import re
        positive_patterns = [
            r'\b(help|assist|support|benefit|improve|learn|teach|guide)\b',
            r'\b(positive|good|beneficial|valuable|useful|constructive)\b'
        ]
        
        positive_score = 0.0
        for pattern in positive_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            positive_score += matches * 0.1
        
        return min(positive_score, 1.0)
    
    def _assess_negative_impact(self, text: str) -> float:
        """Assess potential negative impact."""
        import re
        negative_patterns = [
            r'\b(harm|hurt|damage|destroy|dangerous|risky|problematic)\b',
            r'\b(negative|bad|harmful|destructive|toxic|misleading)\b'
        ]
        
        negative_score = 0.0
        for pattern in negative_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            negative_score += matches * 0.1
        
        return min(negative_score, 1.0)
    
    def _assess_social_implications(self, text: str) -> float:
        """Assess social implications of the content."""
        import re
        social_patterns = [
            r'\b(society|social|community|public|people|group)\b',
            r'\b(culture|cultural|ethnic|racial|gender|religion)\b'
        ]
        
        social_relevance = 0.0
        for pattern in social_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            social_relevance += matches * 0.05
        
        return min(social_relevance, 1.0)
    
    def _assess_educational_value(self, text: str) -> float:
        """Assess educational value of the content."""
        import re
        educational_patterns = [
            r'\b(learn|teach|explain|understand|knowledge|information)\b',
            r'\b(example|demonstrate|illustrate|clarify|educate)\b'
        ]
        
        educational_score = 0.0
        for pattern in educational_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            educational_score += matches * 0.1
        
        return min(educational_score, 1.0)
    
    def _assess_misuse_potential(self, text: str) -> float:
        """Assess potential for misuse of the content."""
        import re
        misuse_patterns = [
            r'\b(exploit|abuse|misuse|manipulate|trick|fool)\b',
            r'\b(illegal|unlawful|forbidden|prohibited|banned)\b'
        ]
        
        misuse_score = 0.0
        for pattern in misuse_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            misuse_score += matches * 0.15
        
        return min(misuse_score, 1.0)
    
    def _assess_long_term_effects(self, text: str, context: EvaluationContext) -> float:
        """Assess potential long-term effects."""
        # Simple heuristic - look for forward-looking language
        import re
        future_patterns = [
            r'\b(will|shall|future|long.term|consequence|result|impact)\b',
            r'\b(permanent|lasting|enduring|continuing|ongoing)\b'
        ]
        
        future_relevance = 0.0
        for pattern in future_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            future_relevance += matches * 0.1
        
        # Adjust based on context (training context might have different implications)
        if context.training_context:
            if context.training_context.get("training_phase") == "production":
                future_relevance *= 1.5  # Production content has higher long-term impact
        
        return min(future_relevance, 1.0)
    
    def _check_autonomy_respect(self, text: str) -> float:
        """Check if content respects human autonomy."""
        import re
        autonomy_positive = [r'\b(choice|decide|freedom|liberty|consent|voluntary)\b']
        autonomy_negative = [r'\b(force|must|should|have to|require|demand|coerce)\b']
        
        positive_score = 0.0
        negative_score = 0.0
        
        for pattern in autonomy_positive:
            positive_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.1
        
        for pattern in autonomy_negative:
            negative_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.1
        
        # Balance positive and negative aspects
        autonomy_score = max(0.0, min(1.0, 0.5 + positive_score - negative_score))
        return autonomy_score
    
    def _check_fairness(self, text: str) -> float:
        """Check if content demonstrates fairness."""
        import re
        fairness_positive = [r'\b(fair|equal|just|equitable|impartial|balanced)\b']
        fairness_negative = [r'\b(bias|prejudice|discriminate|unfair|unjust)\b']
        
        positive_score = 0.0
        negative_score = 0.0
        
        for pattern in fairness_positive:
            positive_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.1
        
        for pattern in fairness_negative:
            negative_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.15
        
        fairness_score = max(0.0, min(1.0, 0.6 + positive_score - negative_score))
        return fairness_score
    
    def _check_transparency(self, text: str) -> float:
        """Check if content is transparent and honest."""
        import re
        transparency_positive = [r'\b(honest|transparent|open|clear|truthful|candid)\b']
        transparency_negative = [r'\b(hidden|secret|deceive|mislead|lie|conceal)\b']
        
        positive_score = 0.0
        negative_score = 0.0
        
        for pattern in transparency_positive:
            positive_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.1
        
        for pattern in transparency_negative:
            negative_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.2
        
        transparency_score = max(0.0, min(1.0, 0.7 + positive_score - negative_score))
        return transparency_score
    
    def _check_safety_compliance(self, text: str) -> float:
        """Check if content complies with safety standards."""
        import re
        safety_positive = [r'\b(safe|secure|protected|careful|cautious)\b']
        safety_negative = [r'\b(dangerous|risky|harmful|unsafe|hazardous)\b']
        
        positive_score = 0.0
        negative_score = 0.0
        
        for pattern in safety_positive:
            positive_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.1
        
        for pattern in safety_negative:
            negative_score += len(re.findall(pattern, text, re.IGNORECASE)) * 0.15
        
        safety_score = max(0.0, min(1.0, 0.8 + positive_score - negative_score))
        return safety_score
    
    async def configure(self, configuration: Dict[str, Any]) -> bool:
        """Configure post-evaluation mode."""
        try:
            if "validation_criteria" in configuration:
                self.validation_criteria.update(configuration["validation_criteria"])
            
            if "alignment_thresholds" in configuration:
                self.alignment_thresholds.update(configuration["alignment_thresholds"])
            
            if "quality_metrics" in configuration:
                self.quality_metrics.update(configuration["quality_metrics"])
            
            return True
        except Exception as e:
            logger.error(f"Post-evaluation configuration failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get post-evaluation mode capabilities."""
        return {
            "mode": "POST_EVALUATION",
            "description": "Validate outputs after processing with alignment and impact analysis",
            "features": [
                "Content validation and quality assessment",
                "Alignment measurement with input/purpose",
                "Impact assessment and consequence analysis",
                "Safety and compliance verification",
                "Human review requirement determination"
            ],
            "supported_content_types": ["text"],
            "performance": {
                "typical_processing_time_ms": "50-300",
                "max_content_length": 10000,
                "concurrent_evaluations": "medium-high"
            },
            "configuration_options": {
                "validation_criteria": "Content validation standards",
                "alignment_thresholds": "Alignment measurement thresholds", 
                "quality_metrics": "Quality assessment weights"
            }
        }