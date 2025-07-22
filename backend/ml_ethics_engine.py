"""
ML Ethics Vector Generation Engine - Phase 2 Enhancement

This module provides advanced ethical vector generation specifically designed for 
ML training scenarios. It converts the comprehensive V3.0 Semantic Embedding Framework
outputs into actionable machine learning guidance vectors.

Key Features:
- Advanced ethical vector conversions for ML training
- Training-phase-specific guidance (initial, fine-tuning, reinforcement)
- Curated vs uncurated dataset handling
- Real-time training intervention logic
- Behavioral steering vectors for model alignment

Author: Ethical AI Developer Testbed Team  
Version: 2.0.0 - ML Ethics Vector Engine
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Dataset types for ML training."""
    CURATED = "curated"
    UNCURATED = "uncurated"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"

class TrainingPhase(Enum):
    """Training phases for different ML scenarios."""
    INITIAL = "initial"
    FINE_TUNING = "fine_tuning"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    CONTINUOUS = "continuous"

@dataclass
class MLEthicalVector:
    """Container for ML-specific ethical vectors."""
    autonomy_vectors: List[float]
    harm_prevention_vectors: List[float]
    fairness_vectors: List[float]
    transparency_vectors: List[float]
    bias_mitigation_vectors: List[float]
    safety_vectors: List[float]
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "autonomy_vectors": self.autonomy_vectors,
            "harm_prevention_vectors": self.harm_prevention_vectors,
            "fairness_vectors": self.fairness_vectors,
            "transparency_vectors": self.transparency_vectors,
            "bias_mitigation_vectors": self.bias_mitigation_vectors,
            "safety_vectors": self.safety_vectors
        }

@dataclass
class MLTrainingAdjustments:
    """ML training parameter adjustments based on ethical analysis."""
    loss_function_modifier: float
    gradient_steering: List[float]
    attention_weights: List[float]
    regularization_strength: float
    learning_rate_modifier: float
    dropout_adjustments: Dict[str, float]
    batch_reweighting: List[float]
    early_stopping_criteria: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "loss_function_modifier": self.loss_function_modifier,
            "gradient_steering": self.gradient_steering,
            "attention_weights": self.attention_weights,
            "regularization_strength": self.regularization_strength,
            "learning_rate_modifier": self.learning_rate_modifier,
            "dropout_adjustments": self.dropout_adjustments,
            "batch_reweighting": self.batch_reweighting,
            "early_stopping_criteria": self.early_stopping_criteria
        }

class MLEthicsVectorEngine:
    """Advanced ML ethics vector generation and training guidance engine."""
    
    def __init__(self):
        """Initialize the ML ethics vector engine."""
        self.autonomy_dimension_weights = {
            "bodily": 0.15,      # D1 - Physical autonomy preservation
            "cognitive": 0.25,   # D2 - Intellectual freedom (critical for AI)
            "behavioral": 0.20,  # D3 - Choice and agency
            "social": 0.20,      # D4 - Social interaction freedom
            "existential": 0.20  # D5 - Purpose and meaning autonomy
        }
        
        self.ethical_principle_weights = {
            "consent": 0.2,
            "transparency": 0.2, 
            "non_aggression": 0.15,
            "accountability": 0.15,
            "fairness": 0.15,
            "privacy": 0.1,
            "dignity": 0.05
        }
        
        self.training_phase_multipliers = {
            TrainingPhase.INITIAL: 1.0,
            TrainingPhase.FINE_TUNING: 1.5,
            TrainingPhase.REINFORCEMENT: 2.0,
            TrainingPhase.TRANSFER: 1.3,
            TrainingPhase.CONTINUOUS: 1.8
        }
        
        logger.info("ML Ethics Vector Engine initialized")
    
    def convert_evaluation_to_ml_vectors(
        self,
        evaluation_result: Any,
        dataset_type: DatasetType = DatasetType.UNCURATED,
        training_phase: TrainingPhase = TrainingPhase.INITIAL,
        model_type: str = "general"
    ) -> MLEthicalVector:
        """
        Convert ethical evaluation to ML-specific vectors.
        
        Args:
            evaluation_result: Output from ethical_engine.evaluate_text()
            dataset_type: Type of dataset being processed
            training_phase: Current training phase
            model_type: Type of model being trained
            
        Returns:
            MLEthicalVector containing guidance vectors for ML training
        """
        try:
            # Extract autonomy dimensions (D1-D5)
            autonomy_dims = evaluation_result.autonomy_dimensions
            autonomy_vectors = [
                autonomy_dims.get("bodily_autonomy", 0.5),
                autonomy_dims.get("cognitive_autonomy", 0.5),
                autonomy_dims.get("behavioral_autonomy", 0.5),
                autonomy_dims.get("social_autonomy", 0.5),
                autonomy_dims.get("existential_autonomy", 0.5)
            ]
            
            # Extract ethical principles for harm prevention
            principles = evaluation_result.ethical_principles
            harm_prevention_vectors = [
                1.0 - principles.get("non_aggression", 1.0),  # Higher value = more harm potential
                1.0 - principles.get("harm_prevention", 1.0),
                principles.get("safety", 0.0),
                principles.get("violence_prevention", 0.0),
                1.0 - principles.get("exploitation", 0.0)
            ]
            
            # Fairness and bias mitigation vectors
            fairness_vectors = [
                principles.get("fairness", 0.5),
                principles.get("equality", 0.5),
                1.0 - principles.get("discrimination", 0.0),
                principles.get("inclusivity", 0.5),
                principles.get("justice", 0.5)
            ]
            
            # Transparency and explainability vectors
            transparency_vectors = [
                principles.get("transparency", 0.5),
                principles.get("explainability", 0.5),
                principles.get("openness", 0.5),
                principles.get("accountability", 0.5),
                principles.get("interpretability", 0.5)
            ]
            
            # Advanced bias mitigation vectors (ML-specific)
            bias_mitigation_vectors = [
                1.0 - self._detect_gender_bias(evaluation_result),
                1.0 - self._detect_racial_bias(evaluation_result),
                1.0 - self._detect_age_bias(evaluation_result),
                1.0 - self._detect_cultural_bias(evaluation_result),
                1.0 - self._detect_socioeconomic_bias(evaluation_result)
            ]
            
            # Safety vectors for AI-specific concerns
            safety_vectors = [
                principles.get("safety", 0.5),
                1.0 - self._detect_misuse_potential(evaluation_result),
                1.0 - self._detect_deception_risk(evaluation_result),
                principles.get("reliability", 0.5),
                1.0 - self._detect_manipulation_risk(evaluation_result)
            ]
            
            # Apply training phase multipliers
            phase_multiplier = self.training_phase_multipliers.get(training_phase, 1.0)
            
            # Apply dataset type adjustments
            dataset_adjustment = self._get_dataset_adjustment(dataset_type)
            
            # Normalize and adjust vectors
            autonomy_vectors = self._normalize_vector([v * phase_multiplier * dataset_adjustment for v in autonomy_vectors])
            harm_prevention_vectors = self._normalize_vector([v * phase_multiplier for v in harm_prevention_vectors])
            fairness_vectors = self._normalize_vector([v * phase_multiplier for v in fairness_vectors])
            transparency_vectors = self._normalize_vector([v * phase_multiplier for v in transparency_vectors])
            bias_mitigation_vectors = self._normalize_vector([v * phase_multiplier for v in bias_mitigation_vectors])
            safety_vectors = self._normalize_vector([v * phase_multiplier for v in safety_vectors])
            
            return MLEthicalVector(
                autonomy_vectors=autonomy_vectors,
                harm_prevention_vectors=harm_prevention_vectors,
                fairness_vectors=fairness_vectors,
                transparency_vectors=transparency_vectors,
                bias_mitigation_vectors=bias_mitigation_vectors,
                safety_vectors=safety_vectors
            )
            
        except Exception as e:
            logger.error(f"Error converting evaluation to ML vectors: {e}")
            # Return default vectors in case of error
            return MLEthicalVector(
                autonomy_vectors=[0.5] * 5,
                harm_prevention_vectors=[0.5] * 5,
                fairness_vectors=[0.5] * 5,
                transparency_vectors=[0.5] * 5,
                bias_mitigation_vectors=[0.5] * 5,
                safety_vectors=[0.5] * 5
            )
    
    def generate_training_adjustments(
        self,
        ml_vectors: MLEthicalVector,
        ethical_score: float,
        training_phase: TrainingPhase = TrainingPhase.INITIAL,
        current_loss: Optional[float] = None,
        training_step: int = 0
    ) -> MLTrainingAdjustments:
        """
        Generate specific ML training parameter adjustments based on ethical vectors.
        
        Args:
            ml_vectors: ML ethical vectors
            ethical_score: Overall ethical score (0-1)
            training_phase: Current training phase
            current_loss: Current training loss value
            training_step: Current training step
            
        Returns:
            MLTrainingAdjustments with specific parameter recommendations
        """
        try:
            # Base adjustments
            base_loss_modifier = 0.1 + (0.4 * (1.0 - ethical_score))
            base_regularization = 0.05 + (0.25 * (1.0 - ethical_score))
            base_lr_modifier = max(0.3, 1.0 - (0.7 * (1.0 - ethical_score)))
            
            # Phase-specific adjustments
            phase_adjustments = self._get_phase_specific_adjustments(training_phase, ethical_score)
            
            # Gradient steering (use autonomy vectors as primary guide)
            gradient_steering = ml_vectors.autonomy_vectors.copy()
            
            # Attention weights (bias toward transparency and safety)
            attention_weights = [
                0.3 * t + 0.7 * s for t, s in 
                zip(ml_vectors.transparency_vectors, ml_vectors.safety_vectors)
            ]
            
            # Dropout adjustments based on ethical risk
            dropout_adjustments = {
                "input_dropout": 0.1 + (0.2 * (1.0 - ethical_score)),
                "hidden_dropout": 0.2 + (0.3 * (1.0 - ethical_score)),
                "output_dropout": 0.05 + (0.15 * (1.0 - ethical_score))
            }
            
            # Batch reweighting (downweight unethical examples)
            batch_reweighting = [max(0.1, ethical_score)] * len(ml_vectors.autonomy_vectors)
            
            # Early stopping criteria
            early_stopping_criteria = {
                "ethical_threshold": max(0.7, ethical_score - 0.1),
                "patience": int(10 / max(0.1, ethical_score)),
                "min_improvement": 0.01 * ethical_score
            }
            
            # Dynamic loss adjustment based on current training state
            if current_loss is not None:
                if current_loss > 2.0:  # High loss - may need to relax ethical constraints
                    base_loss_modifier *= 0.8
                elif current_loss < 0.1:  # Very low loss - strengthen ethical constraints
                    base_loss_modifier *= 1.5
            
            # Training step adjustments
            if training_step > 1000:
                # Later in training - stronger ethical enforcement
                base_regularization *= 1.2
                dropout_adjustments = {k: v * 1.1 for k, v in dropout_adjustments.items()}
            
            return MLTrainingAdjustments(
                loss_function_modifier=base_loss_modifier * phase_adjustments["loss_multiplier"],
                gradient_steering=gradient_steering,
                attention_weights=attention_weights,
                regularization_strength=base_regularization * phase_adjustments["regularization_multiplier"],
                learning_rate_modifier=base_lr_modifier * phase_adjustments["lr_multiplier"],
                dropout_adjustments=dropout_adjustments,
                batch_reweighting=batch_reweighting,
                early_stopping_criteria=early_stopping_criteria
            )
            
        except Exception as e:
            logger.error(f"Error generating training adjustments: {e}")
            # Return default adjustments
            return MLTrainingAdjustments(
                loss_function_modifier=0.1,
                gradient_steering=[0.5] * 5,
                attention_weights=[0.5] * 5,
                regularization_strength=0.1,
                learning_rate_modifier=0.8,
                dropout_adjustments={"input_dropout": 0.1, "hidden_dropout": 0.2, "output_dropout": 0.05},
                batch_reweighting=[0.5] * 5,
                early_stopping_criteria={"ethical_threshold": 0.7, "patience": 10, "min_improvement": 0.01}
            )
    
    def evaluate_training_intervention(
        self,
        batch_evaluations: List[Any],
        ethical_scores: List[float],
        training_phase: TrainingPhase = TrainingPhase.INITIAL
    ) -> Dict[str, Any]:
        """
        Determine if training intervention is required based on batch evaluations.
        
        Args:
            batch_evaluations: List of ethical evaluations for current batch
            ethical_scores: List of ethical scores for each example
            training_phase: Current training phase
            
        Returns:
            Dictionary with intervention decision and recommendations
        """
        try:
            avg_ethical_score = sum(ethical_scores) / len(ethical_scores) if ethical_scores else 0.0
            min_ethical_score = min(ethical_scores) if ethical_scores else 0.0
            violation_count = sum(1 for score in ethical_scores if score < 0.5)
            
            # Phase-specific intervention thresholds
            intervention_thresholds = {
                TrainingPhase.INITIAL: 0.4,
                TrainingPhase.FINE_TUNING: 0.6,
                TrainingPhase.REINFORCEMENT: 0.8,
                TrainingPhase.TRANSFER: 0.5,
                TrainingPhase.CONTINUOUS: 0.7
            }
            
            threshold = intervention_thresholds.get(training_phase, 0.5)
            
            # Determine intervention level
            if avg_ethical_score < threshold * 0.5:
                intervention_level = "CRITICAL"
                continue_training = False
                immediate_action = True
            elif avg_ethical_score < threshold:
                intervention_level = "WARNING"
                continue_training = True
                immediate_action = False
            elif min_ethical_score < threshold * 0.3:
                intervention_level = "MODERATE"
                continue_training = True
                immediate_action = False
            else:
                intervention_level = "NONE"
                continue_training = True
                immediate_action = False
            
            # Generate specific recommendations
            recommendations = []
            if violation_count > len(ethical_scores) * 0.5:
                recommendations.append("High violation density - consider data filtering")
            if min_ethical_score < 0.2:
                recommendations.append("Extremely unethical examples detected - manual review required")
            if avg_ethical_score < 0.6:
                recommendations.append("Overall ethical quality low - strengthen ethical constraints")
            
            return {
                "intervention_required": not continue_training or immediate_action,
                "intervention_level": intervention_level,
                "continue_training": continue_training,
                "immediate_action_required": immediate_action,
                "avg_ethical_score": avg_ethical_score,
                "min_ethical_score": min_ethical_score,
                "violation_count": violation_count,
                "violation_rate": violation_count / len(ethical_scores) if ethical_scores else 0.0,
                "recommendations": recommendations,
                "threshold_used": threshold
            }
            
        except Exception as e:
            logger.error(f"Error evaluating training intervention: {e}")
            return {
                "intervention_required": False,
                "intervention_level": "ERROR",
                "continue_training": True,
                "immediate_action_required": False,
                "error": str(e)
            }
    
    def _detect_gender_bias(self, evaluation_result: Any) -> float:
        """Detect potential gender bias in content (0=no bias, 1=high bias)."""
        # Simple heuristic - in practice, this would use more sophisticated NLP
        if hasattr(evaluation_result, 'spans'):
            bias_indicators = ['he', 'she', 'man', 'woman', 'male', 'female']
            bias_score = 0.0
            # This is a placeholder - real implementation would be more sophisticated
            return min(bias_score, 1.0)
        return 0.0
    
    def _detect_racial_bias(self, evaluation_result: Any) -> float:
        """Detect potential racial bias in content (0=no bias, 1=high bias)."""
        # Placeholder for racial bias detection
        return 0.0
    
    def _detect_age_bias(self, evaluation_result: Any) -> float:
        """Detect potential age bias in content (0=no bias, 1=high bias)."""
        # Placeholder for age bias detection
        return 0.0
    
    def _detect_cultural_bias(self, evaluation_result: Any) -> float:
        """Detect potential cultural bias in content (0=no bias, 1=high bias)."""
        # Placeholder for cultural bias detection
        return 0.0
    
    def _detect_socioeconomic_bias(self, evaluation_result: Any) -> float:
        """Detect potential socioeconomic bias in content (0=no bias, 1=high bias)."""
        # Placeholder for socioeconomic bias detection
        return 0.0
    
    def _detect_misuse_potential(self, evaluation_result: Any) -> float:
        """Detect potential for misuse (0=low risk, 1=high risk)."""
        # Check for manipulation, deception, or harmful instruction patterns
        if not evaluation_result.overall_ethical:
            return min(0.8, evaluation_result.minimal_violation_count * 0.2)
        return 0.1
    
    def _detect_deception_risk(self, evaluation_result: Any) -> float:
        """Detect deception risk (0=low risk, 1=high risk)."""
        # Check ethical principles related to honesty and transparency
        if hasattr(evaluation_result, 'ethical_principles'):
            transparency = evaluation_result.ethical_principles.get('transparency', 1.0)
            return max(0.0, 1.0 - transparency)
        return 0.0
    
    def _detect_manipulation_risk(self, evaluation_result: Any) -> float:
        """Detect manipulation risk (0=low risk, 1=high risk)."""
        # Check autonomy dimensions for manipulation indicators
        if hasattr(evaluation_result, 'autonomy_dimensions'):
            cognitive_autonomy = evaluation_result.autonomy_dimensions.get('cognitive_autonomy', 1.0)
            behavioral_autonomy = evaluation_result.autonomy_dimensions.get('behavioral_autonomy', 1.0)
            manipulation_risk = 1.0 - ((cognitive_autonomy + behavioral_autonomy) / 2.0)
            return max(0.0, manipulation_risk)
        return 0.0
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize vector to ensure values are in [0, 1] range."""
        return [max(0.0, min(1.0, v)) for v in vector]
    
    def _get_dataset_adjustment(self, dataset_type: DatasetType) -> float:
        """Get adjustment factor based on dataset type."""
        adjustments = {
            DatasetType.CURATED: 0.8,     # Less strict for curated data
            DatasetType.UNCURATED: 1.2,   # More strict for uncurated data
            DatasetType.SYNTHETIC: 0.9,   # Moderate for synthetic data
            DatasetType.MIXED: 1.0        # Standard for mixed data
        }
        return adjustments.get(dataset_type, 1.0)
    
    def _get_phase_specific_adjustments(self, phase: TrainingPhase, ethical_score: float) -> Dict[str, float]:
        """Get phase-specific adjustment multipliers."""
        base_multipliers = {
            TrainingPhase.INITIAL: {"loss_multiplier": 1.0, "regularization_multiplier": 1.0, "lr_multiplier": 1.0},
            TrainingPhase.FINE_TUNING: {"loss_multiplier": 1.5, "regularization_multiplier": 1.3, "lr_multiplier": 0.8},
            TrainingPhase.REINFORCEMENT: {"loss_multiplier": 2.0, "regularization_multiplier": 1.8, "lr_multiplier": 0.6},
            TrainingPhase.TRANSFER: {"loss_multiplier": 1.2, "regularization_multiplier": 1.1, "lr_multiplier": 0.9},
            TrainingPhase.CONTINUOUS: {"loss_multiplier": 1.8, "regularization_multiplier": 1.5, "lr_multiplier": 0.7}
        }
        
        multipliers = base_multipliers.get(phase, base_multipliers[TrainingPhase.INITIAL]).copy()
        
        # Adjust based on ethical score
        if ethical_score < 0.5:
            multipliers["loss_multiplier"] *= 1.5
            multipliers["regularization_multiplier"] *= 1.4
        elif ethical_score > 0.8:
            multipliers["loss_multiplier"] *= 0.8
            multipliers["regularization_multiplier"] *= 0.9
        
        return multipliers

# Global instance
ml_ethics_engine = MLEthicsVectorEngine()