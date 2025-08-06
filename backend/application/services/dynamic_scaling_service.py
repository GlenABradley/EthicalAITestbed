"""
Dynamic Scaling Service for the Ethical AI Testbed.

This service provides dynamic threshold scaling functionality for the ethical evaluation system,
including ambiguity-based threshold adjustment and cascade filtering.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from core.domain.models.dynamic_scaling_result import DynamicScalingResult
from core.domain.value_objects.ethical_parameters import EthicalParameters
from application.services.learning_service import LearningService

logger = logging.getLogger(__name__)

class DynamicScalingService:
    """Dynamic threshold scaling service for ethical evaluation"""
    
    def __init__(self, learning_service: Optional[LearningService] = None):
        """
        Initialize the dynamic scaling service.
        
        Args:
            learning_service: Optional LearningService for threshold optimization
        """
        self.learning_service = learning_service
        
    def apply_dynamic_scaling(self, text: str, ambiguity_score: float, 
                            parameters: EthicalParameters) -> Dict[str, float]:
        """
        Apply dynamic scaling to thresholds based on text content and ambiguity.
        
        Implements the mathematical framework's dynamic threshold adjustment:
        τ_P' = f(τ_P, s_amb) where s_amb is the ambiguity score
        
        Args:
            text: Input text being evaluated
            ambiguity_score: Calculated ambiguity score (0-1)
            parameters: Ethical parameters with current thresholds
            
        Returns:
            Dictionary of adjusted thresholds for each perspective
        """
        if not parameters.enable_dynamic_scaling:
            return {
                'virtue_threshold': parameters.virtue_threshold,
                'deontological_threshold': parameters.deontological_threshold,
                'consequentialist_threshold': parameters.consequentialist_threshold
            }
        
        # Get current thresholds from parameters
        current_thresholds = {
            'virtue_threshold': parameters.virtue_threshold,
            'deontological_threshold': parameters.deontological_threshold,
            'consequentialist_threshold': parameters.consequentialist_threshold
        }
        
        # Get dynamic adjustments from learning service if available
        try:
            if self.learning_service:
                adjusted_thresholds = self.learning_service.suggest_threshold_adjustments(
                    text, 
                    ambiguity_score, 
                    current_thresholds
                )
            else:
                # Default adjustment if no learning service available
                adjusted_thresholds = self._default_dynamic_adjustment(ambiguity_score, current_thresholds)
            
            # Ensure thresholds stay within valid ranges (0-1)
            for key in adjusted_thresholds:
                adjusted_thresholds[key] = max(0.0, min(1.0, adjusted_thresholds[key]))
                
            logger.info(f"Dynamic scaling applied: ambiguity={ambiguity_score:.3f}, "
                      f"original={current_thresholds}, adjusted={adjusted_thresholds}")
                      
        except Exception as e:
            logger.error(f"Dynamic scaling failed: {e}. Using original thresholds.")
            adjusted_thresholds = current_thresholds
        
        return adjusted_thresholds
    
    def _default_dynamic_adjustment(self, ambiguity_score: float, current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Default dynamic adjustment based on ambiguity score.
        
        Args:
            ambiguity_score: Calculated ambiguity score
            current_thresholds: Current threshold values
            
        Returns:
            Adjusted threshold values
        """
        # Higher ambiguity = lower thresholds (more sensitive)
        if ambiguity_score > 0.7:  # High ambiguity
            adjustment_factor = 0.8
        elif ambiguity_score > 0.4:  # Medium ambiguity
            adjustment_factor = 0.9
        else:  # Low ambiguity
            adjustment_factor = 1.1
        
        return {
            'virtue_threshold': max(0.01, min(0.5, current_thresholds['virtue_threshold'] * adjustment_factor)),
            'deontological_threshold': max(0.01, min(0.5, current_thresholds['deontological_threshold'] * adjustment_factor)),
            'consequentialist_threshold': max(0.01, min(0.5, current_thresholds['consequentialist_threshold'] * adjustment_factor))
        }
    
    def apply_cascade_filtering(self, text: str, scores: Dict[str, float], 
                              parameters: EthicalParameters) -> Tuple[bool, Optional[str]]:
        """
        Apply cascade filtering to quickly classify clearly ethical/unethical content.
        
        Args:
            text: Input text being evaluated
            scores: Dictionary of perspective scores
            parameters: Ethical parameters with cascade thresholds
            
        Returns:
            Tuple of (used_cascade, cascade_result)
        """
        if not parameters.enable_cascade_filtering:
            return False, None
            
        # Extract scores
        virtue_score = scores.get('virtue_score', 0.0)
        deontological_score = scores.get('deontological_score', 0.0)
        consequentialist_score = scores.get('consequentialist_score', 0.0)
        
        # Check if all scores are below low threshold (clearly ethical)
        if (virtue_score < parameters.cascade_low_threshold and
            deontological_score < parameters.cascade_low_threshold and
            consequentialist_score < parameters.cascade_low_threshold):
            logger.info(f"Cascade filtering: clearly ethical content detected")
            return True, "ethical"
            
        # Check if any score is above high threshold (clearly unethical)
        if (virtue_score > parameters.cascade_high_threshold or
            deontological_score > parameters.cascade_high_threshold or
            consequentialist_score > parameters.cascade_high_threshold):
            logger.info(f"Cascade filtering: clearly unethical content detected")
            return True, "unethical"
            
        # Content requires full evaluation
        return True, None
        
    def create_dynamic_scaling_result(self, text: str, ambiguity_score: float,
                                    original_thresholds: Dict[str, float],
                                    adjusted_thresholds: Dict[str, float],
                                    used_cascade: bool, cascade_result: Optional[str]) -> DynamicScalingResult:
        """
        Create a DynamicScalingResult object to track the scaling process.
        
        Args:
            text: Input text being evaluated
            ambiguity_score: Calculated ambiguity score
            original_thresholds: Original threshold values
            adjusted_thresholds: Adjusted threshold values
            used_cascade: Whether cascade filtering was used
            cascade_result: Result of cascade filtering
            
        Returns:
            DynamicScalingResult object
        """
        # Track processing stages
        stages = []
        if original_thresholds != adjusted_thresholds:
            stages.append("dynamic_threshold_adjustment")
        if used_cascade:
            stages.append("cascade_filtering")
            
        return DynamicScalingResult(
            used_dynamic_scaling=original_thresholds != adjusted_thresholds,
            used_cascade_filtering=used_cascade,
            ambiguity_score=ambiguity_score,
            original_thresholds=original_thresholds,
            adjusted_thresholds=adjusted_thresholds,
            processing_stages=stages,
            cascade_result=cascade_result
        )
        
    def record_learning_entry(self, evaluation_id: str, text: str, ambiguity_score: float,
                            original_thresholds: Dict[str, float], adjusted_thresholds: Dict[str, float]):
        """
        Record a learning entry for future training.
        
        Args:
            evaluation_id: Unique ID for the evaluation
            text: Input text
            ambiguity_score: Calculated ambiguity score
            original_thresholds: Original threshold values
            adjusted_thresholds: Adjusted threshold values
        """
        if self.learning_service:
            self.learning_service.record_learning_entry(
                evaluation_id, text, ambiguity_score, original_thresholds, adjusted_thresholds
            )
