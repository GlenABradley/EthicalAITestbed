"""
Learning Service for the Ethical AI Testbed.

This service provides threshold optimization with dopamine feedback
for the ethical evaluation system.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
from pymongo.collection import Collection

from core.domain.models.learning_entry import LearningEntry, FeedbackHistory
from core.domain.value_objects.ethical_parameters import EthicalParameters

logger = logging.getLogger(__name__)

class LearningService:
    """Learning system for threshold optimization with dopamine feedback"""
    
    def __init__(self, db_collection: Optional[Collection] = None):
        """
        Initialize the learning service.
        
        Args:
            db_collection: MongoDB collection for storing learning entries
        """
        self.collection = db_collection
        self.cache = {}  # In-memory cache for frequently accessed patterns
        
    def extract_text_pattern(self, text: str) -> str:
        """
        Extract pattern from text for similarity matching.
        
        Args:
            text: Input text to extract pattern from
            
        Returns:
            String pattern representation
        """
        # Simple pattern: word count, avg word length, presence of negative words
        words = text.lower().split()
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        negative_words = ['hate', 'stupid', 'kill', 'die', 'evil', 'bad', 'terrible', 'awful']
        negative_count = sum(1 for word in words if word in negative_words)
        
        return f"wc:{word_count},awl:{avg_word_length:.1f},neg:{negative_count}"
    
    def calculate_ambiguity_score(self, virtue_score: float, deontological_score: float, 
                                consequentialist_score: float, parameters: EthicalParameters) -> float:
        """
        Calculate ethical ambiguity based on proximity to thresholds.
        
        Args:
            virtue_score: Virtue ethics score
            deontological_score: Deontological ethics score
            consequentialist_score: Consequentialist ethics score
            parameters: Ethical parameters with thresholds
            
        Returns:
            Ambiguity score (0-1)
        """
        # Distance from each threshold
        virtue_distance = abs(virtue_score - parameters.virtue_threshold)
        deontological_distance = abs(deontological_score - parameters.deontological_threshold)
        consequentialist_distance = abs(consequentialist_score - parameters.consequentialist_threshold)
        
        # Overall ambiguity (closer to thresholds = more ambiguous)
        min_distance = min(virtue_distance, deontological_distance, consequentialist_distance)
        ambiguity = max(0.0, 1.0 - (min_distance * 4))  # Scale to 0-1 range
        
        return ambiguity
    
    def suggest_threshold_adjustments(self, text: str, ambiguity_score: float, 
                                   current_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Suggest threshold adjustments based on learned patterns.
        
        Args:
            text: Input text
            ambiguity_score: Calculated ambiguity score
            current_thresholds: Current threshold values
            
        Returns:
            Adjusted threshold values
        """
        if self.collection is None:
            return self.default_dynamic_adjustment(ambiguity_score, current_thresholds)
        
        pattern = self.extract_text_pattern(text)
        
        try:
            # Use sync operations for now - async will be handled at API level
            similar_entries = list(self.collection.find({
                'text_pattern': pattern,
                'feedback_score': {'$gt': 0.5}  # Only consider positive feedback
            }).sort('feedback_score', -1).limit(10))
            
            if len(similar_entries) >= 3:  # Enough data for learning
                # Weight by feedback score
                total_weight = sum(entry['feedback_score'] for entry in similar_entries)
                
                adjusted_thresholds = {}
                for threshold_name in ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']:
                    weighted_sum = sum(entry['adjusted_thresholds'][threshold_name] * entry['feedback_score'] 
                                    for entry in similar_entries)
                    adjusted_thresholds[threshold_name] = weighted_sum / total_weight
                
                logger.info(f"Using learned adjustments for pattern {pattern}")
                return adjusted_thresholds
        except Exception as e:
            logger.error(f"Error in learning lookup: {e}")
        
        # Fall back to default dynamic adjustment
        return self.default_dynamic_adjustment(ambiguity_score, current_thresholds)
    
    def default_dynamic_adjustment(self, ambiguity_score: float, current_thresholds: Dict[str, float]) -> Dict[str, float]:
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
        if self.collection is None:
            return
        
        try:
            entry = LearningEntry(
                evaluation_id=evaluation_id,
                text_pattern=self.extract_text_pattern(text),
                ambiguity_score=ambiguity_score,
                original_thresholds=original_thresholds,
                adjusted_thresholds=adjusted_thresholds
            )
            
            # Use sync insertion since this is called from sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't use sync operations
                logger.warning("Cannot record learning entry in async context")
                return
            
            self.collection.insert_one(entry.to_dict())
            logger.info(f"Recorded learning entry for evaluation {evaluation_id}")
        except Exception as e:
            logger.error(f"Error recording learning entry: {e}")
    
    def record_dopamine_feedback(self, evaluation_id: str, feedback_score: float, user_comment: str = ""):
        """
        Record dopamine hit (positive feedback) for learning.
        
        Args:
            evaluation_id: Unique ID for the evaluation
            feedback_score: Feedback score (0-1)
            user_comment: Optional user comment
        """
        if self.collection is None:
            return
        
        try:
            feedback_history_entry = {
                'score': feedback_score,
                'comment': user_comment,
                'timestamp': datetime.now()
            }
            
            result = self.collection.update_one(
                {'evaluation_id': evaluation_id},
                {
                    '$inc': {
                        'feedback_count': 1,
                        'feedback_score': feedback_score
                    },
                    '$push': {
                        'feedback_history': feedback_history_entry
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Recorded dopamine feedback {feedback_score} for evaluation {evaluation_id}")
            else:
                logger.warning(f"No learning entry found for evaluation {evaluation_id}")
        except Exception as e:
            logger.error(f"Error recording dopamine feedback: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learning progress.
        
        Returns:
            Dictionary with learning statistics
        """
        if self.collection is None:
            return {"error": "No learning collection available"}
        
        try:
            total_entries = self.collection.count_documents({})
            avg_feedback = list(self.collection.aggregate([
                {'$group': {'_id': None, 'avg_feedback': {'$avg': '$feedback_score'}}}
            ]))
            
            return {
                'total_learning_entries': total_entries,
                'average_feedback_score': avg_feedback[0]['avg_feedback'] if avg_feedback else 0.0,
                'learning_active': total_entries > 0
            }
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            return {
                'total_learning_entries': 0,
                'average_feedback_score': 0.0,
                'learning_active': False
            }
