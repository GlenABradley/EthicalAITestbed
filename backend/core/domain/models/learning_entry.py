"""
Learning Entry Model for the Ethical AI Testbed.

This module defines the LearningEntry model used by the learning system
for threshold optimization with dopamine feedback.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class FeedbackHistory(BaseModel):
    """Feedback history entry"""
    score: float
    comment: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class LearningEntry(BaseModel):
    """Entry for learning system with dopamine feedback"""
    evaluation_id: str
    text_pattern: str
    ambiguity_score: float
    original_thresholds: Dict[str, float]
    adjusted_thresholds: Dict[str, float]
    feedback_score: float = 0.0
    feedback_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    feedback_history: List[FeedbackHistory] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        return {
            'evaluation_id': self.evaluation_id,
            'text_pattern': self.text_pattern,
            'ambiguity_score': self.ambiguity_score,
            'original_thresholds': self.original_thresholds,
            'adjusted_thresholds': self.adjusted_thresholds,
            'feedback_score': self.feedback_score,
            'feedback_count': self.feedback_count,
            'created_at': self.created_at,
            'feedback_history': [fh.model_dump() for fh in self.feedback_history]
        }
