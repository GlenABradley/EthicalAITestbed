"""
Dynamic Scaling Result Model for the Ethical AI Testbed.

This module defines the DynamicScalingResult model used to represent
the results of dynamic threshold scaling operations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel

class DynamicScalingResult(BaseModel):
    """Result of dynamic scaling process"""
    used_dynamic_scaling: bool
    used_cascade_filtering: bool
    ambiguity_score: float
    original_thresholds: Dict[str, float]
    adjusted_thresholds: Dict[str, float]
    processing_stages: List[str]
    cascade_result: Optional[str] = None  # "ethical", "unethical", or None
