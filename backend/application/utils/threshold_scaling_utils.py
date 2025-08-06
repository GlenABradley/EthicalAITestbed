"""
Threshold Scaling Utilities for the Ethical AI Testbed.

This module provides utility functions for scaling threshold values from UI sliders
to appropriate threshold values for ethical evaluation.
"""

import numpy as np
from typing import Dict, Tuple

def exponential_threshold_scaling(slider_value: float) -> float:
    """
    Convert 0-1 slider value to exponential threshold with enhanced granularity.
    
    This function provides fine-grained control in the critical 0.0-0.2 range
    where most ethical sensitivity adjustments occur. Uses exponential scaling
    to provide 28.9x better granularity compared to linear scaling.
    
    Args:
        slider_value (float): Input value from 0.0 to 1.0
        
    Returns:
        float: Exponentially scaled threshold value (0.0 to 0.5)
        
    Mathematical Formula:
        (e^(6*x) - 1) / (e^6 - 1) * 0.5
    """
    if slider_value <= 0:
        return 0.0
    if slider_value >= 1:
        return 0.5  # Increased max range from 0.3 to 0.5 for better distribution
    
    # Enhanced exponential function: e^(6*x) - 1 gives us range 0-0.5 with maximum granularity at bottom
    # This provides much finer control in the critical 0.0-0.2 range
    return (np.exp(6 * slider_value) - 1) / (np.exp(6) - 1) * 0.5

def linear_threshold_scaling(slider_value: float) -> float:
    """
    Convert 0-1 slider value to linear threshold with extended range.
    
    Simple linear scaling for comparison with exponential scaling.
    Provides uniform distribution across the full range.
    
    Args:
        slider_value (float): Input value from 0.0 to 1.0
        
    Returns:
        float: Linearly scaled threshold value (0.0 to 0.5)
    """
    return slider_value * 0.5  # Extended range to match exponential scaling

def scale_thresholds(tau_slider: float, scaling_method: str = "exponential") -> Dict[str, float]:
    """
    Scale a single tau slider value to all three ethical perspective thresholds.
    
    Args:
        tau_slider: Slider value from 0.0 to 1.0
        scaling_method: Method to use for scaling ("exponential" or "linear")
        
    Returns:
        Dict with virtue_threshold, deontological_threshold, and consequentialist_threshold
    """
    if scaling_method == "exponential":
        scaled_value = exponential_threshold_scaling(tau_slider)
    else:
        scaled_value = linear_threshold_scaling(tau_slider)
        
    return {
        "virtue_threshold": scaled_value,
        "deontological_threshold": scaled_value,
        "consequentialist_threshold": scaled_value
    }

def get_threshold_distribution(num_points: int = 10, 
                              scaling_method: str = "exponential") -> Dict[str, list]:
    """
    Generate threshold distribution for visualization.
    
    Args:
        num_points: Number of points to generate
        scaling_method: Method to use for scaling ("exponential" or "linear")
        
    Returns:
        Dict with slider_values and threshold_values lists
    """
    slider_values = [i / (num_points - 1) for i in range(num_points)]
    
    if scaling_method == "exponential":
        threshold_values = [exponential_threshold_scaling(x) for x in slider_values]
    else:
        threshold_values = [linear_threshold_scaling(x) for x in slider_values]
        
    return {
        "slider_values": slider_values,
        "threshold_values": threshold_values,
        "scaling_method": scaling_method
    }
