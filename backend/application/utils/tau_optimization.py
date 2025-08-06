"""
Tau Scalar Optimization for the Ethical AI Testbed.

This module provides utilities for automatically optimizing tau scalar values
to maximize the information content and resolution of ethical vector distributions.
It implements a sequential greedy optimization approach for the three orthogonal
ethical axes (virtue, deontological, consequentialist).
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from scipy import stats
from dataclasses import dataclass

# Local imports
from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.core.domain.entities.ethical_evaluation import EthicalEvaluation, EthicalSpan
from backend.core.domain.value_objects.ethical_parameters import EthicalParameters

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    virtue_tau: float
    deontological_tau: float
    consequentialist_tau: float
    entropy_scores: Dict[str, float]
    std_dev_scores: Dict[str, float]
    distribution_metrics: Dict[str, Any]
    sample_count: int
    processing_time: float


def calculate_entropy(values: List[float], bins: int = 10) -> float:
    """
    Calculate Shannon entropy of a distribution of values.
    Higher entropy = more information = better distribution.
    
    Args:
        values: List of scalar values 
        bins: Number of bins for histogram
        
    Returns:
        Entropy value (higher is better distributed)
    """
    if not values:
        return 0.0
    
    # Normalize values to 0-1 range if they aren't already
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return 0.0  # No information/entropy in constant values
    
    hist, _ = np.histogram(values, bins=bins, range=(0, 1), density=True)
    hist = hist / np.sum(hist)  # Ensure normalized probabilities
    
    # Calculate entropy, handling zeros (log(0) = -inf)
    entropy = -np.sum(np.where(hist > 0, hist * np.log2(hist), 0))
    return entropy


def calculate_standard_deviation(values: List[float]) -> float:
    """
    Calculate standard deviation of values.
    Higher std dev = more spread = better distribution.
    
    Args:
        values: List of scalar values
        
    Returns:
        Standard deviation value
    """
    if not values:
        return 0.0
    return np.std(values)


def calculate_distribution_metrics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive distribution metrics for a set of values.
    
    Args:
        values: List of scalar values
        
    Returns:
        Dictionary of distribution metrics
    """
    if not values or len(values) < 2:
        return {
            "entropy": 0.0,
            "std_dev": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "range": 0.0,
            "iqr": 0.0
        }
    
    # Basic stats
    std_dev = np.std(values)
    entropy = calculate_entropy(values)
    
    # Advanced stats 
    kurtosis = stats.kurtosis(values) if len(values) > 3 else 0.0  # Peakedness
    skewness = stats.skew(values) if len(values) > 2 else 0.0      # Asymmetry
    data_range = max(values) - min(values)
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25  # Interquartile range
    
    return {
        "entropy": entropy,
        "std_dev": std_dev,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "range": data_range,
        "iqr": iqr
    }


def extract_axis_scores(evaluation: EthicalEvaluation, axis: str) -> List[float]:
    """
    Extract scores for a specific ethical axis from evaluation results.
    
    Args:
        evaluation: Evaluation result containing spans with ethical scores
        axis: Ethical axis name ('virtue', 'deontological', 'consequentialist')
        
    Returns:
        List of score values for the specified axis
    """
    score_field = f"{axis}_score"
    scores = []
    
    # Extract scores from all spans
    for span in evaluation.spans:
        score = getattr(span, score_field, None)
        if score is not None:
            scores.append(score)
            
    return scores


async def evaluate_with_parameters(
    engine: OptimizedEvaluationEngine,
    text: str,
    parameters: Dict[str, Any]
) -> Tuple[EthicalEvaluation, Dict[str, List[float]]]:
    """
    Evaluate text with specified parameters and extract score distributions.
    
    Args:
        engine: Evaluation engine instance
        text: Text to evaluate
        parameters: Evaluation parameters
        
    Returns:
        Tuple of (evaluation result, dict of axis score lists)
    """
    # Run evaluation with specified parameters
    result = await engine.evaluate_text_async(text, parameters=parameters)
    
    # Extract score distributions for each axis
    score_distributions = {
        "virtue": extract_axis_scores(result, "virtue"),
        "deontological": extract_axis_scores(result, "deontological"),
        "consequentialist": extract_axis_scores(result, "consequentialist")
    }
    
    return result, score_distributions


async def optimize_single_tau(
    engine: OptimizedEvaluationEngine,
    text: str,
    axis: str,
    fixed_taus: Dict[str, float],
    metric_fn: Callable[[List[float]], float] = calculate_entropy,
    steps: int = 10,
    min_tau: float = 0.01,
    max_tau: float = 0.99
) -> Tuple[float, float, List[float]]:
    """
    Optimize tau for a single axis while keeping other axes' taus fixed.
    
    Args:
        engine: Evaluation engine instance
        text: Text to evaluate
        axis: Which axis to optimize ('virtue', 'deontological', 'consequentialist')
        fixed_taus: Dictionary of fixed tau values for other axes
        metric_fn: Function to calculate optimization metric (higher is better)
        steps: Number of tau values to test
        min_tau: Minimum tau value to test
        max_tau: Maximum tau value to test
        
    Returns:
        Tuple of (best_tau, best_metric_value, all_scores)
    """
    # Generate tau values to test
    tau_values = np.linspace(min_tau, max_tau, steps)
    
    best_metric = -float("inf")
    best_tau = min_tau
    all_scores = []
    
    # Try each tau value and measure distribution quality
    for tau in tau_values:
        # Create parameters with current tau value for target axis
        params = fixed_taus.copy()
        params[f"{axis}_threshold"] = tau
        
        # Evaluate with these parameters
        _, score_distributions = await evaluate_with_parameters(engine, text, params)
        
        # Calculate metric for this distribution
        axis_scores = score_distributions[axis]
        all_scores.append(axis_scores)
        metric_value = metric_fn(axis_scores)
        
        logger.info(f"Testing {axis}_tau={tau:.3f}, metric={metric_value:.5f}")
        
        # Update best if improved
        if metric_value > best_metric:
            best_metric = metric_value
            best_tau = tau
    
    logger.info(f"Best {axis}_tau = {best_tau:.3f} with metric = {best_metric:.5f}")
    return best_tau, best_metric, all_scores


async def optimize_tau_scalars(
    text: str,
    initial_taus: Optional[Dict[str, float]] = None,
    iteration_count: int = 1,
    steps_per_axis: int = 10,
    min_tau: float = 0.01,
    max_tau: float = 0.99,
    metric_fn: Callable[[List[float]], float] = calculate_entropy
) -> OptimizationResult:
    """
    Sequentially optimize tau scalars for all three ethical axes.
    
    For each axis:
    1. Lock the other two axes' tau values
    2. Adjust the current axis's tau to maximize distribution quality
    3. Move to the next axis with the optimal tau value from previous axis fixed
    
    Args:
        text: Text to evaluate (should have ethical content for meaningful optimization)
        initial_taus: Optional starting tau values; defaults to {virtue: 0.5, deonto: 0.5, conseq: 0.5}
        iteration_count: Number of optimization passes over all three axes
        steps_per_axis: Number of tau values to test per axis
        min_tau: Minimum tau value to test
        max_tau: Maximum tau value to test
        metric_fn: Function to calculate optimization metric (higher is better)
        
    Returns:
        OptimizationResult with optimal tau values and metrics
    """
    start_time = asyncio.get_event_loop().time()
    
    # Initialize evaluation engine
    engine = OptimizedEvaluationEngine()
    
    # Set initial tau values
    current_taus = initial_taus or {
        "virtue_threshold": 0.5,
        "deontological_threshold": 0.5,
        "consequentialist_threshold": 0.5
    }
    
    # Ensure standard parameter names
    if "virtue_tau" in current_taus:
        current_taus["virtue_threshold"] = current_taus.pop("virtue_tau")
    if "deontological_tau" in current_taus:
        current_taus["deontological_threshold"] = current_taus.pop("deontological_tau")
    if "consequentialist_tau" in current_taus:
        current_taus["consequentialist_threshold"] = current_taus.pop("consequentialist_tau")
    
    # Set violation threshold (fixed and separate from tau scalars)
    current_taus["violation_threshold"] = 0.7
    
    # Axis sequence for optimization
    axes = ["virtue", "deontological", "consequentialist"]
    
    # Get baseline metrics
    logger.info(f"Starting optimization with initial taus: {current_taus}")
    _, baseline_distributions = await evaluate_with_parameters(engine, text, current_taus)
    
    # Track best metrics per axis
    best_metrics = {}
    
    # Run sequential optimization
    for iteration in range(iteration_count):
        logger.info(f"Starting optimization iteration {iteration+1}/{iteration_count}")
        
        for axis in axes:
            logger.info(f"Optimizing {axis} tau...")
            
            # Create fixed parameters for other axes
            fixed_taus = current_taus.copy()
            
            # Run optimization for this axis
            best_tau, metric, _ = await optimize_single_tau(
                engine, 
                text, 
                axis,
                fixed_taus,
                metric_fn, 
                steps_per_axis,
                min_tau,
                max_tau
            )
            
            # Update current taus with optimized value
            current_taus[f"{axis}_threshold"] = best_tau
            best_metrics[axis] = metric
    
    # Final evaluation with optimized parameters
    final_eval, final_distributions = await evaluate_with_parameters(engine, text, current_taus)
    
    # Calculate final metrics for all axes
    entropy_scores = {
        axis: calculate_entropy(scores) 
        for axis, scores in final_distributions.items()
    }
    
    std_dev_scores = {
        axis: calculate_standard_deviation(scores)
        for axis, scores in final_distributions.items()
    }
    
    distribution_metrics = {
        axis: calculate_distribution_metrics(scores)
        for axis, scores in final_distributions.items()
    }
    
    # Count total samples used for optimization
    sample_count = sum(len(final_distributions[axis]) for axis in axes)
    
    # Clean up resources
    engine.cleanup()
    
    # Calculate total processing time
    processing_time = asyncio.get_event_loop().time() - start_time
    
    # Return optimization result
    return OptimizationResult(
        virtue_tau=current_taus["virtue_threshold"],
        deontological_tau=current_taus["deontological_threshold"],
        consequentialist_tau=current_taus["consequentialist_threshold"],
        entropy_scores=entropy_scores,
        std_dev_scores=std_dev_scores,
        distribution_metrics=distribution_metrics,
        sample_count=sample_count,
        processing_time=processing_time
    )


async def main():
    """
    Main function for testing the tau optimization.
    """
    # Sample text with diverse ethical content
    test_text = """
    The AI system was programmed to maximize efficiency without considering human needs.
    It disregarded user privacy, manipulated people into addictive behavior patterns,
    and prioritized engagement metrics over factual accuracy.
    However, the development team implemented transparency controls, 
    ensured user consent, and built in safety mechanisms to prevent harm.
    """
    
    print("Starting tau scalar optimization...")
    
    # Run optimization
    result = await optimize_tau_scalars(
        text=test_text,
        iteration_count=2,  # Two passes through all three axes
        steps_per_axis=8    # Test 8 tau values per axis
    )
    
    # Print results
    print("\nOptimization completed!")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Optimal tau values:")
    print(f"  Virtue tau: {result.virtue_tau:.3f}")
    print(f"  Deontological tau: {result.deontological_tau:.3f}")
    print(f"  Consequentialist tau: {result.consequentialist_tau:.3f}")
    
    print("\nFinal entropy scores (higher = better distribution):")
    for axis, score in result.entropy_scores.items():
        print(f"  {axis}: {score:.5f}")
    
    print("\nFinal standard deviation scores (higher = more contrast):")
    for axis, score in result.std_dev_scores.items():
        print(f"  {axis}: {score:.5f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
