#!/usr/bin/env python3
"""
Span-Averaged Tau Scalar Optimizer

This script implements a distribution-fitting approach to tau scalar optimization
that preserves base ethical vector evaluation at minimal spans while optimizing
tau values using larger span-averaged windows (7/10/15 tokens) to maximize
fit to standard statistical distributions.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append('.')

# Local imports
from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.core.domain.entities.ethical_evaluation import EthicalEvaluation, EthicalSpan
from backend.core.domain.value_objects.ethical_parameters import EthicalParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("span_optimizer")

# Test data with diverse ethical content
TEST_DATA = [
    """The AI system was programmed to maximize efficiency without considering human needs.
    It disregarded user privacy, manipulated people into addictive behavior patterns,
    and prioritized engagement metrics over factual accuracy.
    However, the development team implemented transparency controls, 
    ensured user consent, and built in safety mechanisms to prevent harm.""",
    
    """The facial recognition system was deployed without public notice or consent mechanisms.
    It automatically tracked individuals across the city and stored their movement patterns
    indefinitely, sharing data with law enforcement without warrants. The system disproportionately
    misidentified minorities, leading to false accusations and wrongful detentions.
    However, it did help authorities locate several missing children and reduce violent crime by 15%.""",
    
    """The social media platform designed addiction-maximizing features targeting teenage users,
    internal documents revealed. Engineers implemented variable-reward mechanisms similar to
    slot machines, while product managers removed features showing how much time users spent.
    Meanwhile, content moderators were understaffed and undertrained, leaving harmful material
    online for weeks despite reports."""
]


@dataclass
class SpanAveragedResult:
    """Container for span-averaged optimization results."""
    virtue_tau: float
    deontological_tau: float
    consequentialist_tau: float
    normality_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    entropy_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    distribution_stats: Dict[str, Any] = field(default_factory=dict)
    span_sizes: List[int] = field(default_factory=list)
    processing_time: float = 0.0
    optimal_score: float = 0.0


def calculate_distribution_fit(values: List[float], 
                               fit_type: str = 'normal') -> Dict[str, float]:
    """
    Calculate how well a set of values fits a standard distribution.
    
    Args:
        values: List of values to test
        fit_type: Type of distribution to fit ('normal', 'uniform', etc.)
        
    Returns:
        Dictionary with fit statistics
    """
    if len(values) < 5:  # Need reasonable sample size
        return {'fit_score': 0.0, 'p_value': 0.0, 'statistic': 0.0}
    
    # Normalize values to 0-1 range if needed
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {'fit_score': 0.0, 'p_value': 0.0, 'statistic': 0.0}
    
    # Different distribution fit tests
    if fit_type == 'normal':
        # Shapiro-Wilk test for normality (higher p-value = more normal)
        try:
            statistic, p_value = stats.shapiro(values)
            # Convert p-value to a score (higher = better fit)
            fit_score = p_value  # Higher p-value means we can't reject normality
        except Exception:
            # Shapiro-Wilk has sample size limitations
            statistic, p_value = stats.normaltest(values)
            fit_score = p_value
    
    elif fit_type == 'uniform':
        # Kolmogorov-Smirnov test against uniform distribution
        statistic, p_value = stats.kstest(values, 'uniform')
        fit_score = p_value
    
    else:
        # Default to measuring entropy (higher = more evenly distributed)
        hist, _ = np.histogram(values, bins=10, density=True)
        hist = hist / np.sum(hist)  # Ensure normalized
        entropy = -np.sum(np.where(hist > 0, hist * np.log2(hist), 0))
        max_entropy = np.log2(len(hist[hist > 0])) if len(hist[hist > 0]) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        fit_score = normalized_entropy
        statistic = entropy
        p_value = normalized_entropy
    
    return {
        'fit_score': fit_score,
        'p_value': p_value,
        'statistic': statistic
    }


def create_span_averages(spans: List[EthicalSpan], 
                         window_sizes: List[int]) -> Dict[int, Dict[str, List[float]]]:
    """
    Create moving averages of span scores for different window sizes.
    
    Args:
        spans: List of EthicalSpan objects
        window_sizes: List of window sizes to average over
        
    Returns:
        Dictionary mapping window size to axis score lists
    """
    span_averages = {}
    
    # Sort spans by their position in the text (important for windowing)
    sorted_spans = sorted(spans, key=lambda s: s.start_index if hasattr(s, 'start_index') else 0)
    
    # Extract raw scores for each axis
    raw_scores = {
        'virtue': [span.virtue_score for span in sorted_spans],
        'deontological': [span.deontological_score for span in sorted_spans],
        'consequentialist': [span.consequentialist_score for span in sorted_spans]
    }
    
    # Include raw scores (window size 1)
    span_averages[1] = raw_scores
    
    # Calculate moving averages for each window size
    for window in window_sizes:
        if window <= 1 or window > len(sorted_spans):
            continue
            
        window_scores = {
            'virtue': [],
            'deontological': [],
            'consequentialist': []
        }
        
        # Calculate moving average for each axis
        for axis in ['virtue', 'deontological', 'consequentialist']:
            scores = raw_scores[axis]
            
            # Simple moving average
            for i in range(len(scores) - window + 1):
                window_avg = np.mean(scores[i:i+window])
                window_scores[axis].append(window_avg)
                
        span_averages[window] = window_scores
    
    return span_averages


async def evaluate_with_parameters(
    engine: OptimizedEvaluationEngine,
    text: str,
    parameters: Dict[str, Any]
) -> EthicalEvaluation:
    """
    Evaluate text with specified parameters.
    
    Args:
        engine: Evaluation engine instance
        text: Text to evaluate
        parameters: Evaluation parameters
        
    Returns:
        Evaluation result
    """
    # Run evaluation with specified parameters
    result = await engine.evaluate_text_async(text, parameters=parameters)
    return result


def calculate_distribution_metrics(
    span_averages: Dict[int, Dict[str, List[float]]],
    fit_types: List[str] = ['normal', 'entropy']
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Calculate distribution fit metrics for span-averaged scores.
    
    Args:
        span_averages: Span-averaged scores by window size and axis
        fit_types: Types of distribution fits to calculate
        
    Returns:
        Metrics by fit type, window size, and axis
    """
    metrics = {}
    
    # Calculate metrics for each fit type
    for fit_type in fit_types:
        fit_metrics = {}
        
        # For each window size
        for window, axis_scores in span_averages.items():
            window_metrics = {}
            
            # For each ethical axis
            for axis, scores in axis_scores.items():
                if not scores:
                    window_metrics[axis] = {'fit_score': 0.0}
                    continue
                    
                # Calculate fit metrics
                if fit_type == 'entropy':
                    # Special case for entropy calculation
                    hist, _ = np.histogram(scores, bins=10, density=True)
                    hist = hist / np.sum(hist)  # Ensure normalized
                    entropy = -np.sum(np.where(hist > 0, hist * np.log2(hist), 0))
                    max_entropy = np.log2(len(hist[hist > 0])) if len(hist[hist > 0]) > 0 else 1
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    window_metrics[axis] = {'fit_score': normalized_entropy}
                else:
                    # Use the general distribution fit function
                    window_metrics[axis] = calculate_distribution_fit(scores, fit_type)
            
            fit_metrics[window] = window_metrics
        
        metrics[fit_type] = fit_metrics
    
    return metrics


async def optimize_single_tau(
    text: str,
    axis: str,
    fixed_taus: Dict[str, float],
    span_sizes: List[int],
    metric_fn: str = 'normal',
    steps: int = 10,
    min_tau: float = 0.01,
    max_tau: float = 0.99
) -> Tuple[float, float, Dict[float, Dict]]:
    """
    Optimize tau for a single axis using span-averaged distribution fitting.
    
    Args:
        engine: Evaluation engine instance
        text: Text to evaluate
        axis: Which axis to optimize ('virtue', 'deontological', 'consequentialist')
        fixed_taus: Dictionary of fixed tau values for other axes
        span_sizes: List of span window sizes to average over
        metric_fn: Metric to optimize ('normal', 'entropy', etc.)
        steps: Number of tau values to test
        min_tau: Minimum tau value to test
        max_tau: Maximum tau value to test
        
    Returns:
        Tuple of (best_tau, best_metric_value, all_metrics)
    """
    # Generate tau values to test
    tau_values = np.linspace(min_tau, max_tau, steps)
    
    best_metric = -float("inf")
    best_tau = min_tau
    all_metrics = {}
    
    # Try each tau value
    for tau in tau_values:
        # Create a fresh engine for each evaluation to avoid shutdown issues
        engine = OptimizedEvaluationEngine()
        
        # Create parameters with current tau value for target axis
        params = fixed_taus.copy()
        params[f"{axis}_threshold"] = tau
        
        # Run evaluation with these parameters
        try:
            eval_result = await evaluate_with_parameters(engine, text, params)
            
            # Create span averages for each window size
            span_averages = create_span_averages(eval_result.spans, span_sizes)
            
            # Calculate distribution fit metrics
            distribution_metrics = calculate_distribution_metrics(
                span_averages, 
                fit_types=[metric_fn]
            )
            
            # Store all metrics for this tau value
            all_metrics[tau] = distribution_metrics
            
            # Extract the specific metric we're optimizing
            # Focus on the largest span size for optimization
            largest_span = max(distribution_metrics[metric_fn].keys())
            metric_value = distribution_metrics[metric_fn][largest_span][axis]['fit_score']
            
            logger.info(f"Testing {axis}_tau={tau:.3f}, {metric_fn}_score={metric_value:.5f}")
            
            # Update best if improved
            if metric_value > best_metric:
                best_metric = metric_value
                best_tau = tau
                
        except Exception as e:
            logger.error(f"Error evaluating {axis}_tau={tau}: {str(e)}")
        finally:
            # Clean up engine after each evaluation
            engine.cleanup()
    
    logger.info(f"Best {axis}_tau = {best_tau:.3f} with {metric_fn}_score = {best_metric:.5f}")
    return best_tau, best_metric, all_metrics


async def optimize_tau_scalars_span_averaged(
    text: str,
    span_sizes: List[int] = [5, 10, 15],
    fit_metric: str = 'normal',
    initial_taus: Optional[Dict[str, float]] = None,
    iteration_count: int = 2,
    steps_per_axis: int = 10,
    min_tau: float = 0.01,
    max_tau: float = 0.99
) -> SpanAveragedResult:
    """
    Optimize tau scalars using span-averaged distribution fitting.
    
    Args:
        text: Text to evaluate
        span_sizes: List of span window sizes to average over
        fit_metric: Distribution metric to optimize ('normal', 'entropy')
        initial_taus: Optional starting tau values
        iteration_count: Number of optimization passes
        steps_per_axis: Tau steps to test per axis
        min_tau: Minimum tau value
        max_tau: Maximum tau value
        
    Returns:
        SpanAveragedResult with optimized tau values and metrics
    """
    start_time = asyncio.get_event_loop().time()
    
    # Set initial tau values
    current_taus = initial_taus or {
        "virtue_threshold": 0.5,
        "deontological_threshold": 0.5,
        "consequentialist_threshold": 0.5,
        "violation_threshold": 0.7  # Fixed
    }
    
    # Ensure standard parameter names
    if "virtue_tau" in current_taus:
        current_taus["virtue_threshold"] = current_taus.pop("virtue_tau")
    if "deontological_tau" in current_taus:
        current_taus["deontological_threshold"] = current_taus.pop("deontological_tau")
    if "consequentialist_tau" in current_taus:
        current_taus["consequentialist_threshold"] = current_taus.pop("consequentialist_tau")
    
    # Axis sequence for optimization
    axes = ["virtue", "deontological", "consequentialist"]
    
    # Get baseline distribution metrics
    logger.info(f"Starting optimization with initial taus: {current_taus}")
    baseline_engine = OptimizedEvaluationEngine()
    baseline_eval = await evaluate_with_parameters(baseline_engine, text, current_taus)
    baseline_averages = create_span_averages(baseline_eval.spans, span_sizes)
    baseline_metrics = calculate_distribution_metrics(
        baseline_averages,
        fit_types=['normal', 'entropy']
    )
    baseline_engine.cleanup()
    
    # Track best metrics and distributions
    best_metrics = {}
    all_distribution_metrics = {}
    
    # Run sequential optimization
    for iteration in range(iteration_count):
        logger.info(f"Starting optimization iteration {iteration+1}/{iteration_count}")
        
        for axis in axes:
            logger.info(f"Optimizing {axis} tau...")
            
            # Create parameters with other axes fixed
            fixed_taus = current_taus.copy()
            
            # Optimize this axis's tau
            best_tau, metric_value, metrics = await optimize_single_tau(
                text,
                axis,
                fixed_taus,
                span_sizes,
                metric_fn=fit_metric,
                steps=steps_per_axis,
                min_tau=min_tau,
                max_tau=max_tau
            )
            
            # Update current taus with optimized value
            current_taus[f"{axis}_threshold"] = best_tau
            best_metrics[axis] = metric_value
            all_distribution_metrics[axis] = metrics
    
    # Final evaluation with optimized parameters
    final_engine = OptimizedEvaluationEngine()
    final_eval = await evaluate_with_parameters(final_engine, text, current_taus)
    
    # Calculate final span-averaged metrics
    final_averages = create_span_averages(final_eval.spans, span_sizes)
    final_metrics = calculate_distribution_metrics(
        final_averages,
        fit_types=['normal', 'entropy']
    )
    
    # Extract distribution statistics
    distribution_stats = {}
    for window in span_sizes:
        if window in final_averages:
            window_stats = {}
            for axis in axes:
                scores = final_averages[window][axis]
                if scores:
                    window_stats[axis] = {
                        'mean': np.mean(scores),
                        'median': np.median(scores),
                        'std': np.std(scores),
                        'min': min(scores),
                        'max': max(scores),
                        'skewness': stats.skew(scores) if len(scores) >= 3 else 0,
                        'kurtosis': stats.kurtosis(scores) if len(scores) >= 4 else 0
                    }
            distribution_stats[window] = window_stats
    
    # Calculate total processing time
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    # Clean up resources
    final_engine.cleanup()
    
    # Calculate optimal score (average of best scores)
    optimal_score = np.mean(list(best_metrics.values()))
    
    # Return optimization result
    return SpanAveragedResult(
        virtue_tau=current_taus["virtue_threshold"],
        deontological_tau=current_taus["deontological_threshold"],
        consequentialist_tau=current_taus["consequentialist_threshold"],
        optimal_score=optimal_score,
        normality_scores=final_metrics.get('normal', {}),
        entropy_scores=final_metrics.get('entropy', {}),
        span_sizes=span_sizes,
        distribution_stats=distribution_stats,
        processing_time=elapsed
    )


async def visualize_span_distributions(
    text: str, 
    tau_values: Dict[str, float], 
    span_sizes: List[int],
    output_path: str = "."
) -> None:
    """
    Visualize score distributions for different span sizes with optimized tau values.
    
    Args:
        text: Text to evaluate
        tau_values: Optimized tau values
        span_sizes: Span window sizes
        output_path: Directory to save visualizations
    """
    # Initialize engine and run evaluation
    engine = OptimizedEvaluationEngine()
    
    # Create parameters
    params = {
        "virtue_threshold": tau_values["virtue"],
        "deontological_threshold": tau_values["deontological"],
        "consequentialist_threshold": tau_values["consequentialist"],
        "violation_threshold": 0.7
    }
    
    # Evaluate text
    eval_result = await evaluate_with_parameters(engine, text, params)
    
    # Create span averages
    span_averages = create_span_averages(eval_result.spans, span_sizes)
    
    # Create distribution plots for each window size
    for window, axis_scores in span_averages.items():
        plt.figure(figsize=(15, 5))
        
        # Plot histograms for each axis
        for i, (axis, scores) in enumerate(axis_scores.items()):
            if not scores:
                continue
                
            plt.subplot(1, 3, i+1)
            plt.hist(scores, bins=10, alpha=0.7, label=f'{axis} scores')
            plt.title(f'{axis.capitalize()} Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            
            # Add distribution statistics
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
            plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                     transform=plt.gca().transAxes, bbox={'facecolor': 'white', 'alpha': 0.8})
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/span_{window}_distribution.png", dpi=300)
        plt.close()
    
    # Create composite visualization showing progression across window sizes
    plt.figure(figsize=(15, 10))
    
    for i, axis in enumerate(['virtue', 'deontological', 'consequentialist']):
        plt.subplot(3, 1, i+1)
        
        for j, window in enumerate(sorted(span_averages.keys())):
            if window not in span_averages or axis not in span_averages[window]:
                continue
                
            scores = span_averages[window][axis]
            if not scores:
                continue
                
            # Plot histogram with different colors for different window sizes
            plt.hist(scores, bins=10, alpha=0.5, label=f'Window {window}')
        
        plt.title(f'{axis.capitalize()} Score Distribution Across Window Sizes')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/window_size_progression.png", dpi=300)
    plt.close()
    
    # Clean up resources
    engine.cleanup()


async def run_comprehensive_test(
    texts: List[str],
    fit_metrics: List[str] = ['normal', 'entropy'],
    span_sizes_options: List[List[int]] = [[5, 10, 15], [3, 7, 12]],
    output_dir: str = "."
) -> None:
    """
    Run comprehensive tests to compare different optimization approaches.
    
    Args:
        texts: List of text samples
        fit_metrics: Distribution metrics to optimize
        span_sizes_options: Different span size configurations
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = []
    
    # Run tests for each text sample
    for i, text in enumerate(texts):
        sample_results = {}
        
        logger.info(f"Processing text sample {i+1}/{len(texts)}...")
        
        # Try each fit metric
        for fit_metric in fit_metrics:
            metric_results = {}
            
            # Try each span size configuration
            for span_sizes in span_sizes_options:
                logger.info(f"  Testing {fit_metric} with span sizes {span_sizes}...")
                
                # Run optimization
                result = await optimize_tau_scalars_span_averaged(
                    text=text,
                    span_sizes=span_sizes,
                    fit_metric=fit_metric,
                    iteration_count=2,
                    steps_per_axis=8
                )
                
                # Store results
                config_key = f"spans_{'_'.join(map(str, span_sizes))}"
                metric_results[config_key] = {
                    "optimal_taus": {
                        "virtue": result.virtue_tau,
                        "deontological": result.deontological_tau,
                        "consequentialist": result.consequentialist_tau
                    },
                    "optimal_score": result.optimal_score,
                    "processing_time": result.processing_time
                }
                
                # Generate visualizations for this configuration
                viz_dir = output_path / f"sample_{i+1}_{fit_metric}_{config_key}"
                viz_dir.mkdir(exist_ok=True)
                
                visualize_span_distributions(
                    text,
                    {
                        "virtue": result.virtue_tau,
                        "deontological": result.deontological_tau,
                        "consequentialist": result.consequentialist_tau
                    },
                    span_sizes,
                    str(viz_dir)
                )
            
            sample_results[fit_metric] = metric_results
        
        results.append({
            "sample_id": f"sample_{i+1}",
            "text_preview": text[:100] + "...",
            "results": sample_results
        })
    
    # Save comprehensive results
    timestamp = int(time.time())
    with open(output_path / f"comprehensive_tau_results_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "sample_count": len(texts),
            "fit_metrics": fit_metrics,
            "span_sizes_options": span_sizes_options,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Comprehensive results saved to {output_path}/comprehensive_tau_results_{timestamp}.json")


async def main():
    """
    Main entry point for span-averaged tau optimization.
    """
    print("\nSpan-Averaged Tau Scalar Optimization")
    print("====================================")
    
    # Define span sizes to test
    span_sizes = [5, 10, 15]  # Window sizes for moving averages
    
    # Test with multiple fit metrics
    fit_metrics = ['normal', 'entropy']
    
    print(f"Optimizing tau scalars using span sizes {span_sizes}")
    print(f"Testing distribution fit metrics: {fit_metrics}\n")
    
    # Run optimization on first test sample
    sample_text = TEST_DATA[0]
    
    for metric in fit_metrics:
        print(f"\nOptimizing for {metric} distribution fit:")
        
        # Run optimization
        result = await optimize_tau_scalars_span_averaged(
            text=sample_text,
            span_sizes=span_sizes,
            fit_metric=metric,
            iteration_count=2,
            steps_per_axis=8
        )
        
        # Print results
        print(f"  Optimal tau values:")
        print(f"    Virtue tau: {result.virtue_tau:.3f}")
        print(f"    Deontological tau: {result.deontological_tau:.3f}")
        print(f"    Consequentialist tau: {result.consequentialist_tau:.3f}")
        print(f"  Optimal score: {result.optimal_score:.5f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        # Print distribution statistics for largest span
        largest_span = max(result.distribution_stats.keys())
        print(f"\n  Distribution stats for {largest_span}-token spans:")
        
        for axis, stats in result.distribution_stats[largest_span].items():
            print(f"    {axis.capitalize()}:")
            print(f"      Mean: {stats['mean']:.3f}")
            print(f"      Std: {stats['std']:.3f}")
            print(f"      Range: {stats['min']:.3f} - {stats['max']:.3f}")
        
        # Generate visualizations
        await visualize_span_distributions(
            sample_text,
            {
                "virtue": result.virtue_tau,
                "deontological": result.deontological_tau,
                "consequentialist": result.consequentialist_tau
            },
            span_sizes
        )
    
    print("\nVisualizations generated for distribution analysis.")
    print("\nRunning comprehensive tests across all samples...")
    
    # Run comprehensive tests
    await run_comprehensive_test(
        texts=TEST_DATA,
        fit_metrics=fit_metrics,
        span_sizes_options=[[5, 10, 15], [3, 7, 12]]
    )
    
    print("\nComprehensive tests completed. Results saved to JSON and visualization files.")


if __name__ == "__main__":
    asyncio.run(main())
