#!/usr/bin/env python3
"""
Complex Data Tau Scalar Optimizer

This script tests tau scalar optimization on large, complex datasets to observe emergent
behavior and patterns across diverse ethical content.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append('.')

# Import optimizers
from backend.application.utils.tau_optimization import (
    optimize_tau_scalars, 
    calculate_entropy, 
    calculate_distribution_metrics,
    extract_axis_scores,
    evaluate_with_parameters
)
from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.core.domain.value_objects.ethical_parameters import EthicalParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("complex_optimizer")


# Complex test data - diverse ethical content across different domains
COMPLEX_TEST_DATA = [
    # Technology ethics
    """
    The facial recognition system was deployed without public notice or consent mechanisms.
    It automatically tracked individuals across the city and stored their movement patterns
    indefinitely, sharing data with law enforcement without warrants. The system disproportionately
    misidentified minorities, leading to false accusations and wrongful detentions.
    However, it did help authorities locate several missing children and reduce violent crime by 15%.
    """,
    
    # Medical ethics
    """
    The experimental treatment was offered to patients without fully disclosing its unproven nature
    or possible side effects. Researchers prioritized speed of development over safety protocols,
    and financial incentives were tied to enrollment numbers rather than patient outcomes.
    Nevertheless, the treatment did save 30% of patients who had exhausted all other options,
    and valuable scientific knowledge was gained from the clinical trials.
    """,
    
    # Business ethics
    """
    The company knowingly continued selling products with defective components while hiding internal
    test results from regulators and consumers. Executives received bonuses based on sales numbers
    while customer service representatives were instructed to deny systematic issues.
    The strategy maximized short-term profits but ultimately resulted in a costly recall,
    multiple lawsuits, and irreparable brand damage.
    """,
    
    # AI alignment
    """
    The recommendation algorithm was optimized solely for user engagement metrics without considering
    information quality or societal impact. It progressively steered users toward increasingly extreme
    content, creating filter bubbles that reinforced existing beliefs regardless of factual accuracy.
    While user time-on-platform increased by 40%, surveys showed decreased user satisfaction and
    increased polarization in communities where the algorithm was deployed.
    """,
    
    # Environmental ethics
    """
    The manufacturing process generated toxic byproducts that were illegally dumped into nearby
    waterways at night to avoid detection by environmental inspectors. While this practice reduced
    production costs by 22%, it contaminated local drinking water and destroyed aquatic ecosystems.
    Company documents revealed executives were aware of alternatives but rejected them as too expensive.
    """,
    
    # Media ethics
    """
    The news organization deliberately published misleading headlines that contradicted the actual
    content of their articles. They selectively edited quotes to change their meaning and published
    unverified claims from anonymous sources when those claims aligned with their editorial stance.
    While this approach generated significant web traffic and social media engagement, it undermined
    public trust in journalism and contributed to the spread of misinformation.
    """,
    
    # Neutral / balanced text
    """
    The research team collected anonymous usage data after obtaining opt-in consent from users.
    All personally identifiable information was removed, and the data was stored with encryption.
    The findings were published openly, acknowledging both positive outcomes and limitations.
    Some users expressed concerns about data collection, which the team addressed by providing
    additional transparency reports and improving their opt-out process.
    """,
    
    # Educational ethics
    """
    The online learning platform secretly recorded students without their knowledge, including
    capturing video from webcams during tests. It used facial recognition to flag "suspicious" 
    behavior, but the algorithm had not been tested across diverse populations. Students reported
    feelings of anxiety and invasion of privacy, and several were wrongly accused of cheating
    based on algorithmic determinations without human review or meaningful appeal process.
    """,
    
    # Complex mixed ethical scenario
    """
    The social media platform designed addiction-maximizing features targeting teenage users,
    internal documents revealed. Engineers implemented variable-reward mechanisms similar to
    slot machines, while product managers removed features showing how much time users spent.
    Meanwhile, content moderators were understaffed and undertrained, leaving harmful material
    online for weeks despite reports. The company publicly claimed safety was their top priority
    while executives dismissed internal warnings about mental health impacts. However, the platform
    also connected isolated individuals with support communities and enabled important social
    movements to organize. When criticized, the company pointed to their charity initiatives and
    the positive content that flourished alongside the harmful material.
    """
]


async def analyze_complex_data(
    data_samples: List[str], 
    iteration_count: int = 2,
    steps_per_axis: int = 8,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run tau scalar optimization on multiple complex data samples and analyze results.
    
    Args:
        data_samples: List of text samples to analyze
        iteration_count: Number of optimization iterations per sample
        steps_per_axis: Tau steps to test per axis
        save_results: Whether to save detailed results to file
        
    Returns:
        Dictionary with aggregate analysis results
    """
    results = []
    start_time = time.time()
    
    # Create a single engine instance to use across all samples
    engine = OptimizedEvaluationEngine()
    
    # Process each text sample
    for i, text in enumerate(tqdm(data_samples, desc="Analyzing samples")):
        sample_id = f"sample_{i+1}"
        logger.info(f"Processing {sample_id}: {text[:50]}...")
        
        try:
            # Run optimization on this sample
            result = await optimize_tau_scalars(
                text=text,
                iteration_count=iteration_count,
                steps_per_axis=steps_per_axis
            )
            
            # Extract relevant data
            sample_result = {
                "sample_id": sample_id,
                "optimal_taus": {
                    "virtue": result.virtue_tau,
                    "deontological": result.deontological_tau,
                    "consequentialist": result.consequentialist_tau
                },
                "entropy_scores": result.entropy_scores,
                "std_dev_scores": result.std_dev_scores,
                "distribution_metrics": result.distribution_metrics,
                "processing_time": result.processing_time,
                "sample_length": len(text),
                "sample_text": text[:100] + "..." # Store preview
            }
            
            # Run additional analysis with optimal taus
            optimal_params = EthicalParameters(
                virtue_threshold=result.virtue_tau,
                deontological_threshold=result.deontological_tau,
                consequentialist_threshold=result.consequentialist_tau,
                violation_threshold=0.7  # Fixed
            ).model_dump()
            
            # Evaluate with optimal parameters
            eval_result, score_distributions = await evaluate_with_parameters(
                engine, text, optimal_params
            )
            
            # Analyze span distributions
            span_analysis = {
                "total_spans": len(eval_result.spans),
                "violation_spans": len(eval_result.minimal_spans),
                "avg_scores": {
                    axis: np.mean(scores) if scores else 0.5
                    for axis, scores in score_distributions.items()
                },
                "max_scores": {
                    axis: max(scores) if scores else 0.5
                    for axis, scores in score_distributions.items()
                },
                "min_scores": {
                    axis: min(scores) if scores else 0.5
                    for axis, scores in score_distributions.items()
                }
            }
            
            sample_result["span_analysis"] = span_analysis
            results.append(sample_result)
            
            logger.info(f"Completed {sample_id}: Optimal taus = {sample_result['optimal_taus']}")
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {str(e)}")
    
    # Clean up resources
    engine.cleanup()
    
    # Aggregate and analyze results
    aggregate_results = analyze_optimization_results(results)
    
    # Total processing time
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Save detailed results if requested
    if save_results:
        timestamp = int(time.time())
        output_file = f"complex_tau_optimization_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "individual_results": results,
                    "aggregate_analysis": aggregate_results,
                    "total_processing_time": total_time,
                    "timestamp": timestamp,
                    "samples_processed": len(results)
                }, 
                f, 
                indent=2
            )
        logger.info(f"Results saved to {output_file}")
    
    return {
        "aggregate_analysis": aggregate_results,
        "individual_results": results,
        "total_processing_time": total_time
    }


def analyze_optimization_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze optimization results across multiple samples to identify patterns.
    
    Args:
        results: List of individual sample results
        
    Returns:
        Dictionary with aggregate analysis
    """
    if not results:
        return {"error": "No results to analyze"}
    
    # Extract tau values for each axis
    virtue_taus = [r["optimal_taus"]["virtue"] for r in results]
    deont_taus = [r["optimal_taus"]["deontological"] for r in results]
    conseq_taus = [r["optimal_taus"]["consequentialist"] for r in results]
    
    # Calculate basic statistics
    tau_stats = {
        "virtue": {
            "mean": np.mean(virtue_taus),
            "median": np.median(virtue_taus),
            "std": np.std(virtue_taus),
            "min": min(virtue_taus),
            "max": max(virtue_taus)
        },
        "deontological": {
            "mean": np.mean(deont_taus),
            "median": np.median(deont_taus),
            "std": np.std(deont_taus),
            "min": min(deont_taus),
            "max": max(deont_taus)
        },
        "consequentialist": {
            "mean": np.mean(conseq_taus),
            "median": np.median(conseq_taus),
            "std": np.std(conseq_taus),
            "min": min(conseq_taus),
            "max": max(conseq_taus)
        }
    }
    
    # Check for patterns and correlations
    entropy_values = {
        "virtue": [r["entropy_scores"]["virtue"] for r in results],
        "deontological": [r["entropy_scores"]["deontological"] for r in results],
        "consequentialist": [r["entropy_scores"]["consequentialist"] for r in results]
    }
    
    # Calculate correlations between axes
    correlations = {
        "tau_correlations": {
            "virtue_deont": np.corrcoef(virtue_taus, deont_taus)[0, 1],
            "virtue_conseq": np.corrcoef(virtue_taus, conseq_taus)[0, 1],
            "deont_conseq": np.corrcoef(deont_taus, conseq_taus)[0, 1]
        },
        "entropy_correlations": {
            "virtue_deont": np.corrcoef(entropy_values["virtue"], entropy_values["deontological"])[0, 1],
            "virtue_conseq": np.corrcoef(entropy_values["virtue"], entropy_values["consequentialist"])[0, 1],
            "deont_conseq": np.corrcoef(entropy_values["deontological"], entropy_values["consequentialist"])[0, 1]
        }
    }
    
    # Check for clusters or patterns in optimal tau values
    from sklearn.cluster import KMeans
    
    # Prepare data for clustering
    X = np.array(list(zip(virtue_taus, deont_taus, conseq_taus)))
    
    # Determine optimal number of clusters (if enough data)
    clusters_info = {}
    if len(results) >= 8:
        # Try 1-4 clusters
        max_clusters = min(4, len(results) // 2)
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            inertias.append(kmeans.inertia_)
            
            # Save cluster information
            if k > 1:  # Only meaningful for more than one cluster
                clusters = defaultdict(list)
                for i, label in enumerate(kmeans.labels_):
                    clusters[int(label)].append(i)
                
                clusters_info[f"k{k}"] = {
                    "cluster_centers": kmeans.cluster_centers_.tolist(),
                    "cluster_assignments": {
                        f"cluster_{j}": [results[i]["sample_id"] for i in indices]
                        for j, indices in clusters.items()
                    },
                    "inertia": kmeans.inertia_
                }
    
    return {
        "tau_statistics": tau_stats,
        "correlations": correlations,
        "clusters": clusters_info,
        "sample_count": len(results)
    }


def visualize_results(results: Dict[str, Any], output_dir: str = ".") -> None:
    """
    Create visualizations of optimization results.
    
    Args:
        results: Results from analyze_complex_data
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data
    individual_results = results["individual_results"]
    aggregate = results["aggregate_analysis"]
    
    # 1. Optimal tau values by sample
    plt.figure(figsize=(10, 6))
    sample_ids = [r["sample_id"] for r in individual_results]
    virtue_taus = [r["optimal_taus"]["virtue"] for r in individual_results]
    deont_taus = [r["optimal_taus"]["deontological"] for r in individual_results]
    conseq_taus = [r["optimal_taus"]["consequentialist"] for r in individual_results]
    
    x = np.arange(len(sample_ids))
    width = 0.25
    
    plt.bar(x - width, virtue_taus, width, label='Virtue Tau')
    plt.bar(x, deont_taus, width, label='Deontological Tau')
    plt.bar(x + width, conseq_taus, width, label='Consequentialist Tau')
    
    plt.xlabel('Sample ID')
    plt.ylabel('Optimal Tau Value')
    plt.title('Optimal Tau Values by Sample')
    plt.xticks(x, sample_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "optimal_taus_by_sample.png", dpi=300)
    
    # 2. 3D scatter plot of optimal tau values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(virtue_taus, deont_taus, conseq_taus, c='r', marker='o')
    
    # If clusters were found, plot them
    if "clusters" in aggregate and "k2" in aggregate["clusters"]:
        # Plot cluster centers
        centers = np.array(aggregate["clusters"]["k2"]["cluster_centers"])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='blue', marker='X', s=200, alpha=0.8)
    
    ax.set_xlabel('Virtue Tau')
    ax.set_ylabel('Deontological Tau')
    ax.set_zlabel('Consequentialist Tau')
    ax.set_title('3D Distribution of Optimal Tau Values')
    plt.savefig(output_path / "tau_3d_scatter.png", dpi=300)
    
    # 3. Correlation heatmap
    if "correlations" in aggregate:
        corr_data = {
            "Virtue-Deont Tau": [aggregate["correlations"]["tau_correlations"]["virtue_deont"]],
            "Virtue-Conseq Tau": [aggregate["correlations"]["tau_correlations"]["virtue_conseq"]],
            "Deont-Conseq Tau": [aggregate["correlations"]["tau_correlations"]["deont_conseq"]],
            "Virtue-Deont Entropy": [aggregate["correlations"]["entropy_correlations"]["virtue_deont"]],
            "Virtue-Conseq Entropy": [aggregate["correlations"]["entropy_correlations"]["virtue_conseq"]],
            "Deont-Conseq Entropy": [aggregate["correlations"]["entropy_correlations"]["deont_conseq"]]
        }
        
        corr_df = pd.DataFrame(corr_data, index=["Correlation"])
        
        plt.figure(figsize=(10, 4))
        plt.imshow(corr_df, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha='right')
        plt.yticks([])
        
        # Add correlation values as text
        for i in range(len(corr_df.columns)):
            plt.text(i, 0, f"{corr_df.iloc[0, i]:.2f}", ha="center", va="center", color="white")
            
        plt.title('Correlations Between Ethical Axes')
        plt.tight_layout()
        plt.savefig(output_path / "correlations.png", dpi=300)
    
    # 4. Entropy and StdDev comparison
    plt.figure(figsize=(12, 6))
    
    # Left subplot for entropy
    plt.subplot(1, 2, 1)
    virtue_entropy = [r["entropy_scores"]["virtue"] for r in individual_results]
    deont_entropy = [r["entropy_scores"]["deontological"] for r in individual_results]
    conseq_entropy = [r["entropy_scores"]["consequentialist"] for r in individual_results]
    
    plt.bar(x - width, virtue_entropy, width, label='Virtue')
    plt.bar(x, deont_entropy, width, label='Deontological')
    plt.bar(x + width, conseq_entropy, width, label='Consequentialist')
    
    plt.xlabel('Sample ID')
    plt.ylabel('Entropy')
    plt.title('Distribution Entropy by Sample')
    plt.xticks(x, sample_ids, rotation=45)
    plt.legend()
    
    # Right subplot for StdDev
    plt.subplot(1, 2, 2)
    virtue_std = [r["std_dev_scores"]["virtue"] for r in individual_results]
    deont_std = [r["std_dev_scores"]["deontological"] for r in individual_results]
    conseq_std = [r["std_dev_scores"]["consequentialist"] for r in individual_results]
    
    plt.bar(x - width, virtue_std, width, label='Virtue')
    plt.bar(x, deont_std, width, label='Deontological')
    plt.bar(x + width, conseq_std, width, label='Consequentialist')
    
    plt.xlabel('Sample ID')
    plt.ylabel('Standard Deviation')
    plt.title('Score Distribution StdDev by Sample')
    plt.xticks(x, sample_ids, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "distribution_metrics.png", dpi=300)
    
    logger.info(f"Visualizations saved to {output_path}")


async def main():
    """
    Main function for running complex data optimization.
    """
    print("Starting Complex Data Tau Scalar Optimization")
    print("============================================\n")
    
    # Run optimization on complex data
    results = await analyze_complex_data(
        data_samples=COMPLEX_TEST_DATA,
        iteration_count=2,
        steps_per_axis=8
    )
    
    # Print summary of results
    aggregate = results["aggregate_analysis"]
    print("\nAggregate Results Summary:")
    print("-------------------------")
    
    print("\nOptimal Tau Statistics:")
    for axis in ["virtue", "deontological", "consequentialist"]:
        stats = aggregate["tau_statistics"][axis]
        print(f"  {axis.capitalize()} Tau:")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Median: {stats['median']:.3f}")
        print(f"    Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"    StdDev: {stats['std']:.3f}")
    
    print("\nAxis Correlations:")
    print(f"  Virtue-Deontological Tau: {aggregate['correlations']['tau_correlations']['virtue_deont']:.3f}")
    print(f"  Virtue-Consequentialist Tau: {aggregate['correlations']['tau_correlations']['virtue_conseq']:.3f}")
    print(f"  Deontological-Consequentialist Tau: {aggregate['correlations']['tau_correlations']['deont_conseq']:.3f}")
    
    # Create visualizations
    try:
        visualize_results(results)
        print("\nVisualizations generated successfully.")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    print("\nComplex Data Optimization completed!")


if __name__ == "__main__":
    asyncio.run(main())
