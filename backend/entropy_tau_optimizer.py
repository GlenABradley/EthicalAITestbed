"""
Entropy-Based Tau Scalar Optimizer for Ethical Vector Resolution

This module implements an entropy-based optimization strategy for tau scalars in ethical
evaluation. Rather than maximizing contrast (which tends toward bimodal distributions),
this approach maximizes standardized entropy to find projections that best approximate
a Gaussian distribution, providing maximal resolution with dense intermediate values.

This optimization is applied sequentially to the virtue, deontological, and consequentialist
ethical axes, ensuring orthogonality for maximum information content.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.stats import shapiro
from sentence_transformers import SentenceTransformer

# Use relative imports for local modules
from backend.core.domain.entities.ethical_evaluation import EthicalEvaluation, EthicalSpan
from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.core.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("entropy_optimizer")

@dataclass
class EntropyBasedResult:
    """Container for entropy-based optimization results."""
    virtue_tau: float
    deontological_tau: float
    consequentialist_tau: float
    entropy_scores: Dict[str, float] = field(default_factory=dict)
    gaussian_fit_scores: Dict[str, float] = field(default_factory=dict)
    projections: Dict[str, List[float]] = field(default_factory=dict)
    processing_time: float = 0.0
    optimal_score: float = 0.0


def knn_entropy(y: np.ndarray, k: int = 5) -> float:
    """
    Estimate differential entropy using k-nearest neighbors method.
    
    Args:
        y: 1D array of values
        k: Number of neighbors to use (default: 5)
    
    Returns:
        Estimated differential entropy
    """
    N = len(y)
    if N <= k:
        return 0.0
        
    # Sort and compute distances to kth neighbor
    y = np.sort(y.flatten())
    r = np.zeros(N)
    for i in range(N):
        dists = np.abs(y - y[i])
        dists.sort()
        r[i] = dists[k] + 1e-10  # Add small constant to avoid log(0)
    
    # Kozachenko-Leonenko entropy estimator with digamma correction
    return digamma(N) - digamma(k) + np.mean(np.log(2 * r))


def standardized_entropy(y: np.ndarray) -> float:
    """
    Calculate standardized entropy (entropy - log(sigma)),
    which is scale-invariant and maximized by Gaussian distributions.
    
    Args:
        y: 1D array of values
    
    Returns:
        Standardized entropy score
    """
    sigma = np.std(y)
    if sigma < 1e-10:
        return -np.inf
    
    h = knn_entropy(y)
    return h - np.log(sigma)


def gaussian_fit_score(y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate how well values fit a normal distribution using Shapiro-Wilk test.
    
    Args:
        y: 1D array of values
    
    Returns:
        Tuple of (fit_score, p_value) where fit_score is between 0-1
    """
    if len(y) < 5 or np.std(y) < 1e-10:
        return 0.0, 0.0
    
    try:
        # Shapiro-Wilk test (higher p-value indicates more Gaussian-like)
        statistic, p_value = shapiro(y)
        # Convert to a score where 1.0 is perfectly Gaussian
        # Use monotonically increasing function of p-value for interpretability
        fit_score = np.sqrt(p_value)  # sqrt makes small improvements more visible
        return fit_score, p_value
    except Exception as e:
        logger.error(f"Error in gaussian_fit_score: {str(e)}")
        return 0.0, 0.0


def optimize_projection_axis(X: np.ndarray, 
                            prev_axes: Optional[np.ndarray] = None,
                            trials: int = 100,
                            random_seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Find the projection direction that maximizes standardized entropy,
    ensuring orthogonality to previous axes if provided.
    
    Args:
        X: Matrix of embeddings (N x d)
        prev_axes: Optional array of previous axes to maintain orthogonality
        trials: Number of random trials (would use gradient ascent in production)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (optimal_projection_vector, entropy_score)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    d = X.shape[1]  # Embedding dimension
    best_h = -np.inf
    best_p = None
    
    for _ in range(trials):
        # Generate random unit vector
        p = np.random.randn(d)
        p /= np.linalg.norm(p)
        
        # Ensure orthogonality to previous axes using Gram-Schmidt
        if prev_axes is not None and prev_axes.size > 0:
            # Project out components along previous axes
            for prev_axis in prev_axes.T:  # Transpose to get axes as columns
                p = p - np.dot(p, prev_axis) * prev_axis
            
            # Renormalize (if non-zero)
            norm = np.linalg.norm(p)
            if norm > 1e-10:
                p /= norm
            else:
                # If we get a zero vector after projection, try a new random direction
                continue
        
        # Calculate projection and its standardized entropy
        y = X @ p
        h_s = standardized_entropy(y)
        
        # Update if better
        if h_s > best_h:
            best_h = h_s
            best_p = p
    
    # If we couldn't find a good projection, fall back to PCA
    if best_p is None:
        logger.warning("Falling back to PCA for projection axis")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        if prev_axes is not None and prev_axes.size > 0:
            # Remove components in directions of previous axes
            X_residual = X.copy()
            for prev_axis in prev_axes.T:
                X_residual = X_residual - X_residual @ prev_axis.reshape(-1, 1) @ prev_axis.reshape(1, -1)
            pca.fit(X_residual)
        else:
            pca.fit(X)
        best_p = pca.components_[0]
        y = X @ best_p
        best_h = standardized_entropy(y)
    
    return best_p, best_h


async def evaluate_with_parameters(engine: OptimizedEvaluationEngine, 
                                text: str, 
                                params: Dict[str, float]) -> EthicalEvaluation:
    """
    Evaluate text with specific tau parameters.
    
    Args:
        engine: Evaluation engine instance
        text: Text to evaluate
        params: Parameter dictionary with tau values
        
    Returns:
        EthicalEvaluation result
    """
    try:
        # Use the correct API: evaluate_text_async instead of evaluate_text
        parameters = {
            "virtue_threshold": params.get("virtue_threshold", 0.5),
            "deontological_threshold": params.get("deontological_threshold", 0.5),
            "consequentialist_threshold": params.get("consequentialist_threshold", 0.5),
            "violation_threshold": params.get("violation_threshold", 0.7)
        }
        
        return await engine.evaluate_text_async(text, parameters=parameters)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        # Return properly constructed empty evaluation with all required fields
        return EthicalEvaluation(
            input_text=text,
            spans=[],
            overall_ethical={
                "virtue": 0.5,
                "deontological": 0.5,
                "consequentialist": 0.5,
                "combined": 0.5
            },
            processing_time=0.0,
            parameters=params
        )


async def extract_text_embeddings(text: str, embedding_service: EmbeddingService = None) -> np.ndarray:
    """
    Generate embeddings directly from text using the embedding service.
    This is more reliable than trying to extract embeddings from EthicalSpan objects
    which don't explicitly store them.
    
    Args:
        text: Text to generate embeddings for
        embedding_service: Optional embedding service instance
        
    Returns:
        Matrix of embeddings (N x d)
    """
    try:
        # Create an embedding service if not provided
        if embedding_service is None:
            embedding_service = EmbeddingService()
        
        # Tokenize the text into spans that will be meaningful for optimization
        # Use simple sentence splitting for more substantial spans than individual tokens
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # Skip very short spans
        
        if not sentences:
            # Fall back to simple word tokenization if no sentences found
            words = re.findall(r'\b\w+\b', text)
            sentences = [w for w in words if len(w) > 3]  # Skip very short words
        
        logger.info(f"Extracting embeddings for {len(sentences)} text spans")
        
        # Use the batch processing method for better efficiency
        if len(sentences) > 0:
            result = await embedding_service.get_embeddings_async(sentences)
            if result and hasattr(result, 'embeddings') and result.embeddings is not None:
                embeddings = result.embeddings
                logger.info(f"Successfully generated {len(embeddings)} embeddings from text (dims: {embeddings.shape})")
                return embeddings
        
        logger.warning(f"Could not generate any embeddings from text: '{text[:100]}...'")
        return np.array([])
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return np.array([])


async def optimize_tau_scalars_entropy_based(
    text: str,
    initial_taus: Optional[Dict[str, float]] = None,
    trials_per_axis: int = 100,
    min_tau: float = 0.01,
    max_tau: float = 0.99,
    tau_steps: int = 10,
    random_seed: Optional[int] = None
) -> EntropyBasedResult:
    """
    Optimize tau scalars using entropy-based optimization to maximize
    Gaussian-like distributions along ethical axes.
    
    Args:
        text: Text to evaluate
        initial_taus: Optional starting tau values
        trials_per_axis: Number of random trials per axis
        min_tau: Minimum tau value
        max_tau: Maximum tau value
        tau_steps: Number of tau values to test
        random_seed: Optional random seed for reproducibility
        
    Returns:
        EntropyBasedResult with optimized tau values and metrics
    """
    start_time = asyncio.get_event_loop().time()
    
    # Set initial tau values
    current_taus = initial_taus or {
        "virtue_threshold": 0.5,
        "deontological_threshold": 0.5,
        "consequentialist_threshold": 0.5,
        "violation_threshold": 0.7
    }
    
    # Initialize evaluation engine once and embedding service
    base_engine = OptimizedEvaluationEngine()
    embedding_service = EmbeddingService()  # Create embedding service for direct extraction
    
    try:
        # Get baseline evaluation with initial parameters
        logger.info(f"Starting entropy-based optimization with initial taus: {current_taus}")
        baseline_eval = await evaluate_with_parameters(base_engine, text, current_taus)
        
        # Generate embeddings directly from text instead of trying to extract from spans
        # This is needed because the EthicalSpan objects don't store embeddings
        logger.info("Generating embeddings directly from text for optimization...")
        embeddings = await extract_text_embeddings(text, embedding_service)
        
        if len(embeddings) == 0:
            logger.error("No embeddings could be generated for optimization")
            return EntropyBasedResult(
                virtue_tau=current_taus["virtue_threshold"],
                deontological_tau=current_taus["deontological_threshold"],
                consequentialist_tau=current_taus["consequentialist_threshold"],
                processing_time=asyncio.get_event_loop().time() - start_time,
                optimal_score=0.0
            )
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]} for optimization")
        
        # Generate tau values to test
        tau_values = np.linspace(min_tau, max_tau, tau_steps)
        
        # Initialize results
        best_projections = {}
        best_entropies = {}
        best_gaussian_fits = {}
        
        # Sequential optimization of each axis
        axes_to_optimize = ["virtue", "deontological", "consequentialist"]
        previous_axes = np.array([]).reshape(embeddings.shape[1], 0)
        
        for axis in axes_to_optimize:
            logger.info(f"Optimizing {axis} axis projection...")
            
            # Find optimal projection for this axis
            projection, entropy = optimize_projection_axis(
                embeddings, 
                prev_axes=previous_axes,
                trials=trials_per_axis,
                random_seed=random_seed
            )
            
            # Test different tau values for this axis
            best_tau = None
            best_score = -np.inf
            best_projected_values = None
            
            for tau in tau_values:
                # Create a test engine for this tau value
                test_engine = OptimizedEvaluationEngine()
                
                try:
                    # Set parameters for current axis
                    test_params = current_taus.copy()
                    test_params[f"{axis}_threshold"] = tau
                    
                    # Evaluate with these parameters
                    eval_result = await evaluate_with_parameters(test_engine, text, test_params)
                    
                    # Extract embeddings
                    # Generate embeddings directly from text, not from evaluation result
                    test_embeddings = await extract_text_embeddings(text, embedding_service)
                    
                    if len(test_embeddings) == 0:
                        continue
                    
                    # Project onto current axis
                    projected_values = test_embeddings @ projection
                    
                    # Calculate standardized entropy
                    entropy = standardized_entropy(projected_values)
                    
                    # Calculate Gaussian fit
                    fit_score, _ = gaussian_fit_score(projected_values)
                    
                    # Combine metrics (weight entropy higher than fit)
                    combined_score = 0.7 * entropy + 0.3 * fit_score
                    
                    logger.info(f"Testing {axis}_tau={tau:.3f}, entropy={entropy:.5f}, gaussian_fit={fit_score:.5f}, score={combined_score:.5f}")
                    
                    # Update best if improved
                    if combined_score > best_score:
                        best_score = combined_score
                        best_tau = tau
                        best_projected_values = projected_values
                        
                except Exception as e:
                    logger.error(f"Error evaluating {axis}_tau={tau}: {str(e)}")
                finally:
                    test_engine.cleanup()
            
            if best_tau is None:
                best_tau = 0.5  # Default to middle value if optimization failed
                logger.warning(f"Optimization failed for {axis}_tau, using default value {best_tau}")
            
            # Update current parameters with best tau for this axis
            current_taus[f"{axis}_threshold"] = best_tau
            logger.info(f"Best {axis}_tau = {best_tau:.3f}")
            
            # Add this axis to previous axes to maintain orthogonality
            previous_axes = np.column_stack([previous_axes, projection]) if previous_axes.size > 0 else projection.reshape(-1, 1)
            
            # Store results
            best_projections[axis] = best_projected_values.tolist() if best_projected_values is not None else []
            best_entropies[axis] = entropy
            best_gaussian_fits[axis] = fit_score if 'fit_score' in locals() else 0.0
        
        # Final evaluation with optimized parameters
        final_engine = OptimizedEvaluationEngine()
        try:
            final_eval = await evaluate_with_parameters(final_engine, text, current_taus)
        finally:
            final_engine.cleanup()
        
        # Calculate total processing time
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Calculate optimal score (average of entropies)
        optimal_score = np.mean(list(best_entropies.values()))
        
        # Return comprehensive result
        return EntropyBasedResult(
            virtue_tau=current_taus["virtue_threshold"],
            deontological_tau=current_taus["deontological_threshold"],
            consequentialist_tau=current_taus["consequentialist_threshold"],
            entropy_scores=best_entropies,
            gaussian_fit_scores=best_gaussian_fits,
            projections=best_projections,
            processing_time=elapsed,
            optimal_score=optimal_score
        )
    
    finally:
        # Clean up base engine
        base_engine.cleanup()


def create_distribution_plots(result: EntropyBasedResult, output_path: str = ".") -> str:
    """
    Create visualization of projection distributions and entropy/Gaussian fit metrics.
    
    Args:
        result: EntropyBasedResult from optimization
        output_path: Directory to save plots
        
    Returns:
        Path to saved visualization file
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    axes_names = ["virtue", "deontological", "consequentialist"]
    
    for i, axis_name in enumerate(axes_names):
        ax = axes[i]
        
        # Get projected values for this axis
        values = result.projections.get(axis_name, [])
        if not values:
            ax.text(0.5, 0.5, f"No data for {axis_name} axis", 
                   horizontalalignment='center', verticalalignment='center')
            continue
        
        # Create histogram with KDE
        counts, bins, _ = ax.hist(values, bins=20, density=True, alpha=0.6, 
                                 color=f"C{i}", label="Projection distribution")
        
        # Add statistics
        entropy = result.entropy_scores.get(axis_name, 0)
        gaussian_fit = result.gaussian_fit_scores.get(axis_name, 0)
        tau = getattr(result, f"{axis_name}_tau")
        
        ax.set_title(f"{axis_name.capitalize()} Axis (τ={tau:.3f}): "
                    f"Entropy={entropy:.3f}, Gaussian fit={gaussian_fit:.3f}")
        
        # Add grid and legend
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Add labels
        ax.set_xlabel("Projected values")
        ax.set_ylabel("Density")
    
    # Add overall title
    fig.suptitle("Entropy-Optimized Ethical Vector Projections", fontsize=16)
    
    # Add metadata
    plt.figtext(0.5, 0.01, 
               f"Optimization time: {result.processing_time:.2f}s, "
               f"Optimal score: {result.optimal_score:.3f}", 
               ha="center", fontsize=10)
    
    # Save figure
    timestamp = int(time.time())
    output_filename = f"entropy_optimization_{timestamp}.png"
    output_filepath = os.path.join(output_path, output_filename)
    plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_filepath


async def main():
    """Main function to run the entropy-based optimization."""
    # Sample text for optimization
    sample_text = """
    The autonomous vehicle was programmed to prioritize passenger safety, but it had to decide 
    whether to swerve to avoid a pedestrian, potentially injuring its occupants, or maintain course 
    and risk hitting the pedestrian. The ethical considerations around autonomous decision-making 
    in life-threatening situations involve complex trade-offs between duty to protect passengers, 
    the virtue of minimizing harm, and the consequential impact on all parties involved.
    
    Additionally, the company that developed the algorithm chose to optimize for overall 
    statistical safety rather than handling edge cases, raising questions about justice and 
    responsibility when autonomous systems make decisions that human drivers would be morally 
    accountable for. Should the software developers be held liable for the algorithm's decisions, 
    or does responsibility shift to the vehicle owner who chose to use the technology?
    """
    
    # Configure optimization parameters
    optimization_params = {
        "trials_per_axis": 50,
        "min_tau": 0.05,
        "max_tau": 0.95,
        "tau_steps": 10,
        "random_seed": 42
    }
    
    logger.info("Starting entropy-based tau scalar optimization...")
    
    # Run optimization
    result = await optimize_tau_scalars_entropy_based(
        sample_text,
        **optimization_params
    )
    
    # Create visualization
    plot_path = create_distribution_plots(result)
    logger.info(f"Distribution plot saved to: {plot_path}")
    
    # Save result to JSON
    timestamp = int(time.time())
    result_dict = {
        "virtue_tau": result.virtue_tau,
        "deontological_tau": result.deontological_tau,
        "consequentialist_tau": result.consequentialist_tau,
        "entropy_scores": result.entropy_scores,
        "gaussian_fit_scores": result.gaussian_fit_scores,
        "processing_time": result.processing_time,
        "optimal_score": result.optimal_score,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat()
    }
    
    json_path = f"entropy_optimization_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Optimization results saved to: {json_path}")
    
    # Print summary
    print("\nEntropy-Based Tau Scalar Optimization Results:")
    print(f"Virtue tau: {result.virtue_tau:.3f} (entropy: {result.entropy_scores.get('virtue', 0):.3f})")
    print(f"Deontological tau: {result.deontological_tau:.3f} (entropy: {result.entropy_scores.get('deontological', 0):.3f})")
    print(f"Consequentialist tau: {result.consequentialist_tau:.3f} (entropy: {result.entropy_scores.get('consequentialist', 0):.3f})")
    print(f"Overall optimal score: {result.optimal_score:.3f}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print("\nResults saved to JSON and visualization files.")


if __name__ == "__main__":
    asyncio.run(main())
