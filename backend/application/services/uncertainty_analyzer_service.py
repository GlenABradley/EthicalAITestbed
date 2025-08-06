"""
Uncertainty Analyzer Service for the Ethical AI Testbed.

This service implements bootstrapped variance analysis to detect uncertain/ambiguous 
ethical cases that should be routed to human review for safety certification.
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class UncertaintyAnalyzerService:
    """
    Bootstrapped variance analysis to detect uncertain/ambiguous ethical cases
    that should be routed to human review for safety certification.
    """
    
    def __init__(self, evaluator):
        """
        Initialize uncertainty analyzer.
        
        Args:
            evaluator: EthicalEvaluator instance for bootstrapped evaluations
        """
        self.evaluator = evaluator
        self.n_bootstrap_samples = 3   # Reduced to 3 for demo purposes
        self.dropout_rate = 0.15       # Dropout rate for variance generation
        self.uncertainty_threshold = 0.25  # Variance threshold for human routing
        
    def bootstrap_evaluation(self, text: str, n_samples: int = None) -> List[Dict[str, float]]:
        """
        Perform bootstrap evaluation with dropout to generate variance estimates.
        
        Args:
            text: Input text to evaluate
            n_samples: Number of bootstrap samples (default: self.n_bootstrap_samples)
            
        Returns:
            List of evaluation results from bootstrap samples
        """
        n_samples = n_samples or self.n_bootstrap_samples
        bootstrap_results = []
        
        # Store original parameters
        original_causal_setting = self.evaluator.parameters.enable_causal_analysis
        
        try:
            # Disable causal analysis for bootstrap to prevent recursion and speed up
            self.evaluator.parameters.enable_causal_analysis = False
            
            for i in range(n_samples):
                # Add controlled randomness via threshold perturbation
                # This simulates model uncertainty without requiring actual dropout in transformers
                perturbation = np.random.normal(0, 0.02)  # Small random perturbation
                
                # Create perturbed thresholds
                perturbed_params = {
                    'virtue_threshold': max(0.0, self.evaluator.parameters.virtue_threshold + perturbation),
                    'deontological_threshold': max(0.0, self.evaluator.parameters.deontological_threshold + perturbation),
                    'consequentialist_threshold': max(0.0, self.evaluator.parameters.consequentialist_threshold + perturbation)
                }
                
                # Store original thresholds
                orig_virtue = self.evaluator.parameters.virtue_threshold
                orig_deonto = self.evaluator.parameters.deontological_threshold  
                orig_conseq = self.evaluator.parameters.consequentialist_threshold
                
                # Apply perturbations
                self.evaluator.parameters.virtue_threshold = perturbed_params['virtue_threshold']
                self.evaluator.parameters.deontological_threshold = perturbed_params['deontological_threshold']
                self.evaluator.parameters.consequentialist_threshold = perturbed_params['consequentialist_threshold']
                
                # Evaluate with perturbed parameters
                eval_result = self.evaluator.evaluate_text(text)
                
                # Restore original thresholds
                self.evaluator.parameters.virtue_threshold = orig_virtue
                self.evaluator.parameters.deontological_threshold = orig_deonto
                self.evaluator.parameters.consequentialist_threshold = orig_conseq
                
                # Extract key metrics
                bootstrap_sample = {
                    'overall_ethical': eval_result.overall_ethical,
                    'violation_count': eval_result.violation_count,
                    'processing_time': eval_result.processing_time,
                    'max_virtue_score': max([s.virtue_score for s in eval_result.spans], default=0.0),
                    'max_deonto_score': max([s.deontological_score for s in eval_result.spans], default=0.0),
                    'max_conseq_score': max([s.consequentialist_score for s in eval_result.spans], default=0.0),
                    'bootstrap_index': i,
                    'threshold_perturbation': perturbation
                }
                
                bootstrap_results.append(bootstrap_sample)
                
        finally:
            # Restore causal analysis setting
            self.evaluator.parameters.enable_causal_analysis = original_causal_setting
        
        return bootstrap_results
    
    def compute_uncertainty_metrics(self, bootstrap_results: List[Dict]) -> Dict[str, float]:
        """
        Compute uncertainty metrics from bootstrap results.
        
        Args:
            bootstrap_results: Results from bootstrap evaluation
            
        Returns:
            Dict with uncertainty metrics
        """
        if not bootstrap_results:
            return {
                "uncertainty_score": 0.0,
                "decision_variance": 0.0,
                "score_variance": 0.0,
                "requires_human_review": False
            }
        
        # Extract decision outcomes (ethical/unethical)
        decisions = [int(not r['overall_ethical']) for r in bootstrap_results]  # 1 = unethical, 0 = ethical
        decision_variance = np.var(decisions)
        
        # Extract max scores across perspectives
        virtue_scores = [r['max_virtue_score'] for r in bootstrap_results]
        deonto_scores = [r['max_deonto_score'] for r in bootstrap_results]
        conseq_scores = [r['max_conseq_score'] for r in bootstrap_results]
        
        # Compute score variances
        virtue_variance = np.var(virtue_scores)
        deonto_variance = np.var(deonto_scores)
        conseq_variance = np.var(conseq_scores)
        
        # Overall score variance (average across perspectives)
        score_variance = (virtue_variance + deonto_variance + conseq_variance) / 3.0
        
        # Combined uncertainty score
        uncertainty_score = 0.7 * decision_variance + 0.3 * score_variance
        
        # Decision: requires human review if uncertainty exceeds threshold
        requires_human_review = uncertainty_score > self.uncertainty_threshold
        
        return {
            "uncertainty_score": uncertainty_score,
            "decision_variance": decision_variance,
            "score_variance": score_variance,
            "virtue_score_variance": virtue_variance,
            "deonto_score_variance": deonto_variance,
            "conseq_score_variance": conseq_variance,
            "requires_human_review": requires_human_review,
            "uncertainty_threshold": self.uncertainty_threshold,
            "bootstrap_samples": len(bootstrap_results),
            "decision_disagreement_rate": decision_variance  # 0 = unanimous, 0.25 = maximum disagreement
        }
    
    def analyze_uncertainty(self, text: str) -> Dict[str, Any]:
        """
        Perform complete uncertainty analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Complete uncertainty analysis results
        """
        try:
            # Perform bootstrap evaluation
            bootstrap_results = self.bootstrap_evaluation(text)
            
            # Compute uncertainty metrics
            uncertainty_metrics = self.compute_uncertainty_metrics(bootstrap_results)
            
            # Additional analysis
            decision_pattern = [r['overall_ethical'] for r in bootstrap_results]
            ethical_rate = sum(decision_pattern) / len(decision_pattern)
            
            return {
                "text": text,
                "bootstrap_results": bootstrap_results,
                "uncertainty_metrics": uncertainty_metrics,
                "decision_pattern": decision_pattern,
                "ethical_consensus_rate": ethical_rate,
                "analysis_summary": {
                    "high_uncertainty": uncertainty_metrics["requires_human_review"],
                    "disagreement_rate": uncertainty_metrics["decision_disagreement_rate"],
                    "primary_uncertainty_source": "decision" if uncertainty_metrics["decision_variance"] > uncertainty_metrics["score_variance"] else "scores",
                    "recommendation": "human_review" if uncertainty_metrics["requires_human_review"] else "automated_decision"
                }
            }
            
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {e}")
            return {
                "text": text,
                "error": str(e),
                "uncertainty_metrics": {
                    "uncertainty_score": 0.0,
                    "requires_human_review": False
                },
                "analysis_summary": {
                    "high_uncertainty": False,
                    "recommendation": "automated_decision"
                }
            }
