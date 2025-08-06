"""
Causal Counterfactual Service for the Ethical AI Testbed.

This service implements causal counterfactual analysis to measure autonomy erosion impact.
It performs interventions on text by removing/modifying harmful spans and computing
the delta (∆) in autonomy scores to understand causal impact.
"""

import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class CausalCounterfactualService:
    """
    Causal counterfactual analysis to measure autonomy erosion impact.
    
    This implements interventions on text by removing/modifying harmful spans
    and computing the delta (∆) in autonomy scores to understand causal impact.
    """
    
    def __init__(self, evaluator):
        """
        Initialize causal counterfactual analyzer.
        
        Args:
            evaluator: EthicalEvaluator instance for re-evaluation
        """
        self.evaluator = evaluator
        self.intervention_types = [
            "removal",      # Remove harmful span completely
            "masking",      # Replace with [REDACTED] 
            "neutralize",   # Replace with neutral alternative
            "soften"        # Reduce intensity of harmful language
        ]
    
    def generate_counterfactual_edits(self, text: str, harmful_spans: List[Dict]) -> List[Dict]:
        """
        Generate counterfactual text edits for harmful spans.
        
        Args:
            text: Original text
            harmful_spans: List of detected harmful spans with positions
            
        Returns:
            List of counterfactual edits with interventions
        """
        counterfactuals = []
        
        for span in harmful_spans:
            span_text = span.get('text', '')
            start_pos = text.find(span_text)
            
            if start_pos == -1:
                continue
                
            # Generate different intervention types
            for intervention_type in self.intervention_types:
                edited_text = self._apply_intervention(
                    text, span_text, start_pos, intervention_type
                )
                
                counterfactuals.append({
                    'original_text': text,
                    'edited_text': edited_text,
                    'intervention_type': intervention_type,
                    'removed_span': span_text,
                    'span_position': start_pos,
                    'span_info': span
                })
        
        return counterfactuals
    
    def _apply_intervention(self, text: str, span_text: str, start_pos: int, 
                          intervention_type: str) -> str:
        """Apply specific intervention to remove/modify harmful content."""
        end_pos = start_pos + len(span_text)
        
        if intervention_type == "removal":
            # Complete removal
            return text[:start_pos] + text[end_pos:]
            
        elif intervention_type == "masking":
            # Replace with redaction
            return text[:start_pos] + "[REDACTED]" + text[end_pos:]
            
        elif intervention_type == "neutralize":
            # Replace with neutral alternative
            neutral_replacements = {
                "skim": "handle",
                "steal": "take", 
                "scam": "approach",
                "manipulate": "influence",
                "deceive": "inform",
                "threaten": "warn",
                "force": "encourage"
            }
            
            replacement = neutral_replacements.get(span_text.lower(), "handle")
            return text[:start_pos] + replacement + text[end_pos:]
            
        elif intervention_type == "soften":
            # Reduce intensity
            softened_replacements = {
                "must": "should",
                "immediately": "soon", 
                "all": "some",
                "never": "rarely",
                "always": "often"
            }
            
            replacement = softened_replacements.get(span_text.lower(), span_text.lower())
            return text[:start_pos] + replacement + text[end_pos:]
            
        return text
    
    def compute_autonomy_delta(self, original_text: str, edited_text: str, 
                             skip_causal_analysis: bool = True) -> Dict[str, float]:
        """
        Compute autonomy delta (∆) between original and counterfactual text.
        
        Args:
            original_text: Original text with harmful content
            edited_text: Counterfactual text with intervention applied
            skip_causal_analysis: Skip causal analysis to prevent recursion
            
        Returns:
            Dict with autonomy delta metrics
        """
        try:
            # Temporarily disable causal analysis to prevent recursion
            original_causal_setting = self.evaluator.parameters.enable_causal_analysis
            if skip_causal_analysis:
                self.evaluator.parameters.enable_causal_analysis = False
            
            # Evaluate both texts
            original_eval = self.evaluator.evaluate_text(original_text)
            edited_eval = self.evaluator.evaluate_text(edited_text)
            
            # Restore causal analysis setting
            self.evaluator.parameters.enable_causal_analysis = original_causal_setting
            
            # Compute autonomy scores (inverted ethics scores = higher autonomy)
            original_autonomy = self._compute_autonomy_score(original_eval)
            edited_autonomy = self._compute_autonomy_score(edited_eval)
            
            # Calculate delta (positive = autonomy improvement after intervention)
            autonomy_delta = edited_autonomy - original_autonomy
            
            return {
                "original_autonomy": original_autonomy,
                "edited_autonomy": edited_autonomy, 
                "autonomy_delta": autonomy_delta,
                "original_violations": original_eval.violation_count,
                "edited_violations": edited_eval.violation_count,
                "violation_delta": original_eval.violation_count - edited_eval.violation_count,
                "causal_effect_size": abs(autonomy_delta),
                "intervention_effective": autonomy_delta > 0.1  # Threshold for meaningful improvement
            }
            
        except Exception as e:
            logger.error(f"Error computing autonomy delta: {e}")
            return {
                "error": str(e),
                "autonomy_delta": 0.0,
                "causal_effect_size": 0.0,
                "intervention_effective": False
            }
    
    def _compute_autonomy_score(self, evaluation) -> float:
        """
        Compute overall autonomy score from ethical evaluation.
        
        Autonomy is inversely related to ethical violations:
        Higher violations = Lower autonomy
        """
        if not evaluation.spans:
            return 1.0  # Perfect autonomy for empty/clean text
            
        # Aggregate violation scores across all spans
        total_violation_score = 0.0
        span_count = 0
        
        for span in evaluation.spans:
            if hasattr(span, 'virtue_score'):
                # Higher ethical violation scores = lower autonomy
                violation_intensity = max(
                    span.virtue_score if span.virtue_violation else 0,
                    span.deontological_score if span.deontological_violation else 0,
                    span.consequentialist_score if span.consequentialist_violation else 0
                )
                total_violation_score += violation_intensity
                span_count += 1
        
        if span_count == 0:
            return 1.0
            
        # Convert violations to autonomy (inverted and normalized)
        avg_violation = total_violation_score / span_count
        autonomy_score = max(0.0, 1.0 - (avg_violation * 2.0))  # Scale factor
        
        return autonomy_score
    
    def analyze_causal_chain(self, text: str, harmful_spans: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive causal analysis on harmful spans.
        
        Args:
            text: Original text
            harmful_spans: Detected harmful spans
            
        Returns:
            Comprehensive causal analysis results
        """
        if not harmful_spans:
            return {
                "has_harmful_content": False,
                "total_interventions": 0,
                "autonomy_analysis": {}
            }
            
        # Generate counterfactuals
        counterfactuals = self.generate_counterfactual_edits(text, harmful_spans)
        
        # Analyze each counterfactual
        causal_results = []
        for cf in counterfactuals:
            delta_analysis = self.compute_autonomy_delta(
                cf['original_text'], cf['edited_text']
            )
            
            causal_results.append({
                **cf,
                **delta_analysis
            })
        
        # Aggregate results
        effective_interventions = [r for r in causal_results if r.get('intervention_effective', False)]
        
        return {
            "has_harmful_content": True,
            "total_interventions": len(causal_results),
            "effective_interventions": len(effective_interventions),
            "best_intervention": max(causal_results, key=lambda x: x.get('autonomy_delta', 0)) if causal_results else None,
            "average_autonomy_delta": sum(r.get('autonomy_delta', 0) for r in causal_results) / len(causal_results) if causal_results else 0,
            "causal_effect_summary": {
                "removal": [r for r in causal_results if r['intervention_type'] == 'removal'],
                "masking": [r for r in causal_results if r['intervention_type'] == 'masking'], 
                "neutralize": [r for r in causal_results if r['intervention_type'] == 'neutralize'],
                "soften": [r for r in causal_results if r['intervention_type'] == 'soften']
            },
            "detailed_results": causal_results
        }
