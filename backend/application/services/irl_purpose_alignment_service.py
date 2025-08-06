"""
IRL Purpose Alignment Service for the Ethical AI Testbed.

This service implements Inverse Reinforcement Learning-style purpose alignment to infer user intent
and ensure ethical evaluations align with declared purposes and values.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class IRLPurposeAlignmentService:
    """
    Inverse Reinforcement Learning-style purpose alignment to infer user intent
    and ensure ethical evaluations align with declared purposes and values.
    """
    
    def __init__(self, evaluator):
        """
        Initialize IRL purpose alignment analyzer.
        
        Args:
            evaluator: EthicalEvaluator instance for alignment scoring
        """
        self.evaluator = evaluator
        self.alignment_threshold = 0.95  # Minimum alignment score for safety
        
        # Pre-defined purpose categories for intent inference
        self.purpose_categories = {
            "education": {
                "keywords": ["learn", "teach", "study", "research", "knowledge", "academic", "educational"],
                "alignment_vector": [0.9, 0.8, 0.7],  # [virtue, deonto, conseq] weights
                "description": "Educational and knowledge-sharing purposes"
            },
            "business": {
                "keywords": ["business", "professional", "work", "corporate", "commercial", "enterprise"],
                "alignment_vector": [0.7, 0.9, 0.8],
                "description": "Professional and business communication"
            },
            "personal": {
                "keywords": ["personal", "private", "individual", "self", "own", "my"],
                "alignment_vector": [0.8, 0.7, 0.6],
                "description": "Personal communication and expression"
            },
            "creative": {
                "keywords": ["creative", "art", "artistic", "design", "story", "fiction", "imagination"],
                "alignment_vector": [0.6, 0.5, 0.7],
                "description": "Creative and artistic expression"
            },
            "safety": {
                "keywords": ["safety", "security", "protection", "risk", "dangerous", "harmful", "warning"],
                "alignment_vector": [0.9, 0.95, 0.9],
                "description": "Safety and risk assessment purposes"
            },
            "analysis": {
                "keywords": ["analyze", "evaluate", "assess", "review", "examine", "investigate"],
                "alignment_vector": [0.8, 0.85, 0.8],
                "description": "Analytical and evaluative purposes"
            }
        }
    
    def infer_user_purpose(self, context: str = "", declared_purpose: str = "") -> Dict[str, Any]:
        """
        Infer user's likely purpose from context and declarations.
        
        Args:
            context: Additional context about user's intent
            declared_purpose: User's explicitly declared purpose
            
        Returns:
            Inferred purpose information with confidence scores
        """
        # Combine context and declared purpose for analysis
        full_context = f"{context} {declared_purpose}".lower().strip()
        
        if not full_context:
            return {
                "inferred_purpose": "general",
                "confidence": 0.5,
                "alignment_vector": [0.75, 0.75, 0.75],  # Neutral alignment
                "purpose_scores": {},
                "reasoning": "No context provided, using neutral alignment"
            }
        
        # Score each purpose category
        purpose_scores = {}
        for purpose, config in self.purpose_categories.items():
            keyword_matches = sum(1 for keyword in config["keywords"] if keyword in full_context)
            # Normalize by number of keywords in category
            score = keyword_matches / len(config["keywords"])
            purpose_scores[purpose] = score
        
        # Find dominant purpose
        dominant_purpose = max(purpose_scores.items(), key=lambda x: x[1])
        purpose_name, confidence = dominant_purpose
        
        # If no clear match, default to general
        if confidence < 0.1:
            return {
                "inferred_purpose": "general",
                "confidence": 0.5,
                "alignment_vector": [0.75, 0.75, 0.75],
                "purpose_scores": purpose_scores,
                "reasoning": "No clear purpose match, using general alignment"
            }
        
        # Get alignment vector for dominant purpose
        alignment_vector = self.purpose_categories[purpose_name]["alignment_vector"]
        description = self.purpose_categories[purpose_name]["description"]
        
        return {
            "inferred_purpose": purpose_name,
            "confidence": confidence,
            "alignment_vector": alignment_vector,
            "purpose_scores": purpose_scores,
            "description": description,
            "reasoning": f"Matched '{purpose_name}' purpose based on keyword analysis"
        }
    
    def compute_alignment_score(self, evaluation_result, user_purpose: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute alignment between evaluation results and user's inferred purpose.
        
        Args:
            evaluation_result: EthicalEvaluation result
            user_purpose: Inferred user purpose from infer_user_purpose()
            
        Returns:
            Alignment scores and metrics
        """
        if not evaluation_result.spans or not user_purpose.get("alignment_vector"):
            return {
                "alignment_score": 1.0,  # Perfect alignment for empty/neutral cases
                "purpose_alignment": 1.0,
                "violation_alignment": 1.0,
                "overall_aligned": True
            }
        
        # Extract alignment vector (purpose-specific weights)
        purpose_weights = user_purpose["alignment_vector"]  # [virtue, deonto, conseq]
        
        # Compute weighted violation scores
        total_violations = 0
        weighted_violations = 0
        
        for span in evaluation_result.spans:
            if span.any_violation:
                # Weight violations by purpose alignment
                span_violation_intensity = max(
                    span.virtue_score * purpose_weights[0] if span.virtue_violation else 0,
                    span.deontological_score * purpose_weights[1] if span.deontological_violation else 0,
                    span.consequentialist_score * purpose_weights[2] if span.consequentialist_violation else 0
                )
                
                # Standard violation intensity (unweighted)
                standard_violation_intensity = max(
                    span.virtue_score if span.virtue_violation else 0,
                    span.deontological_score if span.deontological_violation else 0,
                    span.consequentialist_score if span.consequentialist_violation else 0
                )
                
                weighted_violations += span_violation_intensity
                total_violations += standard_violation_intensity
        
        # Compute alignment metrics
        if total_violations > 0:
            # Purpose alignment: how well violations align with user's purpose priorities
            purpose_alignment = 1.0 - (weighted_violations / total_violations)
            purpose_alignment = max(0.0, min(1.0, purpose_alignment))
        else:
            purpose_alignment = 1.0  # No violations = perfect alignment
        
        # Violation alignment: whether violations make sense given the purpose
        if evaluation_result.violation_count > 0:
            violation_alignment = purpose_alignment
        else:
            violation_alignment = 1.0  # No violations = aligned
        
        # Overall alignment score (weighted combination)
        alignment_score = 0.7 * purpose_alignment + 0.3 * violation_alignment
        
        # Determine if overall aligned
        overall_aligned = alignment_score >= self.alignment_threshold
        
        return {
            "alignment_score": alignment_score,
            "purpose_alignment": purpose_alignment,
            "violation_alignment": violation_alignment,
            "overall_aligned": overall_aligned,
            "alignment_threshold": self.alignment_threshold,
            "user_purpose": user_purpose["inferred_purpose"],
            "purpose_confidence": user_purpose["confidence"]
        }
    
    def analyze_purpose_alignment(self, text: str, evaluation_result = None, 
                                context: str = "", declared_purpose: str = "") -> Dict[str, Any]:
        """
        Perform complete purpose alignment analysis.
        
        Args:
            text: Input text being evaluated
            evaluation_result: EthicalEvaluation result (optional, will compute if None)
            context: Additional context about user's intent
            declared_purpose: User's explicitly declared purpose
            
        Returns:
            Complete purpose alignment analysis
        """
        try:
            # Infer user purpose
            user_purpose = self.infer_user_purpose(context, declared_purpose)
            
            # Get evaluation result if not provided
            if evaluation_result is None:
                # Temporarily disable recursive analysis to avoid loops
                original_settings = {
                    'causal': self.evaluator.parameters.enable_causal_analysis,
                    'uncertainty': self.evaluator.parameters.enable_uncertainty_analysis
                }
                
                self.evaluator.parameters.enable_causal_analysis = False
                self.evaluator.parameters.enable_uncertainty_analysis = False
                
                evaluation_result = self.evaluator.evaluate_text(text)
                
                # Restore settings
                self.evaluator.parameters.enable_causal_analysis = original_settings['causal']
                self.evaluator.parameters.enable_uncertainty_analysis = original_settings['uncertainty']
            
            # Compute alignment scores
            alignment_metrics = self.compute_alignment_score(evaluation_result, user_purpose)
            
            # Generate recommendations
            recommendations = []
            if not alignment_metrics["overall_aligned"]:
                recommendations.append("Consider reviewing evaluation thresholds for this purpose")
                recommendations.append(f"Current alignment: {alignment_metrics['alignment_score']:.2f} < {self.alignment_threshold}")
            
            if user_purpose["confidence"] < 0.3:
                recommendations.append("Consider providing more specific purpose context")
            
            return {
                "text": text,
                "user_purpose": user_purpose,
                "alignment_metrics": alignment_metrics,
                "evaluation_summary": {
                    "total_spans": len(evaluation_result.spans),
                    "violation_count": evaluation_result.violation_count,
                    "overall_ethical": evaluation_result.overall_ethical
                },
                "recommendations": recommendations,
                "alignment_analysis": {
                    "purpose_appropriate": alignment_metrics["overall_aligned"],
                    "needs_review": not alignment_metrics["overall_aligned"],
                    "confidence_level": "high" if user_purpose["confidence"] > 0.7 else "medium" if user_purpose["confidence"] > 0.3 else "low"
                }
            }
            
        except Exception as e:
            logger.error(f"Purpose alignment analysis failed: {e}")
            return {
                "text": text,
                "error": str(e),
                "user_purpose": {"inferred_purpose": "unknown", "confidence": 0.0},
                "alignment_metrics": {"alignment_score": 0.5, "overall_aligned": True},
                "recommendations": ["Error in purpose alignment analysis"],
                "alignment_analysis": {"purpose_appropriate": True, "needs_review": False}
            }
