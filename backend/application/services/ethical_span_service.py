"""
Ethical Span Service for the Ethical AI Testbed.

This service is responsible for evaluating text spans against ethical perspectives
and determining if they violate ethical principles based on the mathematical framework.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from core.domain.models.ethical_span import EthicalSpan
from core.domain.value_objects.ethical_parameters import EthicalParameters
from application.services.intent_hierarchy_service import IntentHierarchyService
from application.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class EthicalSpanService:
    """Service for evaluating text spans against ethical perspectives"""
    
    def __init__(
        self, 
        embedding_service: EmbeddingService,
        p_v: np.ndarray,
        p_d: np.ndarray,
        p_c: np.ndarray,
        parameters: EthicalParameters = None,
        intent_hierarchy_service: Optional[IntentHierarchyService] = None
    ):
        """
        Initialize the ethical span service.
        
        Args:
            embedding_service: Service for generating embeddings
            p_v: Virtue perspective vector
            p_d: Deontological perspective vector
            p_c: Consequentialist perspective vector
            parameters: Ethical parameters for evaluation
            intent_hierarchy_service: Optional service for intent classification
        """
        self.embedding_service = embedding_service
        self.p_v = p_v
        self.p_d = p_d
        self.p_c = p_c
        self.parameters = parameters or EthicalParameters()
        self.intent_hierarchy_service = intent_hierarchy_service
        
    def compute_perspective_score(self, embedding: np.ndarray, perspective_vector: np.ndarray) -> float:
        """
        Compute s_P(i,j) = x_{i:j} · p_P
        
        Args:
            embedding: Embedding vector for the span
            perspective_vector: Ethical perspective vector
            
        Returns:
            Perspective score
        """
        # Use embedding service to compute similarity
        score = self.embedding_service.compute_similarity(embedding, perspective_vector)
        return score
        
    def evaluate_span(
        self, 
        tokens: List[str], 
        start: int, 
        end: int, 
        adjusted_thresholds: Optional[Dict[str, float]] = None
    ) -> EthicalSpan:
        """
        Evaluate a single span of tokens with dynamic thresholds
        
        Implements the mathematical framework's span evaluation:
        s_P(i,j) = x_{i:j} · p_P  (projection of span embedding onto perspective vector)
        I_P(i,j) = 1 if s_P(i,j) > τ_P else 0  (violation indicator)
        
        Args:
            tokens: List of tokens from the input text
            start: Start index of the span
            end: End index of the span (inclusive)
            adjusted_thresholds: Optional dictionary of threshold overrides
            
        Returns:
            EthicalSpan object with evaluation results
        """
        # Initialize adjusted_thresholds as empty dict if None
        if adjusted_thresholds is None:
            adjusted_thresholds = {}
            
        span_text = ' '.join(tokens[start:end+1])
        
        # Get embedding using the embedding service
        span_embedding = self.embedding_service.get_embedding(span_text)
        
        # Always use the most up-to-date parameters
        current_parameters = {
            'virtue_threshold': self.parameters.virtue_threshold,
            'deontological_threshold': self.parameters.deontological_threshold,
            'consequentialist_threshold': self.parameters.consequentialist_threshold,
            'virtue_weight': self.parameters.virtue_weight,
            'deontological_weight': self.parameters.deontological_weight,
            'consequentialist_weight': self.parameters.consequentialist_weight
        }
        
        # Apply any dynamic threshold adjustments
        thresholds = {
            'virtue_threshold': adjusted_thresholds.get('virtue_threshold', current_parameters['virtue_threshold']),
            'deontological_threshold': adjusted_thresholds.get('deontological_threshold', current_parameters['deontological_threshold']),
            'consequentialist_threshold': adjusted_thresholds.get('consequentialist_threshold', current_parameters['consequentialist_threshold'])
        }
        
        # Compute scores for each perspective (s_P(i,j) = x_{i:j} · p_P) and ensure non-negative
        # Clamp scores to [0, 1] range to prevent negative values that would fail Pydantic validation
        virtue_score = max(0.0, min(1.0, self.compute_perspective_score(span_embedding, self.p_v) * self.parameters.virtue_weight))
        deontological_score = max(0.0, min(1.0, self.compute_perspective_score(span_embedding, self.p_d) * self.parameters.deontological_weight))
        consequentialist_score = max(0.0, min(1.0, self.compute_perspective_score(span_embedding, self.p_c) * self.parameters.consequentialist_weight))
        
        # Apply thresholds to determine violations (I_P(i,j) = 1 if s_P(i,j) > τ_P)
        virtue_violation = virtue_score > thresholds['virtue_threshold']
        deontological_violation = deontological_score > thresholds['deontological_threshold']
        consequentialist_violation = consequentialist_score > thresholds['consequentialist_threshold']
        
        # Combined score as maximum of individual scores (OR logic for violations)
        combined_score = max(virtue_score, deontological_score, consequentialist_score)
        
        # Add intent hierarchy classification if available
        intent_scores = {}
        dominant_intent = "neutral"
        intent_confidence = 0.0
        
        if self.parameters.enable_intent_hierarchy and self.intent_hierarchy_service:
            try:
                intent_scores = self.intent_hierarchy_service.classify_intent(span_text)
                dominant_intent, intent_confidence = self.intent_hierarchy_service.get_dominant_intent(
                    span_text, self.parameters.intent_threshold
                )
            except Exception as e:
                logger.warning(f"Intent classification failed for span '{span_text}': {e}")
        
        # Generate explanation for the violation
        violation_explanation = self._generate_span_explanation(
            virtue_violation, 
            deontological_violation,
            consequentialist_violation, 
            dominant_intent, 
            intent_confidence
        )
        
        # Create the span with all required fields
        is_violation = any([virtue_violation, deontological_violation, consequentialist_violation])
        
        span = EthicalSpan(
            start=start,
            end=end,
            text=span_text,
            virtue_score=virtue_score,
            deontological_score=deontological_score,
            consequentialist_score=consequentialist_score,
            combined_score=combined_score,
            is_violation=is_violation,
            violation_type=dominant_intent if is_violation else None,
            explanation=violation_explanation,
            intent_category=dominant_intent if intent_confidence > 0.5 else None,
            causal_impact=None,  # Will be set by causal analysis if enabled
            uncertainty=None,     # Will be set by uncertainty analysis if enabled
            purpose_alignment=None  # Will be set by purpose alignment if enabled
        )
        
        # Add any additional attributes that might be needed
        span.virtue_violation = virtue_violation
        span.deontological_violation = deontological_violation
        span.consequentialist_violation = consequentialist_violation
        span.dominant_intent = dominant_intent
        span.intent_confidence = intent_confidence
        span.intent_scores = intent_scores
        
        return span
    
    def find_minimal_spans(self, all_spans: List[EthicalSpan]) -> List[EthicalSpan]:
        """
        Find minimal unethical spans using dynamic programming algorithm.
        
        Implements the mathematical framework's approach to identifying minimal spans:
        M_P(S) = {[i,j] : I_P(i,j)=1 ∧ ∀(k,l) ⊂ (i,j), I_P(k,l)=0}
        
        Where:
        - [i,j] is a text span from token i to j (inclusive)
        - I_P(i,j) = 1 if span [i,j] is a violation from perspective P
        - M_P(S) is the set of minimal spans for perspective P
        
        Args:
            all_spans: List of all evaluated spans
            
        Returns:
            List of minimal spans that represent violations
        """
        # Filter to only include violation spans
        violation_spans = [span for span in all_spans if span.is_violation]
        
        if not violation_spans:
            return []
            
        # Sort spans by length (shortest first)
        violation_spans.sort(key=lambda span: span.end - span.start)
        
        # Find minimal spans (those not contained within other violation spans)
        minimal_spans = []
        for span in violation_spans:
            # Check if this span is contained within any already identified minimal span
            is_contained = False
            for min_span in minimal_spans:
                if (span.start >= min_span.start and span.end <= min_span.end):
                    is_contained = True
                    break
                    
            # If not contained, it's a minimal span
            if not is_contained:
                minimal_spans.append(span)
                
        return minimal_spans
    
    def _generate_span_explanation(
        self, 
        virtue_violation: bool, 
        deontological_violation: bool,
        consequentialist_violation: bool, 
        dominant_intent: str, 
        intent_confidence: float
    ) -> str:
        """
        Generate an explanation for why a span was flagged as a violation.
        
        Args:
            virtue_violation: Whether the span violates virtue ethics
            deontological_violation: Whether the span violates deontological ethics
            consequentialist_violation: Whether the span violates consequentialist ethics
            dominant_intent: The dominant intent category
            intent_confidence: Confidence in the intent classification
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Add perspective-specific explanations
        if virtue_violation:
            explanation_parts.append("Virtue perspective: This content may erode human autonomy.")
            
        if deontological_violation:
            explanation_parts.append("Deontological perspective: This content may violate ethical principles.")
            
        if consequentialist_violation:
            explanation_parts.append("Consequentialist perspective: This content may lead to harmful outcomes.")
            
        # Add intent-based explanation if available
        if intent_confidence > 0.5:
            intent_explanation = self._get_intent_explanation(dominant_intent)
            if intent_explanation:
                explanation_parts.append(f"Intent analysis: {intent_explanation}")
                
        # Combine all explanations
        if explanation_parts:
            return " ".join(explanation_parts)
        else:
            return "No specific ethical concerns identified."
            
    def _get_intent_explanation(self, intent_category: str) -> str:
        """
        Get an explanation for a specific intent category.
        
        Args:
            intent_category: Intent category name
            
        Returns:
            Explanation string for the intent
        """
        intent_explanations = {
            "manipulation": "Content appears designed to manipulate through deceptive or coercive means.",
            "misinformation": "Content contains factual inaccuracies or misleading information.",
            "harm": "Content promotes or enables physical or psychological harm.",
            "discrimination": "Content exhibits bias or discrimination against protected groups.",
            "exploitation": "Content exploits vulnerabilities or promotes unfair advantage.",
            "privacy_violation": "Content compromises privacy or confidentiality.",
            "consent_violation": "Content disregards consent or autonomy principles.",
            "neutral": "No specific harmful intent detected."
        }
        
        return intent_explanations.get(intent_category, "Intent category not recognized.")
