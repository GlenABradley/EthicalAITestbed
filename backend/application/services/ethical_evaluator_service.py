"""
Ethical Evaluator Service for the Ethical AI Testbed.

This service is responsible for evaluating text against ethical perspectives
using the v3.0 semantic embedding framework. It implements the mathematical framework
for multi-perspective ethical evaluation with orthonormal basis vectors.
"""

import logging
import time
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Import domain models
from core.domain.models.ethical_parameters import EthicalParameters
from core.domain.models.ethical_span import EthicalSpan
from core.domain.models.ethical_evaluation import EthicalEvaluation

# Import services
from application.services.vector_generation_service import VectorGenerationService
from application.services.graph_attention_service import GraphAttention
from application.services.embedding_service import EmbeddingService
from application.services.ethical_span_service import EthicalSpanService
from application.services.text_tokenization_service import TextTokenizationService
from application.services.intent_hierarchy_service import IntentHierarchyService
from application.services.causal_counterfactual_service import CausalCounterfactualService
from application.services.uncertainty_analyzer_service import UncertaintyAnalyzerService
from application.services.irl_purpose_alignment_service import IRLPurposeAlignmentService
from application.services.learning_service import LearningService

logger = logging.getLogger(__name__)

class EthicalEvaluatorService:
    """Main ethical evaluation service implementing the mathematical framework"""
    
    def __init__(self, parameters: EthicalParameters = None, db_collection=None):
        """Initialize the ethical evaluator service.
        
        Args:
            parameters: Configuration parameters for ethical evaluation
            db_collection: MongoDB collection for learning layer
        """
        self.parameters = parameters or EthicalParameters()
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(self.parameters.embedding_model)
        
        # Use the extracted VectorGenerationService instead of EthicalVectorGenerator
        self.vector_service = VectorGenerationService(self.embedding_service.model)
        
        # Initialize learning service
        self.learning_layer = LearningService(db_collection)
        
        # Initialize ethical vectors
        self.p_v, self.p_d, self.p_c = self.vector_service.get_all_vectors()
        
        # v1.1 UPGRADE: Initialize graph attention layer
        embedding_dim = self.embedding_service.embedding_dim
        self.graph_attention = GraphAttention(
            emb_dim=embedding_dim,
            decay_lambda=self.parameters.graph_decay_lambda
        )
        
        # Initialize text tokenization service
        self.tokenization_service = TextTokenizationService()
        
        # Initialize ethical span service
        self.span_service = EthicalSpanService(
            embedding_service=self.embedding_service,
            p_v=self.p_v,
            p_d=self.p_d,
            p_c=self.p_c,
            parameters=self.parameters,
            intent_hierarchy_service=None  # Will be set later when intent_hierarchy is extracted
        )
        
        # v1.1 UPGRADE: Initialize intent hierarchy service
        self.intent_hierarchy = None
        if self.parameters.enable_intent_hierarchy:
            self.intent_hierarchy = IntentHierarchyService(
                base_model=self.embedding_service.model,
                intent_categories=self.parameters.intent_categories
            )
        
        # v1.1 UPGRADE: Initialize causal counterfactual service
        self.causal_analyzer = None
        if self.parameters.enable_causal_analysis:
            self.causal_analyzer = CausalCounterfactualService(self)
        
        # v1.1 UPGRADE: Initialize uncertainty analyzer service
        self.uncertainty_analyzer = None
        if self.parameters.enable_uncertainty_analysis:
            self.uncertainty_analyzer = UncertaintyAnalyzerService(self)
        
        # v1.1 UPGRADE: Initialize purpose alignment service
        self.purpose_alignment = None
        if self.parameters.enable_purpose_alignment:
            self.purpose_alignment = IRLPurposeAlignmentService(self)
    
    def clear_embedding_cache(self):
        """Clear the embedding cache to free memory"""
        self.embedding_service.clear_cache()
        logger.info("Embedding cache cleared")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/tokens"""
        return self.tokenization_service.tokenize(text)
    
    def evaluate_span(self, tokens: List[str], start: int, end: int, 
                     adjusted_thresholds: Dict[str, float] = None) -> EthicalSpan:
        """Evaluate a single span of tokens with dynamic thresholds"""
        return self.span_service.evaluate_span(tokens, start, end, adjusted_thresholds)
    
    def find_minimal_spans(self, all_spans: List[EthicalSpan]) -> List[EthicalSpan]:
        """Find minimal unethical spans using dynamic programming algorithm"""
        return self.span_service.find_minimal_spans(all_spans)
    
    # Additional methods will be added in future refactoring steps
