"""
Vector Generation Service for the Ethical AI Testbed.

This service is responsible for generating orthogonal ethical perspective vectors
using the v3.0 semantic embedding framework. It implements the mathematical foundation
for multi-perspective ethical evaluation with orthonormal basis vectors.
"""

import logging
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from application.services.embedding_service import EmbeddingService
from core.domain.ethical_embedding_statement import EthicalEmbeddingStatement

logger = logging.getLogger(__name__)

class VectorGenerationService:
    """Generates orthogonal ethical perspective vectors using mathematical framework.
    
    Implements the mathematical framework for multi-perspective ethical evaluation
    with orthonormal basis vectors p_v, p_d, p_c that are enforced to be orthogonal
    using Gram-Schmidt orthogonalization.
    
    Mathematical Foundation:
    - Each perspective vector points toward increasing violation
    - Orthogonality ensures independence: p_i · p_j = δ_ij
    - Contrastive learning approach with virtue-vice, rule-compliance, outcome pairs
    - Vectors are normalized to unit length for consistent projections
    """
    
    def __init__(self, model=None, embedding_service=None):
        """Initialize the vector generation service with a model or embedding service.
        
        Args:
            model: SentenceTransformer model for generating embeddings (deprecated)
            embedding_service: EmbeddingService for generating embeddings (preferred)
        """
        if embedding_service:
            self.embedding_service = embedding_service
            self.model = embedding_service.model
        elif model:
            self.model = model
            self.embedding_service = None
        else:
            # Create default embedding service if neither is provided
            self.embedding_service = EmbeddingService()
            self.model = self.embedding_service.model
            
        self._orthogonal_vectors = None
        
    def _get_virtue_examples(self) -> Tuple[List[str], List[str]]:
        """Virtue-based examples derived from Core Axiom: Maximize human autonomy within objective empirical truth"""
        # Get examples directly from the centralized ethical embedding statement
        return EthicalEmbeddingStatement.get_virtue_examples()
    
    def _get_deontological_examples(self) -> Tuple[List[str], List[str]]:
        """Deontological examples based on rule-consistency and procedural integrity"""
        # Get examples directly from the centralized ethical embedding statement
        return EthicalEmbeddingStatement.get_deontological_examples()
    
    def _get_consequentialist_examples(self) -> Tuple[List[str], List[str]]:
        """Consequentialist examples based on outcome analysis and harm/benefit ratio"""
        # Get examples directly from the centralized ethical embedding statement
        return EthicalEmbeddingStatement.get_consequentialist_examples()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.embedding_service:
            return self.embedding_service.get_embedding(text)
        else:
            # Fallback to direct model usage if embedding service not available
            raise ValueError("Embedding service is not available")
    
    def _gram_schmidt_orthogonalization(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Gram-Schmidt orthogonalization to ensure orthogonal basis vectors.
        
        Ensures p_i · p_j = δ_ij (Kronecker delta) for independence.
        
        Args:
            vectors: List of vectors to orthogonalize
            
        Returns:
            List of orthogonalized vectors
        """
        if len(vectors) == 0:
            return []
        
        # Start with first vector normalized
        orthogonal_vectors = [vectors[0] / np.linalg.norm(vectors[0])]
        
        for i in range(1, len(vectors)):
            # Start with current vector
            current_vector = vectors[i].copy()
            
            # Subtract projections onto all previous orthogonal vectors
            for j in range(len(orthogonal_vectors)):
                projection = np.dot(current_vector, orthogonal_vectors[j]) * orthogonal_vectors[j]
                current_vector -= projection
            
            # Normalize the orthogonalized vector
            if np.linalg.norm(current_vector) > 1e-10:  # Avoid division by zero
                orthogonal_vectors.append(current_vector / np.linalg.norm(current_vector))
            else:
                logger.warning(f"Vector {i} became zero during orthogonalization")
                # Create a random orthogonal vector if needed
                random_vector = np.random.randn(vectors[0].shape[0])
                for j in range(len(orthogonal_vectors)):
                    projection = np.dot(random_vector, orthogonal_vectors[j]) * orthogonal_vectors[j]
                    random_vector -= projection
                orthogonal_vectors.append(random_vector / np.linalg.norm(random_vector))
        
        return orthogonal_vectors
    
    def generate_orthogonal_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate orthogonal ethical perspective vectors using v3.0 semantic embedding framework.
        
        Implements the Core Axiom: Maximize human autonomy within objective empirical truth.
        
        Semantic Framework Integration:
        - Virtue: Autonomy-enhancing vs autonomy-eroding behaviors
        - Deontological: Truth prerequisites and ethical principles adherence
        - Consequentialist: Outcomes that maximize/minimize autonomy dimensions
        
        Mathematical Foundation:
        - Core Axiom: Maximize Σ(D_i) within t ≥ 0.95
        - Dimensions: D1-D5 (Bodily, Cognitive, Behavioral, Social, Existential)
        - Truth Prerequisites: T1-T4 (Accuracy, Misinformation Prevention, Objectivity, Distinction)
        - Principles: P1-P8 (Consent, Transparency, Non-Aggression, etc.)
        
        Returns:
            Tuple of three orthogonal vectors (virtue, deontological, consequentialist)
        """
        if self._orthogonal_vectors is None:
            logger.info("Generating orthogonal ethical vectors using v3.0 semantic embedding framework")
            logger.info("Core Axiom: Maximize human autonomy within objective empirical truth")
            
            # Generate raw vectors using v3.0 semantic embedding approach
            virtue_pos, virtue_neg = self._get_virtue_examples()
            deont_pos, deont_neg = self._get_deontological_examples()
            conseq_pos, conseq_neg = self._get_consequentialist_examples()
            
            # Compute embeddings for positive and negative examples
            logger.info("Computing embeddings for autonomy-enhancing vs autonomy-eroding behaviors")
            if self.embedding_service:
                virtue_pos_emb = self.embedding_service.get_embeddings(virtue_pos)
                virtue_neg_emb = self.embedding_service.get_embeddings(virtue_neg)
                
                logger.info("Computing embeddings for truth prerequisites and ethical principles")
                deont_pos_emb = self.embedding_service.get_embeddings(deont_pos)
                deont_neg_emb = self.embedding_service.get_embeddings(deont_neg)
                
                logger.info("Computing embeddings for autonomy dimension outcomes")
                conseq_pos_emb = self.embedding_service.get_embeddings(conseq_pos)
                conseq_neg_emb = self.embedding_service.get_embeddings(conseq_neg)
            else:
                # Fallback to direct model usage
                virtue_pos_emb = self.model.encode(virtue_pos)
                virtue_neg_emb = self.model.encode(virtue_neg)
                
                logger.info("Computing embeddings for truth prerequisites and ethical principles")
                deont_pos_emb = self.model.encode(deont_pos)
                deont_neg_emb = self.model.encode(deont_neg)
                
                logger.info("Computing embeddings for autonomy dimension outcomes")
                conseq_pos_emb = self.model.encode(conseq_pos)
                conseq_neg_emb = self.model.encode(conseq_neg)
            
            # Create direction vectors pointing toward violations of the Core Axiom
            virtue_center_pos = np.mean(virtue_pos_emb, axis=0)
            virtue_center_neg = np.mean(virtue_neg_emb, axis=0)
            virtue_vector = virtue_center_neg - virtue_center_pos
            
            deont_center_pos = np.mean(deont_pos_emb, axis=0)
            deont_center_neg = np.mean(deont_neg_emb, axis=0)
            deont_vector = deont_center_neg - deont_center_pos
            
            conseq_center_pos = np.mean(conseq_pos_emb, axis=0)
            conseq_center_neg = np.mean(conseq_neg_emb, axis=0)
            conseq_vector = conseq_center_neg - conseq_center_pos
            
            # Apply Gram-Schmidt orthogonalization for independence
            logger.info("Applying Gram-Schmidt orthogonalization for vector independence")
            raw_vectors = [virtue_vector, deont_vector, conseq_vector]
            orthogonal_vectors = self._gram_schmidt_orthogonalization(raw_vectors)
            
            # Verify orthogonality with v3.0 semantic embedding
            logger.info("Verifying orthogonality of v3.0 semantic embedding vectors:")
            for i in range(len(orthogonal_vectors)):
                for j in range(i + 1, len(orthogonal_vectors)):
                    dot_product = np.dot(orthogonal_vectors[i], orthogonal_vectors[j])
                    logger.info(f"  p_{i} · p_{j} = {dot_product:.6f} (Core Axiom independence)")
            
            self._orthogonal_vectors = tuple(orthogonal_vectors)
            logger.info("Successfully generated v3.0 semantic embedding orthogonal vectors")
            logger.info("Framework: Autonomy maximization within empirical truth prerequisite")
            
        return self._orthogonal_vectors
    
    def get_all_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all three orthogonal ethical perspective vectors
        
        Returns:
            Tuple of three orthogonal vectors (virtue, deontological, consequentialist)
        """
        return self.generate_orthogonal_vectors()
