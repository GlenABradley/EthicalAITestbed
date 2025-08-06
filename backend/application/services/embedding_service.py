"""
Embedding Service for the Ethical AI Testbed.

This service is responsible for generating and managing embeddings for text analysis.
It provides a clean interface for embedding operations, making the embedding functionality
more accessible and open.

This service implements the Neutral Ethical AI Embedding v1.1 framework, which defines
the principles and structure for ethical evaluation through vector projections.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing embeddings for text analysis"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_cache = {}
        logger.info(f"Initialized EmbeddingService with model {model_name}, embedding dimension: {self.embedding_dim}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Use cache for efficiency if available
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Generate fresh embedding
        with torch.no_grad():
            embedding = self.model.encode([text])[0]
            
        # Cache for future use
        self.embedding_cache[text] = embedding
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        # Check cache for each text
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        
        # Find which texts need embedding
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                embeddings[i] = self.embedding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # Generate embeddings for uncached texts
        if uncached_texts:
            with torch.no_grad():
                new_embeddings = self.model.encode(uncached_texts)
                
            # Update cache and results
            for i, embedding in zip(uncached_indices, new_embeddings):
                text = texts[i]
                self.embedding_cache[text] = embedding
                embeddings[i] = embedding
                
        return np.array(embeddings)
    
    def get_span_embedding(self, tokens: List[str], start: int, end: int) -> np.ndarray:
        """
        Get embedding for a span of tokens.
        
        Args:
            tokens: List of tokens
            start: Start index of the span
            end: End index of the span (inclusive)
            
        Returns:
            Embedding vector for the span
        """
        span_text = ' '.join(tokens[start:end+1])
        return self.get_embedding(span_text)
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize an embedding vector to unit length.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1 = self.normalize_embedding(embedding1)
        embedding2 = self.normalize_embedding(embedding2)
        
        # Compute dot product (cosine similarity for unit vectors)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache = {}
        logger.info(f"Cleared embedding cache ({cache_size} entries)")
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "cache_size": len(self.embedding_cache),
            "model_type": type(self.model).__name__
        }
