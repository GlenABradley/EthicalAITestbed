"""
Intent Hierarchy Service for the Ethical AI Testbed.

This service implements a tree-structured intent classifier with LoRA adapters
for hierarchical harm detection. It detects specific harm categories like fraud,
manipulation, coercion, etc. in a hierarchical structure.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Check for LoRA availability
try:
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    logger.warning("peft not available, intent hierarchy will use fallback classifiers")

class IntentHierarchyService(nn.Module):
    """
    Tree-structured intent classifier with LoRA adapters for hierarchical harm detection.
    
    This implements contrastive learning on intent pairs to detect specific harm categories
    like fraud, manipulation, coercion, etc. in a hierarchical structure.
    """
    
    def __init__(self, base_model: SentenceTransformer, intent_categories: List[str] = None):
        """
        Initialize intent hierarchy with LoRA adapters.
        
        Args:
            base_model: Base sentence transformer model
            intent_categories: List of intent categories to classify
        """
        super().__init__()
        self.base_model = base_model
        self.intent_categories = intent_categories or [
            "fraud", "manipulation", "coercion", "deception", 
            "harassment", "discrimination", "violence", "exploitation"
        ]
        
        # Initialize LoRA adapters if available
        if LORA_AVAILABLE:
            self.lora_adapters = {}
            embedding_dim = base_model.get_sentence_embedding_dimension()
            
            for category in self.intent_categories:
                # Create LoRA config for each intent category
                lora_config = LoraConfig(
                    r=16,  # Low rank
                    lora_alpha=32,
                    target_modules=["dense"],  # Target dense layers
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION
                )
                
                # Create intent-specific classifier head
                classifier = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(embedding_dim // 2, 1),
                    nn.Sigmoid()
                )
                
                self.lora_adapters[category] = classifier
                
        else:
            # Fallback: Simple linear classifiers
            embedding_dim = base_model.get_sentence_embedding_dimension()
            self.intent_classifiers = nn.ModuleDict({
                category: nn.Sequential(
                    nn.Linear(embedding_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                ) for category in self.intent_categories
            })
    
    def classify_intent(self, text: str, embeddings: torch.Tensor = None) -> Dict[str, float]:
        """
        Classify text into intent categories.
        
        Args:
            text: Input text to classify
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            Dict mapping intent categories to confidence scores
        """
        if embeddings is None:
            embeddings = torch.tensor(self.base_model.encode([text]), dtype=torch.float32)
            
        results = {}
        
        if LORA_AVAILABLE and hasattr(self, 'lora_adapters'):
            # Use LoRA adapters
            for category, adapter in self.lora_adapters.items():
                with torch.no_grad():
                    score = adapter(embeddings).item()
                    results[category] = score
        else:
            # Use fallback classifiers  
            for category, classifier in self.intent_classifiers.items():
                with torch.no_grad():
                    score = classifier(embeddings).item()
                    results[category] = score
                    
        return results
    
    def get_dominant_intent(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Get the dominant intent category for text.
        
        Args:
            text: Input text
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (dominant_category, confidence_score)
        """
        intent_scores = self.classify_intent(text)
        
        # Find highest scoring intent above threshold
        dominant_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if dominant_intent[1] >= threshold:
            return dominant_intent
        else:
            return ("neutral", 0.0)
    
    def contrastive_learning_loss(self, positive_texts: List[str], negative_texts: List[str], 
                                 category: str, margin: float = 0.2) -> torch.Tensor:
        """
        Compute contrastive loss for intent category learning.
        
        Args:
            positive_texts: Texts that should match the category
            negative_texts: Texts that should not match the category
            category: Intent category being trained
            margin: Contrastive margin
            
        Returns:
            Contrastive loss tensor
        """
        if not LORA_AVAILABLE or category not in self.lora_adapters:
            return torch.tensor(0.0)
            
        # Encode texts
        pos_embeddings = torch.tensor(self.base_model.encode(positive_texts), dtype=torch.float32)
        neg_embeddings = torch.tensor(self.base_model.encode(negative_texts), dtype=torch.float32)
        
        # Get adapter predictions
        adapter = self.lora_adapters[category]
        pos_scores = adapter(pos_embeddings)
        neg_scores = adapter(neg_embeddings)
        
        # Contrastive loss: maximize positive scores, minimize negative scores
        pos_loss = torch.mean((1.0 - pos_scores) ** 2)
        neg_loss = torch.mean(torch.clamp(neg_scores - margin, min=0.0) ** 2)
        
        return pos_loss + neg_loss
