"""
Graph Attention Service for the Ethical AI Testbed.

This service implements a Graph Attention Network for detecting distributed unethical patterns
that span across multiple text segments beyond local span detection.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Check for torch_geometric availability
try:
    import torch_geometric.nn as pyg_nn
    GRAPH_ATTENTION_AVAILABLE = True
except ImportError:
    GRAPH_ATTENTION_AVAILABLE = False
    logger.warning("torch_geometric not available, falling back to local spans only")

class GraphAttention(nn.Module):
    """
    Graph Attention Network for detecting distributed unethical patterns
    that span across multiple text segments beyond local span detection.
    
    This addresses the v1.0.1 limitation of ~40% distributed recall by adding
    cross-span relationship modeling via graph neural networks.
    """
    
    def __init__(self, emb_dim: int = 384, decay_lambda: float = 5.0):
        """
        Initialize graph attention layer.
        
        Args:
            emb_dim: Embedding dimension (384 for MiniLM-L6-v2)
            decay_lambda: Distance decay parameter for adjacency matrix
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.decay_lambda = decay_lambda
        
        if GRAPH_ATTENTION_AVAILABLE:
            # Graph Convolutional Network layer
            self.gcn = pyg_nn.GCNConv(emb_dim, emb_dim)
            self.attention = pyg_nn.GATConv(emb_dim, emb_dim, heads=4, concat=False)
        else:
            # Fallback to linear layer when torch_geometric not available
            self.linear = nn.Linear(emb_dim, emb_dim)
            
    def create_adjacency_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create adjacency matrix A_ij = cosine_sim(emb_i, emb_j) * exp(-|i-j|/λ)
        
        Args:
            embeddings: Tensor of shape [n_spans, emb_dim]
            
        Returns:
            Adjacency matrix of shape [n_spans, n_spans]
        """
        n_spans = embeddings.shape[0]
        
        # Compute cosine similarity matrix
        embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=1)
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Create distance decay matrix
        indices = torch.arange(n_spans, dtype=torch.float, device=embeddings.device)
        distance_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        decay_matrix = torch.exp(-distance_matrix / self.decay_lambda)
        
        # Combine similarity and decay
        adjacency = cos_sim * decay_matrix
        
        return adjacency
    
    def forward(self, embeddings: torch.Tensor, span_positions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Apply graph attention to embeddings.
        
        Args:
            embeddings: Input embeddings [n_spans, emb_dim]
            span_positions: List of (start, end) positions for each span
            
        Returns:
            Enhanced embeddings with cross-span attention
        """
        if not GRAPH_ATTENTION_AVAILABLE:
            # Fallback: Simple linear transformation
            return self.linear(embeddings)
            
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix(embeddings)
        
        # Convert to edge_index format for torch_geometric
        threshold = 0.1  # Only keep edges above this similarity threshold
        edge_indices = torch.nonzero(adj_matrix > threshold, as_tuple=False).t()
        edge_weights = adj_matrix[adj_matrix > threshold]
        
        # Apply graph attention
        try:
            # Use GAT (Graph Attention) if available
            enhanced_embeddings = self.attention(embeddings, edge_indices)
        except:
            # Fallback to GCN if GAT fails
            enhanced_embeddings = self.gcn(embeddings, edge_indices)
            
        return enhanced_embeddings
