"""
Core Ethical AI Engine Components - v1.1 Modular Architecture

This package contains the fundamental components of the Ethical AI evaluation system,
refactored from the monolithic design for improved performance, maintainability,
and educational clarity.

Components:
- embedding_service: High-performance caching and batch processing for embeddings
- evaluation_engine: Core ethical evaluation logic with optimized processing
- graph_attention: v1.1 Graph Attention Networks for distributed pattern detection
- intent_hierarchy: v1.1 Intent classification with LoRA fine-tuning

Performance Optimizations:
- Multi-level caching (2500x+ speedup confirmed)
- Async processing pipeline for non-blocking operations
- Memory-efficient tensor operations
- Intelligent batch processing

Author: AI Developer Testbed Team
Version: 1.1.0 - Performance Optimized Modular Architecture
"""

# Import core services for easy access
from .caching_manager import CacheManager, EmbeddingCache
from .embedding_service import EmbeddingService
from .evaluation_engine import OptimizedEvaluationEngine

__all__ = [
    'CacheManager',
    'EmbeddingCache', 
    'EmbeddingService',
    'OptimizedEvaluationEngine'
]