"""
Text Tokenization Service for the Ethical AI Testbed.

This service is responsible for tokenizing text into words/tokens for ethical evaluation.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class TextTokenizationService:
    """Service for tokenizing text into words/tokens"""
    
    def __init__(self):
        """Initialize the text tokenization service."""
        pass
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words/tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with more sophisticated methods
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens
    
    def get_span_text(self, tokens: List[str], start: int, end: int) -> str:
        """
        Get the text for a span of tokens.
        
        Args:
            tokens: List of tokens
            start: Start index of the span
            end: End index of the span (inclusive)
            
        Returns:
            Text of the span
        """
        return ' '.join(tokens[start:end+1])
    
    def generate_spans(self, tokens: List[str], min_span_size: int = 3, max_span_size: int = 15) -> List[Tuple[int, int]]:
        """
        Generate spans of tokens for evaluation.
        
        Args:
            tokens: List of tokens
            min_span_size: Minimum span size
            max_span_size: Maximum span size
            
        Returns:
            List of (start, end) tuples representing spans
        """
        spans = []
        n = len(tokens)
        
        # Generate spans of varying sizes
        for span_size in range(min_span_size, min(max_span_size + 1, n + 1)):
            for start in range(0, n - span_size + 1):
                end = start + span_size - 1
                spans.append((start, end))
                
        return spans
    
    def generate_sliding_window_spans(
        self, 
        tokens: List[str], 
        window_size: int = 10, 
        stride: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Generate spans using a sliding window approach.
        
        Args:
            tokens: List of tokens
            window_size: Size of the sliding window
            stride: Step size for the sliding window
            
        Returns:
            List of (start, end) tuples representing spans
        """
        spans = []
        n = len(tokens)
        
        # Generate spans using sliding window
        for start in range(0, n, stride):
            end = min(start + window_size - 1, n - 1)
            spans.append((start, end))
            
            # If we've reached the end, stop
            if end == n - 1:
                break
                
        return spans
    
    def generate_hierarchical_spans(
        self, 
        tokens: List[str], 
        base_span_size: int = 5,
        max_level: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Generate spans using a hierarchical approach.
        
        Args:
            tokens: List of tokens
            base_span_size: Base span size for the lowest level
            max_level: Maximum hierarchical level
            
        Returns:
            List of (start, end) tuples representing spans
        """
        spans = []
        n = len(tokens)
        
        # Generate spans at each hierarchical level
        for level in range(1, max_level + 1):
            span_size = base_span_size * level
            stride = max(1, span_size // 2)
            
            for start in range(0, n, stride):
                end = min(start + span_size - 1, n - 1)
                spans.append((start, end))
                
                # If we've reached the end, move to next level
                if end == n - 1:
                    break
                    
        return spans
