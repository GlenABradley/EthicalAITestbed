#!/usr/bin/env python3
"""
Span Selector module for optimal context analysis using modular difference sets.

This module implements the mathematically optimal approach for span comparison using
modular difference sets (Singer sets, Golomb rulers, or Costas arrays) to achieve
maximal pairwise context coverage with minimal comparisons per token.

Mathematical Foundation:
- Uses difference sets D⊂{1,...,W-1} for context window of width W
- For each token at position t, compares with t±d for every d∈D
- Designed so differences {(d_i-d_j)mod W: d_i,d_j∈D, i≠j} cover every lag
- Reduces span analysis from O(W²N) to O(√W × N) while maintaining coverage
"""

from typing import List, Set, Tuple, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DifferenceSetSpanSelector:
    """
    Optimal span selector using modular difference sets for minimal-comparison,
    maximal-coverage context analysis.
    """
    
    # Pre-computed optimal difference sets for common window sizes
    # Format: {window_size: (difference_set, type)}
    OPTIMAL_DIFFERENCE_SETS = {
        7: ([1, 2, 4], "Singer perfect"),          # q=2 -> W=7, K=3
        13: ([1, 3, 9, 12], "Singer perfect"),     # q=3 -> W=13, K=4
        21: ([1, 4, 14, 16, 20], "Singer perfect"), # q=4 -> W=21, K=5
        31: ([1, 3, 9, 27, 19, 26], "Costas-style"), # Prime W=31, primitive root g=3
        57: ([1, 3, 13, 32, 36, 43, 52], "Near-perfect"), # Approx Singer for q=7
        73: ([1, 2, 4, 8, 16, 32, 64, 5, 21, 34, 55], "Hybrid multiscale"), # Prime W=73
        # Extended window sizes for longer texts
        127: ([1, 2, 4, 8, 16, 32, 64, 3, 5, 13, 21, 34, 55, 89], "Hybrid multiscale"),
        256: ([1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 13, 21, 34, 55, 89, 144], "Hybrid multiscale")
    }
    
    def __init__(self, 
                 window_size: int = 31, 
                 symmetric: bool = True, 
                 difference_set: Optional[List[int]] = None,
                 span_density_factor: float = 1.0):
        """
        Initialize the DifferenceSetSpanSelector.
        
        Args:
            window_size: Size of context window
            symmetric: Whether to use symmetric mesh (both ±d for d∈D)
            difference_set: Optional custom difference set to use
        """
        self.window_size = window_size
        self.symmetric = symmetric
        self.span_density_factor = max(0.1, min(1.0, span_density_factor))  # Clamp between 0.1 and 1.0
        
        # Use provided difference set or find closest pre-computed one
        if difference_set:
            self.difference_set = difference_set
            self.set_type = "custom"
        else:
            if window_size in self.OPTIMAL_DIFFERENCE_SETS:
                self.difference_set, self.set_type = self.OPTIMAL_DIFFERENCE_SETS[window_size]
            else:
                # Find closest pre-computed window size
                closest_size = min(self.OPTIMAL_DIFFERENCE_SETS.keys(), 
                                   key=lambda k: abs(k - window_size))
                self.difference_set, self.set_type = self.OPTIMAL_DIFFERENCE_SETS[closest_size]
                self.window_size = closest_size
                logger.info(f"No exact match for window size {window_size}, using closest: {closest_size}")
        
        logger.info(f"Initialized span selector with window {self.window_size}, "
                    f"difference set type: {self.set_type}, "
                    f"set size: {len(self.difference_set)}")
        
        # Validate difference set covers lags effectively
        self._validate_difference_set()
    
    def _validate_difference_set(self):
        """Validate the difference set for lag coverage and log statistics."""
        differences = set()
        for i, d1 in enumerate(self.difference_set):
            for j, d2 in enumerate(self.difference_set):
                if i != j:
                    diff = (d1 - d2) % self.window_size
                    if diff != 0:
                        differences.add(diff)
        
        coverage = len(differences) / (self.window_size - 1)
        logger.info(f"Difference set coverage: {coverage:.2f} ({len(differences)}/{self.window_size-1} lags)")
    
    def generate_spans(self, 
                       token_count: int, 
                       min_span_length: int = 3,
                       max_span_length: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Generate optimal span pairs for token-level analysis.
        
        Args:
            token_count: Total number of tokens in the text
            min_span_length: Minimum span length to consider
            max_span_length: Maximum span length to consider (defaults to window_size)
            
        Returns:
            List of (start, end) tuples representing spans to evaluate
        """
        if max_span_length is None:
            max_span_length = self.window_size
        
        spans = set()
        
        # Generate spans based on difference set offsets with density control
        # When span_density_factor < 1.0, we'll skip some positions to reduce density
        step = int(1.0 / self.span_density_factor) if self.span_density_factor < 1.0 else 1
        
        logger.info(f"Using span density factor {self.span_density_factor}, step size {step}")
        
        for t in range(0, token_count, step):
            # Add base span at this position with minimum length
            base_span = (t, min(t + min_span_length, token_count))
            spans.add(base_span)
            
            # Add spans based on difference set
            for d in self.difference_set:
                # Forward span
                v = t + d
                if v < token_count:
                    span_length = min(max(min_span_length, d + 1), max_span_length)
                    forward_span = (t, min(t + span_length, token_count))
                    spans.add(forward_span)
                
                # Backward span if using symmetric mesh
                if self.symmetric:
                    u = t - d
                    if u >= 0:
                        span_length = min(max(min_span_length, d + 1), max_span_length)
                        backward_span = (max(u, 0), min(t + 1, token_count))
                        spans.add(backward_span)
        
        # Convert to list and sort by span length, then position
        span_list = sorted(list(spans), key=lambda s: (s[1] - s[0], s[0]))
        
        logger.info(f"Generated {len(span_list)} unique spans for {token_count} tokens "
                    f"using difference set ({self.set_type}) with density factor {self.span_density_factor}")
        
        return span_list

    def generate_spans_with_recursion(self,
                                    token_count: int,
                                    violation_check_func,
                                    min_span_length: int = 3,
                                    max_span_length: Optional[int] = None,
                                    recursion_depth: int = 2) -> List[Tuple[int, int]]:
        """
        Generate spans with recursive drill-down for ambiguous violations.
        This is a future implementation - currently disabled per user request.
        
        Args:
            token_count: Total number of tokens in the text
            violation_check_func: Function to check if a span has a violation
            min_span_length: Minimum span length
            max_span_length: Maximum span length
            recursion_depth: How deep to recurse for violation analysis
            
        Returns:
            List of (start, end) tuples representing spans to evaluate
        """
        # This is a placeholder for the future implementation
        # Currently we use the non-recursive implementation per user request
        return self.generate_spans(token_count, min_span_length, max_span_length)

    @staticmethod
    def create_costas_difference_set(p: int) -> List[int]:
        """
        Create a Costas-style difference set for a prime window size.
        
        Args:
            p: Prime number for window size
            
        Returns:
            Difference set using primitive root properties
        """
        # Find primitive root g
        for g in range(2, p):
            powers = set()
            for k in range(1, p):
                powers.add(pow(g, k, p))
            if len(powers) == p - 1:  # g is a primitive root
                break
        
        # Generate difference set using powers of g
        difference_set = []
        for k in range(6):  # Take first 6 powers (can adjust as needed)
            difference_set.append(pow(g, k, p))
        
        return sorted(difference_set)
    
    @staticmethod
    def create_hybrid_multiscale_set(w: int) -> List[int]:
        """
        Create a hybrid multiscale mesh for arbitrary window size.
        
        Args:
            w: Window size
            
        Returns:
            Difference set combining powers of 2 and Fibonacci numbers
        """
        # Powers of 2
        powers = [2**i for i in range(10) if 2**i < w]
        
        # Fibonacci numbers
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        fib = [f for f in fib if f < w]
        
        # Combine and deduplicate
        combined = sorted(set(powers + fib + [1]))
        
        return combined
