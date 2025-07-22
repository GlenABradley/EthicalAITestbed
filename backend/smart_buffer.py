"""
Smart Buffer System for ML Training Data Streams - Phase 3

This module implements intelligent buffering for continuous ML training data evaluation.
It handles token-by-token streaming analysis, semantic boundary detection, and 
dynamic performance optimization for real-time ethical monitoring during training.

Key Features:
- Token-by-token streaming analysis
- Semantic boundary detection for intelligent chunking
- Dynamic buffer sizing based on content and performance
- Real-time pattern recognition for ethical violations
- Context-aware processing across training batches
- Performance optimization with adaptive thresholds

Author: Ethical AI Developer Testbed Team
Version: 3.0.0 - Smart Buffer System
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re

logger = logging.getLogger(__name__)

class BufferState(Enum):
    """Buffer processing states."""
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    FLUSHING = "flushing"
    ERROR = "error"

class ContentBoundary(Enum):
    """Types of content boundaries for intelligent chunking."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC_UNIT = "semantic_unit"
    TOKEN_LIMIT = "token_limit"
    TIME_LIMIT = "time_limit"

@dataclass
class BufferConfig:
    """Configuration for smart buffer system."""
    max_tokens: int = 512
    min_tokens: int = 32
    max_time_seconds: float = 5.0
    min_time_seconds: float = 0.1
    semantic_threshold: float = 0.7
    performance_threshold_ms: float = 100.0
    max_concurrent_evaluations: int = 5
    auto_resize: bool = True
    context_window: int = 128
    pattern_detection: bool = True
    
@dataclass
class BufferMetrics:
    """Performance and operational metrics."""
    tokens_processed: int = 0
    evaluations_completed: int = 0
    average_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    buffer_utilization: float = 0.0
    boundary_detections: int = 0
    interventions_triggered: int = 0
    total_runtime: float = 0.0
    memory_usage_mb: float = 0.0
    
@dataclass
class TokenChunk:
    """Container for token chunks with metadata."""
    tokens: List[str]
    text: str
    timestamp: float
    chunk_id: str
    training_step: Optional[int] = None
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    boundary_type: Optional[ContentBoundary] = None
    semantic_score: Optional[float] = None
    
@dataclass
class BufferAnalysis:
    """Results from buffer content analysis."""
    ethical_score: float
    violation_count: int
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    intervention_required: bool
    processing_time: float
    patterns_detected: List[str]
    recommendations: List[str]
    autonomy_scores: Dict[str, float]
    vector_analysis: Dict[str, Any]

class PatternDetector:
    """Detects ethical patterns in streaming content."""
    
    def __init__(self):
        """Initialize pattern detection system."""
        self.violation_patterns = [
            r'\b(hate|racist|discriminat\w+)\b',
            r'\b(manipulat\w+|deceiv\w+|mislead\w+)\b',
            r'\b(harm\w*|hurt|damage|destroy)\b',
            r'\b(bias\w*|unfair|prejudic\w+)\b'
        ]
        
        self.ethical_patterns = [
            r'\b(fair\w*|just|equit\w+)\b',
            r'\b(respect\w*|dignity|honor)\b', 
            r'\b(help\w*|assist|support|benefit)\b',
            r'\b(honest\w*|transparent|open)\b'
        ]
        
        self.compiled_violation_patterns = [re.compile(p, re.IGNORECASE) for p in self.violation_patterns]
        self.compiled_ethical_patterns = [re.compile(p, re.IGNORECASE) for p in self.ethical_patterns]
        
    def detect_patterns(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Detect ethical and violation patterns in text.
        
        Returns:
            Tuple of (violations_detected, ethical_patterns_detected)
        """
        violations = []
        ethical = []
        
        for pattern in self.compiled_violation_patterns:
            if pattern.search(text):
                violations.append(pattern.pattern)
                
        for pattern in self.compiled_ethical_patterns:
            if pattern.search(text):
                ethical.append(pattern.pattern)
                
        return violations, ethical

class SemanticBoundaryDetector:
    """Detects semantic boundaries for intelligent chunking."""
    
    def __init__(self):
        """Initialize semantic boundary detection."""
        self.sentence_endings = ['.', '!', '?', '...']
        self.paragraph_indicators = ['\n\n', '\n\r', '\r\n\r\n']
        
    def detect_boundaries(self, text: str) -> List[Tuple[int, ContentBoundary]]:
        """
        Detect semantic boundaries in text.
        
        Returns:
            List of (position, boundary_type) tuples
        """
        boundaries = []
        
        # Detect sentence boundaries
        for i, char in enumerate(text):
            if char in self.sentence_endings:
                if i < len(text) - 1 and text[i + 1] in [' ', '\n', '\r']:
                    boundaries.append((i + 1, ContentBoundary.SENTENCE))
        
        # Detect paragraph boundaries
        for indicator in self.paragraph_indicators:
            pos = 0
            while True:
                pos = text.find(indicator, pos)
                if pos == -1:
                    break
                boundaries.append((pos + len(indicator), ContentBoundary.PARAGRAPH))
                pos += len(indicator)
        
        return sorted(boundaries)
    
    def calculate_semantic_coherence(self, chunk1: str, chunk2: str) -> float:
        """
        Calculate semantic coherence between two chunks.
        Simple implementation - could be enhanced with embeddings.
        """
        # Simple word overlap approach
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0

class SmartBuffer:
    """
    Intelligent buffer for streaming ML training data with ethical analysis.
    
    Handles token accumulation, semantic boundary detection, and real-time 
    ethical evaluation with performance optimization.
    """
    
    def __init__(self, 
                 config: BufferConfig,
                 evaluator_callback: Optional[Callable] = None,
                 ml_ethics_engine: Optional[Any] = None):
        """
        Initialize smart buffer system.
        
        Args:
            config: Buffer configuration
            evaluator_callback: Function to call for ethical evaluation
            ml_ethics_engine: ML ethics engine for vector generation
        """
        self.config = config
        self.evaluator_callback = evaluator_callback
        self.ml_ethics_engine = ml_ethics_engine
        
        # Buffer state
        self.state = BufferState.IDLE
        self.buffer_lock = threading.RLock()
        self.token_buffer = deque(maxlen=config.max_tokens * 2)  # Allow overflow
        self.current_text = ""
        self.chunk_counter = 0
        
        # Analysis components
        self.pattern_detector = PatternDetector()
        self.boundary_detector = SemanticBoundaryDetector()
        
        # Performance tracking
        self.metrics = BufferMetrics()
        self.processing_times = deque(maxlen=100)  # Keep last 100 processing times
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_evaluations)
        self.active_evaluations = 0
        
        # Context management
        self.context_history = deque(maxlen=config.context_window)
        self.last_evaluation_time = 0.0
        
        logger.info(f"Smart Buffer initialized with config: {config}")
    
    async def add_tokens(self, tokens: List[str], metadata: Optional[Dict[str, Any]] = None) -> Optional[BufferAnalysis]:
        """
        Add tokens to the buffer and process if conditions are met.
        
        Args:
            tokens: List of tokens to add
            metadata: Optional metadata for the tokens
            
        Returns:
            BufferAnalysis if processing was triggered, None otherwise
        """
        with self.buffer_lock:
            try:
                # Update state
                if self.state == BufferState.IDLE:
                    self.state = BufferState.ACCUMULATING
                
                # Add tokens to buffer
                for token in tokens:
                    self.token_buffer.append(token)
                    self.current_text += token + " "
                
                # Update metrics
                self.metrics.tokens_processed += len(tokens)
                
                # Check if we should process the buffer
                should_process = await self._should_process_buffer()
                
                if should_process:
                    return await self._process_buffer(metadata)
                    
                return None
                
            except Exception as e:
                logger.error(f"Error adding tokens to buffer: {e}")
                self.state = BufferState.ERROR
                return None
    
    async def _should_process_buffer(self) -> bool:
        """Determine if the buffer should be processed now."""
        current_time = time.time()
        token_count = len(self.token_buffer)
        time_since_last = current_time - self.last_evaluation_time
        
        # Check various conditions
        conditions = {
            "token_limit": token_count >= self.config.max_tokens,
            "min_tokens_and_time": token_count >= self.config.min_tokens and time_since_last >= self.config.min_time_seconds,
            "max_time": time_since_last >= self.config.max_time_seconds,
            "semantic_boundary": self._has_semantic_boundary(),
            "pattern_detected": self._has_critical_pattern()
        }
        
        # Log decision reasoning
        active_conditions = [k for k, v in conditions.items() if v]
        if active_conditions:
            logger.debug(f"Processing triggered by: {active_conditions}")
        
        return any(conditions.values())
    
    def _has_semantic_boundary(self) -> bool:
        """Check if current text has a clear semantic boundary."""
        if len(self.current_text) < 10:  # Too short to have meaningful boundary
            return False
            
        boundaries = self.boundary_detector.detect_boundaries(self.current_text)
        
        # Check if we have a sentence or paragraph boundary near the end
        text_length = len(self.current_text)
        for position, boundary_type in boundaries:
            if text_length - position <= 20:  # Within last 20 characters
                return True
                
        return False
    
    def _has_critical_pattern(self) -> bool:
        """Check if current text contains critical patterns requiring immediate processing."""
        if not self.config.pattern_detection:
            return False
            
        violations, _ = self.pattern_detector.detect_patterns(self.current_text)
        return len(violations) > 0
    
    async def _process_buffer(self, metadata: Optional[Dict[str, Any]] = None) -> BufferAnalysis:
        """Process the current buffer contents."""
        start_time = time.time()
        
        try:
            self.state = BufferState.PROCESSING
            self.active_evaluations += 1
            
            # Create chunk for processing
            chunk = TokenChunk(
                tokens=list(self.token_buffer),
                text=self.current_text.strip(),
                timestamp=start_time,
                chunk_id=f"chunk_{self.chunk_counter}",
                metadata=metadata or {}
            )
            self.chunk_counter += 1
            
            # Analyze content
            analysis = await self._analyze_chunk(chunk)
            
            # Update context history
            self.context_history.append({
                "chunk_id": chunk.chunk_id,
                "ethical_score": analysis.ethical_score,
                "violation_count": analysis.violation_count,
                "timestamp": start_time
            })
            
            # Clear buffer (keep some context if configured)
            await self._flush_buffer(keep_context=True)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
            self.metrics.evaluations_completed += 1
            self.last_evaluation_time = start_time
            
            if analysis.intervention_required:
                self.metrics.interventions_triggered += 1
            
            self.state = BufferState.IDLE
            self.active_evaluations -= 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            self.state = BufferState.ERROR
            self.active_evaluations -= 1
            
            return BufferAnalysis(
                ethical_score=0.0,
                violation_count=0,
                risk_level="ERROR",
                intervention_required=True,
                processing_time=time.time() - start_time,
                patterns_detected=[],
                recommendations=[f"Processing error: {str(e)}"],
                autonomy_scores={},
                vector_analysis={}
            )
    
    async def _analyze_chunk(self, chunk: TokenChunk) -> BufferAnalysis:
        """Analyze a text chunk for ethical content."""
        start_time = time.time()
        
        try:
            # Pattern detection
            violation_patterns, ethical_patterns = self.pattern_detector.detect_patterns(chunk.text)
            
            # Initialize analysis results
            analysis = BufferAnalysis(
                ethical_score=0.5,  # Default neutral
                violation_count=len(violation_patterns),
                risk_level="MEDIUM",
                intervention_required=False,
                processing_time=0.0,
                patterns_detected=violation_patterns + ethical_patterns,
                recommendations=[],
                autonomy_scores={},
                vector_analysis={}
            )
            
            # If we have an evaluator callback, use it for detailed analysis
            if self.evaluator_callback:
                try:
                    # Run evaluation in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    evaluation_result = await loop.run_in_executor(
                        self.executor, 
                        self.evaluator_callback, 
                        chunk.text
                    )
                    
                    if evaluation_result:
                        # Extract ethical scores
                        if hasattr(evaluation_result, 'autonomy_dimensions'):
                            autonomy_scores = evaluation_result.autonomy_dimensions
                            analysis.ethical_score = sum(autonomy_scores.values()) / len(autonomy_scores)
                            analysis.autonomy_scores = autonomy_scores
                        
                        if hasattr(evaluation_result, 'minimal_violation_count'):
                            analysis.violation_count = evaluation_result.minimal_violation_count
                        
                        # Generate ML vectors if ML ethics engine is available
                        if self.ml_ethics_engine:
                            ml_vectors = self.ml_ethics_engine.convert_evaluation_to_ml_vectors(evaluation_result)
                            analysis.vector_analysis = ml_vectors.to_dict()
                    
                except Exception as e:
                    logger.warning(f"Evaluator callback failed: {e}")
            
            # Determine risk level and intervention requirement
            if analysis.ethical_score < 0.3 or analysis.violation_count > 3:
                analysis.risk_level = "CRITICAL"
                analysis.intervention_required = True
                analysis.recommendations.append("CRITICAL: Immediate intervention required")
            elif analysis.ethical_score < 0.5 or analysis.violation_count > 1:
                analysis.risk_level = "HIGH"
                analysis.intervention_required = True
                analysis.recommendations.append("HIGH RISK: Consider intervention")
            elif analysis.ethical_score < 0.7:
                analysis.risk_level = "MEDIUM"
                analysis.recommendations.append("Medium risk: Monitor closely")
            else:
                analysis.risk_level = "LOW"
                analysis.recommendations.append("Low risk: Continue processing")
            
            # Add pattern-based recommendations
            if violation_patterns:
                analysis.recommendations.append(f"Violation patterns detected: {violation_patterns}")
            if ethical_patterns:
                analysis.recommendations.append(f"Positive patterns found: {ethical_patterns}")
            
            analysis.processing_time = time.time() - start_time
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing chunk: {e}")
            return BufferAnalysis(
                ethical_score=0.0,
                violation_count=0,
                risk_level="ERROR",
                intervention_required=True,
                processing_time=time.time() - start_time,
                patterns_detected=[],
                recommendations=[f"Analysis error: {str(e)}"],
                autonomy_scores={},
                vector_analysis={}
            )
    
    async def _flush_buffer(self, keep_context: bool = True):
        """Clear the buffer, optionally keeping some context."""
        if keep_context and self.config.context_window > 0:
            # Keep last portion as context for next evaluation
            context_size = min(self.config.context_window, len(self.token_buffer) // 2)
            context_tokens = list(self.token_buffer)[-context_size:] if context_size > 0 else []
            
            self.token_buffer.clear()
            for token in context_tokens:
                self.token_buffer.append(token)
                
            self.current_text = " ".join(context_tokens) + " "
        else:
            self.token_buffer.clear()
            self.current_text = ""
    
    async def force_flush(self) -> Optional[BufferAnalysis]:
        """Force processing of current buffer contents."""
        if len(self.token_buffer) == 0:
            return None
            
        return await self._process_buffer()
    
    def get_metrics(self) -> BufferMetrics:
        """Get current buffer performance metrics."""
        self.metrics.buffer_utilization = len(self.token_buffer) / self.config.max_tokens
        self.metrics.total_runtime = time.time() - (self.last_evaluation_time or time.time())
        return self.metrics
    
    def update_config(self, new_config: BufferConfig):
        """Update buffer configuration dynamically."""
        with self.buffer_lock:
            old_max_tokens = self.config.max_tokens
            self.config = new_config
            
            # Adjust buffer size if needed
            if new_config.max_tokens != old_max_tokens:
                new_buffer = deque(self.token_buffer, maxlen=new_config.max_tokens * 2)
                self.token_buffer = new_buffer
                
        logger.info(f"Buffer configuration updated: {new_config}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Process any remaining buffer contents
            if len(self.token_buffer) > 0:
                await self.force_flush()
                
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Smart Buffer cleanup completed")
        except Exception as e:
            logger.error(f"Error during buffer cleanup: {e}")

# Factory function for easy buffer creation
def create_smart_buffer(
    max_tokens: int = 512,
    evaluator_callback: Optional[Callable] = None,
    ml_ethics_engine: Optional[Any] = None,
    **kwargs
) -> SmartBuffer:
    """
    Factory function to create a smart buffer with common configurations.
    
    Args:
        max_tokens: Maximum tokens before processing
        evaluator_callback: Function to call for ethical evaluation
        ml_ethics_engine: ML ethics engine instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured SmartBuffer instance
    """
    config = BufferConfig(max_tokens=max_tokens, **kwargs)
    return SmartBuffer(config, evaluator_callback, ml_ethics_engine)