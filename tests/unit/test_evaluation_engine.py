"""
Unit tests for the OptimizedEvaluationEngine class in backend/core/evaluation_engine.py
"""
import pytest
import asyncio
import numpy as np
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from backend.core.evaluation_engine import (
    OptimizedEvaluationEngine,
    EvaluationProgress,
    global_optimized_engine
)
from backend.core.embedding_service import EmbeddingService
from backend.utils.caching_manager import CacheManager
from backend.ethical_engine import EthicalEvaluation, EthicalParameters, EthicalSpan


class TestOptimizedEvaluationEngine:
    """Test cases for the OptimizedEvaluationEngine class."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        with patch('backend.core.embedding_service.EmbeddingService') as mock_cls:
            mock_instance = MagicMock(spec=EmbeddingService)
            mock_instance.get_embedding_async = AsyncMock(return_value=np.random.rand(384).astype(np.float32))
            mock_instance.get_embeddings_async = AsyncMock(return_value=[np.random.rand(384).astype(np.float32)])
            mock_cls.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager that properly implements the CacheManager interface."""
        mock = MagicMock(spec=CacheManager)
        
        # Initialize cache storage
        mock._evaluation_cache = {}
        mock._embedding_cache = {}
        
        # Mock get_embedding to return None (cache miss) by default
        mock.get_embedding.return_value = None
        
        # Mock cache_embedding to store embeddings
        def cache_embedding(text: str, embedding: np.ndarray, model_name: str = "default"):
            key = f"{text}:{model_name}"
            mock._embedding_cache[key] = embedding
            
        def get_embedding(text: str, model_name: str = "default"):
            key = f"{text}:{model_name}"
            return mock._embedding_cache.get(key)
            
        # Mock cache_evaluation to store evaluations
        def cache_evaluation(text: str, parameters: Dict[str, Any], result: Any, ttl: float = 1800):
            key = f"{text}:{str(sorted(parameters.items()) if parameters else '')}"
            mock._evaluation_cache[key] = result
            
        # Mock get_cached_evaluation to check the internal cache
        def get_cached_evaluation(text: str, parameters: Dict[str, Any]):
            key = f"{text}:{str(sorted(parameters.items()) if parameters else '')}"
            return mock._evaluation_cache.get(key)
            
        # By default, don't return anything from the cache unless explicitly set
        mock.get_cached_evaluation.side_effect = get_cached_evaluation
            
        # Mock get_comprehensive_stats to return cache statistics
        def get_comprehensive_stats():
            return {
                'embedding_cache': {
                    'size': len(mock._embedding_cache),
                    'hits': mock.embedding_cache.hits if hasattr(mock, 'embedding_cache') else 0,
                    'misses': mock.embedding_cache.misses if hasattr(mock, 'embedding_cache') else 0,
                    'hit_rate': 0.0,
                    'max_size': 10000
                },
                'evaluation_cache': {
                    'size': len(mock._evaluation_cache),
                    'hits': 0,
                    'misses': 0,
                    'hit_rate': 0.0,
                    'max_size': 1000
                },
                'overall': {
                    'total_queries': 0,
                    'hits': 0,
                    'misses': 0,
                    'hit_rate': 0.0,
                    'memory_usage_estimate_mb': 0.0
                }
            }
            
        # Configure mock methods
        mock.cache_embedding.side_effect = cache_embedding
        mock.get_embedding.side_effect = get_embedding
        mock.cache_evaluation.side_effect = cache_evaluation
        mock.get_cached_evaluation.side_effect = get_cached_evaluation
        mock.get_comprehensive_stats.side_effect = get_comprehensive_stats
        
        # Mock clear_all_caches to clear all caches
        def clear_all_caches():
            mock._evaluation_cache.clear()
            mock._embedding_cache.clear()
            return True
            
        mock.clear_all_caches.side_effect = clear_all_caches
        
        return mock
    
    @pytest.fixture
    def evaluation_engine(self, mock_embedding_service, mock_cache_manager):
        """Create an instance of OptimizedEvaluationEngine with mock dependencies."""
        # Create a real instance with our mocks
        engine = OptimizedEvaluationEngine(
            embedding_service=mock_embedding_service,
            cache_manager=mock_cache_manager,
            max_processing_time=10.0,
            enable_v1_features=True
        )
        
        # Patch the _analyze_core_ethics_async method to return a valid result
        async def mock_analyze_core_ethics(text, embedding, params):
            return {
                'tokens': text.split(),
                'spans': [],
                'minimal_spans': [],
                'overall_ethical': True,
                'virtue_score': 0.1,
                'deontological_score': 0.1,
                'consequentialist_score': 0.1,
                'spans': [{
                    'text': text[:10] + '...' if len(text) > 10 else text,
                    'start': 0,
                    'end': min(10, len(text)),
                    'virtue_score': 0.1,
                    'deontological_score': 0.1,
                    'consequentialist_score': 0.1,
                    'is_violation': False,
                    'explanation': 'No ethical violations detected.'
                }]
            }
            
        # Patch the _apply_v1_features_async method to return empty results
        async def mock_apply_v1_features(text, core_analysis, params):
            return {}
            
        # Apply the patches
        engine._analyze_core_ethics_async = mock_analyze_core_ethics
        engine._apply_v1_features_async = mock_apply_v1_features
        
        return engine
    
    @pytest.mark.asyncio
    async def test_evaluate_text_async_empty_input(self, evaluation_engine):
        """Test evaluation with empty input text."""
        result = await evaluation_engine.evaluate_text_async("")
        assert isinstance(result, EthicalEvaluation)
        assert result.input_text == ""
        assert len(result.tokens) == 0
        assert len(result.spans) == 0
        assert len(result.minimal_spans) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_text_async_basic(self, evaluation_engine, mock_embedding_service):
        """Test basic text evaluation."""
        test_text = "This is a test sentence."
        
        # Mock the embedding service to return a predictable embedding
        test_embedding = MagicMock()
        test_embedding.embeddings = [np.random.rand(384).astype(np.float32)]
        mock_embedding_service.get_embedding_async.return_value = test_embedding
        
        # Create a mock progress callback
        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress)
        
        # Perform the evaluation
        result = await evaluation_engine.evaluate_text_async(
            test_text,
            progress_callback=progress_callback
        )
        
        # Verify the result
        assert isinstance(result, EthicalEvaluation)
        assert result.input_text == test_text
        assert result.overall_ethical is not None
        
        # Verify embedding service was called
        mock_embedding_service.get_embedding_async.assert_called_once()
        
        # Verify evaluation was cached
        evaluation_engine.cache_manager.cache_evaluation.assert_called_once()
        
        # Verify the cached evaluation has the expected structure
        _, cache_args, _ = evaluation_engine.cache_manager.cache_evaluation.mock_calls[0]
        cached_eval = cache_args[2]  # Third argument is the evaluation object
        assert cached_eval.input_text == test_text
        assert cached_eval.overall_ethical is not None
    
    @pytest.mark.asyncio
    async def test_evaluation_progress(self):
        """Test the EvaluationProgress class functionality."""
        progress = EvaluationProgress(total_steps=5)
        assert progress.progress_percent == 0.0
        
        progress.update("Testing", 1)
        assert progress.progress_percent == 20.0
        assert progress.completed_steps == 1
        assert progress.current_step == "Testing"
        assert progress.estimated_time_remaining > 0
        
        # Test converting to dict
        progress_dict = progress.to_dict()
        assert isinstance(progress_dict, dict)
        assert progress_dict["current_step"] == "Testing"
        assert progress_dict["progress_percent"] == 20.0
    
    @pytest.mark.asyncio
    async def test_evaluate_text_async_with_timeout(self, evaluation_engine, mock_embedding_service):
        """Test that evaluation respects the timeout."""
        # Make the embedding service take longer than the timeout
        async def slow_embedding(*args, **kwargs):
            await asyncio.sleep(2)  # Sleep for 2 seconds
            mock = MagicMock()
            mock.embeddings = [np.random.rand(384).astype(np.float32)]
            return mock
            
        mock_embedding_service.get_embedding_async.side_effect = slow_embedding
        
        # Set a very short timeout
        evaluation_engine.max_processing_time = 0.1
        
        # This should complete due to timeout, not raise an exception
        result = await evaluation_engine.evaluate_text_async("This is a test")
        
        # Verify the result is a timeout evaluation
        assert isinstance(result, EthicalEvaluation)
        # Check for timeout indicator in either input_text or evaluation_id
        assert ("timeout" in result.input_text.lower() or 
                "timed out" in result.input_text.lower() or
                "timeout" in result.evaluation_id.lower())
    
    @pytest.mark.asyncio
    async def test_evaluate_text_async_with_cache_hit(self, evaluation_engine, mock_embedding_service, mock_cache_manager):
        """Test that cached results are returned when available."""
        test_text = "This should be cached"
        
        # Create a real evaluation object for the cache
        cached_eval = EthicalEvaluation(
            input_text=test_text,
            tokens=test_text.split(),
            spans=[],
            minimal_spans=[],
            overall_ethical=True,
            processing_time=0.1,
            parameters=EthicalParameters()
        )
        
        # Set up the cache hit by directly adding to the mock's internal cache
        # Note: The engine uses an empty string for parameters when none are provided
        cache_key = f"{test_text}:"  # Matches engine's format for no parameters
        mock_cache_manager._evaluation_cache = {cache_key: cached_eval}
        
        # Add debug prints
        print("\n=== DEBUG: Cache State Before Test ===")
        print(f"Cache keys: {list(mock_cache_manager._evaluation_cache.keys())}")
        print(f"Cache content: {mock_cache_manager._evaluation_cache}")
        
        # Mock the embedding service to verify it's not called
        mock_embedding = MagicMock()
        mock_embedding.embeddings = [np.random.rand(384).astype(np.float32)]
        mock_embedding_service.get_embedding_async.return_value = mock_embedding
        
        # Reset the mock call count to ensure we're only counting calls from this test
        mock_embedding_service.get_embedding_async.reset_mock()
        mock_cache_manager.get_cached_evaluation.reset_mock()
        
        # Print the mock state before the test
        print("\n=== DEBUG: Mock State Before Test ===")
        print(f"get_embedding_async call count: {mock_embedding_service.get_embedding_async.call_count}")
        print(f"get_cached_evaluation call count: {mock_cache_manager.get_cached_evaluation.call_count}")
        
        # Perform the evaluation
        print("\n=== DEBUG: Starting Evaluation ===")
        result = await evaluation_engine.evaluate_text_async(test_text)
        print("=== DEBUG: Evaluation Complete ===")
        
        # Print the mock state after the test
        print("\n=== DEBUG: Mock State After Test ===")
        print(f"get_embedding_async call count: {mock_embedding_service.get_embedding_async.call_count}")
        print(f"get_cached_evaluation call count: {mock_cache_manager.get_cached_evaluation.call_count}")
        
        # Print the cache state after the test
        print("\n=== DEBUG: Cache State After Test ===")
        print(f"Cache keys: {list(mock_cache_manager._evaluation_cache.keys())}")
        print(f"Cache content: {mock_cache_manager._evaluation_cache}")
        
        # Verify the result came from cache by checking key properties
        assert result.input_text == cached_eval.input_text, "Result input text does not match cached evaluation"
        assert result.overall_ethical == cached_eval.overall_ethical, "Result ethical flag does not match cached evaluation"
        assert isinstance(result.parameters, type(cached_eval.parameters)), "Result parameters type does not match cached evaluation"
        
        # Verify the cache was checked with the correct parameters
        mock_cache_manager.get_cached_evaluation.assert_called_once()
        
        # Verify embedding service was not called (proves cache was used)
        mock_embedding_service.get_embedding_async.assert_not_called()
    
    def test_initialization_defaults(self):
        """Test that the engine initializes with default values."""
        engine = OptimizedEvaluationEngine()
        assert engine.max_processing_time == 30.0  # Default value
        assert engine.enable_v1_features is True
        assert engine.embedding_service is not None
        assert engine.cache_manager is not None
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        custom_embedding = MagicMock(spec=EmbeddingService)
        custom_cache = MagicMock(spec=CacheManager)
        
        engine = OptimizedEvaluationEngine(
            embedding_service=custom_embedding,
            cache_manager=custom_cache,
            max_processing_time=10.0,
            enable_v1_features=False
        )
        
        assert engine.embedding_service == custom_embedding
        assert engine.cache_manager == custom_cache
        assert engine.max_processing_time == 10.0
        assert engine.enable_v1_features is False


class TestGlobalEvaluationEngine:
    """Tests for the global evaluation engine instance."""
    
    @pytest.mark.asyncio
    async def test_global_engine_available(self):
        """Test that the global engine is available and functional."""
        assert isinstance(global_optimized_engine, OptimizedEvaluationEngine)
        
        # Test a basic operation
        result = await global_optimized_engine.evaluate_text_async("Test")
        assert isinstance(result, EthicalEvaluation)
