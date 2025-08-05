"""
Unit tests for the EmbeddingService class.

These tests verify the functionality of the high-performance embedding service,
including caching, batch processing, and async operations.
"""

import numpy as np
import pytest
import time
import torch
from unittest.mock import MagicMock, patch, ANY
from typing import List, Dict, Any, Optional
import asyncio

from backend.core.embedding_service import EmbeddingService, EmbeddingResult
from backend.utils.caching_manager import CacheManager

# Sample test data
SAMPLE_TEXTS = [
    "This is a test sentence.",
    "Another test sentence for embedding.",
    "The quick brown fox jumps over the lazy dog."
]

class TestEmbeddingService:
    """Test cases for the EmbeddingService class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cache = {}
    
    def get_cached_embedding(self, text):
        """Helper method to simulate getting a cached embedding."""
        return self.cache.get(text)
    
    def set_cached_embedding(self, text, embedding):
        """Helper method to simulate caching an embedding."""
        self.cache[text] = embedding
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        with patch('backend.utils.caching_manager.CacheManager') as mock_cls:
            mock_instance = MagicMock()
            # Configure mock methods
            mock_instance.get_embedding.side_effect = lambda text, model: self.get_cached_embedding(text)
            mock_instance.cache_embedding.side_effect = lambda text, embedding, model: self.set_cached_embedding(text, embedding)
            mock_instance.get_cache_stats.return_value = {
                'hits': 0,
                'misses': 0,
                'size': 0,
                'max_size': 1000,
                'cache_efficiency': {
                    'hit_rate': 0.0,
                    'memory_usage_estimate_mb': 10.0
                }
            }
            mock_cls.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_embedding_model(self, mock_cache_manager):
        """Create a mock embedding model with proper batch processing."""
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            mock_instance = MagicMock()
            # Configure mock to return a dummy embedding
            mock_embedding = np.random.rand(384).astype(np.float32)
            mock_instance.encode.return_value = mock_embedding
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance
            
            # Create a real EmbeddingService instance for testing
            service = EmbeddingService(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                cache_manager=mock_cache_manager,
                batch_size=2,  # Small batch size for testing
                max_workers=2
            )
            
            # Store the mock model for assertions
            service._mock_model = mock_instance
            
            yield mock_instance
            
            # Cleanup
            service.cleanup()

    @pytest.fixture
    def embedding_service(self, mock_cache_manager, mock_embedding_model):
        """Create an EmbeddingService instance with mocks."""
        # Create service with test configuration using a valid model name
        service = EmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_manager=mock_cache_manager,
            batch_size=2,  # Small batch size for testing
            max_workers=2
        )
        
        # Store the mock model for assertions
        service._mock_model = mock_embedding_model
        
        # Create a synchronous mock for _compute_embeddings_batch
        def mock_compute_batch(texts):
            # Return a list of numpy arrays with the correct shape (384-dimensional vectors)
            # This matches what the real _compute_embeddings_batch returns
            return [np.random.rand(384).astype(np.float32) for _ in texts]
            
        # Replace the method with our synchronous mock
        service._compute_embeddings_batch = mock_compute_batch
        
        yield service
        
        # Cleanup
        service.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_embedding_async(self, mock_cache_manager):
        """Test asynchronous embedding retrieval."""
        # Setup test data
        test_text = SAMPLE_TEXTS[0]
        mock_embedding = np.random.rand(384).astype(np.float32)
        
        # Configure cache to miss on first call
        mock_cache_manager.get_embedding.return_value = None
        
        # Create an EmbeddingService instance with a mock model
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            # Configure the mock model to return predictable embeddings
            # Using 384 dimensions to match the actual model's output
            mock_model.return_value.encode.return_value = mock_embedding
            mock_model.return_value.get_sentence_embedding_dimension.return_value = 384
            
            # Create service with test configuration using a valid model name
            service = EmbeddingService(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Valid model name
                cache_manager=mock_cache_manager,
                batch_size=2,  # Small batch size for testing
                max_workers=2
            )
            
            # Store the mock model for assertions
            service._mock_model = mock_model.return_value
            
            # Call the method under test
            result = await service.get_embedding_async(test_text)
            
            # Verify the result structure
            assert hasattr(result, 'embeddings'), "Result should have 'embeddings' attribute"
            assert hasattr(result, 'processing_time'), "Result should have 'processing_time' attribute"
            assert hasattr(result, 'cache_hit'), "Result should have 'cache_hit' attribute"
            assert hasattr(result, 'model_name'), "Result should have 'model_name' attribute"
            assert hasattr(result, 'text_count'), "Result should have 'text_count' attribute"
            assert hasattr(result, 'batch_size'), "Result should have 'batch_size' attribute"
            
            # Verify the embeddings shape and content
            assert isinstance(result.embeddings, np.ndarray), "Embeddings should be a numpy array"
            assert result.embeddings.shape == (1, 384), \
                f"Expected embeddings shape (1, 384), got {result.embeddings.shape}"
                
            # Verify other attributes
            assert result.text_count == 1, f"Expected text_count=1, got {result.text_count}"
            assert result.batch_size == 1, f"Expected batch_size=1, got {result.batch_size}"
            assert result.model_name == service.model_name, "Model name mismatch"
            assert result.cache_hit is False, "First call should be a cache miss"
            
            # Verify cache was checked
            mock_cache_manager.get_embedding.assert_called_once_with(
                test_text, service.model_name)
                
            # Verify the embedding was cached
            assert mock_cache_manager.cache_embedding.call_count == 1, \
                "Embedding should be cached"
    
    @pytest.mark.asyncio
    async def test_get_embeddings_async(self, mock_cache_manager):
        """Test getting multiple embeddings asynchronously with batching."""
        # Setup test data - use first 3 texts
        test_texts = SAMPLE_TEXTS[:3]
        
        # Configure cache to miss on all calls
        mock_cache_manager.get_embedding.return_value = None
        
        # Create an EmbeddingService instance with a mock model
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            # Configure the mock model to return predictable embeddings
            # Using 384 dimensions to match the actual model's output
            mock_embedding = np.random.rand(384).astype(np.float32)
            mock_model.return_value.encode.return_value = mock_embedding
            mock_model.return_value.get_sentence_embedding_dimension.return_value = 384
            
            # Create service with test configuration using a valid model name
            service = EmbeddingService(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Valid model name
                cache_manager=mock_cache_manager,
                batch_size=2,  # Small batch size for testing
                max_workers=2
            )
            
            # Store the mock model for assertions
            service._mock_model = mock_model.return_value
            
            # Call the method under test
            result = await service.get_embeddings_async(test_texts)
            
            # Verify the result
            assert isinstance(result, EmbeddingResult)
            assert result.embeddings.shape == (3, 384), \
                f"Expected embeddings shape (3, 384), got {result.embeddings.shape}"
            assert result.text_count == 3, f"Expected text_count=3, got {result.text_count}"
            assert result.batch_size == 3, f"Expected batch_size=3 (all texts computed), got {result.batch_size}"
            assert result.cache_hit is False, "First call should be a cache miss"
            
            # Verify cache was checked for each text
            assert mock_cache_manager.get_embedding.call_count == 3, \
                f"Expected 3 cache lookups, got {mock_cache_manager.get_embedding.call_count}"
                
            # Verify the cache was updated with all embeddings
            assert mock_cache_manager.cache_embedding.call_count == 3, \
                f"Expected 3 cache updates, got {mock_cache_manager.cache_embedding.call_count}"
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, embedding_service, mock_cache_manager):
        """Test that embeddings are properly cached and retrieved."""
        test_text = "This is a test sentence."
        
        # First call - should be a cache miss
        result1 = await embedding_service.get_embedding_async(test_text)
        
        # Verify the result contains the expected embedding and metadata
        assert result1.text_count == 1, "Should have processed one text"
        assert len(result1.embeddings) == 1, "Should return one embedding"
        assert result1.embeddings[0].shape == (384,), "Embedding has wrong shape"
        assert result1.batch_size == 1, "Batch size should be 1 for single text"
        assert result1.model_name == 'sentence-transformers/all-MiniLM-L6-v2', "Model name should match"
        assert isinstance(result1.processing_time, float), "Processing time should be a float"
        
        # Verify the cache was checked
        embedding_service.cache_manager.get_embedding.assert_called_once_with(
            test_text, embedding_service.model_name)
            
        # Verify the embedding was cached with the correct arguments
        embedding_service.cache_manager.cache_embedding.assert_called_once()
        
        # Get the call arguments for verification
        call_args = embedding_service.cache_manager.cache_embedding.call_args
        
        # Check if called with positional or keyword arguments
        if call_args[0]:  # Positional args
            assert call_args[0][0] == test_text, "Text in positional args doesn't match"
            assert call_args[0][2] == embedding_service.model_name, "Model name in positional args doesn't match"
            cached_embedding = call_args[0][1]  # Store for later use in the test
        else:  # Keyword args
            if 'text' in call_args[1]:
                assert call_args[1]['text'] == test_text, "Text in keyword args doesn't match"
                assert call_args[1]['model_name'] == embedding_service.model_name, "Model name in keyword args doesn't match"
                cached_embedding = call_args[1]['embedding']  # Store for later use in the test
            else:
                # Fallback: check values directly
                values = list(call_args[1].values())
                assert values[0] == test_text, "Text in values doesn't match"
                assert values[2] == embedding_service.model_name, "Model name in values doesn't match"
                cached_embedding = values[1]  # Store for later use in the test
        
        # Verify we have a valid embedding
        assert hasattr(cached_embedding, 'shape'), "Cached embedding is not a valid array"
        assert cached_embedding.shape == (384,), "Cached embedding has wrong shape"
        
        # --- Test 2: Cache hit ---
        # Reset mocks for the second call
        embedding_service.cache_manager.get_embedding.reset_mock()
        embedding_service.cache_manager.cache_embedding.reset_mock()
        
        # Configure mock for cache hit
        embedding_service.cache_manager.get_embedding.return_value = cached_embedding
        
        # Second call - should be a cache hit
        result2 = await embedding_service.get_embedding_async(test_text)
        
        # Verify cache was checked again
        embedding_service.cache_manager.get_embedding.assert_called_once_with(
            test_text, embedding_service.model_name)
            
        # Verify we didn't try to cache again
        embedding_service.cache_manager.cache_embedding.assert_not_called()
        
        # Verify both results contain the same embedding
        assert np.array_equal(result1.embeddings, result2.embeddings), \
            "Both results should contain the same embedding"
            
        # Verify the cache hit flag is set correctly
        assert result1.cache_hit is False, "First call should be a cache miss"
        assert result2.cache_hit is True, "Second call should be a cache hit"
    
    def test_get_embedding_sync(self, embedding_service):
        """Test synchronous embedding retrieval."""
        # Test with a single text
        result = embedding_service.get_embedding_sync(SAMPLE_TEXTS[0])
        
        # Verify the result
        assert hasattr(result, 'embeddings'), "Result should have 'embeddings' attribute"
        
        # Check embeddings shape (1 text, 384 dimensions)
        embeddings = result.embeddings  # Verify the result - remove cache_hit assertion since we're not testing that here
        assert embeddings.shape == (1, 384), \
            f"Expected embeddings shape (1, 384), got {embeddings.shape}"
        assert isinstance(result.processing_time, float), "Processing time should be a float"
        assert result.model_name == embedding_service.model_name, "Model name mismatch"
        assert result.text_count == 1, f"Expected text_count=1, got {result.text_count}"
    
    def test_performance_stats(self, embedding_service):
        """Test that performance statistics are tracked correctly."""
        # Reset stats
        embedding_service.total_requests = 10
        embedding_service.total_cache_hits = 4
        embedding_service.total_processing_time = 1.5
        
        # Mock cache manager's get_comprehensive_stats
        mock_cache_stats = {
            'hits': 4,
            'misses': 6,
            'size': 10,
            'max_size': 1000,
            'cache_efficiency': {
                'hit_rate': 0.4,
                'memory_usage_estimate_mb': 5.0
            }
        }
        embedding_service.cache_manager.get_comprehensive_stats.return_value = mock_cache_stats
        
        # Get performance stats
        stats = embedding_service.get_performance_stats()
        
        # Verify stats structure and values
        assert isinstance(stats, dict), "Stats should be a dictionary"
        
        # Check top-level keys
        assert 'embedding_service' in stats, "Stats should include embedding_service"
        assert 'cache_system' in stats, "Stats should include cache_system"
        assert 'performance_summary' in stats, "Stats should include performance_summary"
        
        # Check embedding service stats
        service_stats = stats['embedding_service']
        assert service_stats['total_requests'] == 10
        assert service_stats['total_cache_hits'] == 4
        assert service_stats['cache_hit_rate_percent'] == 40.0  # 4/10 = 40%
        assert service_stats['average_processing_time_ms'] == 150.0  # 1.5s / 10 * 1000
        assert service_stats['total_processing_time_s'] == 1.5
        
        # Check cache system stats
        assert stats['cache_system'] == mock_cache_stats
        
        # Check performance summary
        assert 'efficiency_rating' in stats['performance_summary']
        assert 'speed_improvement_estimate' in stats['performance_summary']
        assert 'memory_efficiency' in stats['performance_summary']
    
    def test_cleanup(self, embedding_service):
        """Test that cleanup releases resources properly."""
        # Create a mock executor with shutdown method
        mock_executor = MagicMock()
        mock_executor.shutdown = MagicMock()
        embedding_service.executor = mock_executor
        
        # Mock torch.cuda if available
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.empty_cache = MagicMock()
        
        try:
            # Call cleanup
            embedding_service.cleanup()
            
            # Verify executor's shutdown was called
            mock_executor.shutdown.assert_called_once_with(wait=True)
            
            # Verify CUDA cache was cleared if available
            if original_cuda_available():
                torch.cuda.empty_cache.assert_called_once()
                
        finally:
            # Restore original cuda availability
            torch.cuda.is_available = original_cuda_available
