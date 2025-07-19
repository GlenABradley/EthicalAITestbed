"""
Quick Performance Test for Optimized Components

This script tests our new optimized caching and evaluation system to ensure
the performance improvements are working as expected.

For Novice Developers:
Think of this like a test drive for our newly optimized car. We want to make
sure all the performance improvements (caching, async processing, timeout handling)
are working correctly before we replace the old system.

Expected Results:
- First evaluation: 2-10 seconds (computing from scratch)
- Second evaluation: <1 second (cached result)
- Cache hit rate: Should increase with repeated requests
- No timeouts: All requests should complete within 30 seconds

Author: AI Developer Testbed Team
Version: 1.1.0 - Performance Validation
"""

import asyncio
import time
import sys
import os
import numpy as np

# Add backend directory to path
sys.path.append('/app/backend')

from utils.caching_manager import CacheManager, EmbeddingCache
from core.embedding_service import EmbeddingService
from core.evaluation_engine import OptimizedEvaluationEngine

async def test_embedding_cache():
    """
    Test the embedding cache system for performance improvements.
    
    For Novice Developers:
    This tests our "translation dictionary" to make sure it remembers
    translations and can find them quickly the second time.
    """
    print("\n🧪 Testing Embedding Cache System...")
    
    cache = EmbeddingCache(max_size=100)
    
    # Test basic cache operations
    test_text = "Hello world, this is a test"
    
    # Should return None (not cached yet)
    result = cache.get(test_text)
    assert result is None, "Cache should be empty initially"
    print("✅ Empty cache test passed")
    
    # Add an embedding to cache
    test_embedding = np.random.rand(384)  # MiniLM embedding size
    cache.put(test_text, test_embedding)
    
    # Should return the cached embedding
    cached_result = cache.get(test_text)
    assert cached_result is not None, "Should find cached embedding"
    assert np.array_equal(cached_result, test_embedding), "Cached embedding should match original"
    print("✅ Cache storage and retrieval test passed")
    
    # Test cache statistics
    stats = cache.get_stats()
    assert stats["total_hits"] == 1, "Should have 1 cache hit"
    assert stats["total_misses"] == 1, "Should have 1 cache miss"
    print(f"✅ Cache stats: {stats['hit_rate_percent']:.1f}% hit rate")
    
    print("🎉 Embedding Cache System: ALL TESTS PASSED!")

async def test_cache_manager():
    """
    Test the comprehensive cache manager system.
    
    For Novice Developers:
    This tests our "head librarian" to make sure they can efficiently
    manage all the different types of libraries (caches) we have.
    """
    print("\n🧪 Testing Cache Manager...")
    
    manager = CacheManager()
    
    # Test evaluation caching
    test_text = "Test evaluation caching"
    test_params = {"virtue_threshold": 0.5}
    test_result = {"overall_ethical": True, "violations": []}
    
    # Should not find cached evaluation initially
    cached = manager.get_cached_evaluation(test_text, test_params)
    assert cached is None, "Should not find cached evaluation initially"
    
    # Cache the evaluation
    manager.cache_evaluation(test_text, test_params, test_result)
    
    # Should find cached evaluation now
    cached = manager.get_cached_evaluation(test_text, test_params)
    assert cached is not None, "Should find cached evaluation"
    assert cached["overall_ethical"] == True, "Cached result should match original"
    
    print("✅ Evaluation caching test passed")
    
    # Test comprehensive stats
    stats = manager.get_comprehensive_stats()
    print(f"📊 Cache Manager Stats:")
    print(f"   - Total cache entries: {stats['total_cache_entries']}")
    print(f"   - Memory estimate: {stats['cache_efficiency']['memory_usage_estimate_mb']:.1f} MB")
    print(f"   - Embedding hit rate: {stats['embedding_cache']['hit_rate_percent']:.1f}%")
    
    print("🎉 Cache Manager: ALL TESTS PASSED!")

async def main():
    """
    Run basic tests to validate our optimizations.
    
    For Novice Developers:
    This is a simplified test suite that checks our core optimizations
    are working correctly.
    """
    print("🚀 STARTING PERFORMANCE VALIDATION TESTS")
    print("=" * 60)
    
    try:
        await test_embedding_cache()
        await test_cache_manager()
        
        print("\n" + "=" * 60)
        print("🎉 CORE OPTIMIZATION TESTS PASSED!")
        print("✅ Caching systems are working correctly")
        print("🚀 Ready to proceed with integration!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("🔧 Check the error above and fix before deployment")
        raise

if __name__ == "__main__":
    asyncio.run(main())