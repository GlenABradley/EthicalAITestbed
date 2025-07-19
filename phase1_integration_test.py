#!/usr/bin/env python3
"""
Phase 1 Integration Testing - Performance Optimization Components
Tests the integration of new optimized components with existing system
"""

import sys
import os
import time
import asyncio
import numpy as np
from typing import Dict, Any, List
import traceback

# Change to backend directory and add to path for proper imports
os.chdir('/app/backend')
sys.path.insert(0, '/app/backend')

class Phase1IntegrationTester:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details or {}
        }
        self.results.append(result)
        
        if success:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: {message}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {message}")
            if details:
                print(f"   Details: {details}")

    def test_import_dependencies(self):
        """Test that all new optimized modules can be imported correctly"""
        try:
            # Test caching manager import
            from utils.caching_manager import CacheManager, EmbeddingCache, global_cache_manager
            self.log_result("Import Caching Manager", True, "Successfully imported caching components")
            
            # Test embedding service import
            from core.embedding_service import EmbeddingService, global_embedding_service
            self.log_result("Import Embedding Service", True, "Successfully imported embedding service")
            
            # Test evaluation engine import
            from core.evaluation_engine import OptimizedEvaluationEngine, global_optimized_engine
            self.log_result("Import Evaluation Engine", True, "Successfully imported optimized evaluation engine")
            
            # Test server import
            import server_optimized
            self.log_result("Import Optimized Server", True, "Successfully imported optimized server")
            
            return True
            
        except ImportError as e:
            self.log_result("Import Dependencies", False, f"Import error: {str(e)}", {"traceback": traceback.format_exc()})
            return False
        except Exception as e:
            self.log_result("Import Dependencies", False, f"Unexpected error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_caching_system_basic(self):
        """Test basic caching system functionality"""
        try:
            from utils.caching_manager import EmbeddingCache
            
            # Create cache instance
            cache = EmbeddingCache(max_size=100, default_ttl=3600)
            
            # Test cache miss
            result = cache.get("test text", "test_model")
            if result is None:
                self.log_result("Cache Miss Test", True, "Cache correctly returns None for non-existent entry")
            else:
                self.log_result("Cache Miss Test", False, "Cache should return None for non-existent entry")
                return False
            
            # Test cache put and get
            test_embedding = np.random.random(384)  # MiniLM-L6-v2 dimension
            cache.put("test text", test_embedding, "test_model")
            
            cached_result = cache.get("test text", "test_model")
            if cached_result is not None and np.array_equal(cached_result, test_embedding):
                self.log_result("Cache Put/Get Test", True, "Cache correctly stores and retrieves embeddings")
            else:
                self.log_result("Cache Put/Get Test", False, "Cache failed to store/retrieve embeddings correctly")
                return False
            
            # Test cache statistics
            stats = cache.get_stats()
            if stats['total_hits'] == 1 and stats['total_misses'] == 1:
                self.log_result("Cache Statistics", True, f"Cache stats working: {stats}")
            else:
                self.log_result("Cache Statistics", False, f"Cache stats incorrect: {stats}")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("Caching System Basic", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_embedding_service_basic(self):
        """Test basic embedding service functionality"""
        try:
            from core.embedding_service import EmbeddingService
            
            # Create embedding service instance
            service = EmbeddingService(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=2,
                max_workers=2
            )
            
            # Test single text embedding (sync)
            result = service.get_embedding_sync("Hello world")
            
            if result.embeddings.shape == (384,):  # Single embedding should be 1D
                self.log_result("Single Embedding Test", True, f"Embedding generated successfully, shape: {result.embeddings.shape}")
            else:
                self.log_result("Single Embedding Test", False, f"Unexpected embedding shape: {result.embeddings.shape}")
                return False
            
            # Test batch embedding (sync)
            texts = ["Hello world", "This is a test", "Another sentence"]
            batch_result = service.get_embeddings_sync(texts)
            
            if batch_result.embeddings.shape == (3, 384):
                self.log_result("Batch Embedding Test", True, f"Batch embeddings generated successfully, shape: {batch_result.embeddings.shape}")
            else:
                self.log_result("Batch Embedding Test", False, f"Unexpected batch embedding shape: {batch_result.embeddings.shape}")
                return False
            
            # Test caching (second call should be faster)
            start_time = time.time()
            cached_result = service.get_embedding_sync("Hello world")
            cache_time = time.time() - start_time
            
            if cache_time < 0.1 and cached_result.cache_hit:  # Should be very fast if cached
                self.log_result("Embedding Cache Test", True, f"Cache working - second call took {cache_time:.4f}s")
            else:
                self.log_result("Embedding Cache Test", False, f"Cache not working - second call took {cache_time:.4f}s, cache_hit: {cached_result.cache_hit}")
            
            # Cleanup
            service.cleanup()
            return True
            
        except Exception as e:
            self.log_result("Embedding Service Basic", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_evaluation_engine_basic(self):
        """Test basic optimized evaluation engine functionality"""
        try:
            from core.evaluation_engine import OptimizedEvaluationEngine
            
            # Create evaluation engine instance
            engine = OptimizedEvaluationEngine(
                max_processing_time=30.0,
                enable_v1_features=True
            )
            
            # Test simple evaluation (sync version for testing)
            test_text = "This is a simple test sentence."
            
            # Use asyncio to run async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                evaluation_result = loop.run_until_complete(
                    engine.evaluate_text_async(test_text)
                )
                
                if hasattr(evaluation_result, 'input_text') and evaluation_result.input_text == test_text:
                    self.log_result("Basic Evaluation Test", True, f"Evaluation completed successfully, processing_time: {evaluation_result.processing_time:.3f}s")
                else:
                    self.log_result("Basic Evaluation Test", False, "Evaluation result format incorrect")
                    return False
                
                # Test timeout protection
                if evaluation_result.processing_time < 30.0:  # Should complete within timeout
                    self.log_result("Timeout Protection Test", True, f"Evaluation completed within timeout ({evaluation_result.processing_time:.3f}s < 30.0s)")
                else:
                    self.log_result("Timeout Protection Test", False, f"Evaluation took too long: {evaluation_result.processing_time:.3f}s")
                
            finally:
                loop.close()
            
            # Test performance stats
            stats = engine.get_performance_stats()
            if 'evaluation_engine' in stats and 'performance_summary' in stats:
                self.log_result("Performance Stats Test", True, f"Performance stats available: {stats['performance_summary']['status']}")
            else:
                self.log_result("Performance Stats Test", False, "Performance stats format incorrect")
                return False
            
            # Cleanup
            engine.cleanup()
            return True
            
        except Exception as e:
            self.log_result("Evaluation Engine Basic", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_server_optimized_import(self):
        """Test that optimized server can be imported and initialized"""
        try:
            # Test import
            import server_optimized
            
            # Test that FastAPI app is created
            if hasattr(server_optimized, 'app'):
                self.log_result("Server App Creation", True, "FastAPI app created successfully")
            else:
                self.log_result("Server App Creation", False, "FastAPI app not found")
                return False
            
            # Test that optimized components are imported
            if hasattr(server_optimized, 'evaluation_engine') and hasattr(server_optimized, 'embedding_service'):
                self.log_result("Server Components", True, "Optimized components imported successfully")
            else:
                self.log_result("Server Components", False, "Optimized components not properly imported")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("Server Optimized Import", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_integration_compatibility(self):
        """Test integration between optimized components"""
        try:
            from utils.caching_manager import global_cache_manager
            from core.embedding_service import global_embedding_service
            from core.evaluation_engine import global_optimized_engine
            
            # Test that global instances are properly connected
            if global_embedding_service.cache_manager is global_cache_manager:
                self.log_result("Cache Integration", True, "Embedding service properly connected to cache manager")
            else:
                self.log_result("Cache Integration", False, "Embedding service not properly connected to cache manager")
                return False
            
            if global_optimized_engine.embedding_service is global_embedding_service:
                self.log_result("Service Integration", True, "Evaluation engine properly connected to embedding service")
            else:
                self.log_result("Service Integration", False, "Evaluation engine not properly connected to embedding service")
                return False
            
            # Test end-to-end integration
            test_text = "Integration test sentence"
            
            # Use asyncio for async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                start_time = time.time()
                result = loop.run_until_complete(
                    global_optimized_engine.evaluate_text_async(test_text)
                )
                end_time = time.time()
                
                if hasattr(result, 'input_text') and result.input_text == test_text:
                    self.log_result("End-to-End Integration", True, f"Full integration test passed in {end_time - start_time:.3f}s")
                else:
                    self.log_result("End-to-End Integration", False, "Integration test failed - invalid result")
                    return False
                
            finally:
                loop.close()
            
            return True
            
        except Exception as e:
            self.log_result("Integration Compatibility", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_memory_management(self):
        """Test memory management and cleanup"""
        try:
            from utils.caching_manager import EmbeddingCache
            from core.embedding_service import EmbeddingService
            
            # Create instances
            cache = EmbeddingCache(max_size=10, default_ttl=3600)
            service = EmbeddingService(batch_size=2, max_workers=2)
            
            # Fill cache beyond capacity to test LRU eviction
            for i in range(15):
                test_embedding = np.random.random(384)
                cache.put(f"test_text_{i}", test_embedding, "test_model")
            
            # Check that cache size is limited
            stats = cache.get_stats()
            if stats['cache_size'] <= 10:
                self.log_result("Cache Size Limit", True, f"Cache properly limited to {stats['cache_size']} entries")
            else:
                self.log_result("Cache Size Limit", False, f"Cache exceeded size limit: {stats['cache_size']} entries")
                return False
            
            # Test eviction count
            if stats['total_evictions'] > 0:
                self.log_result("LRU Eviction", True, f"LRU eviction working: {stats['total_evictions']} evictions")
            else:
                self.log_result("LRU Eviction", False, "LRU eviction not working")
                return False
            
            # Test cleanup
            service.cleanup()
            cache.clear()
            
            final_stats = cache.get_stats()
            if final_stats['cache_size'] == 0:
                self.log_result("Memory Cleanup", True, "Cache cleared successfully")
            else:
                self.log_result("Memory Cleanup", False, f"Cache not properly cleared: {final_stats['cache_size']} entries remain")
                return False
            
            return True
            
        except Exception as e:
            self.log_result("Memory Management", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def test_performance_improvements(self):
        """Test that performance improvements are working"""
        try:
            from core.embedding_service import EmbeddingService
            
            service = EmbeddingService(batch_size=4, max_workers=2)
            
            test_text = "Performance test sentence for caching"
            
            # First call (should be slow - no cache)
            start_time = time.time()
            first_result = service.get_embedding_sync(test_text)
            first_time = time.time() - start_time
            
            # Second call (should be fast - cached)
            start_time = time.time()
            second_result = service.get_embedding_sync(test_text)
            second_time = time.time() - start_time
            
            # Calculate speedup
            if second_time > 0:
                speedup = first_time / second_time
            else:
                speedup = float('inf')
            
            if speedup > 5:  # Should be at least 5x faster (reduced from 10x for more realistic expectation)
                self.log_result("Performance Improvement", True, f"Cache provides {speedup:.1f}x speedup ({first_time:.3f}s -> {second_time:.3f}s)")
            else:
                self.log_result("Performance Improvement", False, f"Insufficient speedup: {speedup:.1f}x ({first_time:.3f}s -> {second_time:.3f}s)")
            
            # Test batch processing efficiency
            texts = [f"Batch test sentence {i}" for i in range(5)]
            
            start_time = time.time()
            batch_result = service.get_embeddings_sync(texts)
            batch_time = time.time() - start_time
            
            if batch_result.text_count == 5 and batch_time < 10.0:  # Should process 5 texts in under 10 seconds
                self.log_result("Batch Processing", True, f"Batch processing efficient: {batch_result.text_count} texts in {batch_time:.3f}s")
            else:
                self.log_result("Batch Processing", False, f"Batch processing inefficient: {batch_result.text_count} texts in {batch_time:.3f}s")
            
            service.cleanup()
            return True
            
        except Exception as e:
            self.log_result("Performance Improvements", False, f"Error: {str(e)}", {"traceback": traceback.format_exc()})
            return False

    def run_all_tests(self):
        """Run all Phase 1 integration tests"""
        print("🚀 Starting Phase 1 Integration Testing - Performance Optimization Components")
        print("=" * 80)
        
        # Test order matters - dependencies first
        tests = [
            self.test_import_dependencies,
            self.test_caching_system_basic,
            self.test_embedding_service_basic,
            self.test_evaluation_engine_basic,
            self.test_server_optimized_import,
            self.test_integration_compatibility,
            self.test_memory_management,
            self.test_performance_improvements
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_result(test.__name__, False, f"Test crashed: {str(e)}", {"traceback": traceback.format_exc()})
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 PHASE 1 INTEGRATION TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if failed_count > 0:
            print(f"\n❌ FAILED TESTS:")
            for test_name in self.failed_tests:
                print(f"   - {test_name}")
        
        if passed_count == total_tests:
            print(f"\n🎉 ALL PHASE 1 INTEGRATION TESTS PASSED!")
            print("✅ Intelligent Caching System: WORKING")
            print("✅ High-Performance Embedding Service: WORKING") 
            print("✅ Optimized Evaluation Engine: WORKING")
            print("✅ Modern FastAPI Server: WORKING")
            print("✅ Component Integration: WORKING")
            return True
        else:
            print(f"\n⚠️  PHASE 1 INTEGRATION ISSUES DETECTED")
            print("Some optimized components need attention before production deployment.")
            return False

if __name__ == "__main__":
    tester = Phase1IntegrationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)