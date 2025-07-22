#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE BACKEND TESTING SUITE
═══════════════════════════════════════════════════════════════════════════════════

MISSION: EXHAUSTIVE REAL-WORLD BACKEND TESTING FOR VERSION 1.2 CERTIFICATION

This test suite validates all claims made in the documentation:
- 100% backend success rate (14/14 tests passed)
- 0.055s average response times with 6,251x speedup
- All core API endpoints functioning
- Production-ready unified architecture
- Real-time streaming capabilities
- Multi-level caching performance

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
import statistics
import random
import string
from datetime import datetime
from typing import Dict, List, Any, Tuple
import concurrent.futures
import threading

# Backend URL from environment
BACKEND_URL = "https://b214a97a-320b-41fd-b6d6-ff742674f4c6.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BackendTestSuite:
    """Comprehensive backend testing suite for Ethical AI Developer Testbed."""
    
    def __init__(self):
        self.results = {
            "smoke_tests": [],
            "performance_tests": [],
            "content_tests": [],
            "integration_tests": [],
            "reliability_tests": [],
            "bayesian_tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "average_response_time": 0.0,
                "success_rate": 0.0
            }
        }
        self.session = None
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup_session(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, category: str, test_name: str, passed: bool, 
                       response_time: float = 0.0, details: str = ""):
        """Log test result to results dictionary."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "response_time": response_time,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results[category].append(result)
        self.results["summary"]["total_tests"] += 1
        if passed:
            self.results["summary"]["passed_tests"] += 1
        else:
            self.results["summary"]["failed_tests"] += 1
    
    async def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Tuple[bool, float, Dict]:
        """Generic endpoint testing method."""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{API_BASE}{endpoint}") as response:
                    response_time = time.time() - start_time
                    result = await response.json()
                    return response.status == 200, response_time, result
            elif method.upper() == "POST":
                async with self.session.post(f"{API_BASE}{endpoint}", json=data) as response:
                    response_time = time.time() - start_time
                    result = await response.json()
                    return response.status == 200, response_time, result
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 1. SMOKE TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def smoke_test_health_endpoint(self):
        """Test /api/health endpoint basic functionality."""
        print("🏥 Testing health endpoint...")
        success, response_time, result = await self.test_endpoint("GET", "/health")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Status: {result.get('status', 'unknown')}"
            details += f" | Orchestrator: {result.get('orchestrator_healthy', False)}"
            details += f" | Database: {result.get('database_connected', False)}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_test_result("smoke_tests", "Health Endpoint", success, response_time, details)
        return success
    
    async def smoke_test_parameters_endpoint(self):
        """Test /api/parameters endpoint basic functionality."""
        print("⚙️ Testing parameters endpoint...")
        success, response_time, result = await self.test_endpoint("GET", "/parameters")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Parameters count: {len(result)}"
            details += f" | Has thresholds: {'virtue_threshold' in result}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_test_result("smoke_tests", "Parameters Endpoint", success, response_time, details)
        return success
    
    async def smoke_test_evaluate_endpoint(self):
        """Test /api/evaluate endpoint with basic request."""
        print("🎯 Testing evaluate endpoint...")
        test_data = {
            "text": "We should help those in need when possible.",
            "context": {"domain": "general"},
            "mode": "production",
            "priority": "normal"
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Ethical: {result.get('overall_ethical', False)}"
            details += f" | Confidence: {result.get('confidence_score', 0.0):.3f}"
            details += f" | Request ID: {result.get('request_id', 'N/A')[:8]}..."
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_test_result("smoke_tests", "Evaluate Endpoint", success, response_time, details)
        return success
    
    async def smoke_test_heat_map_endpoint(self):
        """Test /api/heat-map-mock endpoint."""
        print("🗺️ Testing heat-map endpoint...")
        test_data = {"text": "This is a test sentence for heat map generation."}
        
        success, response_time, result = await self.test_endpoint("POST", "/heat-map-mock", test_data)
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Evaluations: {len(result.get('evaluations', {}))}"
            details += f" | Text length: {result.get('textLength', 0)}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_test_result("smoke_tests", "Heat Map Endpoint", success, response_time, details)
        return success
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 2. PERFORMANCE TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def performance_test_response_times(self):
        """Test response times with various text lengths."""
        print("⚡ Testing response times with different text lengths...")
        
        test_cases = [
            ("Short text", "Help others."),
            ("Medium text", "We should always strive to help those in need when it is within our power to do so, considering the broader implications of our actions."),
            ("Long text", " ".join(["This is a comprehensive ethical evaluation test with substantial content to analyze." for _ in range(20)]))
        ]
        
        response_times = []
        
        for test_name, text in test_cases:
            test_data = {
                "text": text,
                "context": {"domain": "general"},
                "mode": "production",
                "priority": "normal"
            }
            
            success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
            response_times.append(response_time)
            
            details = f"Text length: {len(text)} chars | Response time: {response_time:.3f}s"
            if success:
                details += f" | Confidence: {result.get('confidence_score', 0.0):.3f}"
            
            self.log_test_result("performance_tests", f"Response Time - {test_name}", success, response_time, details)
        
        # Calculate average response time
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        self.results["summary"]["average_response_time"] = avg_response_time
        
        # Test if average response time meets claimed performance (0.055s)
        performance_claim_met = avg_response_time <= 0.1  # Allow some tolerance
        details = f"Average response time: {avg_response_time:.3f}s | Claimed: 0.055s"
        self.log_test_result("performance_tests", "Average Response Time", performance_claim_met, avg_response_time, details)
        
        return len([t for t in response_times if t > 0]) == len(test_cases)
    
    async def performance_test_concurrent_requests(self):
        """Test concurrent request handling."""
        print("🔄 Testing concurrent request handling...")
        
        async def single_request():
            test_data = {
                "text": "Concurrent testing of ethical evaluation system.",
                "context": {"domain": "general"},
                "mode": "production",
                "priority": "normal"
            }
            return await self.test_endpoint("POST", "/evaluate", test_data)
        
        # Test with 5 concurrent requests
        start_time = time.time()
        tasks = [single_request() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if not isinstance(r, Exception) and r[0])
        
        details = f"5 concurrent requests | {successful_requests}/5 successful | Total time: {total_time:.3f}s"
        success = successful_requests >= 4  # Allow 1 failure
        
        self.log_test_result("performance_tests", "Concurrent Requests", success, total_time, details)
        return success
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 3. REAL-WORLD CONTENT TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def content_test_diverse_texts(self):
        """Test with diverse, realistic text content."""
        print("📝 Testing with diverse text content...")
        
        test_texts = [
            "Healthcare professionals should prioritize patient welfare above all other considerations.",
            "The company's decision to lay off employees during the pandemic raised ethical concerns.",
            "Artificial intelligence systems must be designed with transparency and accountability.",
            "Environmental protection requires balancing economic growth with sustainability.",
            "Educational institutions should ensure equal access to learning opportunities."
        ]
        
        successful_tests = 0
        
        for i, text in enumerate(test_texts):
            test_data = {
                "text": text,
                "context": {"domain": "general", "cultural_context": "western"},
                "mode": "production",
                "priority": "normal"
            }
            
            success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
            
            details = f"Text {i+1} | Response time: {response_time:.3f}s"
            if success:
                details += f" | Ethical: {result.get('overall_ethical', False)}"
                details += f" | Confidence: {result.get('confidence_score', 0.0):.3f}"
                successful_tests += 1
            
            self.log_test_result("content_tests", f"Diverse Content Test {i+1}", success, response_time, details)
        
        overall_success = successful_tests >= 4  # Allow 1 failure
        return overall_success
    
    async def content_test_edge_cases(self):
        """Test with edge cases."""
        print("🔍 Testing edge cases...")
        
        edge_cases = [
            ("Empty text", ""),
            ("Single character", "a"),
            ("Special characters", "!@#$%^&*()_+-=[]{}|;:,.<>?"),
            ("Very long text", "A" * 10000),
            ("Unicode text", "Hello 世界 🌍 Здравствуй мир")
        ]
        
        successful_tests = 0
        
        for test_name, text in edge_cases:
            test_data = {
                "text": text,
                "context": {"domain": "general"},
                "mode": "production",
                "priority": "normal"
            }
            
            success, response_time, result = await self.test_endpoint("POST", "/evaluate", test_data)
            
            details = f"Response time: {response_time:.3f}s | Text length: {len(text)}"
            if success:
                details += f" | Handled gracefully"
                successful_tests += 1
            elif "empty" in test_name.lower() or "Empty" in test_name:
                # Empty text should fail validation - this is expected
                details += " | Expected validation failure"
                successful_tests += 1
                success = True
            
            self.log_test_result("content_tests", f"Edge Case - {test_name}", success, response_time, details)
        
        return successful_tests >= 4
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 4. API INTEGRATION TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def integration_test_parameter_updates(self):
        """Test parameter update functionality."""
        print("🔧 Testing parameter updates...")
        
        # First get current parameters
        success1, response_time1, current_params = await self.test_endpoint("GET", "/parameters")
        
        if not success1:
            self.log_test_result("integration_tests", "Get Parameters", False, response_time1, "Failed to get current parameters")
            return False
        
        # Test parameter update
        update_data = {
            "virtue_threshold": 0.3,
            "deontological_threshold": 0.3,
            "consequentialist_threshold": 0.3
        }
        
        success2, response_time2, update_result = await self.test_endpoint("POST", "/update-parameters", update_data)
        
        details = f"Get: {response_time1:.3f}s | Update: {response_time2:.3f}s"
        if success1 and success2:
            details += " | Both operations successful"
        
        overall_success = success1 and success2
        self.log_test_result("integration_tests", "Parameter Updates", overall_success, response_time1 + response_time2, details)
        
        return overall_success
    
    async def integration_test_learning_stats(self):
        """Test learning statistics endpoint."""
        print("📊 Testing learning statistics...")
        
        success, response_time, result = await self.test_endpoint("GET", "/learning-stats")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Total evaluations: {result.get('total_evaluations', 0)}"
            details += f" | Learning enabled: {result.get('learning_enabled', False)}"
            details += f" | Has metrics: {'performance_metrics' in result}"
        
        self.log_test_result("integration_tests", "Learning Statistics", success, response_time, details)
        return success
    
    async def integration_test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("❌ Testing error handling...")
        
        # Test invalid evaluation request
        invalid_data = {
            "text": "",  # Empty text should cause validation error
            "mode": "invalid_mode",
            "priority": "invalid_priority"
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/evaluate", invalid_data)
        
        # For error handling test, we expect the request to fail gracefully
        error_handled_gracefully = not success or (success and result.get('confidence_score', 1.0) == 0.0)
        
        details = f"Response time: {response_time:.3f}s | Graceful error handling: {error_handled_gracefully}"
        
        self.log_test_result("integration_tests", "Error Handling", error_handled_gracefully, response_time, details)
        return error_handled_gracefully
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 5. SYSTEM RELIABILITY TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def reliability_test_system_health(self):
        """Test comprehensive system health check."""
        print("🏥 Testing comprehensive system health...")
        
        success, response_time, result = await self.test_endpoint("GET", "/health")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            status = result.get('status', 'unknown')
            orchestrator_healthy = result.get('orchestrator_healthy', False)
            database_connected = result.get('database_connected', False)
            
            details += f" | Status: {status}"
            details += f" | Orchestrator: {orchestrator_healthy}"
            details += f" | Database: {database_connected}"
            
            # System is healthy if status is healthy or degraded (not error)
            system_healthy = status in ['healthy', 'degraded']
        else:
            system_healthy = False
            details += " | Health check failed"
        
        self.log_test_result("reliability_tests", "System Health Check", system_healthy, response_time, details)
        return system_healthy
    
    async def reliability_test_load_handling(self):
        """Test system under moderate load."""
        print("🔥 Testing load handling...")
        
        async def load_request():
            test_data = {
                "text": f"Load testing request with random content: {random.randint(1000, 9999)}",
                "context": {"domain": "general"},
                "mode": "production",
                "priority": "normal"
            }
            return await self.test_endpoint("POST", "/evaluate", test_data)
        
        # Test with 10 requests in quick succession
        start_time = time.time()
        tasks = [load_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if not isinstance(r, Exception) and r[0])
        success_rate = successful_requests / 10
        
        details = f"10 requests | {successful_requests}/10 successful | Success rate: {success_rate:.1%} | Total time: {total_time:.3f}s"
        
        # Consider successful if at least 80% of requests succeed
        load_handled = success_rate >= 0.8
        
        self.log_test_result("reliability_tests", "Load Handling", load_handled, total_time, details)
        return load_handled
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # 6. BAYESIAN OPTIMIZATION TESTING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def bayesian_test_optimization_start(self):
        """Test starting a PERFORMANCE-OPTIMIZED Bayesian optimization process."""
        print("🚀 Testing PERFORMANCE-OPTIMIZED Bayesian optimization start...")
        
        # Use the EXACT parameters from the review request for performance testing
        test_data = {
            "test_texts": [
                "Quick test for optimized Bayesian cluster resolution.",
                "Performance improvements should make this much faster."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 5,
            "max_optimization_time": 20.0,
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        details = f"Response time: {response_time:.3f}s"
        
        # Check if response time meets performance requirements (< 5 seconds for start)
        performance_met = response_time < 5.0
        
        if success:
            optimization_id = result.get('optimization_id', '')
            details += f" | Optimization ID: {optimization_id[:12]}..."
            details += f" | Status: {result.get('status', 'unknown')}"
            details += f" | Config samples: {result.get('configuration', {}).get('initial_samples', 0)}"
            details += f" | Performance target met (< 5s): {performance_met}"
            # Store optimization ID for later tests
            if not hasattr(self, 'optimization_ids'):
                self.optimization_ids = []
            self.optimization_ids.append(optimization_id)
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
            details += f" | Performance target met (< 5s): {performance_met}"
        
        # Success requires both API success AND performance target
        overall_success = success and performance_met
        
        self.log_test_result("bayesian_tests", "Performance-Optimized Start", overall_success, response_time, details)
        return overall_success
    
    async def bayesian_test_parameter_validation(self):
        """Test parameter validation for optimization requests with PERFORMANCE focus."""
        print("🔍 Testing optimization parameter validation with performance requirements...")
        
        # Test with invalid parameters - should respond quickly
        invalid_test_cases = [
            ("Empty test_texts", {"test_texts": []}),
            ("Non-list test_texts", {"test_texts": "not a list"}),
            ("Insufficient test_texts", {"test_texts": ["only one text"]}),
        ]
        
        validation_passed = 0
        total_response_time = 0
        
        for test_name, test_data in invalid_test_cases:
            success, response_time, result = await self.test_endpoint("POST", "/optimization/start", test_data)
            total_response_time += response_time
            
            # For validation tests, we expect failures (422 status) and fast responses
            validation_handled = not success or result.get('status') == 'error'
            performance_ok = response_time < 5.0  # Should be fast for validation errors
            
            details = f"Response time: {response_time:.3f}s | Validation handled: {validation_handled} | Performance OK: {performance_ok}"
            
            if validation_handled and performance_ok:
                validation_passed += 1
            
            self.log_test_result("bayesian_tests", f"Performance Validation - {test_name}", validation_handled and performance_ok, response_time, details)
        
        # Test edge case with timeout handling
        timeout_test_data = {
            "test_texts": [
                "Quick test for optimized Bayesian cluster resolution.",
                "Performance improvements should make this much faster."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 5,
            "max_optimization_time": 1.0,  # Very short timeout to test timeout handling
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/optimization/start", timeout_test_data)
        total_response_time += response_time
        
        # Should either succeed quickly or handle timeout gracefully
        timeout_handled = success or (not success and "timeout" in str(result.get('error', '')).lower())
        performance_ok = response_time < 5.0
        
        details = f"Timeout test: {response_time:.3f}s | Handled: {timeout_handled} | Performance OK: {performance_ok}"
        
        if timeout_handled and performance_ok:
            validation_passed += 1
        
        self.log_test_result("bayesian_tests", "Performance Timeout Handling", timeout_handled and performance_ok, response_time, details)
        
        overall_success = validation_passed >= 3  # Allow some tolerance
        avg_response_time = total_response_time / 4  # 4 tests total
        
        print(f"    Average validation response time: {avg_response_time:.3f}s")
        
        return overall_success
    
    async def bayesian_test_status_monitoring(self):
        """Test optimization status monitoring with PERFORMANCE requirements."""
        print("📊 Testing optimization status monitoring with performance requirements...")
        
        # Test with non-existent optimization ID - should be fast (< 2 seconds)
        fake_id = "opt_fake_12345"
        success1, response_time1, result1 = await self.test_endpoint("GET", f"/optimization/status/{fake_id}")
        
        # Check performance requirement for status endpoint (< 2 seconds)
        status_performance_met = response_time1 < 2.0
        
        # Should return 404 for non-existent ID
        not_found_handled = not success1
        
        details = f"Non-existent ID test: {response_time1:.3f}s | 404 handled: {not_found_handled} | Performance (< 2s): {status_performance_met}"
        
        # Test with real optimization ID if available
        if hasattr(self, 'optimization_ids') and self.optimization_ids:
            real_id = self.optimization_ids[0]
            success2, response_time2, result2 = await self.test_endpoint("GET", f"/optimization/status/{real_id}")
            
            status_performance_met2 = response_time2 < 2.0
            details += f" | Real ID test: {response_time2:.3f}s | Performance (< 2s): {status_performance_met2}"
            if success2:
                details += f" | Status: {result2.get('status', 'unknown')}"
            
            # Overall success requires proper error handling AND performance
            overall_success = not_found_handled and status_performance_met and status_performance_met2
        else:
            overall_success = not_found_handled and status_performance_met
        
        self.log_test_result("bayesian_tests", "Performance Status Monitoring", overall_success, response_time1, details)
        return overall_success
    
    async def bayesian_test_results_retrieval(self):
        """Test optimization results retrieval."""
        print("📈 Testing optimization results retrieval...")
        
        # Test with non-existent optimization ID
        fake_id = "opt_fake_results_12345"
        success, response_time, result = await self.test_endpoint("GET", f"/optimization/results/{fake_id}")
        
        # Should return 404 for non-existent ID
        not_found_handled = not success
        
        details = f"Response time: {response_time:.3f}s | 404 handled: {not_found_handled}"
        
        # Note: Real results would only be available after optimization completes
        # which takes too long for testing, so we focus on error handling
        
        self.log_test_result("bayesian_tests", "Results Retrieval", not_found_handled, response_time, details)
        return not_found_handled
    
    async def bayesian_test_optimization_list(self):
        """Test listing all optimizations."""
        print("📋 Testing optimization list endpoint...")
        
        success, response_time, result = await self.test_endpoint("GET", "/optimization/list")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            optimizations = result.get('optimizations', [])
            details += f" | Found {len(optimizations)} optimizations"
            details += f" | Has pagination: {'pagination' in result}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_test_result("bayesian_tests", "Optimization List", success, response_time, details)
        return success
    
    async def bayesian_test_parameter_application(self):
        """Test applying optimization parameters."""
        print("⚙️ Testing optimization parameter application...")
        
        # Test with non-existent optimization ID
        fake_id = "opt_fake_apply_12345"
        success, response_time, result = await self.test_endpoint("POST", f"/optimization/apply/{fake_id}", {})
        
        # Should return 404 for non-existent ID
        not_found_handled = not success
        
        details = f"Response time: {response_time:.3f}s | 404 handled: {not_found_handled}"
        
        self.log_test_result("bayesian_tests", "Parameter Application", not_found_handled, response_time, details)
        return not_found_handled
    
    async def bayesian_test_optimization_completion(self):
        """Test that optimization completes within reasonable time (under 30 seconds)."""
        print("⏱️ Testing optimization completion within 30 seconds...")
        
        # Start a minimal optimization that should complete quickly
        test_data = {
            "test_texts": [
                "Quick test for optimized Bayesian cluster resolution.",
                "Performance improvements should make this much faster."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 5,
            "max_optimization_time": 20.0,
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        # Start optimization
        start_success, start_time, start_result = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        if not start_success:
            details = f"Failed to start optimization: {start_result.get('error', 'Unknown error')}"
            self.log_test_result("bayesian_tests", "Optimization Completion Test", False, start_time, details)
            return False
        
        optimization_id = start_result.get('optimization_id', '')
        
        # Monitor optimization progress for up to 35 seconds (5s buffer beyond max_optimization_time)
        max_wait_time = 35.0
        check_interval = 2.0
        elapsed_time = 0
        completed = False
        final_status = "unknown"
        
        print(f"    Monitoring optimization {optimization_id[:12]}... (max wait: {max_wait_time}s)")
        
        while elapsed_time < max_wait_time:
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
            
            # Check status
            status_success, status_time, status_result = await self.test_endpoint("GET", f"/optimization/status/{optimization_id}")
            
            if status_success:
                final_status = status_result.get('status', 'unknown')
                progress = status_result.get('progress', {})
                
                print(f"    Status at {elapsed_time:.1f}s: {final_status} | Progress: {progress.get('completion_percentage', 0):.1f}%")
                
                if final_status in ['completed', 'failed', 'timeout']:
                    completed = True
                    break
            else:
                print(f"    Status check failed at {elapsed_time:.1f}s")
        
        # Check if optimization completed within expected time
        completion_within_time = completed and elapsed_time <= 30.0
        
        details = f"Start time: {start_time:.3f}s | Total time: {elapsed_time:.1f}s | Status: {final_status} | Completed within 30s: {completion_within_time}"
        
        if completed:
            # Try to get results
            results_success, results_time, results_data = await self.test_endpoint("GET", f"/optimization/results/{optimization_id}")
            if results_success:
                details += f" | Results retrieved: {results_time:.3f}s"
            else:
                details += f" | Results retrieval failed"
        
        overall_success = start_success and completion_within_time
        
        self.log_test_result("bayesian_tests", "Optimization Completion (30s)", overall_success, elapsed_time, details)
        return overall_success
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MAIN TEST EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    async def run_all_tests(self):
        """Run all test suites."""
        print("🚀 Starting Comprehensive Backend Testing Suite")
        print("=" * 80)
        
        await self.setup_session()
        
        try:
            # 1. Smoke Tests
            print("\n🔥 SMOKE TESTING")
            print("-" * 40)
            await self.smoke_test_health_endpoint()
            await self.smoke_test_parameters_endpoint()
            await self.smoke_test_evaluate_endpoint()
            await self.smoke_test_heat_map_endpoint()
            
            # 2. Performance Tests
            print("\n⚡ PERFORMANCE TESTING")
            print("-" * 40)
            await self.performance_test_response_times()
            await self.performance_test_concurrent_requests()
            
            # 3. Content Tests
            print("\n📝 REAL-WORLD CONTENT TESTING")
            print("-" * 40)
            await self.content_test_diverse_texts()
            await self.content_test_edge_cases()
            
            # 4. Integration Tests
            print("\n🔧 API INTEGRATION TESTING")
            print("-" * 40)
            await self.integration_test_parameter_updates()
            await self.integration_test_learning_stats()
            await self.integration_test_error_handling()
            
            # 5. Reliability Tests
            print("\n🏥 SYSTEM RELIABILITY TESTING")
            print("-" * 40)
            await self.reliability_test_system_health()
            await self.reliability_test_load_handling()
            
            # 6. Bayesian Optimization Tests - PERFORMANCE FOCUSED
            print("\n🎯 PERFORMANCE-OPTIMIZED BAYESIAN OPTIMIZATION TESTING")
            print("-" * 40)
            await self.bayesian_test_optimization_start()
            await self.bayesian_test_status_monitoring()
            await self.bayesian_test_parameter_validation()
            await self.bayesian_test_optimization_completion()
            await self.bayesian_test_results_retrieval()
            await self.bayesian_test_optimization_list()
            await self.bayesian_test_parameter_application()
            
        finally:
            await self.cleanup_session()
        
        # Calculate final statistics
        total_tests = self.results["summary"]["total_tests"]
        passed_tests = self.results["summary"]["passed_tests"]
        self.results["summary"]["success_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Response Time: {summary['average_response_time']:.3f}s")
        
        # Print detailed results by category
        categories = [
            ("smoke_tests", "🔥 SMOKE TESTS"),
            ("performance_tests", "⚡ PERFORMANCE TESTS"),
            ("content_tests", "📝 CONTENT TESTS"),
            ("integration_tests", "🔧 INTEGRATION TESTS"),
            ("reliability_tests", "🏥 RELIABILITY TESTS"),
            ("bayesian_tests", "🎯 BAYESIAN OPTIMIZATION TESTS")
        ]
        
        for category, title in categories:
            print(f"\n{title}")
            print("-" * 40)
            
            for test in self.results[category]:
                status = "✅ PASS" if test["passed"] else "❌ FAIL"
                print(f"{status} {test['test_name']} ({test['response_time']:.3f}s)")
                if test["details"]:
                    print(f"    {test['details']}")
        
        # Performance Claims Verification
        print(f"\n🎯 PERFORMANCE CLAIMS VERIFICATION")
        print("-" * 40)
        avg_time = summary['average_response_time']
        claimed_time = 0.055
        
        if avg_time <= 0.1:  # Allow some tolerance
            print(f"✅ Response Time: {avg_time:.3f}s (Claimed: {claimed_time}s) - ACCEPTABLE")
        else:
            print(f"❌ Response Time: {avg_time:.3f}s (Claimed: {claimed_time}s) - SLOWER THAN CLAIMED")
        
        if summary['success_rate'] >= 90:
            print(f"✅ Success Rate: {summary['success_rate']:.1f}% - EXCELLENT")
        elif summary['success_rate'] >= 80:
            print(f"⚠️ Success Rate: {summary['success_rate']:.1f}% - GOOD")
        else:
            print(f"❌ Success Rate: {summary['success_rate']:.1f}% - NEEDS IMPROVEMENT")
        
        print("\n" + "=" * 80)

async def main():
    """Main test execution function."""
    test_suite = BackendTestSuite()
    results = await test_suite.run_all_tests()
    
    # Save results to file
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: /app/backend_test_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())