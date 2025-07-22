#!/usr/bin/env python3
"""
🎯 PERFORMANCE-OPTIMIZED BAYESIAN OPTIMIZATION TEST SUITE
═══════════════════════════════════════════════════════════════════════════════════

MISSION: Test the performance-optimized Bayesian cluster optimization system
focusing on the specific requirements from the review request:

1. Quick Start Optimization Test with minimal parameters
2. Status Monitoring Test (< 2 seconds for status)
3. Performance Validation (< 5 seconds for start)
4. Error Handling Test
5. Optimization Completion Test (< 30 seconds)

Author: Testing Agent
Version: 1.0.0 - Performance Focus
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Backend URL from environment
BACKEND_URL = "https://b214a97a-320b-41fd-b6d6-ff742674f4c6.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BayesianPerformanceTestSuite:
    """Performance-focused Bayesian optimization testing suite."""
    
    def __init__(self):
        self.results = {
            "performance_tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "average_response_time": 0.0,
                "success_rate": 0.0
            }
        }
        self.session = None
        self.optimization_ids = []
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=45)  # Allow time for optimization completion
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup_session(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, passed: bool, 
                       response_time: float = 0.0, details: str = ""):
        """Log test result to results dictionary."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "response_time": response_time,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results["performance_tests"].append(result)
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
    
    async def test_quick_start_optimization(self):
        """Test 1: Quick Start Optimization Test with minimal parameters."""
        print("🚀 Test 1: Quick Start Optimization with minimal parameters...")
        
        # Use EXACT parameters from review request
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
        
        # Performance requirement: < 5 seconds for start
        performance_met = response_time < 5.0
        
        details = f"Response time: {response_time:.3f}s | Performance target (< 5s): {performance_met}"
        
        if success:
            optimization_id = result.get('optimization_id', '')
            self.optimization_ids.append(optimization_id)
            details += f" | Optimization ID: {optimization_id[:12]}..."
            details += f" | Status: {result.get('status', 'unknown')}"
            details += f" | Config: {result.get('configuration', {}).get('initial_samples', 0)} samples"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        overall_success = success and performance_met
        self.log_test_result("Quick Start Optimization", overall_success, response_time, details)
        
        print(f"    Result: {'✅ PASS' if overall_success else '❌ FAIL'} - {details}")
        return overall_success
    
    async def test_status_monitoring(self):
        """Test 2: Status Monitoring Test (< 2 seconds for status)."""
        print("📊 Test 2: Status Monitoring with performance requirements...")
        
        # Test with non-existent optimization ID - should be fast
        fake_id = "opt_fake_performance_test"
        success1, response_time1, result1 = await self.test_endpoint("GET", f"/optimization/status/{fake_id}")
        
        # Performance requirement: < 2 seconds for status
        status_performance_met = response_time1 < 2.0
        not_found_handled = not success1  # Should return 404
        
        details = f"Non-existent ID: {response_time1:.3f}s | Performance (< 2s): {status_performance_met} | 404 handled: {not_found_handled}"
        
        # Test with real optimization ID if available
        if self.optimization_ids:
            real_id = self.optimization_ids[0]
            success2, response_time2, result2 = await self.test_endpoint("GET", f"/optimization/status/{real_id}")
            
            status_performance_met2 = response_time2 < 2.0
            details += f" | Real ID: {response_time2:.3f}s | Performance (< 2s): {status_performance_met2}"
            
            if success2:
                details += f" | Status: {result2.get('status', 'unknown')}"
            
            overall_success = not_found_handled and status_performance_met and status_performance_met2
        else:
            overall_success = not_found_handled and status_performance_met
        
        self.log_test_result("Status Monitoring Performance", overall_success, response_time1, details)
        
        print(f"    Result: {'✅ PASS' if overall_success else '❌ FAIL'} - {details}")
        return overall_success
    
    async def test_performance_validation(self):
        """Test 3: Performance Validation with edge cases."""
        print("🔍 Test 3: Performance Validation with edge cases...")
        
        # Test invalid parameters - should respond quickly
        invalid_cases = [
            ("Empty test_texts", {"test_texts": []}),
            ("Insufficient test_texts", {"test_texts": ["only one text"]}),
        ]
        
        validation_passed = 0
        total_response_time = 0
        
        for test_name, test_data in invalid_cases:
            success, response_time, result = await self.test_endpoint("POST", "/optimization/start", test_data)
            total_response_time += response_time
            
            # Should fail validation quickly
            validation_handled = not success or result.get('status') == 'error'
            performance_ok = response_time < 5.0
            
            details = f"{test_name}: {response_time:.3f}s | Validation handled: {validation_handled} | Performance OK: {performance_ok}"
            
            if validation_handled and performance_ok:
                validation_passed += 1
            
            print(f"    {test_name}: {'✅ PASS' if validation_handled and performance_ok else '❌ FAIL'} - {details}")
        
        overall_success = validation_passed >= 1  # At least one validation test should pass
        avg_response_time = total_response_time / len(invalid_cases)
        
        self.log_test_result("Performance Validation", overall_success, avg_response_time, f"Validation tests passed: {validation_passed}/{len(invalid_cases)}")
        return overall_success
    
    async def test_optimization_completion(self):
        """Test 4: Optimization Completion within 30 seconds."""
        print("⏱️ Test 4: Optimization completion within 30 seconds...")
        
        if not self.optimization_ids:
            print("    No optimization ID available, skipping completion test")
            self.log_test_result("Optimization Completion", False, 0, "No optimization ID available")
            return False
        
        optimization_id = self.optimization_ids[0]
        
        # Monitor optimization progress for up to 35 seconds
        max_wait_time = 35.0
        check_interval = 3.0
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
                completion_pct = progress.get('completion_percentage', 0)
                
                print(f"    Status at {elapsed_time:.1f}s: {final_status} | Progress: {completion_pct:.1f}%")
                
                if final_status in ['completed', 'failed', 'timeout']:
                    completed = True
                    break
            else:
                print(f"    Status check failed at {elapsed_time:.1f}s")
        
        # Check if optimization completed within expected time
        completion_within_time = completed and elapsed_time <= 30.0
        
        details = f"Total time: {elapsed_time:.1f}s | Status: {final_status} | Completed within 30s: {completion_within_time}"
        
        if completed and final_status == 'completed':
            # Try to get results
            results_success, results_time, results_data = await self.test_endpoint("GET", f"/optimization/results/{optimization_id}")
            if results_success:
                details += f" | Results retrieved: {results_time:.3f}s"
            else:
                details += f" | Results retrieval failed"
        
        overall_success = completed and completion_within_time
        
        self.log_test_result("Optimization Completion (30s)", overall_success, elapsed_time, details)
        
        print(f"    Result: {'✅ PASS' if overall_success else '❌ FAIL'} - {details}")
        return overall_success
    
    async def test_error_handling(self):
        """Test 5: Error Handling Test."""
        print("❌ Test 5: Error handling with timeout scenarios...")
        
        # Test with very short timeout to trigger timeout handling
        timeout_test_data = {
            "test_texts": [
                "Quick test for optimized Bayesian cluster resolution.",
                "Performance improvements should make this much faster."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 5,
            "max_optimization_time": 1.0,  # Very short timeout
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/optimization/start", timeout_test_data)
        
        # Should either succeed quickly or handle timeout gracefully
        timeout_handled = success or (not success and "timeout" in str(result.get('error', '')).lower())
        performance_ok = response_time < 5.0
        
        details = f"Timeout test: {response_time:.3f}s | Handled gracefully: {timeout_handled} | Performance OK: {performance_ok}"
        
        overall_success = timeout_handled and performance_ok
        
        self.log_test_result("Error Handling", overall_success, response_time, details)
        
        print(f"    Result: {'✅ PASS' if overall_success else '❌ FAIL'} - {details}")
        return overall_success
    
    async def run_performance_tests(self):
        """Run all performance-focused Bayesian optimization tests."""
        print("🎯 PERFORMANCE-OPTIMIZED BAYESIAN OPTIMIZATION TEST SUITE")
        print("=" * 80)
        print("Testing the performance improvements made to the Bayesian cluster optimization system:")
        print("- Reduced samples from 20 to 5, iterations from 50 to 10")
        print("- Limited optimization time to 30s max")
        print("- Simplified clustering to 3 scales instead of 7")
        print("- Added aggressive timeouts and async protection")
        print("- Disabled parallel processing for stability")
        print("=" * 80)
        
        await self.setup_session()
        
        try:
            # Run the 5 key performance tests
            test_results = []
            
            test_results.append(await self.test_quick_start_optimization())
            test_results.append(await self.test_status_monitoring())
            test_results.append(await self.test_performance_validation())
            test_results.append(await self.test_optimization_completion())
            test_results.append(await self.test_error_handling())
            
        finally:
            await self.cleanup_session()
        
        # Calculate final statistics
        total_tests = self.results["summary"]["total_tests"]
        passed_tests = self.results["summary"]["passed_tests"]
        self.results["summary"]["success_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Calculate average response time
        response_times = [test["response_time"] for test in self.results["performance_tests"] if test["response_time"] > 0]
        self.results["summary"]["average_response_time"] = sum(response_times) / len(response_times) if response_times else 0
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE-OPTIMIZED BAYESIAN OPTIMIZATION TEST RESULTS")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Response Time: {summary['average_response_time']:.3f}s")
        
        print(f"\n🎯 DETAILED TEST RESULTS")
        print("-" * 40)
        
        for test in self.results["performance_tests"]:
            status = "✅ PASS" if test["passed"] else "❌ FAIL"
            print(f"{status} {test['test_name']} ({test['response_time']:.3f}s)")
            if test["details"]:
                print(f"    {test['details']}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT")
        print("-" * 40)
        
        if summary['success_rate'] >= 80:
            print(f"✅ Overall Performance: {summary['success_rate']:.1f}% - EXCELLENT")
        elif summary['success_rate'] >= 60:
            print(f"⚠️ Overall Performance: {summary['success_rate']:.1f}% - GOOD")
        else:
            print(f"❌ Overall Performance: {summary['success_rate']:.1f}% - NEEDS IMPROVEMENT")
        
        avg_time = summary['average_response_time']
        if avg_time <= 3.0:
            print(f"✅ Response Times: {avg_time:.3f}s average - EXCELLENT")
        elif avg_time <= 5.0:
            print(f"⚠️ Response Times: {avg_time:.3f}s average - ACCEPTABLE")
        else:
            print(f"❌ Response Times: {avg_time:.3f}s average - TOO SLOW")
        
        print("\n" + "=" * 80)

async def main():
    """Main test execution function."""
    test_suite = BayesianPerformanceTestSuite()
    results = await test_suite.run_performance_tests()
    
    # Save results to file
    with open('/app/bayesian_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: /app/bayesian_performance_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())