#!/usr/bin/env python3
"""
🎯 FOCUSED BAYESIAN OPTIMIZATION TESTING
═══════════════════════════════════════════════════════════════════════════════════

MISSION: Test the newly implemented lightweight Bayesian cluster optimization system
specifically to verify if the performance issues have been resolved.

Focus Areas:
- Quick Start Test (< 3 seconds response time)
- Status Monitoring (< 1 second response time) 
- Optimization Completion (15-30 seconds total)
- Results Retrieval
- Parameter Application
- List Optimizations

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple

# Backend URL from environment
BACKEND_URL = "https://efb05ca6-d049-4715-907b-1090362ca79b.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BayesianOptimizationTester:
    """Focused testing for Bayesian optimization system."""
    
    def __init__(self):
        self.results = []
        self.session = None
        self.optimization_ids = []
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup_session(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, passed: bool, response_time: float, details: str = ""):
        """Log test result."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "response_time": response_time,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name} ({response_time:.3f}s)")
        if details:
            print(f"    {details}")
    
    async def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Tuple[bool, float, Dict]:
        """Generic endpoint testing method."""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{API_BASE}{endpoint}") as response:
                    response_time = time.time() - start_time
                    if response.content_type == 'application/json':
                        result = await response.json()
                    else:
                        result = {"text": await response.text()}
                    return response.status == 200, response_time, result
            elif method.upper() == "POST":
                async with self.session.post(f"{API_BASE}{endpoint}", json=data) as response:
                    response_time = time.time() - start_time
                    if response.content_type == 'application/json':
                        result = await response.json()
                    else:
                        result = {"text": await response.text()}
                    return response.status == 200, response_time, result
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, {"error": str(e)}
    
    async def test_quick_start(self):
        """Test 1: Quick Start Test (< 3 seconds response time)"""
        print("🚀 Testing Quick Start (Target: < 3 seconds)...")
        
        # Use the exact parameters from the review request
        test_data = {
            "test_texts": [
                "Quick test for lightweight Bayesian cluster optimization.",
                "This should complete much faster than the previous system."
            ],
            "n_initial_samples": 3,
            "n_optimization_iterations": 3,
            "max_optimization_time": 15.0
        }
        
        success, response_time, result = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        # Check if response time meets the < 3 seconds requirement
        performance_met = response_time < 3.0
        
        details = f"Response time: {response_time:.3f}s (Target: < 3.0s)"
        if success:
            optimization_id = result.get('optimization_id', '')
            self.optimization_ids.append(optimization_id)
            details += f" | Optimization ID: {optimization_id[:12]}..."
            details += f" | Status: {result.get('status', 'unknown')}"
            details += f" | Performance target met: {performance_met}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
            details += f" | Performance target met: {performance_met}"
        
        overall_success = success and performance_met
        self.log_result("Quick Start Test", overall_success, response_time, details)
        return overall_success
    
    async def test_status_monitoring(self):
        """Test 2: Status Monitoring (< 1 second response time)"""
        print("📊 Testing Status Monitoring (Target: < 1 second)...")
        
        if not self.optimization_ids:
            # Test with a fake ID first
            fake_id = "opt_fake_12345"
            success, response_time, result = await self.test_endpoint("GET", f"/optimization/status/{fake_id}")
            
            performance_met = response_time < 1.0
            not_found_handled = not success  # Should return 404
            
            details = f"Fake ID test: {response_time:.3f}s (Target: < 1.0s) | 404 handled: {not_found_handled} | Performance met: {performance_met}"
            
            overall_success = not_found_handled and performance_met
            self.log_result("Status Monitoring (Fake ID)", overall_success, response_time, details)
            return overall_success
        
        # Test with real optimization ID
        optimization_id = self.optimization_ids[0]
        success, response_time, result = await self.test_endpoint("GET", f"/optimization/status/{optimization_id}")
        
        performance_met = response_time < 1.0
        
        details = f"Response time: {response_time:.3f}s (Target: < 1.0s)"
        if success:
            status = result.get('status', 'unknown')
            progress = result.get('progress', {})
            details += f" | Status: {status}"
            details += f" | Progress: {progress.get('progress_percent', 0):.1f}%"
            details += f" | Performance target met: {performance_met}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
            details += f" | Performance target met: {performance_met}"
        
        overall_success = success and performance_met
        self.log_result("Status Monitoring", overall_success, response_time, details)
        return overall_success
    
    async def test_optimization_completion(self):
        """Test 3: Optimization Completion (15-30 seconds total)"""
        print("⏱️ Testing Optimization Completion (Target: 15-30 seconds)...")
        
        if not self.optimization_ids:
            self.log_result("Optimization Completion", False, 0.0, "No optimization ID available")
            return False
        
        optimization_id = self.optimization_ids[0]
        
        # Monitor optimization progress for up to 35 seconds
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
                
                print(f"    Status at {elapsed_time:.1f}s: {final_status} | Progress: {progress.get('progress_percent', 0):.1f}%")
                
                if final_status in ['completed', 'failed', 'timeout']:
                    completed = True
                    break
            else:
                print(f"    Status check failed at {elapsed_time:.1f}s")
        
        # Check if optimization completed within expected time (15-30 seconds)
        completion_within_time = completed and 15.0 <= elapsed_time <= 30.0
        
        details = f"Total time: {elapsed_time:.1f}s | Status: {final_status} | Completed in 15-30s: {completion_within_time}"
        
        overall_success = completed and completion_within_time
        self.log_result("Optimization Completion", overall_success, elapsed_time, details)
        return overall_success
    
    async def test_results_retrieval(self):
        """Test 4: Results Retrieval"""
        print("📈 Testing Results Retrieval...")
        
        if not self.optimization_ids:
            # Test with fake ID
            fake_id = "opt_fake_results_12345"
            success, response_time, result = await self.test_endpoint("GET", f"/optimization/results/{fake_id}")
            
            not_found_handled = not success  # Should return 404
            details = f"Fake ID test: {response_time:.3f}s | 404 handled: {not_found_handled}"
            
            self.log_result("Results Retrieval (Fake ID)", not_found_handled, response_time, details)
            return not_found_handled
        
        # Test with real optimization ID
        optimization_id = self.optimization_ids[0]
        success, response_time, result = await self.test_endpoint("GET", f"/optimization/results/{optimization_id}")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Has results: {'best_parameters' in result}"
            details += f" | Has score: {'best_score' in result}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_result("Results Retrieval", success, response_time, details)
        return success
    
    async def test_parameter_application(self):
        """Test 5: Parameter Application"""
        print("⚙️ Testing Parameter Application...")
        
        if not self.optimization_ids:
            # Test with fake ID
            fake_id = "opt_fake_apply_12345"
            success, response_time, result = await self.test_endpoint("POST", f"/optimization/apply/{fake_id}", {})
            
            not_found_handled = not success  # Should return 404
            details = f"Fake ID test: {response_time:.3f}s | 404 handled: {not_found_handled}"
            
            self.log_result("Parameter Application (Fake ID)", not_found_handled, response_time, details)
            return not_found_handled
        
        # Test with real optimization ID
        optimization_id = self.optimization_ids[0]
        success, response_time, result = await self.test_endpoint("POST", f"/optimization/apply/{optimization_id}", {})
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            details += f" | Applied: {result.get('applied', False)}"
            details += f" | Message: {result.get('message', 'N/A')}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_result("Parameter Application", success, response_time, details)
        return success
    
    async def test_list_optimizations(self):
        """Test 6: List Optimizations"""
        print("📋 Testing List Optimizations...")
        
        success, response_time, result = await self.test_endpoint("GET", "/optimization/list")
        
        details = f"Response time: {response_time:.3f}s"
        if success:
            optimizations = result.get('optimizations', [])
            details += f" | Found {len(optimizations)} optimizations"
            details += f" | Has pagination: {'pagination' in result}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_result("List Optimizations", success, response_time, details)
        return success
    
    async def run_all_tests(self):
        """Run all Bayesian optimization tests."""
        print("🎯 FOCUSED BAYESIAN OPTIMIZATION TESTING")
        print("=" * 60)
        print("Testing the newly implemented lightweight Bayesian cluster optimization system")
        print("Performance Expectations:")
        print("- Start endpoint: < 3 seconds response time")
        print("- Status endpoint: < 1 second response time")
        print("- Optimization completion: 15-30 seconds total")
        print("- No timeout errors or hanging processes")
        print("=" * 60)
        
        await self.setup_session()
        
        try:
            # Run tests in sequence
            test_results = []
            
            # Test 1: Quick Start
            result1 = await self.test_quick_start()
            test_results.append(result1)
            
            # Test 2: Status Monitoring
            result2 = await self.test_status_monitoring()
            test_results.append(result2)
            
            # Test 3: Optimization Completion (only if we have an optimization running)
            if self.optimization_ids:
                result3 = await self.test_optimization_completion()
                test_results.append(result3)
            
            # Test 4: Results Retrieval
            result4 = await self.test_results_retrieval()
            test_results.append(result4)
            
            # Test 5: Parameter Application
            result5 = await self.test_parameter_application()
            test_results.append(result5)
            
            # Test 6: List Optimizations
            result6 = await self.test_list_optimizations()
            test_results.append(result6)
            
        finally:
            await self.cleanup_session()
        
        # Print summary
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 BAYESIAN OPTIMIZATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n🎯 PERFORMANCE ANALYSIS:")
        print("-" * 30)
        
        # Analyze key performance metrics
        start_test = next((r for r in self.results if "Quick Start" in r["test_name"]), None)
        if start_test:
            target_met = "✅ MET" if start_test["response_time"] < 3.0 else "❌ FAILED"
            print(f"Start Endpoint: {start_test['response_time']:.3f}s (Target: < 3.0s) {target_met}")
        
        status_test = next((r for r in self.results if "Status Monitoring" in r["test_name"] and "Fake" not in r["test_name"]), None)
        if status_test:
            target_met = "✅ MET" if status_test["response_time"] < 1.0 else "❌ FAILED"
            print(f"Status Endpoint: {status_test['response_time']:.3f}s (Target: < 1.0s) {target_met}")
        
        completion_test = next((r for r in self.results if "Completion" in r["test_name"]), None)
        if completion_test:
            target_met = "✅ MET" if 15.0 <= completion_test["response_time"] <= 30.0 else "❌ FAILED"
            print(f"Optimization Completion: {completion_test['response_time']:.1f}s (Target: 15-30s) {target_met}")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print("-" * 30)
        if success_rate >= 80:
            print("✅ LIGHTWEIGHT BAYESIAN OPTIMIZATION SYSTEM IS WORKING")
            print("   Performance improvements have resolved the timeout issues!")
        else:
            print("❌ LIGHTWEIGHT BAYESIAN OPTIMIZATION SYSTEM NEEDS WORK")
            print("   Performance issues persist and require further optimization.")
        
        print("=" * 60)

async def main():
    """Main test execution function."""
    tester = BayesianOptimizationTester()
    results = await tester.run_all_tests()
    
    # Save results to file
    with open('/app/bayesian_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: /app/bayesian_test_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())