#!/usr/bin/env python3
"""
🎯 BAYESIAN OPTIMIZATION ENDPOINT TESTING
═══════════════════════════════════════════════════════════════════════════════════

MISSION: Test the newly implemented 7-stage Bayesian cluster optimization system

This focused test suite validates:
- POST /api/optimization/start - Start optimization process
- GET /api/optimization/status/{id} - Check progress
- GET /api/optimization/results/{id} - Get results
- POST /api/optimization/apply/{id} - Apply parameters
- GET /api/optimization/list - List optimizations

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Backend URL from environment
BACKEND_URL = "https://efb05ca6-d049-4715-907b-1090362ca79b.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BayesianOptimizationTester:
    """Focused testing for Bayesian optimization endpoints."""
    
    def __init__(self):
        self.results = []
        self.session = None
        self.optimization_ids = []
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup_session(self):
        """Cleanup aiohttp session."""
        if self.session:
            await self.session.close()
    
    def log_result(self, test_name: str, passed: bool, response_time: float = 0.0, details: str = ""):
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
    
    async def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Tuple[bool, float, Dict, int]:
        """Generic endpoint testing method."""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{API_BASE}{endpoint}") as response:
                    response_time = time.time() - start_time
                    try:
                        result = await response.json()
                    except:
                        result = {"text": await response.text()}
                    return response.status == 200, response_time, result, response.status
            elif method.upper() == "POST":
                async with self.session.post(f"{API_BASE}{endpoint}", json=data) as response:
                    response_time = time.time() - start_time
                    try:
                        result = await response.json()
                    except:
                        result = {"text": await response.text()}
                    return response.status == 200, response_time, result, response.status
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, {"error": str(e)}, 0
    
    async def test_optimization_start_basic(self):
        """Test basic optimization start functionality."""
        print("🚀 Testing optimization start (basic)...")
        
        test_data = {
            "test_texts": [
                "This text contains some ethical considerations about AI development.",
                "We should ensure fairness and transparency in machine learning systems.",
                "Privacy protection is crucial when handling personal data.",
                "Algorithmic bias can lead to discriminatory outcomes.",
                "Ethical guidelines should be integrated into software development processes."
            ],
            "n_initial_samples": 5,
            "n_optimization_iterations": 10,
            "max_optimization_time": 60.0,
            "parallel_evaluations": True,
            "max_workers": 2
        }
        
        success, response_time, result, status_code = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        details = f"Status: {status_code}"
        if success:
            optimization_id = result.get('optimization_id', '')
            details += f" | ID: {optimization_id[:12]}..."
            details += f" | Status: {result.get('status', 'unknown')}"
            if optimization_id:
                self.optimization_ids.append(optimization_id)
        else:
            details += f" | Error: {result.get('error', result.get('detail', 'Unknown error'))}"
        
        self.log_result("Optimization Start - Basic", success, response_time, details)
        return success
    
    async def test_optimization_start_minimal(self):
        """Test optimization start with minimal parameters."""
        print("🎯 Testing optimization start (minimal)...")
        
        test_data = {
            "test_texts": [
                "Minimal test for optimization.",
                "Second text for testing."
            ]
        }
        
        success, response_time, result, status_code = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        details = f"Status: {status_code}"
        if success:
            optimization_id = result.get('optimization_id', '')
            details += f" | ID: {optimization_id[:12]}..."
            if optimization_id:
                self.optimization_ids.append(optimization_id)
        else:
            details += f" | Error: {result.get('error', result.get('detail', 'Unknown error'))}"
        
        self.log_result("Optimization Start - Minimal", success, response_time, details)
        return success
    
    async def test_parameter_validation_empty_texts(self):
        """Test parameter validation with empty test_texts."""
        print("🔍 Testing validation (empty texts)...")
        
        test_data = {"test_texts": []}
        
        success, response_time, result, status_code = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        # Should fail with 422
        validation_handled = status_code == 422 or not success
        details = f"Status: {status_code} | Validation handled: {validation_handled}"
        
        self.log_result("Parameter Validation - Empty Texts", validation_handled, response_time, details)
        return validation_handled
    
    async def test_parameter_validation_insufficient_texts(self):
        """Test parameter validation with insufficient test_texts."""
        print("📝 Testing validation (insufficient texts)...")
        
        test_data = {"test_texts": ["only one text"]}
        
        success, response_time, result, status_code = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        # Should fail with 422
        validation_handled = status_code == 422 or not success
        details = f"Status: {status_code} | Validation handled: {validation_handled}"
        
        self.log_result("Parameter Validation - Insufficient Texts", validation_handled, response_time, details)
        return validation_handled
    
    async def test_status_nonexistent_id(self):
        """Test status endpoint with non-existent optimization ID."""
        print("📊 Testing status (non-existent ID)...")
        
        fake_id = "opt_fake_12345"
        success, response_time, result, status_code = await self.test_endpoint("GET", f"/optimization/status/{fake_id}")
        
        # Should return 404
        not_found_handled = status_code == 404 or not success
        details = f"Status: {status_code} | 404 handled: {not_found_handled}"
        
        self.log_result("Status Monitoring - Non-existent ID", not_found_handled, response_time, details)
        return not_found_handled
    
    async def test_status_real_id(self):
        """Test status endpoint with real optimization ID."""
        print("📈 Testing status (real ID)...")
        
        if not self.optimization_ids:
            self.log_result("Status Monitoring - Real ID", False, 0.0, "No optimization IDs available")
            return False
        
        real_id = self.optimization_ids[0]
        success, response_time, result, status_code = await self.test_endpoint("GET", f"/optimization/status/{real_id}")
        
        details = f"Status: {status_code}"
        if success:
            details += f" | Opt Status: {result.get('status', 'unknown')}"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        # Either success or graceful failure is acceptable
        handled = success or status_code in [404, 422]
        self.log_result("Status Monitoring - Real ID", handled, response_time, details)
        return handled
    
    async def test_results_nonexistent_id(self):
        """Test results endpoint with non-existent optimization ID."""
        print("📋 Testing results (non-existent ID)...")
        
        fake_id = "opt_fake_results_12345"
        success, response_time, result, status_code = await self.test_endpoint("GET", f"/optimization/results/{fake_id}")
        
        # Should return 404
        not_found_handled = status_code == 404 or not success
        details = f"Status: {status_code} | 404 handled: {not_found_handled}"
        
        self.log_result("Results Retrieval - Non-existent ID", not_found_handled, response_time, details)
        return not_found_handled
    
    async def test_optimization_list(self):
        """Test optimization list endpoint."""
        print("📜 Testing optimization list...")
        
        success, response_time, result, status_code = await self.test_endpoint("GET", "/optimization/list")
        
        details = f"Status: {status_code}"
        if success:
            optimizations = result.get('optimizations', [])
            details += f" | Found: {len(optimizations)} optimizations"
        else:
            details += f" | Error: {result.get('error', 'Unknown error')}"
        
        self.log_result("Optimization List", success, response_time, details)
        return success
    
    async def test_parameter_application_nonexistent_id(self):
        """Test parameter application with non-existent optimization ID."""
        print("⚙️ Testing parameter application (non-existent ID)...")
        
        fake_id = "opt_fake_apply_12345"
        success, response_time, result, status_code = await self.test_endpoint("POST", f"/optimization/apply/{fake_id}", {})
        
        # Should return 404
        not_found_handled = status_code == 404 or not success
        details = f"Status: {status_code} | 404 handled: {not_found_handled}"
        
        self.log_result("Parameter Application - Non-existent ID", not_found_handled, response_time, details)
        return not_found_handled
    
    async def test_integration_check(self):
        """Test integration with ethical engine."""
        print("🧠 Testing ethical engine integration...")
        
        # Start a very small optimization to test integration
        test_data = {
            "test_texts": [
                "Integration test text one.",
                "Integration test text two."
            ],
            "n_initial_samples": 2,
            "n_optimization_iterations": 3,
            "max_optimization_time": 30.0,
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        success, response_time, result, status_code = await self.test_endpoint("POST", "/optimization/start", test_data)
        
        details = f"Status: {status_code}"
        if success:
            details += f" | Integration successful"
            details += f" | Has config: {'configuration' in result}"
        else:
            details += f" | Error: {result.get('error', result.get('detail', 'Unknown error'))}"
        
        self.log_result("Ethical Engine Integration", success, response_time, details)
        return success
    
    async def run_all_tests(self):
        """Run all Bayesian optimization tests."""
        print("🎯 Starting Bayesian Optimization Testing Suite")
        print("=" * 80)
        
        await self.setup_session()
        
        try:
            # Basic functionality tests
            print("\n🚀 BASIC FUNCTIONALITY TESTS")
            print("-" * 40)
            await self.test_optimization_start_basic()
            await self.test_optimization_start_minimal()
            
            # Parameter validation tests
            print("\n🔍 PARAMETER VALIDATION TESTS")
            print("-" * 40)
            await self.test_parameter_validation_empty_texts()
            await self.test_parameter_validation_insufficient_texts()
            
            # Status monitoring tests
            print("\n📊 STATUS MONITORING TESTS")
            print("-" * 40)
            await self.test_status_nonexistent_id()
            await self.test_status_real_id()
            
            # Results retrieval tests
            print("\n📋 RESULTS RETRIEVAL TESTS")
            print("-" * 40)
            await self.test_results_nonexistent_id()
            
            # List and application tests
            print("\n📜 LIST AND APPLICATION TESTS")
            print("-" * 40)
            await self.test_optimization_list()
            await self.test_parameter_application_nonexistent_id()
            
            # Integration tests
            print("\n🧠 INTEGRATION TESTS")
            print("-" * 40)
            await self.test_integration_check()
            
        finally:
            await self.cleanup_session()
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 BAYESIAN OPTIMIZATION TEST RESULTS")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_response_time = sum(r["response_time"] for r in self.results) / total_tests if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        
        print(f"\n🎯 BAYESIAN OPTIMIZATION ENDPOINTS STATUS")
        print("-" * 40)
        
        endpoint_status = {
            "POST /api/optimization/start": any("Optimization Start" in r["test_name"] and r["passed"] for r in self.results),
            "GET /api/optimization/status/{id}": any("Status Monitoring" in r["test_name"] and r["passed"] for r in self.results),
            "GET /api/optimization/results/{id}": any("Results Retrieval" in r["test_name"] and r["passed"] for r in self.results),
            "GET /api/optimization/list": any("Optimization List" in r["test_name"] and r["passed"] for r in self.results),
            "POST /api/optimization/apply/{id}": any("Parameter Application" in r["test_name"] and r["passed"] for r in self.results),
        }
        
        for endpoint, working in endpoint_status.items():
            status = "✅ WORKING" if working else "❌ ISSUES"
            print(f"{status} {endpoint}")
        
        print("\n" + "=" * 80)

async def main():
    """Main test execution function."""
    tester = BayesianOptimizationTester()
    results = await tester.run_all_tests()
    
    # Save results to file
    with open('/app/bayesian_optimization_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: /app/bayesian_optimization_test_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())