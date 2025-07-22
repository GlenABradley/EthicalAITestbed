#!/usr/bin/env python3
"""
🎯 BAYESIAN OPTIMIZATION ENDPOINT VALIDATION TEST
═══════════════════════════════════════════════════════════════════════════════════

Quick validation test for Bayesian optimization endpoints focusing on:
- Endpoint existence and basic response
- Parameter validation
- Error handling
- System integration checks

Author: Testing Agent
Version: 1.0.0
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

# Backend URL from environment
BACKEND_URL = "https://b214a97a-320b-41fd-b6d6-ff742674f4c6.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BayesianOptimizationValidator:
    """Quick validation for Bayesian optimization endpoints."""
    
    def __init__(self):
        self.results = []
        self.session = None
    
    async def setup_session(self):
        """Setup aiohttp session for testing."""
        timeout = aiohttp.ClientTimeout(total=10)  # Short timeout for validation
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
    
    async def test_endpoint_quick(self, method: str, endpoint: str, data: dict = None, timeout: float = 10.0):
        """Quick endpoint test with timeout."""
        start_time = time.time()
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{API_BASE}{endpoint}") as response:
                    response_time = time.time() - start_time
                    try:
                        result = await response.json()
                    except:
                        result = {"text": await response.text()}
                    return response.status, response_time, result
            elif method.upper() == "POST":
                async with self.session.post(f"{API_BASE}{endpoint}", json=data) as response:
                    response_time = time.time() - start_time
                    try:
                        result = await response.json()
                    except:
                        result = {"text": await response.text()}
                    return response.status, response_time, result
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return 408, response_time, {"error": "Request timeout"}
        except Exception as e:
            response_time = time.time() - start_time
            return 500, response_time, {"error": str(e)}
    
    async def test_optimization_list_endpoint(self):
        """Test the optimization list endpoint (should be quick)."""
        print("📋 Testing optimization list endpoint...")
        
        status_code, response_time, result = await self.test_endpoint_quick("GET", "/optimization/list")
        
        details = f"Status: {status_code}"
        if status_code == 200:
            details += f" | Response received"
            passed = True
        elif status_code == 408:
            details += f" | Timeout (endpoint exists but slow)"
            passed = False  # Timeout indicates implementation issue
        elif status_code == 404:
            details += f" | Endpoint not found"
            passed = False
        else:
            details += f" | Error: {result.get('error', 'Unknown')}"
            passed = False
        
        self.log_result("Optimization List Endpoint", passed, response_time, details)
        return passed
    
    async def test_optimization_status_nonexistent(self):
        """Test status endpoint with non-existent ID (should be quick)."""
        print("📊 Testing status endpoint with fake ID...")
        
        fake_id = "opt_fake_12345"
        status_code, response_time, result = await self.test_endpoint_quick("GET", f"/optimization/status/{fake_id}")
        
        details = f"Status: {status_code}"
        if status_code == 404:
            details += f" | Correctly returned 404 for non-existent ID"
            passed = True
        elif status_code == 408:
            details += f" | Timeout (endpoint exists but slow)"
            passed = False
        elif status_code == 200:
            details += f" | Unexpected success for fake ID"
            passed = False
        else:
            details += f" | Error: {result.get('error', 'Unknown')}"
            passed = False
        
        self.log_result("Status Endpoint - Non-existent ID", passed, response_time, details)
        return passed
    
    async def test_optimization_results_nonexistent(self):
        """Test results endpoint with non-existent ID (should be quick)."""
        print("📈 Testing results endpoint with fake ID...")
        
        fake_id = "opt_fake_results_12345"
        status_code, response_time, result = await self.test_endpoint_quick("GET", f"/optimization/results/{fake_id}")
        
        details = f"Status: {status_code}"
        if status_code == 404:
            details += f" | Correctly returned 404 for non-existent ID"
            passed = True
        elif status_code == 408:
            details += f" | Timeout (endpoint exists but slow)"
            passed = False
        else:
            details += f" | Error: {result.get('error', 'Unknown')}"
            passed = False
        
        self.log_result("Results Endpoint - Non-existent ID", passed, response_time, details)
        return passed
    
    async def test_parameter_validation_empty_texts(self):
        """Test parameter validation with empty test_texts."""
        print("🔍 Testing parameter validation (empty texts)...")
        
        test_data = {"test_texts": []}
        status_code, response_time, result = await self.test_endpoint_quick("POST", "/optimization/start", test_data)
        
        details = f"Status: {status_code}"
        if status_code == 422:
            details += f" | Correctly validated empty texts"
            passed = True
        elif status_code == 408:
            details += f" | Timeout (validation should be quick)"
            passed = False
        else:
            details += f" | Unexpected response: {result.get('error', result.get('detail', 'Unknown'))}"
            passed = False
        
        self.log_result("Parameter Validation - Empty Texts", passed, response_time, details)
        return passed
    
    async def test_parameter_validation_insufficient_texts(self):
        """Test parameter validation with insufficient test_texts."""
        print("📝 Testing parameter validation (insufficient texts)...")
        
        test_data = {"test_texts": ["only one text"]}
        status_code, response_time, result = await self.test_endpoint_quick("POST", "/optimization/start", test_data)
        
        details = f"Status: {status_code}"
        if status_code == 422:
            details += f" | Correctly validated insufficient texts"
            passed = True
        elif status_code == 408:
            details += f" | Timeout (validation should be quick)"
            passed = False
        else:
            details += f" | Unexpected response: {result.get('error', result.get('detail', 'Unknown'))}"
            passed = False
        
        self.log_result("Parameter Validation - Insufficient Texts", passed, response_time, details)
        return passed
    
    async def test_optimization_start_timeout_behavior(self):
        """Test optimization start endpoint timeout behavior."""
        print("⏰ Testing optimization start timeout behavior...")
        
        test_data = {
            "test_texts": [
                "Test text one for optimization.",
                "Test text two for optimization."
            ],
            "n_initial_samples": 2,
            "n_optimization_iterations": 3,
            "max_optimization_time": 5.0,  # Very short time
            "parallel_evaluations": False,
            "max_workers": 1
        }
        
        status_code, response_time, result = await self.test_endpoint_quick("POST", "/optimization/start", test_data, timeout=8.0)
        
        details = f"Status: {status_code}"
        if status_code == 200:
            details += f" | Started successfully (background process)"
            passed = True
        elif status_code == 408:
            details += f" | Timeout - indicates computational complexity"
            passed = False  # This is an issue for user experience
        elif status_code == 422:
            details += f" | Validation error: {result.get('detail', 'Unknown')}"
            passed = False
        else:
            details += f" | Error: {result.get('error', result.get('detail', 'Unknown'))}"
            passed = False
        
        self.log_result("Optimization Start - Timeout Behavior", passed, response_time, details)
        return passed
    
    async def run_validation_tests(self):
        """Run all validation tests."""
        print("🎯 Starting Bayesian Optimization Validation Tests")
        print("=" * 80)
        
        await self.setup_session()
        
        try:
            # Quick endpoint tests
            print("\n📋 ENDPOINT EXISTENCE TESTS")
            print("-" * 40)
            await self.test_optimization_list_endpoint()
            await self.test_optimization_status_nonexistent()
            await self.test_optimization_results_nonexistent()
            
            # Parameter validation tests
            print("\n🔍 PARAMETER VALIDATION TESTS")
            print("-" * 40)
            await self.test_parameter_validation_empty_texts()
            await self.test_parameter_validation_insufficient_texts()
            
            # Timeout behavior test
            print("\n⏰ TIMEOUT BEHAVIOR TESTS")
            print("-" * 40)
            await self.test_optimization_start_timeout_behavior()
            
        finally:
            await self.cleanup_session()
        
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 BAYESIAN OPTIMIZATION VALIDATION RESULTS")
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
        
        print(f"\n🎯 BAYESIAN OPTIMIZATION SYSTEM STATUS")
        print("-" * 40)
        
        # Analyze results
        endpoint_exists = any("Endpoint" in r["test_name"] and r["passed"] for r in self.results)
        validation_works = any("Validation" in r["test_name"] and r["passed"] for r in self.results)
        timeout_issues = any("Timeout" in r["test_name"] and not r["passed"] for r in self.results)
        
        if endpoint_exists:
            print("✅ Bayesian optimization endpoints are implemented")
        else:
            print("❌ Bayesian optimization endpoints have issues")
        
        if validation_works:
            print("✅ Parameter validation is working")
        else:
            print("❌ Parameter validation has issues")
        
        if timeout_issues:
            print("⚠️ System has timeout issues (computationally intensive)")
        else:
            print("✅ No timeout issues detected")
        
        print("\n" + "=" * 80)

async def main():
    """Main validation execution function."""
    validator = BayesianOptimizationValidator()
    results = await validator.run_validation_tests()
    
    # Save results to file
    with open('/app/bayesian_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Validation results saved to: /app/bayesian_validation_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())