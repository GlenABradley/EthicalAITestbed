#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Ethical AI Developer Testbed
Tests all API endpoints and core functionality
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class BackendTester:
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
    
    def test_health_check(self):
        """Test /api/health endpoint"""
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'status' in data and data['status'] == 'healthy':
                    self.log_result(
                        "Health Check", 
                        True, 
                        f"Service is healthy, evaluator_initialized: {data.get('evaluator_initialized', 'unknown')}"
                    )
                else:
                    self.log_result("Health Check", False, "Invalid health response format", data)
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Health Check", False, f"Connection error: {str(e)}")
    
    def test_get_parameters(self):
        """Test /api/parameters endpoint"""
        try:
            response = requests.get(f"{API_BASE}/parameters", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'parameters' in data:
                    params = data['parameters']
                    required_params = ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']
                    
                    if all(param in params for param in required_params):
                        self.log_result("Get Parameters", True, "Parameters retrieved successfully", params)
                        return params
                    else:
                        self.log_result("Get Parameters", False, "Missing required parameters", data)
                else:
                    self.log_result("Get Parameters", False, "Invalid parameters response format", data)
            else:
                self.log_result("Get Parameters", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Get Parameters", False, f"Connection error: {str(e)}")
        
        return None
    
    def test_update_parameters(self):
        """Test /api/update-parameters endpoint"""
        try:
            # Test parameter update
            new_params = {
                "virtue_threshold": 0.4,
                "deontological_threshold": 0.35,
                "consequentialist_threshold": 0.45
            }
            
            response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": new_params},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.log_result("Update Parameters", True, "Parameters updated successfully", data)
                    
                    # Verify parameters were actually updated
                    verify_response = requests.get(f"{API_BASE}/parameters", timeout=10)
                    if verify_response.status_code == 200:
                        verify_data = verify_response.json()
                        current_params = verify_data.get('parameters', {})
                        
                        if current_params.get('virtue_threshold') == 0.4:
                            self.log_result("Parameter Verification", True, "Parameter update verified")
                        else:
                            self.log_result("Parameter Verification", False, "Parameters not properly updated", current_params)
                    
                else:
                    self.log_result("Update Parameters", False, "Update failed", data)
            else:
                self.log_result("Update Parameters", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Update Parameters", False, f"Connection error: {str(e)}")
    
    def test_evaluate_neutral_text(self):
        """Test /api/evaluate with neutral text"""
        try:
            test_text = "The weather is nice today"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                
                if all(field in data for field in required_fields):
                    evaluation = data['evaluation']
                    
                    # Check if text is evaluated as ethical (should be for neutral text)
                    if evaluation.get('overall_ethical', False):
                        self.log_result(
                            "Evaluate Neutral Text", 
                            True, 
                            f"Neutral text correctly evaluated as ethical. Processing time: {evaluation.get('processing_time', 'unknown')}s"
                        )
                    else:
                        self.log_result(
                            "Evaluate Neutral Text", 
                            False, 
                            "Neutral text incorrectly flagged as unethical", 
                            evaluation
                        )
                else:
                    self.log_result("Evaluate Neutral Text", False, "Missing required response fields", data)
            else:
                self.log_result("Evaluate Neutral Text", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Evaluate Neutral Text", False, f"Connection error: {str(e)}")
    
    def test_evaluate_problematic_text(self):
        """Test /api/evaluate with potentially problematic text"""
        try:
            test_text = "You are stupid and worthless"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                
                # This text should likely be flagged as unethical
                if not evaluation.get('overall_ethical', True):
                    minimal_spans = evaluation.get('minimal_spans', [])
                    self.log_result(
                        "Evaluate Problematic Text", 
                        True, 
                        f"Problematic text correctly flagged as unethical. Found {len(minimal_spans)} violations"
                    )
                else:
                    self.log_result(
                        "Evaluate Problematic Text", 
                        False, 
                        "Problematic text not flagged as unethical", 
                        evaluation
                    )
            else:
                self.log_result("Evaluate Problematic Text", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Evaluate Problematic Text", False, f"Connection error: {str(e)}")
    
    def test_evaluate_mixed_content(self):
        """Test /api/evaluate with mixed content text"""
        try:
            test_text = "I love helping people but hate idiots"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                clean_text = data.get('clean_text', '')
                
                # Should detect some violations but preserve positive parts
                minimal_spans = evaluation.get('minimal_spans', [])
                
                self.log_result(
                    "Evaluate Mixed Content", 
                    True, 
                    f"Mixed content processed. Violations: {len(minimal_spans)}, Clean text: '{clean_text}'"
                )
            else:
                self.log_result("Evaluate Mixed Content", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Evaluate Mixed Content", False, f"Connection error: {str(e)}")
    
    def test_get_evaluations(self):
        """Test /api/evaluations endpoint"""
        try:
            response = requests.get(f"{API_BASE}/evaluations", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'evaluations' in data and 'count' in data:
                    count = data['count']
                    evaluations = data['evaluations']
                    
                    self.log_result(
                        "Get Evaluations", 
                        True, 
                        f"Retrieved {count} evaluations from database"
                    )
                    
                    # Verify evaluation structure if any exist
                    if count > 0 and evaluations:
                        eval_sample = evaluations[0]
                        required_fields = ['id', 'input_text', 'result', 'timestamp']
                        
                        if all(field in eval_sample for field in required_fields):
                            self.log_result("Evaluation Structure", True, "Evaluation records have correct structure")
                        else:
                            self.log_result("Evaluation Structure", False, "Evaluation records missing required fields", eval_sample)
                    
                else:
                    self.log_result("Get Evaluations", False, "Invalid evaluations response format", data)
            else:
                self.log_result("Get Evaluations", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Get Evaluations", False, f"Connection error: {str(e)}")
    
    def test_create_calibration_test(self):
        """Test /api/calibration-test endpoint"""
        try:
            test_case = {
                "text": "This is a test for calibration purposes",
                "expected_result": "ethical"
            }
            
            response = requests.post(
                f"{API_BASE}/calibration-test",
                json=test_case,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'test_case' in data:
                    test_id = data['test_case']['id']
                    self.log_result("Create Calibration Test", True, f"Calibration test created with ID: {test_id}")
                    return test_id
                else:
                    self.log_result("Create Calibration Test", False, "Invalid calibration test response", data)
            else:
                self.log_result("Create Calibration Test", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Create Calibration Test", False, f"Connection error: {str(e)}")
        
        return None
    
    def test_run_calibration_test(self, test_id: str):
        """Test /api/run-calibration-test/{test_id} endpoint"""
        if not test_id:
            self.log_result("Run Calibration Test", False, "No test ID provided")
            return
            
        try:
            response = requests.post(f"{API_BASE}/run-calibration-test/{test_id}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    passed = data.get('passed', False)
                    expected = data.get('expected', 'unknown')
                    actual = data.get('actual', 'unknown')
                    
                    self.log_result(
                        "Run Calibration Test", 
                        True, 
                        f"Calibration test executed. Passed: {passed}, Expected: {expected}, Actual: {actual}"
                    )
                else:
                    self.log_result("Run Calibration Test", False, "Calibration test execution failed", data)
            else:
                self.log_result("Run Calibration Test", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Run Calibration Test", False, f"Connection error: {str(e)}")
    
    def test_get_calibration_tests(self):
        """Test /api/calibration-tests endpoint"""
        try:
            response = requests.get(f"{API_BASE}/calibration-tests", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'tests' in data and 'count' in data:
                    count = data['count']
                    self.log_result("Get Calibration Tests", True, f"Retrieved {count} calibration tests")
                else:
                    self.log_result("Get Calibration Tests", False, "Invalid calibration tests response format", data)
            else:
                self.log_result("Get Calibration Tests", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Get Calibration Tests", False, f"Connection error: {str(e)}")
    
    def test_performance_metrics(self):
        """Test /api/performance-metrics endpoint"""
        try:
            response = requests.get(f"{API_BASE}/performance-metrics", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we have metrics or no evaluations message
                if 'metrics' in data:
                    metrics = data['metrics']
                    required_metrics = ['total_evaluations', 'average_processing_time']
                    
                    if any(metric in metrics for metric in required_metrics):
                        self.log_result(
                            "Performance Metrics", 
                            True, 
                            f"Performance metrics retrieved. Total evaluations: {metrics.get('total_evaluations', 0)}"
                        )
                    else:
                        self.log_result("Performance Metrics", False, "Invalid metrics format", metrics)
                        
                elif 'message' in data and 'No evaluations found' in data['message']:
                    self.log_result("Performance Metrics", True, "No evaluations found (expected for fresh system)")
                    
                else:
                    self.log_result("Performance Metrics", False, "Invalid performance metrics response", data)
            else:
                self.log_result("Performance Metrics", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Performance Metrics", False, f"Connection error: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        try:
            # Test with empty text
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ""},
                timeout=10
            )
            
            if response.status_code in [200, 400, 422]:
                self.log_result("Error Handling - Empty Text", True, f"Handled empty text appropriately (HTTP {response.status_code})")
            else:
                self.log_result("Error Handling - Empty Text", False, f"Unexpected response to empty text: HTTP {response.status_code}")
            
            # Test with invalid JSON
            try:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    data="invalid json",
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code in [400, 422]:
                    self.log_result("Error Handling - Invalid JSON", True, f"Handled invalid JSON appropriately (HTTP {response.status_code})")
                else:
                    self.log_result("Error Handling - Invalid JSON", False, f"Unexpected response to invalid JSON: HTTP {response.status_code}")
            except:
                self.log_result("Error Handling - Invalid JSON", True, "Request properly rejected invalid JSON")
            
            # Test non-existent calibration test
            response = requests.post(f"{API_BASE}/run-calibration-test/nonexistent-id", timeout=10)
            
            if response.status_code == 404:
                self.log_result("Error Handling - Not Found", True, "Properly returned 404 for non-existent calibration test")
            else:
                self.log_result("Error Handling - Not Found", False, f"Unexpected response for non-existent test: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Error Handling", False, f"Error during error handling tests: {str(e)}")
    
    def test_stress_large_text(self):
        """Test with extremely large text input to find breaking points"""
        try:
            # Test with 10KB text
            large_text = "This is a test sentence. " * 400  # ~10KB
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": large_text},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get('evaluation', {}).get('processing_time', 0)
                self.log_result("Stress Test - Large Text (10KB)", True, f"Processed large text successfully in {processing_time:.2f}s")
            else:
                self.log_result("Stress Test - Large Text (10KB)", False, f"Failed with HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Stress Test - Large Text (10KB)", False, f"Error: {str(e)}")
    
    def test_stress_very_large_text(self):
        """Test with very large text input (100KB) to find memory limits"""
        try:
            # Test with 100KB text
            very_large_text = "This is a test sentence with some potentially problematic content like hate and anger. " * 1000  # ~100KB
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": very_large_text},
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                processing_time = data.get('evaluation', {}).get('processing_time', 0)
                self.log_result("Stress Test - Very Large Text (100KB)", True, f"Processed very large text successfully in {processing_time:.2f}s")
            elif response.status_code == 413:
                self.log_result("Stress Test - Very Large Text (100KB)", True, "Properly rejected oversized request (413)")
            else:
                self.log_result("Stress Test - Very Large Text (100KB)", False, f"Failed with HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Stress Test - Very Large Text (100KB)", False, f"Error: {str(e)}")
    
    def test_stress_concurrent_requests(self):
        """Test concurrent requests to find race conditions"""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request(thread_id):
            try:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": f"Concurrent test request {thread_id} with some content"},
                    timeout=30
                )
                results.append((thread_id, response.status_code))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        try:
            # Launch 10 concurrent requests
            threads = []
            for i in range(10):
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            success_count = sum(1 for _, status in results if status == 200)
            
            if success_count >= 8:  # Allow some failures due to concurrency
                self.log_result("Stress Test - Concurrent Requests", True, f"{success_count}/10 requests succeeded")
            else:
                self.log_result("Stress Test - Concurrent Requests", False, f"Only {success_count}/10 requests succeeded. Errors: {errors}")
                
        except Exception as e:
            self.log_result("Stress Test - Concurrent Requests", False, f"Error: {str(e)}")
    
    def test_stress_extreme_parameters(self):
        """Test with extreme parameter values"""
        try:
            # Test with extreme threshold values
            extreme_params = {
                "virtue_threshold": 0.0,
                "deontological_threshold": 1.0,
                "consequentialist_threshold": 0.5
            }
            
            response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": extreme_params},
                timeout=10
            )
            
            if response.status_code == 200:
                # Test evaluation with extreme parameters
                eval_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": "This is a test with extreme parameters"},
                    timeout=30
                )
                
                if eval_response.status_code == 200:
                    self.log_result("Stress Test - Extreme Parameters", True, "System handled extreme parameter values")
                else:
                    self.log_result("Stress Test - Extreme Parameters", False, f"Evaluation failed with extreme params: HTTP {eval_response.status_code}")
            else:
                self.log_result("Stress Test - Extreme Parameters", False, f"Parameter update failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Stress Test - Extreme Parameters", False, f"Error: {str(e)}")
    
    def test_stress_malformed_requests(self):
        """Test with various malformed requests"""
        try:
            # Test with missing required fields
            response = requests.post(f"{API_BASE}/evaluate", json={}, timeout=10)
            if response.status_code in [400, 422]:
                self.log_result("Stress Test - Missing Fields", True, f"Properly rejected missing fields (HTTP {response.status_code})")
            else:
                self.log_result("Stress Test - Missing Fields", False, f"Unexpected response: HTTP {response.status_code}")
            
            # Test with wrong data types
            response = requests.post(f"{API_BASE}/evaluate", json={"text": 123}, timeout=10)
            if response.status_code in [400, 422]:
                self.log_result("Stress Test - Wrong Data Types", True, f"Properly rejected wrong data types (HTTP {response.status_code})")
            else:
                self.log_result("Stress Test - Wrong Data Types", False, f"Unexpected response: HTTP {response.status_code}")
            
            # Test with null values
            response = requests.post(f"{API_BASE}/evaluate", json={"text": None}, timeout=10)
            if response.status_code in [400, 422]:
                self.log_result("Stress Test - Null Values", True, f"Properly rejected null values (HTTP {response.status_code})")
            else:
                self.log_result("Stress Test - Null Values", False, f"Unexpected response: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Stress Test - Malformed Requests", False, f"Error: {str(e)}")
    
    def test_stress_unicode_and_special_chars(self):
        """Test with unicode and special characters"""
        try:
            # Test with unicode characters
            unicode_text = "Testing with émojis 🚀 and spëcial chäractërs ñ 中文 العربية русский"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": unicode_text},
                timeout=30
            )
            
            if response.status_code == 200:
                self.log_result("Stress Test - Unicode Characters", True, "Successfully processed unicode characters")
            else:
                self.log_result("Stress Test - Unicode Characters", False, f"Failed with unicode: HTTP {response.status_code}")
            
            # Test with special characters and escape sequences
            special_text = "Testing with \n\t\r special chars and \"quotes\" and 'apostrophes' and \\backslashes"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": special_text},
                timeout=30
            )
            
            if response.status_code == 200:
                self.log_result("Stress Test - Special Characters", True, "Successfully processed special characters")
            else:
                self.log_result("Stress Test - Special Characters", False, f"Failed with special chars: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Stress Test - Unicode/Special Chars", False, f"Error: {str(e)}")
    
    def test_stress_database_limits(self):
        """Test database operation limits"""
        try:
            # Test retrieving large number of evaluations
            response = requests.get(f"{API_BASE}/evaluations?limit=1000", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                self.log_result("Stress Test - Large DB Query", True, f"Retrieved {count} evaluations successfully")
            else:
                self.log_result("Stress Test - Large DB Query", False, f"Failed: HTTP {response.status_code}")
            
            # Test creating many calibration tests rapidly
            success_count = 0
            for i in range(5):
                response = requests.post(
                    f"{API_BASE}/calibration-test",
                    json={
                        "text": f"Rapid test creation {i}",
                        "expected_result": "ethical"
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    success_count += 1
            
            if success_count >= 4:
                self.log_result("Stress Test - Rapid DB Writes", True, f"{success_count}/5 rapid writes succeeded")
            else:
                self.log_result("Stress Test - Rapid DB Writes", False, f"Only {success_count}/5 rapid writes succeeded")
                
        except Exception as e:
            self.log_result("Stress Test - Database Limits", False, f"Error: {str(e)}")
    
    def test_edge_case_empty_and_whitespace(self):
        """Test edge cases with empty and whitespace-only content"""
        try:
            test_cases = [
                ("", "Empty string"),
                ("   ", "Whitespace only"),
                ("\n\t\r", "Newlines and tabs only"),
                (".", "Single character"),
                ("a", "Single letter")
            ]
            
            for text, description in test_cases:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=10
                )
                
                if response.status_code in [200, 400, 422]:
                    self.log_result(f"Edge Case - {description}", True, f"Handled appropriately (HTTP {response.status_code})")
                else:
                    self.log_result(f"Edge Case - {description}", False, f"Unexpected response: HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Edge Case - Empty/Whitespace", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all backend tests"""
        print(f"🚀 Starting comprehensive backend testing for: {API_BASE}")
        print("=" * 80)
        
        # Core functionality tests
        self.test_health_check()
        self.test_get_parameters()
        self.test_update_parameters()
        
        # Evaluation tests
        self.test_evaluate_neutral_text()
        self.test_evaluate_problematic_text()
        self.test_evaluate_mixed_content()
        
        # Database operations
        self.test_get_evaluations()
        
        # Calibration system
        test_id = self.test_create_calibration_test()
        self.test_run_calibration_test(test_id)
        self.test_get_calibration_tests()
        
        # Performance and metrics
        self.test_performance_metrics()
        
        # Error handling
        self.test_error_handling()
        
        # Summary
        print("\n" + "=" * 80)
        print("🏁 TESTING COMPLETE")
        print("=" * 80)
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        print(f"📊 Total: {len(self.results)}")
        
        if self.failed_tests:
            print(f"\n❌ Failed tests: {', '.join(self.failed_tests)}")
        
        return len(self.failed_tests) == 0

def main():
    """Main test execution"""
    tester = BackendTester()
    success = tester.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()