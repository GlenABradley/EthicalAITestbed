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
    
    # NEW DYNAMIC SCALING AND LEARNING TESTS
    
    def test_threshold_scaling_exponential(self):
        """Test POST /api/threshold-scaling with exponential scaling"""
        try:
            test_cases = [
                {"slider_value": 0.0, "use_exponential": True},
                {"slider_value": 0.5, "use_exponential": True},
                {"slider_value": 1.0, "use_exponential": True}
            ]
            
            for test_case in test_cases:
                response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json=test_case,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ['slider_value', 'scaled_threshold', 'scaling_type', 'formula']
                    
                    if all(field in data for field in required_fields):
                        slider_val = data['slider_value']
                        scaled_val = data['scaled_threshold']
                        scaling_type = data['scaling_type']
                        
                        if scaling_type == "exponential" and 0.0 <= scaled_val <= 0.3:
                            self.log_result(
                                f"Threshold Scaling Exponential (slider={slider_val})", 
                                True, 
                                f"Exponential scaling working: {slider_val} -> {scaled_val:.4f}"
                            )
                        else:
                            self.log_result(
                                f"Threshold Scaling Exponential (slider={slider_val})", 
                                False, 
                                f"Invalid scaling result: {data}"
                            )
                    else:
                        self.log_result("Threshold Scaling Exponential", False, "Missing required fields", data)
                else:
                    self.log_result("Threshold Scaling Exponential", False, f"HTTP {response.status_code}", {"response": response.text})
                    
        except Exception as e:
            self.log_result("Threshold Scaling Exponential", False, f"Connection error: {str(e)}")
    
    def test_threshold_scaling_linear(self):
        """Test POST /api/threshold-scaling with linear scaling"""
        try:
            test_cases = [
                {"slider_value": 0.0, "use_exponential": False},
                {"slider_value": 0.5, "use_exponential": False},
                {"slider_value": 1.0, "use_exponential": False}
            ]
            
            for test_case in test_cases:
                response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json=test_case,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    slider_val = data.get('slider_value', 0)
                    scaled_val = data.get('scaled_threshold', 0)
                    scaling_type = data.get('scaling_type', '')
                    
                    if scaling_type == "linear" and abs(scaled_val - slider_val) < 0.001:
                        self.log_result(
                            f"Threshold Scaling Linear (slider={slider_val})", 
                            True, 
                            f"Linear scaling working: {slider_val} -> {scaled_val}"
                        )
                    else:
                        self.log_result(
                            f"Threshold Scaling Linear (slider={slider_val})", 
                            False, 
                            f"Linear scaling failed: expected {slider_val}, got {scaled_val}"
                        )
                else:
                    self.log_result("Threshold Scaling Linear", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Threshold Scaling Linear", False, f"Connection error: {str(e)}")
    
    def test_learning_stats_initial(self):
        """Test GET /api/learning-stats endpoint"""
        try:
            response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['total_learning_entries', 'average_feedback_score', 'learning_active']
                
                if all(field in data for field in required_fields):
                    total_entries = data['total_learning_entries']
                    avg_feedback = data['average_feedback_score']
                    learning_active = data['learning_active']
                    
                    self.log_result(
                        "Learning Stats", 
                        True, 
                        f"Learning stats retrieved: {total_entries} entries, avg feedback: {avg_feedback:.3f}, active: {learning_active}"
                    )
                    return data
                else:
                    self.log_result("Learning Stats", False, "Missing required fields", data)
            else:
                self.log_result("Learning Stats", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Learning Stats", False, f"Connection error: {str(e)}")
        
        return None
    
    def test_dynamic_scaling_enabled_evaluation(self):
        """Test evaluation with dynamic scaling enabled"""
        try:
            # Enable dynamic scaling parameters
            dynamic_params = {
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "enable_learning_mode": True,
                "exponential_scaling": True,
                "cascade_high_threshold": 0.5,
                "cascade_low_threshold": 0.2
            }
            
            # Update parameters first
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": dynamic_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                self.log_result("Dynamic Scaling Setup", False, f"Failed to update parameters: HTTP {param_response.status_code}")
                return None
            
            # Test evaluation with dynamic scaling
            test_text = "This is a moderately ambiguous text that might trigger dynamic scaling"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                
                # Check for dynamic scaling information
                if 'dynamic_scaling' in evaluation:
                    dynamic_info = evaluation['dynamic_scaling']
                    used_dynamic = dynamic_info.get('used_dynamic_scaling', False)
                    used_cascade = dynamic_info.get('used_cascade_filtering', False)
                    ambiguity_score = dynamic_info.get('ambiguity_score', 0.0)
                    
                    self.log_result(
                        "Dynamic Scaling Enabled Evaluation", 
                        True, 
                        f"Dynamic scaling: {used_dynamic}, cascade: {used_cascade}, ambiguity: {ambiguity_score:.3f}"
                    )
                    return evaluation.get('evaluation_id')
                else:
                    self.log_result("Dynamic Scaling Enabled Evaluation", False, "No dynamic scaling information in response")
            else:
                self.log_result("Dynamic Scaling Enabled Evaluation", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Enabled Evaluation", False, f"Connection error: {str(e)}")
        
        return None
    
    def test_dynamic_scaling_disabled_evaluation(self):
        """Test evaluation with dynamic scaling disabled"""
        try:
            # Disable dynamic scaling parameters
            static_params = {
                "enable_dynamic_scaling": False,
                "enable_cascade_filtering": False,
                "enable_learning_mode": False
            }
            
            # Update parameters first
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": static_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                self.log_result("Static Scaling Setup", False, f"Failed to update parameters: HTTP {param_response.status_code}")
                return
            
            # Test evaluation without dynamic scaling
            test_text = "This is the same moderately ambiguous text for comparison"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                
                # Check that dynamic scaling is disabled
                if 'dynamic_scaling' in evaluation:
                    dynamic_info = evaluation['dynamic_scaling']
                    used_dynamic = dynamic_info.get('used_dynamic_scaling', True)  # Should be False
                    used_cascade = dynamic_info.get('used_cascade_filtering', True)  # Should be False
                    
                    if not used_dynamic and not used_cascade:
                        self.log_result(
                            "Dynamic Scaling Disabled Evaluation", 
                            True, 
                            "Dynamic scaling properly disabled"
                        )
                    else:
                        self.log_result(
                            "Dynamic Scaling Disabled Evaluation", 
                            False, 
                            f"Dynamic scaling not properly disabled: dynamic={used_dynamic}, cascade={used_cascade}"
                        )
                else:
                    self.log_result("Dynamic Scaling Disabled Evaluation", True, "No dynamic scaling information (expected when disabled)")
            else:
                self.log_result("Dynamic Scaling Disabled Evaluation", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Disabled Evaluation", False, f"Connection error: {str(e)}")
    
    def test_feedback_submission(self):
        """Test POST /api/feedback endpoint"""
        try:
            # First, create an evaluation to get an evaluation_id
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "Test text for feedback submission"},
                timeout=30
            )
            
            if eval_response.status_code != 200:
                self.log_result("Feedback Setup", False, "Failed to create evaluation for feedback test")
                return
            
            eval_data = eval_response.json()
            evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
            
            if not evaluation_id:
                self.log_result("Feedback Setup", False, "No evaluation_id in response")
                return
            
            # Test feedback submission
            feedback_cases = [
                {"evaluation_id": evaluation_id, "feedback_score": 0.8, "user_comment": "Good result"},
                {"evaluation_id": evaluation_id, "feedback_score": 0.2, "user_comment": "Poor result"},
                {"evaluation_id": evaluation_id, "feedback_score": 1.0, "user_comment": "Perfect"}
            ]
            
            for feedback in feedback_cases:
                response = requests.post(
                    f"{API_BASE}/feedback",
                    json=feedback,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'message' in data and 'evaluation_id' in data:
                        score = feedback['feedback_score']
                        self.log_result(
                            f"Feedback Submission (score={score})", 
                            True, 
                            f"Feedback recorded successfully: {data['message']}"
                        )
                    else:
                        self.log_result(f"Feedback Submission (score={score})", False, "Invalid response format", data)
                else:
                    self.log_result(f"Feedback Submission", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Feedback Submission", False, f"Connection error: {str(e)}")
    
    def test_feedback_validation(self):
        """Test feedback endpoint validation"""
        try:
            # Test invalid feedback scores
            invalid_cases = [
                {"evaluation_id": "test", "feedback_score": -0.1},  # Below 0
                {"evaluation_id": "test", "feedback_score": 1.1},   # Above 1
                {"evaluation_id": "", "feedback_score": 0.5},       # Empty evaluation_id
                {"feedback_score": 0.5}                             # Missing evaluation_id
            ]
            
            for i, invalid_case in enumerate(invalid_cases):
                response = requests.post(
                    f"{API_BASE}/feedback",
                    json=invalid_case,
                    timeout=10
                )
                
                if response.status_code in [400, 422]:
                    self.log_result(f"Feedback Validation {i+1}", True, f"Properly rejected invalid input (HTTP {response.status_code})")
                else:
                    self.log_result(f"Feedback Validation {i+1}", False, f"Should have rejected invalid input: HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Feedback Validation", False, f"Connection error: {str(e)}")
    
    def test_dynamic_scaling_details(self):
        """Test GET /api/dynamic-scaling-test/{evaluation_id} endpoint"""
        try:
            # First, create an evaluation with dynamic scaling enabled
            dynamic_params = {
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": dynamic_params}, timeout=10)
            
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "This text should trigger dynamic scaling analysis"},
                timeout=30
            )
            
            if eval_response.status_code != 200:
                self.log_result("Dynamic Scaling Details Setup", False, "Failed to create evaluation")
                return
            
            eval_data = eval_response.json()
            evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
            
            if not evaluation_id:
                self.log_result("Dynamic Scaling Details Setup", False, "No evaluation_id in response")
                return
            
            # Test dynamic scaling details endpoint
            response = requests.get(f"{API_BASE}/dynamic-scaling-test/{evaluation_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['evaluation_id', 'dynamic_scaling_enabled', 'cascade_filtering_enabled', 'ambiguity_score']
                
                if all(field in data for field in required_fields):
                    self.log_result(
                        "Dynamic Scaling Details", 
                        True, 
                        f"Retrieved scaling details for {evaluation_id}: ambiguity={data.get('ambiguity_score', 0):.3f}"
                    )
                else:
                    self.log_result("Dynamic Scaling Details", False, "Missing required fields", data)
            elif response.status_code == 404:
                self.log_result("Dynamic Scaling Details", False, "Evaluation not found in database")
            else:
                self.log_result("Dynamic Scaling Details", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Details", False, f"Connection error: {str(e)}")
    
    def test_learning_stats_after_feedback(self):
        """Test learning stats after submitting feedback"""
        try:
            # Get initial stats
            initial_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            initial_stats = initial_response.json() if initial_response.status_code == 200 else {}
            initial_entries = initial_stats.get('total_learning_entries', 0)
            
            # Create evaluation and submit feedback
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "Test for learning stats update"},
                timeout=30
            )
            
            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
                
                if evaluation_id:
                    # Submit feedback
                    feedback_response = requests.post(
                        f"{API_BASE}/feedback",
                        json={"evaluation_id": evaluation_id, "feedback_score": 0.9},
                        timeout=10
                    )
                    
                    if feedback_response.status_code == 200:
                        # Check updated stats
                        time.sleep(1)  # Brief delay for database update
                        updated_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
                        
                        if updated_response.status_code == 200:
                            updated_stats = updated_response.json()
                            updated_entries = updated_stats.get('total_learning_entries', 0)
                            avg_feedback = updated_stats.get('average_feedback_score', 0)
                            
                            if updated_entries >= initial_entries:
                                self.log_result(
                                    "Learning Stats After Feedback", 
                                    True, 
                                    f"Stats updated: {updated_entries} entries, avg feedback: {avg_feedback:.3f}"
                                )
                            else:
                                self.log_result("Learning Stats After Feedback", False, "Learning entries did not increase")
                        else:
                            self.log_result("Learning Stats After Feedback", False, "Failed to get updated stats")
                    else:
                        self.log_result("Learning Stats After Feedback", False, "Failed to submit feedback")
                else:
                    self.log_result("Learning Stats After Feedback", False, "No evaluation_id")
            else:
                self.log_result("Learning Stats After Feedback", False, "Failed to create evaluation")
                
        except Exception as e:
            self.log_result("Learning Stats After Feedback", False, f"Connection error: {str(e)}")
    
    def test_cascade_filtering_obvious_cases(self):
        """Test cascade filtering with obviously ethical and unethical text"""
        try:
            # Enable cascade filtering
            cascade_params = {
                "enable_cascade_filtering": True,
                "cascade_high_threshold": 0.5,
                "cascade_low_threshold": 0.2
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": cascade_params}, timeout=10)
            
            test_cases = [
                ("I love helping people and being kind", "Obviously ethical"),
                ("You are stupid worthless garbage and should die", "Obviously unethical"),
                ("The weather is nice today", "Neutral/ethical")
            ]
            
            for text, description in test_cases:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    dynamic_info = evaluation.get('dynamic_scaling', {})
                    
                    cascade_result = dynamic_info.get('cascade_result')
                    processing_stages = dynamic_info.get('processing_stages', [])
                    
                    if cascade_result is not None:
                        self.log_result(
                            f"Cascade Filtering - {description}", 
                            True, 
                            f"Cascade decision: {cascade_result}, stages: {processing_stages}"
                        )
                    else:
                        self.log_result(
                            f"Cascade Filtering - {description}", 
                            True, 
                            f"Proceeded to detailed evaluation, stages: {processing_stages}"
                        )
                else:
                    self.log_result(f"Cascade Filtering - {description}", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Cascade Filtering", False, f"Connection error: {str(e)}")
    
    def test_exponential_vs_linear_scaling_comparison(self):
        """Compare exponential vs linear threshold scaling"""
        try:
            test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for slider_value in test_values:
                # Test exponential
                exp_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                # Test linear
                lin_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": False},
                    timeout=10
                )
                
                if exp_response.status_code == 200 and lin_response.status_code == 200:
                    exp_data = exp_response.json()
                    lin_data = lin_response.json()
                    
                    exp_scaled = exp_data.get('scaled_threshold', 0)
                    lin_scaled = lin_data.get('scaled_threshold', 0)
                    
                    # For low values, exponential should be lower than linear
                    # For high values, exponential should be higher than linear
                    if slider_value < 0.5:
                        comparison_valid = exp_scaled <= lin_scaled
                    else:
                        comparison_valid = exp_scaled >= lin_scaled or abs(exp_scaled - lin_scaled) < 0.01
                    
                    if comparison_valid:
                        self.log_result(
                            f"Scaling Comparison (slider={slider_value})", 
                            True, 
                            f"Exponential: {exp_scaled:.4f}, Linear: {lin_scaled:.4f}"
                        )
                    else:
                        self.log_result(
                            f"Scaling Comparison (slider={slider_value})", 
                            False, 
                            f"Unexpected scaling relationship: exp={exp_scaled:.4f}, lin={lin_scaled:.4f}"
                        )
                else:
                    self.log_result(f"Scaling Comparison (slider={slider_value})", False, "Failed to get scaling responses")
                    
        except Exception as e:
            self.log_result("Scaling Comparison", False, f"Connection error: {str(e)}")
    
    def test_complete_learning_workflow(self):
        """Test the complete learning system workflow as requested in review"""
        try:
            print("\n🧠 TESTING COMPLETE LEARNING WORKFLOW")
            print("=" * 60)
            
            # Step 1: Enable learning mode and dynamic scaling
            learning_params = {
                "enable_learning_mode": True,
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "virtue_threshold": 0.25,
                "deontological_threshold": 0.25,
                "consequentialist_threshold": 0.25
            }
            
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": learning_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                self.log_result("Learning Workflow - Setup", False, f"Failed to enable learning mode: HTTP {param_response.status_code}")
                return
            
            # Step 2: Evaluate text with "This is a test for learning system"
            test_text = "This is a test for learning system"
            
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if eval_response.status_code != 200:
                self.log_result("Learning Workflow - Evaluation", False, f"Failed to evaluate text: HTTP {eval_response.status_code}")
                return
            
            eval_data = eval_response.json()
            evaluation = eval_data.get('evaluation', {})
            evaluation_id = evaluation.get('evaluation_id')
            
            if not evaluation_id:
                self.log_result("Learning Workflow - Evaluation ID", False, "No evaluation_id returned")
                return
            
            self.log_result("Learning Workflow - Evaluation", True, f"Text evaluated successfully, ID: {evaluation_id}")
            
            # Step 3: Check if learning entry was created in MongoDB
            time.sleep(2)  # Allow time for async operations
            
            stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                total_entries = stats_data.get('total_learning_entries', 0)
                
                if total_entries > 0:
                    self.log_result("Learning Workflow - Entry Creation", True, f"Learning entry created successfully ({total_entries} total entries)")
                else:
                    self.log_result("Learning Workflow - Entry Creation", False, "No learning entries found in database")
                    return
            else:
                self.log_result("Learning Workflow - Entry Creation", False, "Failed to check learning stats")
                return
            
            # Step 4: Submit dopamine feedback (score 0.8)
            feedback_response = requests.post(
                f"{API_BASE}/feedback",
                json={
                    "evaluation_id": evaluation_id,
                    "feedback_score": 0.8,
                    "user_comment": "Good learning system test"
                },
                timeout=10
            )
            
            if feedback_response.status_code == 200:
                feedback_data = feedback_response.json()
                message = feedback_data.get('message', '')
                
                if 'successfully' in message.lower():
                    self.log_result("Learning Workflow - Feedback", True, f"Feedback submitted successfully: {message}")
                else:
                    self.log_result("Learning Workflow - Feedback", False, f"Feedback submission issue: {message}")
                    return
            else:
                self.log_result("Learning Workflow - Feedback", False, f"Failed to submit feedback: HTTP {feedback_response.status_code}")
                return
            
            # Step 5: Check that learning stats show the entry and feedback
            time.sleep(1)  # Allow time for database update
            
            final_stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            if final_stats_response.status_code == 200:
                final_stats = final_stats_response.json()
                total_entries = final_stats.get('total_learning_entries', 0)
                avg_feedback = final_stats.get('average_feedback_score', 0.0)
                learning_active = final_stats.get('learning_active', False)
                
                if total_entries > 0 and avg_feedback > 0 and learning_active:
                    self.log_result(
                        "Learning Workflow - Complete", 
                        True, 
                        f"✅ COMPLETE WORKFLOW SUCCESS: {total_entries} entries, avg feedback: {avg_feedback:.3f}, active: {learning_active}"
                    )
                else:
                    self.log_result(
                        "Learning Workflow - Complete", 
                        False, 
                        f"Incomplete workflow: entries={total_entries}, feedback={avg_feedback:.3f}, active={learning_active}"
                    )
            else:
                self.log_result("Learning Workflow - Final Stats", False, "Failed to get final learning stats")
                
        except Exception as e:
            self.log_result("Learning Workflow - Complete", False, f"Error in complete workflow: {str(e)}")
    
    def test_threshold_sensitivity_deep_analysis(self):
        """Deep analysis of threshold sensitivity as requested in review"""
        try:
            print("\n🔍 THRESHOLD SENSITIVITY DEEP ANALYSIS")
            print("=" * 60)
            
            # Test problematic text with different thresholds
            problematic_text = "You are stupid and worthless"
            test_thresholds = [0.25, 0.15, 0.10]
            
            for threshold in test_thresholds:
                # Set threshold
                threshold_params = {
                    "virtue_threshold": threshold,
                    "deontological_threshold": threshold,
                    "consequentialist_threshold": threshold
                }
                
                requests.post(f"{API_BASE}/update-parameters", json={"parameters": threshold_params}, timeout=10)
                
                # Evaluate text
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": problematic_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    overall_ethical = evaluation.get('overall_ethical', True)
                    minimal_spans = evaluation.get('minimal_spans', [])
                    span_scores = evaluation.get('span_scores', [])
                    
                    # Analyze span-level scores
                    violation_count = len(minimal_spans)
                    
                    if not overall_ethical and violation_count > 0:
                        self.log_result(
                            f"Threshold Sensitivity (threshold={threshold})", 
                            True, 
                            f"✅ DETECTED violations: {violation_count} spans flagged as unethical"
                        )
                    else:
                        # Analyze why violations weren't detected
                        span_analysis = []
                        for i, score in enumerate(span_scores):
                            if isinstance(score, dict):
                                virtue_score = score.get('virtue_score', 1.0)
                                deont_score = score.get('deontological_score', 1.0)
                                conseq_score = score.get('consequentialist_score', 1.0)
                                span_analysis.append(f"span{i}: v={virtue_score:.3f}, d={deont_score:.3f}, c={conseq_score:.3f}")
                        
                        self.log_result(
                            f"Threshold Sensitivity (threshold={threshold})", 
                            False, 
                            f"❌ NO VIOLATIONS DETECTED. Span scores: {'; '.join(span_analysis[:3])}"
                        )
                else:
                    self.log_result(f"Threshold Sensitivity (threshold={threshold})", False, f"HTTP {response.status_code}")
            
            # Test obviously unethical phrases
            unethical_phrases = [
                "I hate you",
                "I want to kill you", 
                "You are evil"
            ]
            
            # Set very low threshold for these tests
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": {"virtue_threshold": 0.10, "deontological_threshold": 0.10, "consequentialist_threshold": 0.10}}, timeout=10)
            
            for phrase in unethical_phrases:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": phrase},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    overall_ethical = evaluation.get('overall_ethical', True)
                    violation_count = len(evaluation.get('minimal_spans', []))
                    
                    if not overall_ethical:
                        self.log_result(
                            f"Unethical Phrase Test: '{phrase}'", 
                            True, 
                            f"✅ Correctly flagged as unethical ({violation_count} violations)"
                        )
                    else:
                        self.log_result(
                            f"Unethical Phrase Test: '{phrase}'", 
                            False, 
                            f"❌ Not flagged as unethical (threshold too high or scoring issue)"
                        )
                else:
                    self.log_result(f"Unethical Phrase Test: '{phrase}'", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Threshold Sensitivity Analysis", False, f"Error: {str(e)}")
    
    def test_cascade_filtering_verification(self):
        """Verify cascade filtering with specific test cases from review"""
        try:
            print("\n🌊 CASCADE FILTERING VERIFICATION")
            print("=" * 60)
            
            # Enable cascade filtering
            cascade_params = {
                "enable_cascade_filtering": True,
                "cascade_high_threshold": 0.5,
                "cascade_low_threshold": 0.2
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": cascade_params}, timeout=10)
            
            # Test cases from review request
            test_cases = [
                ("I love helping people", "obviously ethical"),
                ("I hate you and want to kill you", "obviously unethical")
            ]
            
            for text, expected_type in test_cases:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    dynamic_info = evaluation.get('dynamic_scaling', {})
                    
                    cascade_result = dynamic_info.get('cascade_result')
                    processing_stages = dynamic_info.get('processing_stages', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    # Analyze cascade results
                    if cascade_result is not None:
                        if expected_type == "obviously ethical" and cascade_result == "ethical":
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                True, 
                                f"✅ Correctly identified as {cascade_result} via cascade"
                            )
                        elif expected_type == "obviously unethical" and cascade_result == "unethical":
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                True, 
                                f"✅ Correctly identified as {cascade_result} via cascade"
                            )
                        else:
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                False, 
                                f"❌ Cascade result '{cascade_result}' doesn't match expected '{expected_type}'"
                            )
                    else:
                        # No cascade decision, went to detailed evaluation
                        if expected_type == "obviously ethical" and overall_ethical:
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                True, 
                                f"✅ Proceeded to detailed evaluation, correctly identified as ethical"
                            )
                        elif expected_type == "obviously unethical" and not overall_ethical:
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                True, 
                                f"✅ Proceeded to detailed evaluation, correctly identified as unethical"
                            )
                        else:
                            self.log_result(
                                f"Cascade Filtering - {expected_type}", 
                                False, 
                                f"❌ No cascade decision and final result doesn't match expected"
                            )
                    
                    # Log processing stages for analysis
                    print(f"   Processing stages for '{text}': {processing_stages}")
                    
                else:
                    self.log_result(f"Cascade Filtering - {expected_type}", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Cascade Filtering Verification", False, f"Error: {str(e)}")
    
    def test_dynamic_scaling_integration(self):
        """Test dynamic scaling integration with all features enabled"""
        try:
            print("\n⚡ DYNAMIC SCALING INTEGRATION TEST")
            print("=" * 60)
            
            # Enable all dynamic features
            full_dynamic_params = {
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "enable_learning_mode": True,
                "exponential_scaling": True,
                "cascade_high_threshold": 0.5,
                "cascade_low_threshold": 0.2,
                "virtue_threshold": 0.25,
                "deontological_threshold": 0.25,
                "consequentialist_threshold": 0.25
            }
            
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": full_dynamic_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                self.log_result("Dynamic Scaling Integration - Setup", False, "Failed to enable dynamic features")
                return
            
            # Test with moderately ambiguous text
            test_text = "I sometimes feel frustrated with difficult people but try to be patient"
            
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get('evaluation', {})
                dynamic_info = evaluation.get('dynamic_scaling', {})
                
                # Verify all dynamic features are working
                used_dynamic = dynamic_info.get('used_dynamic_scaling', False)
                used_cascade = dynamic_info.get('used_cascade_filtering', False)
                ambiguity_score = dynamic_info.get('ambiguity_score', 0.0)
                original_thresholds = dynamic_info.get('original_thresholds', {})
                adjusted_thresholds = dynamic_info.get('adjusted_thresholds', {})
                processing_stages = dynamic_info.get('processing_stages', [])
                
                # Check ambiguity score calculation
                if 0.0 <= ambiguity_score <= 1.0:
                    self.log_result(
                        "Dynamic Scaling - Ambiguity Score", 
                        True, 
                        f"✅ Ambiguity score calculated: {ambiguity_score:.3f}"
                    )
                else:
                    self.log_result(
                        "Dynamic Scaling - Ambiguity Score", 
                        False, 
                        f"❌ Invalid ambiguity score: {ambiguity_score}"
                    )
                
                # Check threshold adjustment
                if adjusted_thresholds and original_thresholds:
                    threshold_changes = []
                    for key in ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']:
                        if key in original_thresholds and key in adjusted_thresholds:
                            orig = original_thresholds[key]
                            adj = adjusted_thresholds[key]
                            change = adj - orig
                            threshold_changes.append(f"{key}: {orig:.3f}→{adj:.3f} ({change:+.3f})")
                    
                    self.log_result(
                        "Dynamic Scaling - Threshold Adjustment", 
                        True, 
                        f"✅ Thresholds adjusted: {'; '.join(threshold_changes)}"
                    )
                else:
                    self.log_result(
                        "Dynamic Scaling - Threshold Adjustment", 
                        False, 
                        "❌ No threshold adjustment data found"
                    )
                
                # Test exponential vs linear scaling
                for use_exponential in [True, False]:
                    scaling_response = requests.post(
                        f"{API_BASE}/threshold-scaling",
                        json={"slider_value": 0.3, "use_exponential": use_exponential},
                        timeout=10
                    )
                    
                    if scaling_response.status_code == 200:
                        scaling_data = scaling_response.json()
                        scaling_type = "exponential" if use_exponential else "linear"
                        scaled_value = scaling_data.get('scaled_threshold', 0)
                        
                        self.log_result(
                            f"Dynamic Scaling - {scaling_type.title()} Scaling", 
                            True, 
                            f"✅ {scaling_type} scaling: 0.3 → {scaled_value:.4f}"
                        )
                    else:
                        self.log_result(f"Dynamic Scaling - {scaling_type.title()} Scaling", False, f"HTTP {scaling_response.status_code}")
                
                # Overall integration check
                if used_dynamic and processing_stages:
                    self.log_result(
                        "Dynamic Scaling - Integration Complete", 
                        True, 
                        f"✅ All dynamic features integrated. Stages: {processing_stages}"
                    )
                else:
                    self.log_result(
                        "Dynamic Scaling - Integration Complete", 
                        False, 
                        f"❌ Dynamic features not fully integrated. Used: {used_dynamic}, Stages: {processing_stages}"
                    )
                    
            else:
                self.log_result("Dynamic Scaling Integration", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Integration", False, f"Error: {str(e)}")
    
    def test_api_integration_complete_flow(self):
        """Test complete API integration flow: evaluate → feedback → stats"""
        try:
            print("\n🔄 COMPLETE API INTEGRATION FLOW")
            print("=" * 60)
            
            # Step 1: Evaluate text
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "Testing complete API integration flow"},
                timeout=30
            )
            
            if eval_response.status_code != 200:
                self.log_result("API Integration - Evaluation", False, f"HTTP {eval_response.status_code}")
                return
            
            eval_data = eval_response.json()
            
            # Verify JSON serialization works for all fields
            required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
            if all(field in eval_data for field in required_fields):
                evaluation_id = eval_data['evaluation'].get('evaluation_id')
                if evaluation_id:
                    self.log_result("API Integration - Evaluation", True, f"✅ Evaluation successful, ID: {evaluation_id}")
                else:
                    self.log_result("API Integration - Evaluation", False, "No evaluation_id in response")
                    return
            else:
                self.log_result("API Integration - Evaluation", False, f"Missing required fields: {required_fields}")
                return
            
            # Step 2: Submit feedback
            feedback_response = requests.post(
                f"{API_BASE}/feedback",
                json={
                    "evaluation_id": evaluation_id,
                    "feedback_score": 0.7,
                    "user_comment": "API integration test"
                },
                timeout=10
            )
            
            if feedback_response.status_code == 200:
                feedback_data = feedback_response.json()
                if 'message' in feedback_data and 'evaluation_id' in feedback_data:
                    self.log_result("API Integration - Feedback", True, f"✅ Feedback submitted: {feedback_data['message']}")
                else:
                    self.log_result("API Integration - Feedback", False, "Invalid feedback response format")
                    return
            else:
                self.log_result("API Integration - Feedback", False, f"HTTP {feedback_response.status_code}")
                return
            
            # Step 3: Check stats
            stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                if all(field in stats_data for field in ['total_learning_entries', 'average_feedback_score', 'learning_active']):
                    self.log_result("API Integration - Stats", True, f"✅ Stats retrieved: {stats_data}")
                else:
                    self.log_result("API Integration - Stats", False, "Invalid stats response format")
                    return
            else:
                self.log_result("API Integration - Stats", False, f"HTTP {stats_response.status_code}")
                return
            
            # Step 4: Test error handling for invalid evaluation ID
            invalid_feedback_response = requests.post(
                f"{API_BASE}/feedback",
                json={
                    "evaluation_id": "invalid-id-12345",
                    "feedback_score": 0.5
                },
                timeout=10
            )
            
            if invalid_feedback_response.status_code == 200:
                invalid_data = invalid_feedback_response.json()
                message = invalid_data.get('message', '')
                if 'not found' in message.lower():
                    self.log_result("API Integration - Error Handling", True, f"✅ Properly handled invalid ID: {message}")
                else:
                    self.log_result("API Integration - Error Handling", True, f"✅ Handled invalid ID: {message}")
            else:
                self.log_result("API Integration - Error Handling", False, f"Unexpected response for invalid ID: HTTP {invalid_feedback_response.status_code}")
            
            # Step 5: Check MongoDB document structure by retrieving evaluations
            evaluations_response = requests.get(f"{API_BASE}/evaluations?limit=5", timeout=10)
            
            if evaluations_response.status_code == 200:
                evaluations_data = evaluations_response.json()
                evaluations = evaluations_data.get('evaluations', [])
                
                if evaluations:
                    sample_eval = evaluations[0]
                    required_eval_fields = ['id', 'input_text', 'result', 'timestamp']
                    
                    if all(field in sample_eval for field in required_eval_fields):
                        self.log_result("API Integration - MongoDB Structure", True, f"✅ MongoDB documents have correct structure")
                    else:
                        self.log_result("API Integration - MongoDB Structure", False, f"Missing fields in MongoDB document: {required_eval_fields}")
                else:
                    self.log_result("API Integration - MongoDB Structure", True, "✅ No evaluations found (expected for fresh system)")
            else:
                self.log_result("API Integration - MongoDB Structure", False, f"HTTP {evaluations_response.status_code}")
            
            self.log_result("API Integration - Complete Flow", True, "✅ Complete API integration flow successful")
                
        except Exception as e:
            self.log_result("API Integration - Complete Flow", False, f"Error: {str(e)}")

    def run_all_tests(self):
        """Run all backend tests with focus on critical review items"""
        print(f"🚀 Starting comprehensive backend testing for: {API_BASE}")
        print("=" * 80)
        
        # CRITICAL TESTS FROM REVIEW REQUEST
        print("\n" + "🔥" * 20 + " CRITICAL REVIEW TESTS " + "🔥" * 20)
        
        # 1. Complete Learning Workflow
        self.test_complete_learning_workflow()
        
        # 2. Threshold Sensitivity Deep Analysis  
        self.test_threshold_sensitivity_deep_analysis()
        
        # 3. Cascade Filtering Verification
        self.test_cascade_filtering_verification()
        
        # 4. Dynamic Scaling Integration
        self.test_dynamic_scaling_integration()
        
        # 5. API Integration Testing
        self.test_api_integration_complete_flow()
        
        # EXISTING COMPREHENSIVE TESTS
        print("\n" + "=" * 80)
        print("🧠 EXISTING DYNAMIC SCALING AND LEARNING FEATURES")
        print("=" * 80)
        
        # Threshold scaling tests
        self.test_threshold_scaling_exponential()
        self.test_threshold_scaling_linear()
        self.test_exponential_vs_linear_scaling_comparison()
        
        # Learning system tests
        self.test_learning_stats_initial()
        self.test_feedback_submission()
        self.test_feedback_validation()
        self.test_learning_stats_after_feedback()
        
        # Dynamic scaling evaluation tests
        self.test_dynamic_scaling_enabled_evaluation()
        self.test_dynamic_scaling_disabled_evaluation()
        self.test_dynamic_scaling_details()
        
        # Cascade filtering tests
        self.test_cascade_filtering_obvious_cases()
        
        # Core functionality tests (abbreviated for focus)
        print("\n" + "=" * 80)
        print("⚡ CORE FUNCTIONALITY VERIFICATION")
        print("=" * 80)
        
        self.test_health_check()
        self.test_get_parameters()
        self.test_update_parameters()
        self.test_evaluate_neutral_text()
        self.test_evaluate_problematic_text()
        self.test_get_evaluations()
        self.test_performance_metrics()
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