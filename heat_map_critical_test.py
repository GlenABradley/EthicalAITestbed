#!/usr/bin/env python3
"""
Critical Heat-Map Functionality Testing for Ethical AI Developer Testbed
Tests the specific fixes mentioned in the review request:
1. Heat-map mock endpoint performance and structure
2. Main evaluation endpoint timeout handling
3. API health check
4. Data structure validation
5. Performance comparison between mock and real endpoints
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

class HeatMapCriticalTester:
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

    def test_api_health_evaluator_status(self):
        """Test /api/health endpoint for evaluator initialization status"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['status', 'evaluator_initialized', 'timestamp']
                
                if all(field in data for field in required_fields):
                    status = data['status']
                    evaluator_init = data['evaluator_initialized']
                    
                    if status == 'healthy' and evaluator_init:
                        self.log_result(
                            "API Health Check", 
                            True, 
                            f"Service healthy, evaluator initialized, response time: {response_time:.1f}ms",
                            {"status": status, "evaluator_initialized": evaluator_init}
                        )
                    else:
                        self.log_result(
                            "API Health Check", 
                            False, 
                            f"Service issues: status={status}, evaluator_init={evaluator_init}",
                            data
                        )
                else:
                    self.log_result("API Health Check", False, "Missing required fields in health response", data)
            else:
                self.log_result("API Health Check", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("API Health Check", False, f"Connection error: {str(e)}")

    def test_heat_map_mock_performance_short_text(self):
        """Test /api/heat-map-mock with short text for fast response (<1 second)"""
        try:
            test_text = "Hello world"
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response time (should be <1000ms)
                if response_time < 1000:
                    self.log_result(
                        "Heat-Map Mock - Short Text Performance", 
                        True, 
                        f"Fast response: {response_time:.1f}ms (target: <1000ms)",
                        {"response_time_ms": response_time, "text_length": len(test_text)}
                    )
                    return data
                else:
                    self.log_result(
                        "Heat-Map Mock - Short Text Performance", 
                        False, 
                        f"Slow response: {response_time:.1f}ms (target: <1000ms)"
                    )
            else:
                self.log_result("Heat-Map Mock - Short Text Performance", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Heat-Map Mock - Short Text Performance", False, f"Connection error: {str(e)}")
        
        return None

    def test_heat_map_mock_performance_medium_text(self):
        """Test /api/heat-map-mock with medium text for fast response"""
        try:
            test_text = "This is a medium length text that should still respond very quickly from the mock endpoint. " * 3
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response time (should be <1000ms)
                if response_time < 1000:
                    self.log_result(
                        "Heat-Map Mock - Medium Text Performance", 
                        True, 
                        f"Fast response: {response_time:.1f}ms (target: <1000ms)",
                        {"response_time_ms": response_time, "text_length": len(test_text)}
                    )
                    return data
                else:
                    self.log_result(
                        "Heat-Map Mock - Medium Text Performance", 
                        False, 
                        f"Slow response: {response_time:.1f}ms (target: <1000ms)"
                    )
            else:
                self.log_result("Heat-Map Mock - Medium Text Performance", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Heat-Map Mock - Medium Text Performance", False, f"Connection error: {str(e)}")
        
        return None

    def test_heat_map_mock_performance_long_text(self):
        """Test /api/heat-map-mock with long text for fast response"""
        try:
            test_text = "This is a longer text that tests the mock endpoint's ability to handle substantial content while maintaining fast response times. " * 10
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response time (should be <1000ms)
                if response_time < 1000:
                    self.log_result(
                        "Heat-Map Mock - Long Text Performance", 
                        True, 
                        f"Fast response: {response_time:.1f}ms (target: <1000ms)",
                        {"response_time_ms": response_time, "text_length": len(test_text)}
                    )
                    return data
                else:
                    self.log_result(
                        "Heat-Map Mock - Long Text Performance", 
                        False, 
                        f"Slow response: {response_time:.1f}ms (target: <1000ms)"
                    )
            else:
                self.log_result("Heat-Map Mock - Long Text Performance", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Heat-Map Mock - Long Text Performance", False, f"Connection error: {str(e)}")
        
        return None

    def test_heat_map_mock_empty_text_handling(self):
        """Test /api/heat-map-mock with empty text"""
        try:
            test_text = ""
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Should handle empty text gracefully
                evaluations = data.get('evaluations', {})
                if all(len(evaluations.get(span_type, {}).get('spans', [])) == 0 for span_type in ['short', 'medium', 'long', 'stochastic']):
                    self.log_result(
                        "Heat-Map Mock - Empty Text Handling", 
                        True, 
                        f"Empty text handled correctly: {response_time:.1f}ms, 0 spans generated"
                    )
                else:
                    self.log_result(
                        "Heat-Map Mock - Empty Text Handling", 
                        False, 
                        "Empty text generated unexpected spans",
                        data
                    )
            elif response.status_code == 422:
                self.log_result(
                    "Heat-Map Mock - Empty Text Handling", 
                    True, 
                    f"Empty text properly rejected: HTTP {response.status_code}"
                )
            else:
                self.log_result("Heat-Map Mock - Empty Text Handling", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Heat-Map Mock - Empty Text Handling", False, f"Connection error: {str(e)}")

    def test_heat_map_mock_special_characters(self):
        """Test /api/heat-map-mock with special characters and emojis"""
        try:
            test_text = "Testing émojis 🚀 and spëcial chäractërs @#$% with unicode 中文"
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response time and proper handling
                if response_time < 1000:
                    self.log_result(
                        "Heat-Map Mock - Special Characters", 
                        True, 
                        f"Special characters handled: {response_time:.1f}ms",
                        {"response_time_ms": response_time, "text_length": len(test_text)}
                    )
                else:
                    self.log_result(
                        "Heat-Map Mock - Special Characters", 
                        False, 
                        f"Slow response with special chars: {response_time:.1f}ms"
                    )
            else:
                self.log_result("Heat-Map Mock - Special Characters", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Heat-Map Mock - Special Characters", False, f"Connection error: {str(e)}")

    def validate_heat_map_data_structure(self, data: Dict, test_name: str):
        """Validate the heat-map data structure"""
        try:
            # Check top-level structure
            required_top_level = ['evaluations', 'overallGrades', 'textLength', 'originalEvaluation']
            missing_top = [field for field in required_top_level if field not in data]
            
            if missing_top:
                self.log_result(f"{test_name} - Data Structure", False, f"Missing top-level fields: {missing_top}")
                return False
            
            # Check evaluations structure
            evaluations = data['evaluations']
            required_eval_types = ['short', 'medium', 'long', 'stochastic']
            missing_eval_types = [eval_type for eval_type in required_eval_types if eval_type not in evaluations]
            
            if missing_eval_types:
                self.log_result(f"{test_name} - Data Structure", False, f"Missing evaluation types: {missing_eval_types}")
                return False
            
            # Check each evaluation type structure
            for eval_type in required_eval_types:
                eval_data = evaluations[eval_type]
                required_eval_fields = ['spans', 'averageScore', 'metadata']
                missing_eval_fields = [field for field in required_eval_fields if field not in eval_data]
                
                if missing_eval_fields:
                    self.log_result(f"{test_name} - Data Structure", False, f"Missing {eval_type} fields: {missing_eval_fields}")
                    return False
                
                # Check spans structure
                spans = eval_data['spans']
                for i, span in enumerate(spans):
                    required_span_fields = ['span', 'text', 'scores', 'uncertainty']
                    missing_span_fields = [field for field in required_span_fields if field not in span]
                    
                    if missing_span_fields:
                        self.log_result(f"{test_name} - Data Structure", False, f"Missing span fields in {eval_type}[{i}]: {missing_span_fields}")
                        return False
                    
                    # Check scores structure (V/A/C dimensions)
                    scores = span['scores']
                    required_dimensions = ['V', 'A', 'C']
                    missing_dimensions = [dim for dim in required_dimensions if dim not in scores]
                    
                    if missing_dimensions:
                        self.log_result(f"{test_name} - Data Structure", False, f"Missing dimensions in {eval_type}[{i}]: {missing_dimensions}")
                        return False
                    
                    # Validate score ranges (0.0-1.0)
                    for dim, score in scores.items():
                        if not (0.0 <= score <= 1.0):
                            self.log_result(f"{test_name} - Data Structure", False, f"Invalid score range in {eval_type}[{i}].{dim}: {score}")
                            return False
                    
                    # Validate uncertainty range (0.0-1.0)
                    uncertainty = span['uncertainty']
                    if not (0.0 <= uncertainty <= 1.0):
                        self.log_result(f"{test_name} - Data Structure", False, f"Invalid uncertainty range in {eval_type}[{i}]: {uncertainty}")
                        return False
                    
                    # Validate span positions
                    span_pos = span['span']
                    if not (isinstance(span_pos, list) and len(span_pos) == 2 and span_pos[0] <= span_pos[1]):
                        self.log_result(f"{test_name} - Data Structure", False, f"Invalid span position in {eval_type}[{i}]: {span_pos}")
                        return False
            
            # Check overallGrades structure
            overall_grades = data['overallGrades']
            for eval_type in required_eval_types:
                if eval_type not in overall_grades:
                    self.log_result(f"{test_name} - Data Structure", False, f"Missing grade for {eval_type}")
                    return False
                
                grade = overall_grades[eval_type]
                # Grade should be in format like "A+ (95%)" or "F (45%)"
                if not isinstance(grade, str) or '(' not in grade or ')' not in grade:
                    self.log_result(f"{test_name} - Data Structure", False, f"Invalid grade format for {eval_type}: {grade}")
                    return False
            
            # Check originalEvaluation structure
            original_eval = data['originalEvaluation']
            required_original_fields = ['input_text', 'overall_ethical', 'violation_count', 'processing_time', 'evaluation_id']
            missing_original_fields = [field for field in required_original_fields if field not in original_eval]
            
            if missing_original_fields:
                self.log_result(f"{test_name} - Data Structure", False, f"Missing originalEvaluation fields: {missing_original_fields}")
                return False
            
            self.log_result(f"{test_name} - Data Structure", True, "All data structure validation passed")
            return True
            
        except Exception as e:
            self.log_result(f"{test_name} - Data Structure", False, f"Structure validation error: {str(e)}")
            return False

    def test_heat_map_data_structure_validation(self):
        """Test heat-map mock data structure with various inputs"""
        test_cases = [
            ("Hello world", "Short Text"),
            ("This is a medium length text for testing purposes.", "Medium Text"),
            ("This is a longer text that should generate multiple spans across different categories for comprehensive testing of the heat-map data structure.", "Long Text")
        ]
        
        for text, description in test_cases:
            try:
                response = requests.post(
                    f"{API_BASE}/heat-map-mock",
                    json={"text": text},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.validate_heat_map_data_structure(data, f"Heat-Map Structure - {description}")
                else:
                    self.log_result(f"Heat-Map Structure - {description}", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_result(f"Heat-Map Structure - {description}", False, f"Connection error: {str(e)}")

    def test_main_evaluation_timeout_handling(self):
        """Test /api/evaluate endpoint with timeout handling (2 minutes)"""
        try:
            test_text = "This is a test to verify the main evaluation endpoint works within reasonable time limits"
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=120  # 2 minutes as mentioned in review
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                
                if all(field in data for field in required_fields):
                    evaluation = data['evaluation']
                    processing_time = evaluation.get('processing_time', response_time)
                    
                    if response_time < 120:  # Within 2 minute timeout
                        self.log_result(
                            "Main Evaluation Timeout Handling", 
                            True, 
                            f"Evaluation completed within timeout: {response_time:.1f}s (processing: {processing_time:.1f}s)"
                        )
                    else:
                        self.log_result(
                            "Main Evaluation Timeout Handling", 
                            False, 
                            f"Evaluation took too long: {response_time:.1f}s"
                        )
                else:
                    self.log_result("Main Evaluation Timeout Handling", False, "Missing required response fields", data)
            else:
                self.log_result("Main Evaluation Timeout Handling", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except requests.exceptions.Timeout:
            self.log_result("Main Evaluation Timeout Handling", False, "Request timed out after 2 minutes")
        except Exception as e:
            self.log_result("Main Evaluation Timeout Handling", False, f"Connection error: {str(e)}")

    def test_performance_comparison_mock_vs_real(self):
        """Compare response times between mock heat-map endpoint vs real heat-map-visualization endpoint"""
        try:
            test_text = "This is a test text for comparing performance between mock and real heat-map endpoints"
            
            # Test mock endpoint
            mock_start = time.time()
            mock_response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=10
            )
            mock_time = (time.time() - mock_start) * 1000
            
            # Test real endpoint (expect this to be much slower or timeout)
            real_start = time.time()
            try:
                real_response = requests.post(
                    f"{API_BASE}/heat-map-visualization",
                    json={"text": test_text},
                    timeout=30  # Shorter timeout since we expect it to be slow
                )
                real_time = (time.time() - real_start) * 1000
                real_success = real_response.status_code == 200
            except requests.exceptions.Timeout:
                real_time = 30000  # 30 seconds timeout
                real_success = False
                real_response = None
            
            # Analyze results
            if mock_response.status_code == 200:
                if real_success and real_response.status_code == 200:
                    # Both succeeded - compare times
                    speedup = real_time / mock_time if mock_time > 0 else float('inf')
                    self.log_result(
                        "Performance Comparison - Mock vs Real", 
                        True, 
                        f"Mock: {mock_time:.1f}ms, Real: {real_time:.1f}ms, Speedup: {speedup:.1f}x"
                    )
                else:
                    # Mock succeeded, real failed/timed out (expected)
                    self.log_result(
                        "Performance Comparison - Mock vs Real", 
                        True, 
                        f"Mock: {mock_time:.1f}ms (fast), Real: timed out/failed (expected for full evaluation)"
                    )
            else:
                self.log_result("Performance Comparison - Mock vs Real", False, f"Mock endpoint failed: HTTP {mock_response.status_code}")
                
        except Exception as e:
            self.log_result("Performance Comparison - Mock vs Real", False, f"Connection error: {str(e)}")

    def test_data_source_indicator_validation(self):
        """Test that mock data includes proper data source indicators"""
        try:
            test_text = "Test for data source indicator validation"
            
            response = requests.post(
                f"{API_BASE}/heat-map-mock",
                json={"text": test_text},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for data source indicators in metadata
                evaluations = data.get('evaluations', {})
                data_sources_found = []
                
                for eval_type in ['short', 'medium', 'long', 'stochastic']:
                    if eval_type in evaluations:
                        metadata = evaluations[eval_type].get('metadata', {})
                        dataset_source = metadata.get('dataset_source', '')
                        
                        if 'mock' in dataset_source.lower():
                            data_sources_found.append(f"{eval_type}: {dataset_source}")
                
                if data_sources_found:
                    self.log_result(
                        "Data Source Indicator Validation", 
                        True, 
                        f"Mock data source indicators found: {', '.join(data_sources_found)}"
                    )
                else:
                    self.log_result(
                        "Data Source Indicator Validation", 
                        False, 
                        "No mock data source indicators found in metadata"
                    )
            else:
                self.log_result("Data Source Indicator Validation", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Data Source Indicator Validation", False, f"Connection error: {str(e)}")

    def run_all_critical_tests(self):
        """Run all critical heat-map functionality tests"""
        print("🔥 CRITICAL HEAT-MAP FUNCTIONALITY TESTING")
        print("=" * 80)
        print(f"Testing against: {API_BASE}")
        print("=" * 80)
        
        # 1. API Health Check
        print("\n1. API HEALTH CHECK")
        print("-" * 40)
        self.test_api_health_evaluator_status()
        
        # 2. Heat-Map Mock Endpoint Performance Tests
        print("\n2. HEAT-MAP MOCK ENDPOINT PERFORMANCE")
        print("-" * 40)
        self.test_heat_map_mock_performance_short_text()
        self.test_heat_map_mock_performance_medium_text()
        self.test_heat_map_mock_performance_long_text()
        self.test_heat_map_mock_empty_text_handling()
        self.test_heat_map_mock_special_characters()
        
        # 3. Heat-Map Data Structure Validation
        print("\n3. HEAT-MAP DATA STRUCTURE VALIDATION")
        print("-" * 40)
        self.test_heat_map_data_structure_validation()
        
        # 4. Main Evaluation Endpoint Timeout
        print("\n4. MAIN EVALUATION ENDPOINT TIMEOUT HANDLING")
        print("-" * 40)
        self.test_main_evaluation_timeout_handling()
        
        # 5. Performance Comparison
        print("\n5. PERFORMANCE COMPARISON (MOCK VS REAL)")
        print("-" * 40)
        self.test_performance_comparison_mock_vs_real()
        
        # 6. Data Source Indicators
        print("\n6. DATA SOURCE INDICATOR VALIDATION")
        print("-" * 40)
        self.test_data_source_indicator_validation()
        
        # Summary
        print("\n" + "=" * 80)
        print("CRITICAL TESTING SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        if self.passed_tests:
            print(f"\n✅ PASSED TESTS:")
            for test in self.passed_tests:
                print(f"   - {test}")
        
        return {
            'total': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': (passed_count/total_tests)*100,
            'failed_tests': self.failed_tests,
            'passed_tests': self.passed_tests,
            'results': self.results
        }

if __name__ == "__main__":
    tester = HeatMapCriticalTester()
    results = tester.run_all_critical_tests()
    
    # Exit with error code if any tests failed
    sys.exit(0 if results['failed'] == 0 else 1)