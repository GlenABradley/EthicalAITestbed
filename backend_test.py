#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Ethical AI Developer Testbed
Focus: Testing the fixed Ethical AI Developer Testbed backend after EthicalSpan has_violation() compatibility fix

This test suite specifically validates:
1. Primary Evaluation Endpoint (/api/evaluate) with various text inputs
2. Core Functionality - unethical content flagging and ethical content passing
3. Response Structure validation
4. Performance and system stability
5. Integration between optimized and original evaluation engines
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

# Get backend URL from environment - use localhost for testing
BACKEND_URL = 'http://localhost:8001'  # Use localhost for direct backend testing
API_BASE = f"{BACKEND_URL}/api"

class EthicalEvaluationTester:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        self.passed_tests = []
        self.performance_metrics = []
        
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
            start_time = time.time()
            response = requests.get(f"{API_BASE}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                evaluator_initialized = data.get('evaluator_initialized', False)
                db_connected = data.get('database_connected', False)
                
                if evaluator_initialized and db_connected:
                    self.log_result(
                        "Health Check", 
                        True, 
                        f"Service healthy, evaluator initialized, DB connected ({response_time:.3f}s)",
                        {"response_time": response_time, "data": data}
                    )
                else:
                    self.log_result(
                        "Health Check", 
                        False, 
                        f"Service issues: evaluator={evaluator_initialized}, db={db_connected}",
                        {"data": data}
                    )
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Health Check", False, f"Request failed: {str(e)}")
    
    def test_parameters_endpoint(self):
        """Test /api/parameters endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/parameters", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['virtue_threshold', 'deontological_threshold', 'consequentialist_threshold']
                
                if all(field in data for field in required_fields):
                    self.log_result(
                        "Parameters Endpoint", 
                        True, 
                        f"Parameters retrieved successfully ({response_time:.3f}s)",
                        {"response_time": response_time, "parameters": data}
                    )
                else:
                    self.log_result(
                        "Parameters Endpoint", 
                        False, 
                        "Missing required parameter fields",
                        {"data": data, "required": required_fields}
                    )
            else:
                self.log_result("Parameters Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Parameters Endpoint", False, f"Request failed: {str(e)}")
    
    def test_learning_stats_endpoint(self):
        """Test /api/learning-stats endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['total_evaluations', 'total_feedback', 'learning_enabled']
                
                if all(field in data for field in required_fields):
                    self.log_result(
                        "Learning Stats Endpoint", 
                        True, 
                        f"Learning stats retrieved successfully ({response_time:.3f}s)",
                        {"response_time": response_time, "stats": data}
                    )
                else:
                    self.log_result(
                        "Learning Stats Endpoint", 
                        False, 
                        "Missing required stats fields",
                        {"data": data, "required": required_fields}
                    )
            else:
                self.log_result("Learning Stats Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Learning Stats Endpoint", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_unethical_content(self):
        """Test /api/evaluate with clearly unethical content"""
        unethical_text = "You are stupid and worthless, and you should die"
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": unethical_text},
                timeout=120  # Allow up to 2 minutes for evaluation
            )
            response_time = time.time() - start_time
            self.performance_metrics.append(("unethical_evaluation", response_time))
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                if not all(field in data for field in required_fields):
                    self.log_result(
                        "Unethical Content Evaluation - Structure", 
                        False, 
                        "Missing required response fields",
                        {"data": data, "required": required_fields}
                    )
                    return
                
                # Check evaluation details
                evaluation = data.get('evaluation', {})
                delta_summary = data.get('delta_summary', {})
                
                overall_ethical = evaluation.get('overall_ethical', True)
                violation_count = len(evaluation.get('minimal_spans', []))
                ethical_status = delta_summary.get('ethical_status', True)
                
                # This should be flagged as unethical
                if not overall_ethical and violation_count > 0:
                    self.log_result(
                        "Unethical Content Evaluation - Detection", 
                        True, 
                        f"Correctly flagged as unethical: {violation_count} violations detected ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "ethical_status": ethical_status,
                            "engine_version": delta_summary.get('engine_version', 'unknown')
                        }
                    )
                else:
                    self.log_result(
                        "Unethical Content Evaluation - Detection", 
                        False, 
                        f"CRITICAL: Failed to flag unethical content (overall_ethical={overall_ethical}, violations={violation_count})",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "evaluation": evaluation,
                            "text": unethical_text
                        }
                    )
                
                # Check response structure completeness
                self.log_result(
                    "Unethical Content Evaluation - Structure", 
                    True, 
                    "Response structure complete with all required fields",
                    {"response_time": response_time, "structure_valid": True}
                )
                
            else:
                self.log_result(
                    "Unethical Content Evaluation", 
                    False, 
                    f"HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Unethical Content Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_ethical_content(self):
        """Test /api/evaluate with clearly ethical content"""
        ethical_text = "Thank you for your help, I appreciate it"
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ethical_text},
                timeout=120  # Allow up to 2 minutes for evaluation
            )
            response_time = time.time() - start_time
            self.performance_metrics.append(("ethical_evaluation", response_time))
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ['evaluation', 'clean_text', 'explanation', 'delta_summary']
                if not all(field in data for field in required_fields):
                    self.log_result(
                        "Ethical Content Evaluation - Structure", 
                        False, 
                        "Missing required response fields",
                        {"data": data, "required": required_fields}
                    )
                    return
                
                # Check evaluation details
                evaluation = data.get('evaluation', {})
                delta_summary = data.get('delta_summary', {})
                
                overall_ethical = evaluation.get('overall_ethical', False)
                violation_count = len(evaluation.get('minimal_spans', []))
                ethical_status = delta_summary.get('ethical_status', False)
                
                # This should pass as ethical
                if overall_ethical and violation_count == 0:
                    self.log_result(
                        "Ethical Content Evaluation - Detection", 
                        True, 
                        f"Correctly passed as ethical: {violation_count} violations detected ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "ethical_status": ethical_status,
                            "engine_version": delta_summary.get('engine_version', 'unknown')
                        }
                    )
                else:
                    self.log_result(
                        "Ethical Content Evaluation - Detection", 
                        False, 
                        f"Incorrectly flagged ethical content (overall_ethical={overall_ethical}, violations={violation_count})",
                        {
                            "response_time": response_time,
                            "overall_ethical": overall_ethical,
                            "violation_count": violation_count,
                            "evaluation": evaluation,
                            "text": ethical_text
                        }
                    )
                
                # Check response structure completeness
                self.log_result(
                    "Ethical Content Evaluation - Structure", 
                    True, 
                    "Response structure complete with all required fields",
                    {"response_time": response_time, "structure_valid": True}
                )
                
            else:
                self.log_result(
                    "Ethical Content Evaluation", 
                    False, 
                    f"HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Ethical Content Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_empty_text(self):
        """Test /api/evaluate with empty text"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": ""},
                timeout=30
            )
            response_time = time.time() - start_time
            
            # Should return 400 Bad Request for empty text
            if response.status_code == 400:
                self.log_result(
                    "Empty Text Evaluation", 
                    True, 
                    f"Correctly rejected empty text with HTTP 400 ({response_time:.3f}s)",
                    {"response_time": response_time, "status_code": response.status_code}
                )
            else:
                self.log_result(
                    "Empty Text Evaluation", 
                    False, 
                    f"Unexpected response for empty text: HTTP {response.status_code}",
                    {"response": response.text, "response_time": response_time}
                )
                
        except Exception as e:
            self.log_result("Empty Text Evaluation", False, f"Request failed: {str(e)}")
    
    def test_evaluation_with_edge_cases(self):
        """Test /api/evaluate with edge cases"""
        edge_cases = [
            ("Single word", "Hello"),
            ("Numbers only", "12345"),
            ("Special characters", "!@#$%^&*()"),
            ("Mixed content", "Hello! How are you? 123 @#$")
        ]
        
        for case_name, text in edge_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        True, 
                        f"Successfully processed '{text}' ({response_time:.3f}s)",
                        {
                            "response_time": response_time,
                            "overall_ethical": evaluation.get('overall_ethical', 'unknown'),
                            "violation_count": len(evaluation.get('minimal_spans', []))
                        }
                    )
                else:
                    self.log_result(
                        f"Edge Case - {case_name}", 
                        False, 
                        f"Failed to process '{text}': HTTP {response.status_code}",
                        {"response": response.text, "response_time": response_time}
                    )
                    
            except Exception as e:
                self.log_result(f"Edge Case - {case_name}", False, f"Request failed: {str(e)}")
    
    def test_performance_stats_endpoint(self):
        """Test /api/performance-stats endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE}/performance-stats", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for optimization status
                optimization_available = data.get('optimization_available', False)
                
                self.log_result(
                    "Performance Stats Endpoint", 
                    True, 
                    f"Performance stats retrieved, optimization_available={optimization_available} ({response_time:.3f}s)",
                    {"response_time": response_time, "stats": data}
                )
            else:
                self.log_result("Performance Stats Endpoint", False, f"HTTP {response.status_code}", {"response": response.text})
                
        except Exception as e:
            self.log_result("Performance Stats Endpoint", False, f"Request failed: {str(e)}")
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        try:
            # Test the complete workflow: health -> parameters -> evaluate -> stats
            workflow_start = time.time()
            
            # 1. Health check
            health_response = requests.get(f"{API_BASE}/health", timeout=10)
            if health_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Health check failed in workflow")
                return
            
            # 2. Get parameters
            params_response = requests.get(f"{API_BASE}/parameters", timeout=10)
            if params_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Parameters retrieval failed in workflow")
                return
            
            # 3. Evaluate text
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "This is a test message for integration workflow"},
                timeout=60
            )
            if eval_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Evaluation failed in workflow")
                return
            
            # 4. Get learning stats
            stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            if stats_response.status_code != 200:
                self.log_result("Integration Workflow", False, "Learning stats failed in workflow")
                return
            
            workflow_time = time.time() - workflow_start
            
            self.log_result(
                "Integration Workflow", 
                True, 
                f"Complete workflow successful ({workflow_time:.3f}s total)",
                {"workflow_time": workflow_time, "steps_completed": 4}
            )
            
        except Exception as e:
            self.log_result("Integration Workflow", False, f"Workflow failed: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive Backend Testing for Ethical AI Developer Testbed")
        print(f"🔗 Testing backend at: {BACKEND_URL}")
        print("=" * 80)
        
        # Core endpoint tests
        self.test_health_check()
        self.test_parameters_endpoint()
        self.test_learning_stats_endpoint()
        
        print("\n📊 EVALUATION SYSTEM TESTS (Focus: EthicalSpan has_violation() fix)")
        print("-" * 60)
        
        # Primary evaluation tests (focus of this testing session)
        self.test_evaluation_with_unethical_content()
        self.test_evaluation_with_ethical_content()
        self.test_evaluation_with_empty_text()
        self.test_evaluation_with_edge_cases()
        
        print("\n⚡ PERFORMANCE & INTEGRATION TESTS")
        print("-" * 40)
        
        # Performance and integration tests
        self.test_performance_stats_endpoint()
        self.test_integration_workflow()
        
        # Summary
        print("\n" + "=" * 80)
        print("📋 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.performance_metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            for metric_name, response_time in self.performance_metrics:
                print(f"   {metric_name}: {response_time:.3f}s")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test_name in self.failed_tests:
                print(f"   - {test_name}")
        
        print("\n🎯 FOCUS AREA RESULTS (EthicalSpan has_violation() fix):")
        evaluation_tests = [t for t in self.results if 'Evaluation' in t['test']]
        eval_passed = len([t for t in evaluation_tests if t['success']])
        eval_total = len(evaluation_tests)
        
        if eval_total > 0:
            print(f"   Evaluation Tests: {eval_passed}/{eval_total} passed ({(eval_passed/eval_total)*100:.1f}%)")
            
            # Check if critical functionality is working
            unethical_detection = any(t['success'] for t in evaluation_tests if 'Unethical Content' in t['test'] and 'Detection' in t['test'])
            ethical_detection = any(t['success'] for t in evaluation_tests if 'Ethical Content' in t['test'] and 'Detection' in t['test'])
            
            if unethical_detection and ethical_detection:
                print("   ✅ CRITICAL: Both ethical and unethical content detection working")
            elif unethical_detection:
                print("   ⚠️  PARTIAL: Unethical detection working, ethical detection issues")
            elif ethical_detection:
                print("   ⚠️  PARTIAL: Ethical detection working, unethical detection issues")
            else:
                print("   ❌ CRITICAL: Both ethical and unethical detection failing")
        
        return passed_count == total_tests

if __name__ == "__main__":
    tester = EthicalEvaluationTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)