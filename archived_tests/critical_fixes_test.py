#!/usr/bin/env python3
"""
CRITICAL FIXES VERIFICATION TEST
Tests the specific critical issues identified and fixed in the review request:
1. Improved Ethical Detection with enhanced thresholds (0.20)
2. Enhanced Cascade Filtering with new thresholds (high=0.35, low=0.15)
3. Mathematical Framework Validation
4. Integration Testing with learning mode
"""

import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class CriticalFixesTester:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: dict = None):
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
                print(f"   Details: {json.dumps(details, indent=2)}")

    def test_improved_ethical_detection_with_new_thresholds(self):
        """Test improved ethical detection with new default thresholds (0.20)"""
        print("\n🎯 TESTING IMPROVED ETHICAL DETECTION")
        print("=" * 60)
        
        try:
            # Set new default thresholds as mentioned in review (0.20)
            new_thresholds = {
                "virtue_threshold": 0.20,
                "deontological_threshold": 0.20,
                "consequentialist_threshold": 0.20,
                "enable_dynamic_scaling": False,  # Test static thresholds first
                "enable_cascade_filtering": False,
                "enable_learning_mode": False
            }
            
            response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": new_thresholds},
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_result("Threshold Setup", False, f"Failed to set thresholds: HTTP {response.status_code}")
                return
            
            # Test cases from review request
            test_cases = [
                ("You are stupid and worthless", "Should be flagged as unethical with 0.20 threshold"),
                ("I hate you", "Should be flagged with improved vectors"),
                ("I want to kill you", "Should be flagged with enhanced detection"),
                ("You are evil and should die", "Should be flagged with strengthened examples")
            ]
            
            for text, expectation in test_cases:
                eval_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=30
                )
                
                if eval_response.status_code == 200:
                    data = eval_response.json()
                    evaluation = data.get('evaluation', {})
                    overall_ethical = evaluation.get('overall_ethical', True)
                    minimal_spans = evaluation.get('minimal_spans', [])
                    span_scores = evaluation.get('span_scores', [])
                    
                    violation_count = len(minimal_spans)
                    
                    # Analyze span-level scores for detailed feedback
                    span_analysis = []
                    for i, score in enumerate(span_scores):
                        if isinstance(score, dict):
                            virtue_score = score.get('virtue_score', 1.0)
                            deont_score = score.get('deontological_score', 1.0)
                            conseq_score = score.get('consequentialist_score', 1.0)
                            span_analysis.append({
                                'span': i,
                                'virtue': virtue_score,
                                'deontological': deont_score,
                                'consequentialist': conseq_score,
                                'min_score': min(virtue_score, deont_score, conseq_score)
                            })
                    
                    if not overall_ethical and violation_count > 0:
                        self.log_result(
                            f"Ethical Detection - '{text[:20]}...'",
                            True,
                            f"✅ CORRECTLY FLAGGED as unethical ({violation_count} violations)",
                            {
                                'threshold': 0.20,
                                'violations': violation_count,
                                'span_analysis': span_analysis[:3]  # First 3 spans
                            }
                        )
                    else:
                        self.log_result(
                            f"Ethical Detection - '{text[:20]}...'",
                            False,
                            f"❌ NOT FLAGGED as unethical (should be with 0.20 threshold)",
                            {
                                'threshold': 0.20,
                                'overall_ethical': overall_ethical,
                                'violations': violation_count,
                                'span_analysis': span_analysis[:3],
                                'expectation': expectation
                            }
                        )
                else:
                    self.log_result(f"Ethical Detection - '{text[:20]}...'", False, f"HTTP {eval_response.status_code}")
                    
        except Exception as e:
            self.log_result("Improved Ethical Detection", False, f"Error: {str(e)}")

    def test_enhanced_cascade_filtering(self):
        """Test enhanced cascade filtering with new thresholds (high=0.35, low=0.15)"""
        print("\n🌊 TESTING ENHANCED CASCADE FILTERING")
        print("=" * 60)
        
        try:
            # Enable cascade filtering with new thresholds from review
            cascade_params = {
                "enable_cascade_filtering": True,
                "cascade_high_threshold": 0.35,
                "cascade_low_threshold": 0.15,
                "enable_dynamic_scaling": True,
                "virtue_threshold": 0.25,
                "deontological_threshold": 0.25,
                "consequentialist_threshold": 0.25
            }
            
            response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": cascade_params},
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_result("Cascade Setup", False, f"Failed to enable cascade: HTTP {response.status_code}")
                return
            
            # Test cases from review request
            test_cases = [
                ("I love helping people", "ethical", "Should be ethical via cascade"),
                ("I hate you and want to kill you", "unethical", "Should be unethical via cascade"),
                ("You are worthless garbage", "unethical", "Should be unethical via cascade")
            ]
            
            for text, expected_result, expectation in test_cases:
                eval_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=30
                )
                
                if eval_response.status_code == 200:
                    data = eval_response.json()
                    evaluation = data.get('evaluation', {})
                    dynamic_scaling = evaluation.get('dynamic_scaling', {})
                    
                    cascade_result = dynamic_scaling.get('cascade_result')
                    used_cascade = dynamic_scaling.get('used_cascade_filtering', False)
                    processing_stages = dynamic_scaling.get('processing_stages', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    # Check if cascade made the correct decision
                    if cascade_result is not None:
                        cascade_correct = (
                            (expected_result == "ethical" and cascade_result == "ethical") or
                            (expected_result == "unethical" and cascade_result == "unethical")
                        )
                        
                        if cascade_correct:
                            self.log_result(
                                f"Cascade Filtering - '{text[:25]}...'",
                                True,
                                f"✅ CORRECT cascade decision: {cascade_result}",
                                {
                                    'cascade_result': cascade_result,
                                    'expected': expected_result,
                                    'used_cascade': used_cascade,
                                    'processing_stages': processing_stages
                                }
                            )
                        else:
                            self.log_result(
                                f"Cascade Filtering - '{text[:25]}...'",
                                False,
                                f"❌ INCORRECT cascade decision: got {cascade_result}, expected {expected_result}",
                                {
                                    'cascade_result': cascade_result,
                                    'expected': expected_result,
                                    'overall_ethical': overall_ethical,
                                    'processing_stages': processing_stages
                                }
                            )
                    else:
                        # No cascade decision - check if final result is correct
                        final_correct = (
                            (expected_result == "ethical" and overall_ethical) or
                            (expected_result == "unethical" and not overall_ethical)
                        )
                        
                        if final_correct:
                            self.log_result(
                                f"Cascade Filtering - '{text[:25]}...'",
                                True,
                                f"✅ No cascade, but CORRECT final result: {'ethical' if overall_ethical else 'unethical'}",
                                {
                                    'cascade_result': None,
                                    'final_result': 'ethical' if overall_ethical else 'unethical',
                                    'expected': expected_result,
                                    'processing_stages': processing_stages
                                }
                            )
                        else:
                            self.log_result(
                                f"Cascade Filtering - '{text[:25]}...'",
                                False,
                                f"❌ No cascade AND incorrect final result: got {'ethical' if overall_ethical else 'unethical'}, expected {expected_result}",
                                {
                                    'cascade_result': None,
                                    'final_result': 'ethical' if overall_ethical else 'unethical',
                                    'expected': expected_result,
                                    'processing_stages': processing_stages
                                }
                            )
                else:
                    self.log_result(f"Cascade Filtering - '{text[:25]}...'", False, f"HTTP {eval_response.status_code}")
                    
        except Exception as e:
            self.log_result("Enhanced Cascade Filtering", False, f"Error: {str(e)}")

    def test_mathematical_framework_validation(self):
        """Test mathematical framework validation - virtue, deontological, consequentialist vectors"""
        print("\n🧮 TESTING MATHEMATICAL FRAMEWORK VALIDATION")
        print("=" * 60)
        
        try:
            # Test with various threshold levels to validate mathematical framework
            test_thresholds = [0.30, 0.20, 0.10]
            test_text = "You are stupid and worthless"
            
            for threshold in test_thresholds:
                # Set threshold
                params = {
                    "virtue_threshold": threshold,
                    "deontological_threshold": threshold,
                    "consequentialist_threshold": threshold,
                    "enable_dynamic_scaling": False
                }
                
                requests.post(f"{API_BASE}/update-parameters", json={"parameters": params}, timeout=10)
                
                # Evaluate
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": test_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    span_scores = evaluation.get('span_scores', [])
                    minimal_spans = evaluation.get('minimal_spans', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    # Analyze mathematical framework
                    framework_analysis = {
                        'threshold': threshold,
                        'violations_detected': len(minimal_spans),
                        'overall_ethical': overall_ethical,
                        'span_details': []
                    }
                    
                    for i, score in enumerate(span_scores[:5]):  # First 5 spans
                        if isinstance(score, dict):
                            virtue_score = score.get('virtue_score', 1.0)
                            deont_score = score.get('deontological_score', 1.0)
                            conseq_score = score.get('consequentialist_score', 1.0)
                            
                            framework_analysis['span_details'].append({
                                'span': i,
                                'virtue': round(virtue_score, 4),
                                'deontological': round(deont_score, 4),
                                'consequentialist': round(conseq_score, 4),
                                'below_threshold': {
                                    'virtue': virtue_score < threshold,
                                    'deontological': deont_score < threshold,
                                    'consequentialist': conseq_score < threshold
                                }
                            })
                    
                    # Check if framework is working correctly
                    violations_expected = any(
                        any(detail['below_threshold'].values()) 
                        for detail in framework_analysis['span_details']
                    )
                    
                    framework_working = (
                        (violations_expected and len(minimal_spans) > 0) or
                        (not violations_expected and len(minimal_spans) == 0)
                    )
                    
                    if framework_working:
                        self.log_result(
                            f"Mathematical Framework (threshold={threshold})",
                            True,
                            f"✅ Framework working correctly: {len(minimal_spans)} violations detected",
                            framework_analysis
                        )
                    else:
                        self.log_result(
                            f"Mathematical Framework (threshold={threshold})",
                            False,
                            f"❌ Framework inconsistency: expected violations but got {len(minimal_spans)}",
                            framework_analysis
                        )
                else:
                    self.log_result(f"Mathematical Framework (threshold={threshold})", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Mathematical Framework Validation", False, f"Error: {str(e)}")

    def test_integration_with_learning_mode(self):
        """Test complete workflow with learning mode enabled"""
        print("\n🧠 TESTING INTEGRATION WITH LEARNING MODE")
        print("=" * 60)
        
        try:
            # Enable learning mode with improved parameters
            learning_params = {
                "enable_learning_mode": True,
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "virtue_threshold": 0.20,
                "deontological_threshold": 0.20,
                "consequentialist_threshold": 0.20,
                "cascade_high_threshold": 0.35,
                "cascade_low_threshold": 0.15
            }
            
            response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": learning_params},
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_result("Learning Integration Setup", False, f"Failed to enable learning: HTTP {response.status_code}")
                return
            
            # Test with problematic text that should trigger learning
            test_text = "You are stupid and worthless"
            
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if eval_response.status_code != 200:
                self.log_result("Learning Integration - Evaluation", False, f"HTTP {eval_response.status_code}")
                return
            
            eval_data = eval_response.json()
            evaluation = eval_data.get('evaluation', {})
            evaluation_id = evaluation.get('evaluation_id')
            dynamic_scaling = evaluation.get('dynamic_scaling', {})
            
            if not evaluation_id:
                self.log_result("Learning Integration - Evaluation ID", False, "No evaluation_id returned")
                return
            
            # Check if learning entry was created
            time.sleep(2)  # Allow async operations
            
            stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                total_entries = stats_data.get('total_learning_entries', 0)
                learning_active = stats_data.get('learning_active', False)
                
                if total_entries > 0 and learning_active:
                    self.log_result(
                        "Learning Integration - Entry Creation",
                        True,
                        f"✅ Learning entry created: {total_entries} total entries",
                        {
                            'evaluation_id': evaluation_id,
                            'total_entries': total_entries,
                            'learning_active': learning_active,
                            'dynamic_scaling_used': dynamic_scaling.get('used_dynamic_scaling', False)
                        }
                    )
                    
                    # Test feedback submission
                    feedback_response = requests.post(
                        f"{API_BASE}/feedback",
                        json={
                            "evaluation_id": evaluation_id,
                            "feedback_score": 0.2,  # Poor score for unethical text
                            "user_comment": "This should be flagged as unethical"
                        },
                        timeout=10
                    )
                    
                    if feedback_response.status_code == 200:
                        feedback_data = feedback_response.json()
                        if 'successfully' in feedback_data.get('message', '').lower():
                            self.log_result(
                                "Learning Integration - Feedback",
                                True,
                                "✅ Feedback submitted successfully",
                                {'feedback_score': 0.2, 'message': feedback_data.get('message')}
                            )
                            
                            # Verify updated stats
                            time.sleep(1)
                            final_stats_response = requests.get(f"{API_BASE}/learning-stats", timeout=10)
                            if final_stats_response.status_code == 200:
                                final_stats = final_stats_response.json()
                                avg_feedback = final_stats.get('average_feedback_score', 0.0)
                                
                                self.log_result(
                                    "Learning Integration - Complete Workflow",
                                    True,
                                    f"✅ COMPLETE INTEGRATION SUCCESS: avg feedback: {avg_feedback:.3f}",
                                    final_stats
                                )
                            else:
                                self.log_result("Learning Integration - Final Stats", False, "Failed to get final stats")
                        else:
                            self.log_result("Learning Integration - Feedback", False, f"Feedback issue: {feedback_data}")
                    else:
                        self.log_result("Learning Integration - Feedback", False, f"HTTP {feedback_response.status_code}")
                else:
                    self.log_result("Learning Integration - Entry Creation", False, f"No learning entries created: {stats_data}")
            else:
                self.log_result("Learning Integration - Stats Check", False, f"HTTP {stats_response.status_code}")
                
        except Exception as e:
            self.log_result("Learning Integration", False, f"Error: {str(e)}")

    def test_json_serialization_working(self):
        """Test that JSON serialization is working correctly"""
        print("\n📄 TESTING JSON SERIALIZATION")
        print("=" * 60)
        
        try:
            # Test evaluations endpoint
            response = requests.get(f"{API_BASE}/evaluations", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                evaluations = data.get('evaluations', [])
                
                if evaluations:
                    # Check that ObjectId fields are properly serialized
                    sample_eval = evaluations[0]
                    has_object_id_issues = False
                    
                    for key, value in sample_eval.items():
                        if isinstance(value, dict) and 'ObjectId' in str(value):
                            has_object_id_issues = True
                            break
                    
                    if not has_object_id_issues:
                        self.log_result(
                            "JSON Serialization - Evaluations",
                            True,
                            f"✅ Proper JSON serialization: {len(evaluations)} evaluations retrieved",
                            {'sample_keys': list(sample_eval.keys())[:5]}
                        )
                    else:
                        self.log_result(
                            "JSON Serialization - Evaluations",
                            False,
                            "❌ ObjectId serialization issues detected",
                            {'sample_eval': sample_eval}
                        )
                else:
                    self.log_result("JSON Serialization - Evaluations", True, "No evaluations to test (empty database)")
            else:
                self.log_result("JSON Serialization - Evaluations", False, f"HTTP {response.status_code}")
            
            # Test calibration tests endpoint
            response = requests.get(f"{API_BASE}/calibration-tests", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tests = data.get('tests', [])
                
                self.log_result(
                    "JSON Serialization - Calibration Tests",
                    True,
                    f"✅ Calibration tests serialization working: {len(tests)} tests"
                )
            else:
                self.log_result("JSON Serialization - Calibration Tests", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("JSON Serialization", False, f"Error: {str(e)}")

    def run_all_critical_tests(self):
        """Run all critical fix verification tests"""
        print("🔥 CRITICAL FIXES VERIFICATION - FOCUSED TESTING")
        print("=" * 80)
        print("Testing specific critical issues identified and fixed:")
        print("1. Improved Ethical Detection with enhanced thresholds (0.20)")
        print("2. Enhanced Cascade Filtering with new thresholds (high=0.35, low=0.15)")
        print("3. Mathematical Framework Validation")
        print("4. Integration Testing with learning mode enabled")
        print("=" * 80)
        
        # Run all critical tests
        self.test_improved_ethical_detection_with_new_thresholds()
        self.test_enhanced_cascade_filtering()
        self.test_mathematical_framework_validation()
        self.test_integration_with_learning_mode()
        self.test_json_serialization_working()
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 CRITICAL FIXES VERIFICATION SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        if self.passed_tests:
            print("\n✅ PASSED TESTS:")
            for test in self.passed_tests:
                print(f"  - {test}")
        
        # Critical assessment
        critical_issues = []
        for result in self.results:
            if not result['success'] and any(keyword in result['test'].lower() for keyword in ['ethical detection', 'cascade filtering', 'mathematical framework']):
                critical_issues.append(result['test'])
        
        if critical_issues:
            print(f"\n🚨 CRITICAL SAFETY ISSUES FOUND: {len(critical_issues)}")
            print("These issues affect system safety and must be resolved before production:")
            for issue in critical_issues:
                print(f"  - {issue}")
        else:
            print("\n✅ NO CRITICAL SAFETY ISSUES DETECTED")
            print("System appears ready for production use with improved safety measures.")
        
        return {
            'total_tests': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'critical_issues': critical_issues,
            'results': self.results
        }

if __name__ == "__main__":
    tester = CriticalFixesTester()
    results = tester.run_all_critical_tests()
    
    # Exit with error code if critical issues found
    if results['critical_issues']:
        sys.exit(1)
    else:
        sys.exit(0)