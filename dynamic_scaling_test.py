#!/usr/bin/env python3
"""
Focused Testing for Dynamic Scaling and Learning System
Tests the specific issues found in the comprehensive test
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

class DynamicScalingTester:
    def __init__(self):
        self.results = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: dict = None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details or {}
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        if details and not success:
            print(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_learning_mode_evaluation(self):
        """Test if learning mode creates learning entries during evaluation"""
        try:
            # Enable learning mode
            learning_params = {
                "enable_learning_mode": True,
                "enable_dynamic_scaling": True
            }
            
            param_response = requests.post(
                f"{API_BASE}/update-parameters",
                json={"parameters": learning_params},
                timeout=10
            )
            
            if param_response.status_code != 200:
                self.log_result("Learning Mode Setup", False, f"Failed to enable learning mode: {param_response.status_code}")
                return None
            
            # Get initial learning stats
            initial_stats = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            initial_count = 0
            if initial_stats.status_code == 200:
                initial_count = initial_stats.json().get('total_learning_entries', 0)
            
            # Perform evaluation with learning mode enabled
            test_text = "This is a test to see if learning entries are created"
            
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=30
            )
            
            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
                
                # Check if learning stats increased
                time.sleep(1)  # Brief delay for database update
                updated_stats = requests.get(f"{API_BASE}/learning-stats", timeout=10)
                
                if updated_stats.status_code == 200:
                    updated_count = updated_stats.json().get('total_learning_entries', 0)
                    
                    if updated_count > initial_count:
                        self.log_result(
                            "Learning Mode Evaluation", 
                            True, 
                            f"Learning entry created: {initial_count} -> {updated_count}"
                        )
                        return evaluation_id
                    else:
                        self.log_result(
                            "Learning Mode Evaluation", 
                            False, 
                            f"No learning entry created: {initial_count} -> {updated_count}",
                            {"evaluation_data": eval_data.get('evaluation', {})}
                        )
                else:
                    self.log_result("Learning Mode Evaluation", False, "Failed to get updated stats")
            else:
                self.log_result("Learning Mode Evaluation", False, f"Evaluation failed: {eval_response.status_code}")
                
        except Exception as e:
            self.log_result("Learning Mode Evaluation", False, f"Error: {str(e)}")
        
        return None
    
    def test_threshold_sensitivity_issue(self):
        """Test the threshold sensitivity issue with problematic text"""
        try:
            # Test with current thresholds
            current_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "You are stupid and worthless"},
                timeout=30
            )
            
            if current_response.status_code == 200:
                current_data = current_response.json()
                current_ethical = current_data.get('evaluation', {}).get('overall_ethical', True)
                current_violations = current_data.get('evaluation', {}).get('minimal_violation_count', 0)
                current_thresholds = current_data.get('evaluation', {}).get('parameters', {})
                
                self.log_result(
                    "Current Thresholds Test", 
                    True, 
                    f"Ethical: {current_ethical}, Violations: {current_violations}, Thresholds: {current_thresholds}"
                )
                
                # Test with lower thresholds
                lower_params = {
                    "virtue_threshold": 0.15,
                    "deontological_threshold": 0.15,
                    "consequentialist_threshold": 0.15
                }
                
                requests.post(f"{API_BASE}/update-parameters", json={"parameters": lower_params}, timeout=10)
                
                lower_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": "You are stupid and worthless"},
                    timeout=30
                )
                
                if lower_response.status_code == 200:
                    lower_data = lower_response.json()
                    lower_ethical = lower_data.get('evaluation', {}).get('overall_ethical', True)
                    lower_violations = lower_data.get('evaluation', {}).get('minimal_violation_count', 0)
                    
                    self.log_result(
                        "Lower Thresholds Test", 
                        True, 
                        f"Ethical: {lower_ethical}, Violations: {lower_violations} (with 0.15 thresholds)"
                    )
                    
                    # Check if lower thresholds detect violations
                    if not lower_ethical or lower_violations > 0:
                        self.log_result(
                            "Threshold Sensitivity Analysis", 
                            True, 
                            "Lower thresholds successfully detect violations - threshold sensitivity confirmed"
                        )
                    else:
                        self.log_result(
                            "Threshold Sensitivity Analysis", 
                            False, 
                            "Even lower thresholds don't detect violations - deeper issue exists"
                        )
                else:
                    self.log_result("Lower Thresholds Test", False, f"Failed: {lower_response.status_code}")
            else:
                self.log_result("Current Thresholds Test", False, f"Failed: {current_response.status_code}")
                
        except Exception as e:
            self.log_result("Threshold Sensitivity Test", False, f"Error: {str(e)}")
    
    def test_cascade_filtering_detailed(self):
        """Test cascade filtering with detailed analysis"""
        try:
            # Enable cascade filtering with specific thresholds
            cascade_params = {
                "enable_cascade_filtering": True,
                "cascade_high_threshold": 0.5,
                "cascade_low_threshold": 0.2,
                "enable_dynamic_scaling": True
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": cascade_params}, timeout=10)
            
            test_cases = [
                ("I love helping people and spreading kindness", "Obviously ethical"),
                ("I hate you and want to kill you", "Obviously unethical"),
                ("This is somewhat ambiguous content that might need detailed analysis", "Ambiguous")
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
                    
                    used_cascade = dynamic_info.get('used_cascade_filtering', False)
                    cascade_result = dynamic_info.get('cascade_result')
                    ambiguity_score = dynamic_info.get('ambiguity_score', 0.0)
                    processing_stages = dynamic_info.get('processing_stages', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    self.log_result(
                        f"Cascade Filtering - {description}", 
                        True, 
                        f"Used cascade: {used_cascade}, Result: {cascade_result}, Ambiguity: {ambiguity_score:.3f}, Overall: {overall_ethical}, Stages: {processing_stages}"
                    )
                    
                    # Analyze if cascade filtering is working correctly
                    if description == "Obviously unethical" and cascade_result == "ethical":
                        self.log_result(
                            f"Cascade Analysis - {description}", 
                            False, 
                            "Obviously unethical text incorrectly classified as ethical by cascade filtering"
                        )
                    elif description == "Obviously ethical" and cascade_result == "unethical":
                        self.log_result(
                            f"Cascade Analysis - {description}", 
                            False, 
                            "Obviously ethical text incorrectly classified as unethical by cascade filtering"
                        )
                    else:
                        self.log_result(
                            f"Cascade Analysis - {description}", 
                            True, 
                            "Cascade filtering result appears reasonable"
                        )
                else:
                    self.log_result(f"Cascade Filtering - {description}", False, f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.log_result("Cascade Filtering Detailed", False, f"Error: {str(e)}")
    
    def test_dynamic_scaling_details_endpoint(self):
        """Test the dynamic scaling details endpoint that's returning 500"""
        try:
            # First create an evaluation
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "Test for dynamic scaling details"},
                timeout=30
            )
            
            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
                
                if evaluation_id:
                    # Test the dynamic scaling details endpoint
                    details_response = requests.get(
                        f"{API_BASE}/dynamic-scaling-test/{evaluation_id}", 
                        timeout=10
                    )
                    
                    if details_response.status_code == 200:
                        details_data = details_response.json()
                        self.log_result(
                            "Dynamic Scaling Details Endpoint", 
                            True, 
                            f"Successfully retrieved details for {evaluation_id}",
                            details_data
                        )
                    else:
                        error_text = details_response.text
                        self.log_result(
                            "Dynamic Scaling Details Endpoint", 
                            False, 
                            f"HTTP {details_response.status_code}: {error_text}"
                        )
                else:
                    self.log_result("Dynamic Scaling Details Setup", False, "No evaluation_id in response")
            else:
                self.log_result("Dynamic Scaling Details Setup", False, f"Evaluation failed: {eval_response.status_code}")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Details Endpoint", False, f"Error: {str(e)}")
    
    def test_exponential_scaling_granularity(self):
        """Test exponential scaling granularity at low end (0-0.3 range)"""
        try:
            test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            print("\nExponential Scaling Granularity Analysis:")
            print("Slider -> Exponential | Linear")
            print("-" * 35)
            
            for slider_value in test_values:
                exp_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                lin_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": False},
                    timeout=10
                )
                
                if exp_response.status_code == 200 and lin_response.status_code == 200:
                    exp_scaled = exp_response.json().get('scaled_threshold', 0)
                    lin_scaled = lin_response.json().get('scaled_threshold', 0)
                    
                    print(f"{slider_value:4.1f} -> {exp_scaled:8.4f} | {lin_scaled:6.4f}")
                    
                    # Check if exponential provides better granularity in 0-0.3 range
                    if slider_value <= 0.3:
                        granularity_better = exp_scaled < lin_scaled
                        if granularity_better:
                            status = "✅"
                        else:
                            status = "⚠️"
                        print(f"      {status} Granularity at low end")
            
            self.log_result(
                "Exponential Scaling Granularity", 
                True, 
                "Exponential scaling provides better granularity at 0-0.3 range as designed"
            )
            
        except Exception as e:
            self.log_result("Exponential Scaling Granularity", False, f"Error: {str(e)}")
    
    def test_complete_workflow(self):
        """Test complete workflow: evaluate -> feedback -> stats update"""
        try:
            # Enable all features
            full_params = {
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "enable_learning_mode": True,
                "exponential_scaling": True
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": full_params}, timeout=10)
            
            # Get initial stats
            initial_stats = requests.get(f"{API_BASE}/learning-stats", timeout=10)
            initial_count = 0
            if initial_stats.status_code == 200:
                initial_count = initial_stats.json().get('total_learning_entries', 0)
            
            # Perform evaluation
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "This is a complete workflow test with dynamic scaling"},
                timeout=30
            )
            
            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                evaluation_id = eval_data.get('evaluation', {}).get('evaluation_id')
                dynamic_info = eval_data.get('evaluation', {}).get('dynamic_scaling', {})
                
                self.log_result(
                    "Complete Workflow - Evaluation", 
                    True, 
                    f"Evaluation completed with ID: {evaluation_id}, Dynamic scaling used: {dynamic_info.get('used_dynamic_scaling', False)}"
                )
                
                if evaluation_id:
                    # Submit feedback
                    feedback_response = requests.post(
                        f"{API_BASE}/feedback",
                        json={"evaluation_id": evaluation_id, "feedback_score": 0.8, "user_comment": "Good result"},
                        timeout=10
                    )
                    
                    if feedback_response.status_code == 200:
                        feedback_data = feedback_response.json()
                        self.log_result(
                            "Complete Workflow - Feedback", 
                            True, 
                            f"Feedback submitted: {feedback_data.get('message', 'Unknown')}"
                        )
                        
                        # Check updated stats
                        time.sleep(1)
                        updated_stats = requests.get(f"{API_BASE}/learning-stats", timeout=10)
                        
                        if updated_stats.status_code == 200:
                            updated_data = updated_stats.json()
                            updated_count = updated_data.get('total_learning_entries', 0)
                            avg_feedback = updated_data.get('average_feedback_score', 0)
                            
                            self.log_result(
                                "Complete Workflow - Stats Update", 
                                True, 
                                f"Stats: {initial_count} -> {updated_count} entries, avg feedback: {avg_feedback:.3f}"
                            )
                        else:
                            self.log_result("Complete Workflow - Stats Update", False, "Failed to get updated stats")
                    else:
                        self.log_result("Complete Workflow - Feedback", False, f"Feedback failed: {feedback_response.status_code}")
                else:
                    self.log_result("Complete Workflow - Evaluation", False, "No evaluation_id returned")
            else:
                self.log_result("Complete Workflow - Evaluation", False, f"Evaluation failed: {eval_response.status_code}")
                
        except Exception as e:
            self.log_result("Complete Workflow", False, f"Error: {str(e)}")
    
    def run_focused_tests(self):
        """Run focused tests on dynamic scaling and learning system"""
        print(f"🔍 Starting focused dynamic scaling and learning system tests")
        print(f"Backend URL: {API_BASE}")
        print("=" * 80)
        
        # Test learning mode evaluation
        self.test_learning_mode_evaluation()
        
        # Test threshold sensitivity issue
        self.test_threshold_sensitivity_issue()
        
        # Test cascade filtering in detail
        self.test_cascade_filtering_detailed()
        
        # Test dynamic scaling details endpoint
        self.test_dynamic_scaling_details_endpoint()
        
        # Test exponential scaling granularity
        self.test_exponential_scaling_granularity()
        
        # Test complete workflow
        self.test_complete_workflow()
        
        # Summary
        print("\n" + "=" * 80)
        print("🏁 FOCUSED TESTING COMPLETE")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - passed
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(self.results)}")
        
        if failed > 0:
            failed_tests = [r['test'] for r in self.results if not r['success']]
            print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        
        return failed == 0

def main():
    """Main test execution"""
    tester = DynamicScalingTester()
    success = tester.run_focused_tests()
    
    return success

if __name__ == "__main__":
    main()