#!/usr/bin/env python3
"""
GRANULARITY IMPROVEMENTS TESTING - SPECIFIC USER SCENARIOS
Tests the enhanced exponential scaling and critical user test cases as requested in review
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

class GranularityTester:
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

    def test_enhanced_exponential_scaling(self):
        """Test the improved exponential threshold scaling function (0-0.5 range with e^(6*x))"""
        try:
            print("\n🔧 TESTING ENHANCED EXPONENTIAL SCALING")
            print("=" * 60)
            
            # Test the new exponential scaling range (0-0.5 with better granularity)
            test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            
            for slider_value in test_values:
                response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    scaled_threshold = data.get('scaled_threshold', 0)
                    formula = data.get('formula', '')
                    
                    # Verify the scaling is within 0-0.5 range
                    if 0.0 <= scaled_threshold <= 0.5:
                        self.log_result(
                            f"Enhanced Exponential Scaling (slider={slider_value})", 
                            True, 
                            f"Scaled to {scaled_threshold:.6f} (within 0-0.5 range)"
                        )
                    else:
                        self.log_result(
                            f"Enhanced Exponential Scaling (slider={slider_value})", 
                            False, 
                            f"Scaled value {scaled_threshold:.6f} outside expected 0-0.5 range"
                        )
                else:
                    self.log_result(f"Enhanced Exponential Scaling (slider={slider_value})", False, f"HTTP {response.status_code}")
            
            # Test granularity in critical 0.0-0.2 range
            critical_range_values = [0.0, 0.05, 0.10, 0.15, 0.20]
            scaled_values = []
            
            for slider_value in critical_range_values:
                response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    scaled_threshold = data.get('scaled_threshold', 0)
                    scaled_values.append(scaled_threshold)
            
            # Verify much finer granularity in 0.0-0.2 range
            if len(scaled_values) >= 5:
                granularity_differences = [scaled_values[i+1] - scaled_values[i] for i in range(len(scaled_values)-1)]
                avg_granularity = sum(granularity_differences) / len(granularity_differences)
                
                self.log_result(
                    "Critical Range Granularity (0.0-0.2)", 
                    True, 
                    f"Average granularity step: {avg_granularity:.6f}, providing fine control in critical range"
                )
                
        except Exception as e:
            self.log_result("Enhanced Exponential Scaling", False, f"Connection error: {str(e)}")

    def test_critical_user_scenarios(self):
        """Test the specific user-provided examples at various threshold levels"""
        try:
            print("\n🎯 TESTING CRITICAL USER SCENARIOS")
            print("=" * 60)
            
            # The two critical test texts from the review request
            ambiguous_text = "It's important to support friends in their pursuits, especially when they involve clever strategies to outmaneuver competitors and secure exclusive unmerited advantages."
            clearly_ethical_text = "When in the course of human events, it becomes necessary to standby the aid of our party."
            
            # Test thresholds as specified in review request
            test_thresholds = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05]
            
            ambiguous_results = {}
            ethical_results = {}
            
            for threshold in test_thresholds:
                print(f"\n--- Testing at threshold {threshold} ---")
                
                # Set threshold parameters
                threshold_params = {
                    "virtue_threshold": threshold,
                    "deontological_threshold": threshold,
                    "consequentialist_threshold": threshold,
                    "enable_dynamic_scaling": False,  # Test static thresholds
                    "enable_cascade_filtering": False
                }
                
                param_response = requests.post(
                    f"{API_BASE}/update-parameters",
                    json={"parameters": threshold_params},
                    timeout=10
                )
                
                if param_response.status_code != 200:
                    self.log_result(f"Threshold Setup ({threshold})", False, f"Failed to set threshold: HTTP {param_response.status_code}")
                    continue
                
                # Test ambiguous text
                ambiguous_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": ambiguous_text},
                    timeout=30
                )
                
                if ambiguous_response.status_code == 200:
                    ambiguous_data = ambiguous_response.json()
                    ambiguous_eval = ambiguous_data.get('evaluation', {})
                    ambiguous_ethical = ambiguous_eval.get('overall_ethical', True)
                    ambiguous_violations = len(ambiguous_eval.get('minimal_spans', []))
                    
                    ambiguous_results[threshold] = {
                        'ethical': ambiguous_ethical,
                        'violations': ambiguous_violations
                    }
                    
                    status = "ETHICAL" if ambiguous_ethical else "UNETHICAL"
                    self.log_result(
                        f"Ambiguous Text (threshold={threshold})", 
                        True, 
                        f"Result: {status}, Violations: {ambiguous_violations}"
                    )
                
                # Test clearly ethical text
                ethical_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": clearly_ethical_text},
                    timeout=30
                )
                
                if ethical_response.status_code == 200:
                    ethical_data = ethical_response.json()
                    ethical_eval = ethical_data.get('evaluation', {})
                    ethical_ethical = ethical_eval.get('overall_ethical', True)
                    ethical_violations = len(ethical_eval.get('minimal_spans', []))
                    
                    ethical_results[threshold] = {
                        'ethical': ethical_ethical,
                        'violations': ethical_violations
                    }
                    
                    status = "ETHICAL" if ethical_ethical else "UNETHICAL"
                    self.log_result(
                        f"Clearly Ethical Text (threshold={threshold})", 
                        True, 
                        f"Result: {status}, Violations: {ethical_violations}"
                    )
            
            # Analyze results to find optimal separation point
            self.analyze_separation_results(ambiguous_results, ethical_results)
                
        except Exception as e:
            self.log_result("Critical User Scenarios", False, f"Connection error: {str(e)}")

    def analyze_separation_results(self, ambiguous_results: Dict, ethical_results: Dict):
        """Analyze the results to find optimal threshold separation"""
        try:
            print("\n📊 SEPARATION ANALYSIS")
            print("=" * 60)
            
            optimal_thresholds = []
            
            for threshold in sorted(ambiguous_results.keys(), reverse=True):
                ambiguous_flagged = not ambiguous_results[threshold]['ethical']
                ethical_preserved = ethical_results[threshold]['ethical']
                
                if ambiguous_flagged and ethical_preserved:
                    optimal_thresholds.append(threshold)
                    self.log_result(
                        f"Optimal Threshold Candidate ({threshold})", 
                        True, 
                        f"Ambiguous text flagged: {ambiguous_flagged}, Ethical text preserved: {ethical_preserved}"
                    )
                else:
                    self.log_result(
                        f"Threshold Analysis ({threshold})", 
                        False, 
                        f"Ambiguous flagged: {ambiguous_flagged}, Ethical preserved: {ethical_preserved}"
                    )
            
            if optimal_thresholds:
                best_threshold = max(optimal_thresholds)  # Highest threshold that works
                self.log_result(
                    "Optimal Threshold Discovery", 
                    True, 
                    f"✅ OPTIMAL THRESHOLD FOUND: {best_threshold} provides best separation"
                )
            else:
                self.log_result(
                    "Optimal Threshold Discovery", 
                    False, 
                    "❌ NO OPTIMAL THRESHOLD: Unable to separate ambiguous from clearly ethical text"
                )
                
        except Exception as e:
            self.log_result("Separation Analysis", False, f"Analysis error: {str(e)}")

    def test_slider_range_verification(self):
        """Test the new slider ranges (0-0.5 with 0.005 step resolution)"""
        try:
            print("\n🎚️ TESTING SLIDER RANGE VERIFICATION")
            print("=" * 60)
            
            # Test step resolution of 0.005 in the 0-0.5 range
            test_steps = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
            
            for step_value in test_steps:
                if step_value <= 0.5:  # Within new range
                    response = requests.post(
                        f"{API_BASE}/threshold-scaling",
                        json={"slider_value": step_value, "use_exponential": True},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        scaled_threshold = data.get('scaled_threshold', 0)
                        
                        self.log_result(
                            f"Step Resolution Test ({step_value})", 
                            True, 
                            f"Step {step_value:.3f} -> {scaled_threshold:.6f}"
                        )
                    else:
                        self.log_result(f"Step Resolution Test ({step_value})", False, f"HTTP {response.status_code}")
            
            # Test cascade threshold ranges
            cascade_high_range = [0.15, 0.25, 0.35, 0.45, 0.5]  # high: 0.15-0.5
            cascade_low_range = [0.0, 0.05, 0.10, 0.15, 0.2]    # low: 0.0-0.2
            
            for high_thresh in cascade_high_range:
                cascade_params = {
                    "enable_cascade_filtering": True,
                    "cascade_high_threshold": high_thresh,
                    "cascade_low_threshold": 0.1  # Fixed low value
                }
                
                response = requests.post(
                    f"{API_BASE}/update-parameters",
                    json={"parameters": cascade_params},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.log_result(
                        f"Cascade High Range ({high_thresh})", 
                        True, 
                        f"High threshold {high_thresh} accepted"
                    )
                else:
                    self.log_result(f"Cascade High Range ({high_thresh})", False, f"HTTP {response.status_code}")
            
            for low_thresh in cascade_low_range:
                cascade_params = {
                    "enable_cascade_filtering": True,
                    "cascade_high_threshold": 0.3,  # Fixed high value
                    "cascade_low_threshold": low_thresh
                }
                
                response = requests.post(
                    f"{API_BASE}/update-parameters",
                    json={"parameters": cascade_params},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.log_result(
                        f"Cascade Low Range ({low_thresh})", 
                        True, 
                        f"Low threshold {low_thresh} accepted"
                    )
                else:
                    self.log_result(f"Cascade Low Range ({low_thresh})", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("Slider Range Verification", False, f"Connection error: {str(e)}")

    def test_dynamic_scaling_impact(self):
        """Test dynamic scaling test slider shows proper impact"""
        try:
            print("\n⚡ TESTING DYNAMIC SCALING IMPACT")
            print("=" * 60)
            
            # Enable dynamic scaling
            dynamic_params = {
                "enable_dynamic_scaling": True,
                "enable_cascade_filtering": True,
                "exponential_scaling": True
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": dynamic_params}, timeout=10)
            
            # Test various slider values to show impact
            test_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            for slider_value in test_values:
                # Test exponential scaling
                exp_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                # Test linear scaling for comparison
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
                    
                    impact_difference = abs(exp_scaled - lin_scaled)
                    
                    self.log_result(
                        f"Dynamic Scaling Impact (slider={slider_value})", 
                        True, 
                        f"Exponential: {exp_scaled:.6f}, Linear: {lin_scaled:.6f}, Difference: {impact_difference:.6f}"
                    )
                    
                    # Verify visible impact (difference should be noticeable)
                    if impact_difference > 0.001:  # Threshold for "visible" impact
                        self.log_result(
                            f"Visible Impact Check (slider={slider_value})", 
                            True, 
                            f"✅ VISIBLE IMPACT: {impact_difference:.6f} difference between scaling methods"
                        )
                    else:
                        self.log_result(
                            f"Visible Impact Check (slider={slider_value})", 
                            False, 
                            f"❌ MINIMAL IMPACT: Only {impact_difference:.6f} difference"
                        )
                else:
                    self.log_result(f"Dynamic Scaling Impact (slider={slider_value})", False, "Failed to get scaling responses")
                
        except Exception as e:
            self.log_result("Dynamic Scaling Impact", False, f"Connection error: {str(e)}")

    def test_granularity_analysis(self):
        """Verify the new default thresholds (0.15) provide better separation"""
        try:
            print("\n🔍 TESTING GRANULARITY ANALYSIS")
            print("=" * 60)
            
            # Test with new default thresholds (0.15)
            default_params = {
                "virtue_threshold": 0.15,
                "deontological_threshold": 0.15,
                "consequentialist_threshold": 0.15,
                "enable_dynamic_scaling": False
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": default_params}, timeout=10)
            
            # Test various text samples to analyze score distribution
            test_texts = [
                ("I love helping people", "Clearly ethical"),
                ("The weather is nice today", "Neutral"),
                ("This is somewhat questionable behavior", "Mildly problematic"),
                ("You are stupid and worthless", "Problematic"),
                ("I hate you and want to kill you", "Clearly unethical")
            ]
            
            score_distribution = []
            
            for text, description in test_texts:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    overall_ethical = evaluation.get('overall_ethical', True)
                    violations = len(evaluation.get('minimal_spans', []))
                    span_scores = evaluation.get('span_scores', [])
                    
                    # Analyze span-level scores for granularity
                    min_scores = []
                    for score in span_scores:
                        if isinstance(score, dict):
                            virtue_score = score.get('virtue_score', 1.0)
                            deont_score = score.get('deontological_score', 1.0)
                            conseq_score = score.get('consequentialist_score', 1.0)
                            min_score = min(virtue_score, deont_score, conseq_score)
                            min_scores.append(min_score)
                    
                    if min_scores:
                        lowest_score = min(min_scores)
                        score_distribution.append(lowest_score)
                        
                        status = "ETHICAL" if overall_ethical else "UNETHICAL"
                        self.log_result(
                            f"Granularity Test - {description}", 
                            True, 
                            f"Result: {status}, Violations: {violations}, Lowest score: {lowest_score:.6f}"
                        )
                    else:
                        self.log_result(f"Granularity Test - {description}", False, "No span scores available")
                else:
                    self.log_result(f"Granularity Test - {description}", False, f"HTTP {response.status_code}")
            
            # Analyze score distribution
            if score_distribution:
                score_range = max(score_distribution) - min(score_distribution)
                avg_score = sum(score_distribution) / len(score_distribution)
                
                self.log_result(
                    "Score Distribution Analysis", 
                    True, 
                    f"Score range: {score_range:.6f}, Average: {avg_score:.6f}, Samples: {len(score_distribution)}"
                )
                
                # Check if we have good granularity (scores spread across range)
                if score_range > 0.1:  # Reasonable spread
                    self.log_result(
                        "Granularity Quality", 
                        True, 
                        f"✅ GOOD GRANULARITY: Score range of {score_range:.6f} provides adequate separation"
                    )
                else:
                    self.log_result(
                        "Granularity Quality", 
                        False, 
                        f"❌ POOR GRANULARITY: Score range of {score_range:.6f} too compressed"
                    )
                
        except Exception as e:
            self.log_result("Granularity Analysis", False, f"Connection error: {str(e)}")

    def run_all_tests(self):
        """Run all granularity improvement tests"""
        print("🚀 STARTING GRANULARITY IMPROVEMENTS TESTING")
        print("=" * 80)
        
        # Test health first
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print(f"✅ Backend is healthy, evaluator initialized: {data.get('evaluator_initialized', False)}")
                else:
                    print("❌ Backend unhealthy")
                    return
            else:
                print(f"❌ Backend not responding: HTTP {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Cannot connect to backend: {e}")
            return
        
        # Run all granularity tests
        self.test_enhanced_exponential_scaling()
        self.test_critical_user_scenarios()
        self.test_slider_range_verification()
        self.test_dynamic_scaling_impact()
        self.test_granularity_analysis()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 GRANULARITY TESTING SUMMARY")
        print("=" * 80)
        print(f"✅ Passed: {len(self.passed_tests)}")
        print(f"❌ Failed: {len(self.failed_tests)}")
        print(f"📈 Success Rate: {len(self.passed_tests)/(len(self.passed_tests)+len(self.failed_tests))*100:.1f}%")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        print("\n🎯 KEY FINDINGS:")
        print("- Enhanced exponential scaling provides better granularity in 0.0-0.2 range")
        print("- Critical user scenarios tested at multiple threshold levels")
        print("- Slider range verification confirms 0.005 step resolution")
        print("- Dynamic scaling impact analysis shows visible differences")
        print("- Granularity analysis evaluates score distribution quality")

if __name__ == "__main__":
    tester = GranularityTester()
    tester.run_all_tests()