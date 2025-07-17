#!/usr/bin/env python3
"""
CALIBRATION ISSUES INVESTIGATION - FOCUSED DIAGNOSTIC
Based on review request for investigating critical calibration problems
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

class CalibrationInvestigator:
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
                print(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_caching_behavior_verification(self):
        """1. Caching Behavior Verification - Test same text evaluation twice"""
        print("\n🔍 1. CACHING BEHAVIOR VERIFICATION")
        print("=" * 60)
        
        try:
            test_text = "This is a test for caching behavior analysis"
            
            # First evaluation
            start_time1 = time.time()
            response1 = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=60
            )
            end_time1 = time.time()
            processing_time1 = end_time1 - start_time1
            
            if response1.status_code != 200:
                self.log_result("Caching Test - First Evaluation", False, f"HTTP {response1.status_code}")
                return
            
            data1 = response1.json()
            eval1 = data1.get('evaluation', {})
            backend_time1 = eval1.get('processing_time', 0)
            
            # Wait a moment then do second evaluation
            time.sleep(1)
            
            # Second evaluation (same text)
            start_time2 = time.time()
            response2 = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": test_text},
                timeout=60
            )
            end_time2 = time.time()
            processing_time2 = end_time2 - start_time2
            
            if response2.status_code != 200:
                self.log_result("Caching Test - Second Evaluation", False, f"HTTP {response2.status_code}")
                return
            
            data2 = response2.json()
            eval2 = data2.get('evaluation', {})
            backend_time2 = eval2.get('processing_time', 0)
            
            # Analyze caching behavior
            time_difference = abs(backend_time1 - backend_time2)
            request_time_diff = abs(processing_time1 - processing_time2)
            
            # If caching is working, second evaluation should be significantly faster
            if backend_time2 < backend_time1 * 0.5:  # 50% faster indicates caching
                self.log_result(
                    "Caching Behavior Analysis", 
                    True, 
                    f"✅ CACHING DETECTED: First: {backend_time1:.3f}s, Second: {backend_time2:.3f}s (speedup: {backend_time1/backend_time2:.1f}x)",
                    {
                        "first_backend_time": backend_time1,
                        "second_backend_time": backend_time2,
                        "first_request_time": processing_time1,
                        "second_request_time": processing_time2,
                        "speedup_factor": backend_time1/backend_time2 if backend_time2 > 0 else "infinite"
                    }
                )
            else:
                self.log_result(
                    "Caching Behavior Analysis", 
                    False, 
                    f"❌ NO CACHING DETECTED: First: {backend_time1:.3f}s, Second: {backend_time2:.3f}s (difference: {time_difference:.3f}s)",
                    {
                        "first_backend_time": backend_time1,
                        "second_backend_time": backend_time2,
                        "time_difference": time_difference,
                        "expected_caching": "Second evaluation should be significantly faster if embedding cache is working"
                    }
                )
                
        except Exception as e:
            self.log_result("Caching Behavior Verification", False, f"Error: {str(e)}")
    
    def test_threshold_scaling_slider_investigation(self):
        """2. Threshold Scaling Test Slider Investigation"""
        print("\n🎚️ 2. THRESHOLD SCALING SLIDER INVESTIGATION")
        print("=" * 60)
        
        try:
            # Test different slider values with both exponential and linear scaling
            test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            exponential_results = []
            linear_results = []
            
            for slider_value in test_values:
                # Test exponential scaling
                exp_response = requests.post(
                    f"{API_BASE}/threshold-scaling",
                    json={"slider_value": slider_value, "use_exponential": True},
                    timeout=10
                )
                
                # Test linear scaling
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
                    
                    exponential_results.append((slider_value, exp_scaled))
                    linear_results.append((slider_value, lin_scaled))
                    
                    print(f"Slider {slider_value:.1f}: Exponential={exp_scaled:.4f}, Linear={lin_scaled:.4f}")
                else:
                    self.log_result(f"Threshold Scaling (slider={slider_value})", False, "Failed to get scaling response")
            
            # Analyze exponential vs linear behavior
            if len(exponential_results) == len(test_values):
                # Check if exponential scaling provides better granularity at 0-0.3 range
                low_range_exp = [result[1] for result in exponential_results if result[0] <= 0.3]
                low_range_lin = [result[1] for result in linear_results if result[0] <= 0.3]
                
                exp_granularity = max(low_range_exp) - min(low_range_exp) if low_range_exp else 0
                lin_granularity = max(low_range_lin) - min(low_range_lin) if low_range_lin else 0
                
                self.log_result(
                    "Threshold Scaling Analysis", 
                    True, 
                    f"✅ SCALING WORKING: Exponential granularity (0-0.3): {exp_granularity:.4f}, Linear: {lin_granularity:.4f}",
                    {
                        "exponential_results": exponential_results,
                        "linear_results": linear_results,
                        "exponential_granularity_0_to_0.3": exp_granularity,
                        "linear_granularity_0_to_0.3": lin_granularity,
                        "exponential_better_granularity": exp_granularity > lin_granularity
                    }
                )
                
                # Check if Dynamic Scaling tab slider is connected properly
                # Test with a specific slider value to see if it affects actual evaluation
                test_slider_value = 0.2
                exp_threshold = next((result[1] for result in exponential_results if result[0] == test_slider_value), 0.2)
                
                # Update parameters with the scaled threshold
                threshold_params = {
                    "virtue_threshold": exp_threshold,
                    "deontological_threshold": exp_threshold,
                    "consequentialist_threshold": exp_threshold,
                    "exponential_scaling": True
                }
                
                param_response = requests.post(
                    f"{API_BASE}/update-parameters",
                    json={"parameters": threshold_params},
                    timeout=10
                )
                
                if param_response.status_code == 200:
                    self.log_result(
                        "Dynamic Scaling Tab Connection", 
                        True, 
                        f"✅ SLIDER CONNECTED: Successfully updated thresholds to {exp_threshold:.4f} from slider value {test_slider_value}"
                    )
                else:
                    self.log_result("Dynamic Scaling Tab Connection", False, f"Failed to update parameters: HTTP {param_response.status_code}")
            else:
                self.log_result("Threshold Scaling Analysis", False, "Incomplete scaling test results")
                
        except Exception as e:
            self.log_result("Threshold Scaling Investigation", False, f"Error: {str(e)}")
    
    def test_critical_sensitivity_range_analysis(self):
        """3. Critical Sensitivity Range Analysis - Test specific texts at various thresholds"""
        print("\n🎯 3. CRITICAL SENSITIVITY RANGE ANALYSIS")
        print("=" * 60)
        
        try:
            # Test texts from review request
            ambiguous_text = "It's important to support friends in their pursuits, especially when they involve clever strategies to outmaneuver competitors and secure exclusive unmerited advantages."
            clearly_ethical_text = "When in the course of human events, it becomes necessary to standby the aid of our party."
            
            # Test thresholds from review request
            test_thresholds = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
            
            ambiguous_results = []
            ethical_results = []
            
            print(f"\nTesting AMBIGUOUS text: '{ambiguous_text[:50]}...'")
            print(f"Testing CLEARLY ETHICAL text: '{clearly_ethical_text[:50]}...'")
            print()
            
            for threshold in test_thresholds:
                print(f"Testing threshold {threshold:.2f}...")
                
                # Set threshold for all perspectives
                threshold_params = {
                    "virtue_threshold": threshold,
                    "deontological_threshold": threshold,
                    "consequentialist_threshold": threshold,
                    "enable_dynamic_scaling": False,  # Disable to test pure threshold behavior
                    "enable_cascade_filtering": False
                }
                
                param_response = requests.post(
                    f"{API_BASE}/update-parameters",
                    json={"parameters": threshold_params},
                    timeout=10
                )
                
                if param_response.status_code != 200:
                    continue
                
                # Test ambiguous text
                amb_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": ambiguous_text},
                    timeout=60
                )
                
                # Test clearly ethical text
                eth_response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": clearly_ethical_text},
                    timeout=60
                )
                
                if amb_response.status_code == 200 and eth_response.status_code == 200:
                    amb_data = amb_response.json()
                    eth_data = eth_response.json()
                    
                    amb_eval = amb_data.get('evaluation', {})
                    eth_eval = eth_data.get('evaluation', {})
                    
                    amb_ethical = amb_eval.get('overall_ethical', True)
                    eth_ethical = eth_eval.get('overall_ethical', True)
                    
                    amb_violations = len(amb_eval.get('minimal_spans', []))
                    eth_violations = len(eth_eval.get('minimal_spans', []))
                    
                    ambiguous_results.append({
                        "threshold": threshold,
                        "ethical": amb_ethical,
                        "violations": amb_violations,
                        "span_scores": amb_eval.get('span_scores', [])
                    })
                    
                    ethical_results.append({
                        "threshold": threshold,
                        "ethical": eth_ethical,
                        "violations": eth_violations,
                        "span_scores": eth_eval.get('span_scores', [])
                    })
                    
                    print(f"  Ambiguous: {'ETHICAL' if amb_ethical else 'UNETHICAL'} ({amb_violations} violations)")
                    print(f"  Clearly Ethical: {'ETHICAL' if eth_ethical else 'UNETHICAL'} ({eth_violations} violations)")
                
                time.sleep(0.5)  # Brief pause between tests
            
            # Analyze results
            print("\n📊 SENSITIVITY ANALYSIS RESULTS:")
            print("=" * 40)
            
            # Find threshold where ambiguous text gets flagged
            ambiguous_flagged_at = None
            for result in ambiguous_results:
                if not result["ethical"]:
                    ambiguous_flagged_at = result["threshold"]
                    break
            
            # Check if clearly ethical text ever gets flagged
            ethical_flagged_thresholds = [r["threshold"] for r in ethical_results if not r["ethical"]]
            
            if ambiguous_flagged_at is not None:
                self.log_result(
                    "Ambiguous Text Sensitivity", 
                    True, 
                    f"✅ AMBIGUOUS TEXT FLAGGED at threshold {ambiguous_flagged_at:.2f}",
                    {"flagged_at_threshold": ambiguous_flagged_at, "results": ambiguous_results}
                )
            else:
                self.log_result(
                    "Ambiguous Text Sensitivity", 
                    False, 
                    f"❌ AMBIGUOUS TEXT NEVER FLAGGED - even at lowest threshold {min(test_thresholds):.2f}",
                    {"results": ambiguous_results}
                )
            
            if not ethical_flagged_thresholds:
                self.log_result(
                    "Clearly Ethical Text Protection", 
                    True, 
                    f"✅ CLEARLY ETHICAL TEXT NEVER FLAGGED - properly protected at all thresholds",
                    {"results": ethical_results}
                )
            else:
                self.log_result(
                    "Clearly Ethical Text Protection", 
                    False, 
                    f"❌ CLEARLY ETHICAL TEXT INCORRECTLY FLAGGED at thresholds: {ethical_flagged_thresholds}",
                    {"flagged_at_thresholds": ethical_flagged_thresholds, "results": ethical_results}
                )
            
            # Analyze sensitivity compression
            sensitivity_range = []
            for result in ambiguous_results:
                if result["violations"] > 0:
                    sensitivity_range.append(result["threshold"])
            
            if sensitivity_range:
                min_sensitive = min(sensitivity_range)
                max_sensitive = max(sensitivity_range)
                compression_range = max_sensitive - min_sensitive
                
                self.log_result(
                    "Sensitivity Range Analysis", 
                    compression_range > 0.1, 
                    f"Sensitivity range: {min_sensitive:.2f} to {max_sensitive:.2f} (span: {compression_range:.2f})",
                    {
                        "sensitivity_range": sensitivity_range,
                        "compression_span": compression_range,
                        "compressed_to_0_to_0.2": compression_range <= 0.2,
                        "analysis": "Range should be wider than 0.2 for good granularity"
                    }
                )
            else:
                self.log_result(
                    "Sensitivity Range Analysis", 
                    False, 
                    "❌ NO SENSITIVITY DETECTED - text never flagged at any threshold"
                )
                
        except Exception as e:
            self.log_result("Critical Sensitivity Range Analysis", False, f"Error: {str(e)}")
    
    def test_granularity_analysis(self):
        """4. Granularity Analysis - Analyze mathematical framework"""
        print("\n🔬 4. GRANULARITY ANALYSIS - MATHEMATICAL FRAMEWORK")
        print("=" * 60)
        
        try:
            # Test with a range of texts to understand scoring behavior
            test_texts = [
                ("I love helping people", "Clearly positive"),
                ("The weather is nice", "Neutral"),
                ("This is somewhat concerning", "Mildly negative"),
                ("You are stupid and worthless", "Clearly negative"),
                ("I hate you", "Very negative")
            ]
            
            # Set a moderate threshold to see scoring distribution
            threshold = 0.15
            threshold_params = {
                "virtue_threshold": threshold,
                "deontological_threshold": threshold,
                "consequentialist_threshold": threshold,
                "enable_dynamic_scaling": False,
                "enable_cascade_filtering": False
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": threshold_params}, timeout=10)
            
            scoring_analysis = []
            
            for text, description in test_texts:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    
                    span_scores = evaluation.get('span_scores', [])
                    minimal_spans = evaluation.get('minimal_spans', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    # Analyze individual span scores
                    virtue_scores = []
                    deontological_scores = []
                    consequentialist_scores = []
                    
                    for score in span_scores:
                        if isinstance(score, dict):
                            virtue_scores.append(score.get('virtue_score', 1.0))
                            deontological_scores.append(score.get('deontological_score', 1.0))
                            consequentialist_scores.append(score.get('consequentialist_score', 1.0))
                    
                    analysis = {
                        "text": text,
                        "description": description,
                        "overall_ethical": overall_ethical,
                        "violations": len(minimal_spans),
                        "virtue_scores": virtue_scores,
                        "deontological_scores": deontological_scores,
                        "consequentialist_scores": consequentialist_scores,
                        "min_virtue": min(virtue_scores) if virtue_scores else 1.0,
                        "min_deontological": min(deontological_scores) if deontological_scores else 1.0,
                        "min_consequentialist": min(consequentialist_scores) if consequentialist_scores else 1.0
                    }
                    
                    scoring_analysis.append(analysis)
                    
                    print(f"\n{description}: '{text}'")
                    print(f"  Overall: {'ETHICAL' if overall_ethical else 'UNETHICAL'} ({len(minimal_spans)} violations)")
                    print(f"  Min scores - Virtue: {analysis['min_virtue']:.3f}, Deont: {analysis['min_deontological']:.3f}, Conseq: {analysis['min_consequentialist']:.3f}")
            
            # Analyze score distribution and compression
            all_virtue_scores = []
            all_deont_scores = []
            all_conseq_scores = []
            
            for analysis in scoring_analysis:
                all_virtue_scores.extend(analysis['virtue_scores'])
                all_deont_scores.extend(analysis['deontological_scores'])
                all_conseq_scores.extend(analysis['consequentialist_scores'])
            
            if all_virtue_scores:
                virtue_range = max(all_virtue_scores) - min(all_virtue_scores)
                deont_range = max(all_deont_scores) - min(all_deont_scores)
                conseq_range = max(all_conseq_scores) - min(all_conseq_scores)
                
                print(f"\n📊 SCORE DISTRIBUTION ANALYSIS:")
                print(f"Virtue scores range: {min(all_virtue_scores):.3f} to {max(all_virtue_scores):.3f} (span: {virtue_range:.3f})")
                print(f"Deontological scores range: {min(all_deont_scores):.3f} to {max(all_deont_scores):.3f} (span: {deont_range:.3f})")
                print(f"Consequentialist scores range: {min(all_conseq_scores):.3f} to {max(all_conseq_scores):.3f} (span: {conseq_range:.3f})")
                
                # Check if scores are compressed to 0.0-0.2 range
                compressed_virtue = max(all_virtue_scores) <= 0.2
                compressed_deont = max(all_deont_scores) <= 0.2
                compressed_conseq = max(all_conseq_scores) <= 0.2
                
                if compressed_virtue or compressed_deont or compressed_conseq:
                    self.log_result(
                        "Score Compression Analysis", 
                        False, 
                        f"❌ SCORE COMPRESSION DETECTED: Virtue compressed: {compressed_virtue}, Deont: {compressed_deont}, Conseq: {compressed_conseq}",
                        {
                            "virtue_range": virtue_range,
                            "deontological_range": deont_range,
                            "consequentialist_range": conseq_range,
                            "virtue_compressed_to_0_to_0.2": compressed_virtue,
                            "deontological_compressed_to_0_to_0.2": compressed_deont,
                            "consequentialist_compressed_to_0_to_0.2": compressed_conseq,
                            "scoring_analysis": scoring_analysis
                        }
                    )
                else:
                    self.log_result(
                        "Score Compression Analysis", 
                        True, 
                        f"✅ GOOD SCORE DISTRIBUTION: Adequate range across all perspectives",
                        {
                            "virtue_range": virtue_range,
                            "deontological_range": deont_range,
                            "consequentialist_range": conseq_range,
                            "scoring_analysis": scoring_analysis
                        }
                    )
            else:
                self.log_result("Granularity Analysis", False, "No scoring data available")
                
        except Exception as e:
            self.log_result("Granularity Analysis", False, f"Error: {str(e)}")
    
    def test_vector_distance_investigation(self):
        """5. Vector Distance Investigation - Examine actual similarity scores"""
        print("\n🧮 5. VECTOR DISTANCE INVESTIGATION")
        print("=" * 60)
        
        try:
            # Test the same texts from sensitivity analysis to get detailed vector information
            test_texts = [
                "It's important to support friends in their pursuits, especially when they involve clever strategies to outmaneuver competitors and secure exclusive unmerited advantages.",
                "When in the course of human events, it becomes necessary to standby the aid of our party.",
                "You are stupid and worthless",
                "I love helping people"
            ]
            
            # Set a low threshold to capture more detailed scoring
            threshold = 0.10
            threshold_params = {
                "virtue_threshold": threshold,
                "deontological_threshold": threshold,
                "consequentialist_threshold": threshold,
                "enable_dynamic_scaling": False,
                "enable_cascade_filtering": False
            }
            
            requests.post(f"{API_BASE}/update-parameters", json={"parameters": threshold_params}, timeout=10)
            
            vector_analysis = []
            
            for text in test_texts:
                response = requests.post(
                    f"{API_BASE}/evaluate",
                    json={"text": text},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    evaluation = data.get('evaluation', {})
                    
                    span_scores = evaluation.get('span_scores', [])
                    minimal_spans = evaluation.get('minimal_spans', [])
                    overall_ethical = evaluation.get('overall_ethical', True)
                    
                    # Detailed analysis of each span
                    span_analysis = []
                    for i, score in enumerate(span_scores):
                        if isinstance(score, dict):
                            span_text = score.get('span_text', f'span_{i}')
                            virtue_score = score.get('virtue_score', 1.0)
                            deont_score = score.get('deontological_score', 1.0)
                            conseq_score = score.get('consequentialist_score', 1.0)
                            
                            # Check which perspectives flagged this span
                            virtue_violation = virtue_score < threshold
                            deont_violation = deont_score < threshold
                            conseq_violation = conseq_score < threshold
                            
                            span_analysis.append({
                                "span_text": span_text,
                                "virtue_score": virtue_score,
                                "deontological_score": deont_score,
                                "consequentialist_score": conseq_score,
                                "virtue_violation": virtue_violation,
                                "deontological_violation": deont_violation,
                                "consequentialist_violation": conseq_violation,
                                "any_violation": virtue_violation or deont_violation or conseq_violation
                            })
                    
                    text_analysis = {
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "overall_ethical": overall_ethical,
                        "total_violations": len(minimal_spans),
                        "span_analysis": span_analysis,
                        "lowest_virtue": min([s["virtue_score"] for s in span_analysis]) if span_analysis else 1.0,
                        "lowest_deontological": min([s["deontological_score"] for s in span_analysis]) if span_analysis else 1.0,
                        "lowest_consequentialist": min([s["consequentialist_score"] for s in span_analysis]) if span_analysis else 1.0
                    }
                    
                    vector_analysis.append(text_analysis)
                    
                    print(f"\n📝 Text: '{text[:50]}...'")
                    print(f"   Overall: {'ETHICAL' if overall_ethical else 'UNETHICAL'} ({len(minimal_spans)} violations)")
                    print(f"   Lowest scores - V: {text_analysis['lowest_virtue']:.3f}, D: {text_analysis['lowest_deontological']:.3f}, C: {text_analysis['lowest_consequentialist']:.3f}")
                    
                    # Show spans that violated threshold
                    violated_spans = [s for s in span_analysis if s["any_violation"]]
                    if violated_spans:
                        print(f"   Violated spans:")
                        for span in violated_spans[:3]:  # Show first 3
                            violations = []
                            if span["virtue_violation"]: violations.append(f"V:{span['virtue_score']:.3f}")
                            if span["deontological_violation"]: violations.append(f"D:{span['deontological_score']:.3f}")
                            if span["consequentialist_violation"]: violations.append(f"C:{span['consequentialist_score']:.3f}")
                            print(f"     '{span['span_text'][:30]}...' -> {', '.join(violations)}")
            
            # Analyze why threshold range is compressed
            all_scores = []
            for analysis in vector_analysis:
                for span in analysis["span_analysis"]:
                    all_scores.extend([
                        span["virtue_score"],
                        span["deontological_score"],
                        span["consequentialist_score"]
                    ])
            
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                score_range = max_score - min_score
                
                # Check if most scores are clustered in a narrow range
                scores_below_0_2 = sum(1 for score in all_scores if score < 0.2)
                total_scores = len(all_scores)
                compression_ratio = scores_below_0_2 / total_scores if total_scores > 0 else 0
                
                print(f"\n🔍 VECTOR DISTANCE ANALYSIS:")
                print(f"Score range: {min_score:.3f} to {max_score:.3f} (span: {score_range:.3f})")
                print(f"Scores below 0.2: {scores_below_0_2}/{total_scores} ({compression_ratio:.1%})")
                
                if compression_ratio > 0.8:  # More than 80% of scores below 0.2
                    self.log_result(
                        "Vector Distance Analysis", 
                        False, 
                        f"❌ SEVERE COMPRESSION: {compression_ratio:.1%} of scores below 0.2, range only {score_range:.3f}",
                        {
                            "min_score": min_score,
                            "max_score": max_score,
                            "score_range": score_range,
                            "compression_ratio": compression_ratio,
                            "vector_analysis": vector_analysis,
                            "recommendation": "Ethical vectors may need recalibration to provide better score distribution"
                        }
                    )
                else:
                    self.log_result(
                        "Vector Distance Analysis", 
                        True, 
                        f"✅ ADEQUATE DISTRIBUTION: {compression_ratio:.1%} of scores below 0.2, range {score_range:.3f}",
                        {
                            "min_score": min_score,
                            "max_score": max_score,
                            "score_range": score_range,
                            "compression_ratio": compression_ratio,
                            "vector_analysis": vector_analysis
                        }
                    )
            else:
                self.log_result("Vector Distance Analysis", False, "No vector distance data available")
                
        except Exception as e:
            self.log_result("Vector Distance Investigation", False, f"Error: {str(e)}")
    
    def run_complete_investigation(self):
        """Run all calibration investigation tests"""
        print("🔍 CALIBRATION ISSUES INVESTIGATION - FOCUSED DIAGNOSTIC")
        print("=" * 80)
        print("Investigating critical calibration problems as requested in review")
        print("=" * 80)
        
        # Run all investigation tests
        self.test_caching_behavior_verification()
        self.test_threshold_scaling_slider_investigation()
        self.test_critical_sensitivity_range_analysis()
        self.test_granularity_analysis()
        self.test_vector_distance_investigation()
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 CALIBRATION INVESTIGATION SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = len(self.passed_tests)
        failed_tests = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 DETAILED FINDINGS:")
        for result in self.results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
        
        print("\n🔧 RECOMMENDATIONS:")
        
        # Analyze results and provide recommendations
        caching_working = any("CACHING DETECTED" in r['message'] for r in self.results if r['test'] == "Caching Behavior Analysis")
        sensitivity_compressed = any("COMPRESSION DETECTED" in r['message'] for r in self.results)
        ambiguous_flagged = any("AMBIGUOUS TEXT FLAGGED" in r['message'] for r in self.results)
        ethical_protected = any("CLEARLY ETHICAL TEXT NEVER FLAGGED" in r['message'] for r in self.results)
        
        if not caching_working:
            print("• Implement embedding caching to improve performance")
        
        if sensitivity_compressed:
            print("• Recalibrate ethical vectors to improve score distribution")
            print("• Expand sensitivity range beyond 0.0-0.2 compression")
        
        if not ambiguous_flagged:
            print("• Lower default thresholds to catch ambiguous content")
            print("• Improve ethical vector examples for better detection")
        
        if not ethical_protected:
            print("• Adjust thresholds to protect clearly ethical content")
        
        print("• Consider exponential scaling for better granularity at low thresholds")
        print("• Review cascade filtering thresholds for better accuracy")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests/total_tests)*100,
            "results": self.results,
            "caching_working": caching_working,
            "sensitivity_compressed": sensitivity_compressed,
            "ambiguous_flagged": ambiguous_flagged,
            "ethical_protected": ethical_protected
        }

if __name__ == "__main__":
    investigator = CalibrationInvestigator()
    results = investigator.run_complete_investigation()
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)