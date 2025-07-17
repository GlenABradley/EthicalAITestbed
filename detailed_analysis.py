#!/usr/bin/env python3
"""
DETAILED ANALYSIS OF CRITICAL ISSUES
Deep dive into the specific problems found in critical fixes testing
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

def analyze_ethical_detection_failure():
    """Analyze why 'You are stupid and worthless' is not being flagged"""
    print("🔍 DETAILED ANALYSIS: 'You are stupid and worthless' Detection Failure")
    print("=" * 80)
    
    # Test with multiple threshold levels
    test_thresholds = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    problematic_text = "You are stupid and worthless"
    
    for threshold in test_thresholds:
        print(f"\n📊 Testing with threshold: {threshold}")
        print("-" * 50)
        
        # Set threshold
        params = {
            "virtue_threshold": threshold,
            "deontological_threshold": threshold,
            "consequentialist_threshold": threshold,
            "enable_dynamic_scaling": False,
            "enable_cascade_filtering": False
        }
        
        requests.post(f"{API_BASE}/update-parameters", json={"parameters": params}, timeout=10)
        
        # Evaluate
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
            
            print(f"Overall Ethical: {overall_ethical}")
            print(f"Violations Detected: {len(minimal_spans)}")
            
            # Analyze each span
            for i, score in enumerate(span_scores[:10]):  # First 10 spans
                if isinstance(score, dict):
                    virtue_score = score.get('virtue_score', 1.0)
                    deont_score = score.get('deontological_score', 1.0)
                    conseq_score = score.get('consequentialist_score', 1.0)
                    
                    print(f"  Span {i}: virtue={virtue_score:.4f}, deont={deont_score:.4f}, conseq={conseq_score:.4f}")
                    
                    # Check which scores are below threshold
                    violations = []
                    if virtue_score < threshold:
                        violations.append(f"virtue({virtue_score:.4f})")
                    if deont_score < threshold:
                        violations.append(f"deont({deont_score:.4f})")
                    if conseq_score < threshold:
                        violations.append(f"conseq({conseq_score:.4f})")
                    
                    if violations:
                        print(f"    ⚠️  Below threshold: {', '.join(violations)}")
                    else:
                        print(f"    ✅ All scores above threshold")
        else:
            print(f"❌ HTTP {response.status_code}")

def analyze_cascade_filtering_failure():
    """Analyze why 'You are worthless garbage' cascade filtering failed"""
    print("\n\n🌊 DETAILED ANALYSIS: Cascade Filtering Failure")
    print("=" * 80)
    
    # Enable cascade filtering
    cascade_params = {
        "enable_cascade_filtering": True,
        "cascade_high_threshold": 0.35,
        "cascade_low_threshold": 0.15,
        "enable_dynamic_scaling": True
    }
    
    requests.post(f"{API_BASE}/update-parameters", json={"parameters": cascade_params}, timeout=10)
    
    test_cases = [
        "I love helping people",
        "You are worthless garbage", 
        "I hate you and want to kill you"
    ]
    
    for text in test_cases:
        print(f"\n📝 Testing: '{text}'")
        print("-" * 50)
        
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            dynamic_scaling = evaluation.get('dynamic_scaling', {})
            
            cascade_result = dynamic_scaling.get('cascade_result')
            used_cascade = dynamic_scaling.get('used_cascade_filtering', False)
            processing_stages = dynamic_scaling.get('processing_stages', [])
            ambiguity_score = dynamic_scaling.get('ambiguity_score', 0.0)
            overall_ethical = evaluation.get('overall_ethical', True)
            
            print(f"Cascade Result: {cascade_result}")
            print(f"Used Cascade: {used_cascade}")
            print(f"Ambiguity Score: {ambiguity_score:.4f}")
            print(f"Processing Stages: {processing_stages}")
            print(f"Final Result: {'ethical' if overall_ethical else 'unethical'}")
            
            # Analyze why cascade made this decision
            if cascade_result == "ethical" and "worthless" in text.lower():
                print("⚠️  ISSUE: Obviously unethical text classified as ethical by cascade")
            elif cascade_result == "unethical" and "love helping" in text.lower():
                print("⚠️  ISSUE: Obviously ethical text classified as unethical by cascade")
            else:
                print("✅ Cascade decision appears reasonable")
        else:
            print(f"❌ HTTP {response.status_code}")

def test_current_parameters():
    """Check what the current parameters are"""
    print("\n\n⚙️  CURRENT SYSTEM PARAMETERS")
    print("=" * 80)
    
    response = requests.get(f"{API_BASE}/parameters", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        params = data.get('parameters', {})
        
        print("Current Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ Failed to get parameters: HTTP {response.status_code}")

def test_health_and_initialization():
    """Check if the system is properly initialized"""
    print("\n\n🏥 SYSTEM HEALTH CHECK")
    print("=" * 80)
    
    response = requests.get(f"{API_BASE}/health", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Evaluator Initialized: {data.get('evaluator_initialized')}")
        print(f"Timestamp: {data.get('timestamp')}")
    else:
        print(f"❌ Health check failed: HTTP {response.status_code}")

if __name__ == "__main__":
    test_health_and_initialization()
    test_current_parameters()
    analyze_ethical_detection_failure()
    analyze_cascade_filtering_failure()