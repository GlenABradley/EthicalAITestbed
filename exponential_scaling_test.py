#!/usr/bin/env python3
"""
EXPONENTIAL SCALING VERIFICATION TEST
Tests the enhanced exponential scaling function and granularity improvements
"""

import requests
import json
import math
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

# Get backend URL from environment
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_exponential_scaling_formula():
    """Test the exponential scaling formula and verify granularity improvements"""
    print("🔧 TESTING ENHANCED EXPONENTIAL SCALING")
    print("=" * 60)
    
    # Test values across the 0-0.5 range with focus on 0.0-0.2 critical range
    test_values = [
        0.0, 0.01, 0.02, 0.03, 0.04, 0.05,  # Critical range 0.0-0.05
        0.10, 0.15, 0.20,                    # Critical range 0.05-0.20
        0.25, 0.30, 0.35, 0.40, 0.45, 0.50  # Extended range 0.20-0.50
    ]
    
    print(f"{'Slider':<8} {'Exponential':<12} {'Linear':<12} {'Difference':<12} {'Granularity':<12}")
    print("-" * 70)
    
    exponential_values = []
    linear_values = []
    
    for slider_value in test_values:
        try:
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
                
                exponential_values.append(exp_scaled)
                linear_values.append(lin_scaled)
                
                difference = abs(exp_scaled - lin_scaled)
                
                # Calculate granularity (step size from previous value)
                granularity = ""
                if len(exponential_values) > 1:
                    step_size = exponential_values[-1] - exponential_values[-2]
                    granularity = f"{step_size:.6f}"
                
                print(f"{slider_value:<8.2f} {exp_scaled:<12.6f} {lin_scaled:<12.6f} {difference:<12.6f} {granularity:<12}")
                
            else:
                print(f"{slider_value:<8.2f} ERROR        ERROR        ERROR        ERROR")
                
        except Exception as e:
            print(f"{slider_value:<8.2f} TIMEOUT      TIMEOUT      TIMEOUT      TIMEOUT")
    
    # Analyze granularity in critical 0.0-0.2 range
    print("\n" + "=" * 60)
    print("📊 GRANULARITY ANALYSIS")
    print("=" * 60)
    
    if len(exponential_values) >= 10:
        # Focus on first 9 values (0.0-0.20 range)
        critical_range_exp = exponential_values[:9]
        critical_range_linear = linear_values[:9]
        
        # Calculate step sizes for exponential scaling in critical range
        exp_steps = [critical_range_exp[i+1] - critical_range_exp[i] for i in range(len(critical_range_exp)-1)]
        lin_steps = [critical_range_linear[i+1] - critical_range_linear[i] for i in range(len(critical_range_linear)-1)]
        
        avg_exp_step = sum(exp_steps) / len(exp_steps)
        avg_lin_step = sum(lin_steps) / len(lin_steps)
        
        print(f"Critical Range (0.0-0.2) Analysis:")
        print(f"  Exponential average step: {avg_exp_step:.6f}")
        print(f"  Linear average step:      {avg_lin_step:.6f}")
        print(f"  Granularity improvement:  {avg_lin_step/avg_exp_step:.1f}x finer")
        
        # Check if exponential provides finer granularity in critical range
        if avg_exp_step < avg_lin_step:
            print("  ✅ ENHANCED GRANULARITY: Exponential provides finer control in critical range")
        else:
            print("  ❌ NO IMPROVEMENT: Exponential doesn't provide finer granularity")
        
        # Verify 0-0.5 range constraint
        max_exp_value = max(exponential_values)
        min_exp_value = min(exponential_values)
        
        print(f"\nRange Verification:")
        print(f"  Exponential range: {min_exp_value:.6f} to {max_exp_value:.6f}")
        
        if min_exp_value >= 0.0 and max_exp_value <= 0.5:
            print("  ✅ RANGE CONSTRAINT: All values within 0-0.5 range")
        else:
            print("  ❌ RANGE VIOLATION: Values outside expected 0-0.5 range")
        
        # Test formula verification (should be e^(6*x) based on review request)
        print(f"\nFormula Verification:")
        formula = exp_data.get('formula', 'Unknown')
        print(f"  Reported formula: {formula}")
        
        # Verify exponential behavior (should be non-linear)
        if len(exponential_values) >= 3:
            # Check if exponential curve (non-linear growth)
            first_diff = exponential_values[1] - exponential_values[0]
            last_diff = exponential_values[-1] - exponential_values[-2]
            
            if last_diff > first_diff * 2:  # Exponential should accelerate
                print("  ✅ EXPONENTIAL BEHAVIOR: Non-linear growth confirmed")
            else:
                print("  ⚠️ LINEAR-LIKE BEHAVIOR: Growth pattern seems linear")

def test_step_resolution():
    """Test the 0.005 step resolution mentioned in review"""
    print("\n🎚️ TESTING STEP RESOLUTION (0.005)")
    print("=" * 60)
    
    # Test fine step resolution
    fine_steps = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
    
    print(f"{'Step':<8} {'Exponential':<12} {'Step Size':<12} {'Resolution':<12}")
    print("-" * 50)
    
    previous_value = None
    
    for step in fine_steps:
        try:
            response = requests.post(
                f"{API_BASE}/threshold-scaling",
                json={"slider_value": step, "use_exponential": True},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                scaled_value = data.get('scaled_threshold', 0)
                
                step_size = ""
                resolution = ""
                
                if previous_value is not None:
                    step_size = f"{scaled_value - previous_value:.6f}"
                    # Check if resolution is adequate (should be detectable)
                    if scaled_value - previous_value > 0.000001:  # 1 micro-threshold
                        resolution = "✅ DETECTABLE"
                    else:
                        resolution = "❌ TOO FINE"
                
                print(f"{step:<8.3f} {scaled_value:<12.6f} {step_size:<12} {resolution:<12}")
                previous_value = scaled_value
                
            else:
                print(f"{step:<8.3f} ERROR        ERROR        ERROR")
                
        except Exception as e:
            print(f"{step:<8.3f} TIMEOUT      TIMEOUT      TIMEOUT")

if __name__ == "__main__":
    # Test health first
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print(f"✅ Backend is healthy, evaluator initialized: {data.get('evaluator_initialized', False)}")
                test_exponential_scaling_formula()
                test_step_resolution()
            else:
                print("❌ Backend unhealthy")
        else:
            print(f"❌ Backend not responding: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")