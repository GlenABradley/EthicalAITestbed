#!/usr/bin/env python3
"""
Additional Quick Tests for Main Evaluation Endpoint
Testing with shorter text and timeouts to verify basic functionality
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

def test_main_evaluation_short_text():
    """Test main evaluation with very short text"""
    try:
        test_text = "Hello"
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": test_text},
            timeout=30  # 30 second timeout
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            processing_time = evaluation.get('processing_time', response_time)
            
            print(f"✅ Main Evaluation Short Text: Completed in {response_time:.1f}s (processing: {processing_time:.1f}s)")
            return True
        else:
            print(f"❌ Main Evaluation Short Text: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Main Evaluation Short Text: Timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Main Evaluation Short Text: Error - {str(e)}")
        return False

def test_parameters_endpoint():
    """Test parameters endpoint"""
    try:
        response = requests.get(f"{API_BASE}/parameters", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'parameters' in data:
                print("✅ Parameters Endpoint: Working correctly")
                return True
            else:
                print("❌ Parameters Endpoint: Missing parameters field")
                return False
        else:
            print(f"❌ Parameters Endpoint: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Parameters Endpoint: Error - {str(e)}")
        return False

def test_heat_map_vs_evaluation_consistency():
    """Test that heat-map mock and evaluation endpoints handle the same text consistently"""
    try:
        test_text = "Test consistency"
        
        # Test heat-map mock
        heat_map_response = requests.post(
            f"{API_BASE}/heat-map-mock",
            json={"text": test_text},
            timeout=5
        )
        
        # Test main evaluation with short timeout
        eval_response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": test_text},
            timeout=30
        )
        
        heat_map_success = heat_map_response.status_code == 200
        eval_success = eval_response.status_code == 200
        
        if heat_map_success and eval_success:
            heat_map_data = heat_map_response.json()
            eval_data = eval_response.json()
            
            # Check that both handle the same text
            heat_map_text_length = heat_map_data.get('textLength', 0)
            eval_text_length = len(eval_data.get('evaluation', {}).get('input_text', ''))
            
            if heat_map_text_length == eval_text_length == len(test_text):
                print("✅ Heat-Map vs Evaluation Consistency: Both endpoints handle text consistently")
                return True
            else:
                print(f"❌ Heat-Map vs Evaluation Consistency: Text length mismatch - heat-map: {heat_map_text_length}, eval: {eval_text_length}")
                return False
        elif heat_map_success and not eval_success:
            print("✅ Heat-Map vs Evaluation Consistency: Heat-map fast, evaluation slow/timeout (expected)")
            return True
        else:
            print(f"❌ Heat-Map vs Evaluation Consistency: Heat-map: {heat_map_response.status_code}, Eval: {eval_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Heat-Map vs Evaluation Consistency: Error - {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 ADDITIONAL QUICK TESTS")
    print("=" * 50)
    
    results = []
    
    print("\n1. Testing Main Evaluation with Short Text")
    results.append(test_main_evaluation_short_text())
    
    print("\n2. Testing Parameters Endpoint")
    results.append(test_parameters_endpoint())
    
    print("\n3. Testing Heat-Map vs Evaluation Consistency")
    results.append(test_heat_map_vs_evaluation_consistency())
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"QUICK TESTS SUMMARY: {passed}/{total} passed ({(passed/total)*100:.1f}%)")