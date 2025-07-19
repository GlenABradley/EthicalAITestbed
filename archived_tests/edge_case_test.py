#!/usr/bin/env python3
"""
Edge Case Testing for Ethical AI Developer Testbed
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_edge_cases():
    """Test various edge cases"""
    print(f"🔥 Testing edge cases for: {API_BASE}")
    print("=" * 60)
    
    # Test 1: Empty text
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"text": ""}, timeout=30)
        print(f"Empty text: HTTP {response.status_code}")
    except Exception as e:
        print(f"Empty text: Error - {str(e)}")
    
    # Test 2: Very short text
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"text": "a"}, timeout=30)
        print(f"Single character: HTTP {response.status_code}")
    except Exception as e:
        print(f"Single character: Error - {str(e)}")
    
    # Test 3: Unicode text
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"text": "Hello 🌍 世界"}, timeout=60)
        print(f"Unicode text: HTTP {response.status_code}")
    except Exception as e:
        print(f"Unicode text: Error - {str(e)}")
    
    # Test 4: Invalid JSON structure
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"wrong_field": "test"}, timeout=30)
        print(f"Invalid JSON structure: HTTP {response.status_code}")
    except Exception as e:
        print(f"Invalid JSON structure: Error - {str(e)}")
    
    # Test 5: Null text
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"text": None}, timeout=30)
        print(f"Null text: HTTP {response.status_code}")
    except Exception as e:
        print(f"Null text: Error - {str(e)}")
    
    # Test 6: Non-existent calibration test
    try:
        response = requests.post(f"{API_BASE}/run-calibration-test/fake-id", timeout=30)
        print(f"Non-existent calibration test: HTTP {response.status_code}")
    except Exception as e:
        print(f"Non-existent calibration test: Error - {str(e)}")
    
    # Test 7: Performance metrics
    try:
        response = requests.get(f"{API_BASE}/performance-metrics", timeout=30)
        print(f"Performance metrics: HTTP {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"  Total evaluations: {metrics.get('total_evaluations', 0)}")
                print(f"  Avg processing time: {metrics.get('average_processing_time', 0):.2f}s")
    except Exception as e:
        print(f"Performance metrics: Error - {str(e)}")
    
    # Test 8: Calibration tests list
    try:
        response = requests.get(f"{API_BASE}/calibration-tests", timeout=30)
        print(f"Calibration tests list: HTTP {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {data.get('count', 0)} calibration tests")
    except Exception as e:
        print(f"Calibration tests list: Error - {str(e)}")

if __name__ == "__main__":
    test_edge_cases()