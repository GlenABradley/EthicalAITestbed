#!/usr/bin/env python3
"""
Stress Testing for Ethical AI Developer Testbed
"""

import requests
import json
import os
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_parameter_extremes():
    """Test with extreme parameter values"""
    print("Testing extreme parameter values...")
    
    # Test with extreme thresholds
    extreme_params = {
        "virtue_threshold": 0.0,
        "deontological_threshold": 1.0,
        "consequentialist_threshold": 0.5
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/update-parameters",
            json={"parameters": extreme_params},
            timeout=30
        )
        print(f"Extreme parameters update: HTTP {response.status_code}")
        
        if response.status_code == 200:
            # Test evaluation with extreme parameters
            eval_response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": "This is a test with extreme parameters"},
                timeout=60
            )
            print(f"Evaluation with extreme params: HTTP {eval_response.status_code}")
            
            if eval_response.status_code == 200:
                data = eval_response.json()
                evaluation = data.get('evaluation', {})
                print(f"  Result: ethical={evaluation.get('overall_ethical')}, processing_time={evaluation.get('processing_time', 0):.2f}s")
        
    except Exception as e:
        print(f"Extreme parameters test: Error - {str(e)}")

def test_concurrent_requests():
    """Test concurrent requests"""
    print("Testing concurrent requests...")
    
    results = []
    errors = []
    
    def make_request(thread_id):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/evaluate",
                json={"text": f"Concurrent test {thread_id}"},
                timeout=120
            )
            end_time = time.time()
            results.append((thread_id, response.status_code, end_time - start_time))
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Launch 3 concurrent requests (reduced from 10 due to processing time)
    threads = []
    start_time = time.time()
    
    for i in range(3):
        thread = threading.Thread(target=make_request, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    success_count = sum(1 for _, status, _ in results if status == 200)
    
    print(f"Concurrent requests: {success_count}/3 succeeded in {total_time:.2f}s")
    for thread_id, status, duration in results:
        print(f"  Thread {thread_id}: HTTP {status} in {duration:.2f}s")
    
    if errors:
        print(f"  Errors: {errors}")

def test_malformed_data():
    """Test with malformed data"""
    print("Testing malformed data...")
    
    test_cases = [
        ({"text": 123}, "Integer instead of string"),
        ({"text": []}, "Array instead of string"),
        ({"text": {"nested": "object"}}, "Object instead of string"),
        ({}, "Missing text field"),
        ({"text": "valid", "extra": "field"}, "Extra fields"),
    ]
    
    for payload, description in test_cases:
        try:
            response = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=30)
            print(f"  {description}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  {description}: Error - {str(e)}")

def test_database_stress():
    """Test database operations under stress"""
    print("Testing database stress...")
    
    # Create multiple calibration tests rapidly
    success_count = 0
    for i in range(3):
        try:
            response = requests.post(
                f"{API_BASE}/calibration-test",
                json={
                    "text": f"Stress test calibration {i}",
                    "expected_result": "ethical"
                },
                timeout=30
            )
            if response.status_code == 200:
                success_count += 1
        except Exception as e:
            print(f"  Calibration test {i}: Error - {str(e)}")
    
    print(f"Rapid calibration test creation: {success_count}/3 succeeded")
    
    # Test large limit on evaluations
    try:
        response = requests.get(f"{API_BASE}/evaluations?limit=50", timeout=30)
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            print(f"Large evaluation query: Retrieved {count} records")
        else:
            print(f"Large evaluation query: HTTP {response.status_code}")
    except Exception as e:
        print(f"Large evaluation query: Error - {str(e)}")

def main():
    """Run stress tests"""
    print(f"🔥 Starting stress testing for: {API_BASE}")
    print("=" * 60)
    
    test_parameter_extremes()
    print()
    
    test_concurrent_requests()
    print()
    
    test_malformed_data()
    print()
    
    test_database_stress()
    print()
    
    print("🏁 Stress testing complete")

if __name__ == "__main__":
    main()