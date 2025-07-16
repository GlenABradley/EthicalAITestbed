#!/usr/bin/env python3
"""
Final Comprehensive Test Summary
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/frontend/.env')

BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
API_BASE = f"{BACKEND_URL}/api"

def test_critical_functionality():
    """Test critical functionality that must work"""
    print(f"🎯 Final Critical Functionality Test for: {API_BASE}")
    print("=" * 70)
    
    critical_tests = []
    
    # 1. Health Check
    try:
        response = requests.get(f"{API_BASE}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            evaluator_init = data.get('evaluator_initialized', False)
            critical_tests.append(("Health Check", True, f"Service healthy, evaluator: {evaluator_init}"))
        else:
            critical_tests.append(("Health Check", False, f"HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Health Check", False, f"Error: {str(e)}"))
    
    # 2. Parameter Management
    try:
        response = requests.get(f"{API_BASE}/parameters", timeout=30)
        if response.status_code == 200:
            critical_tests.append(("Parameter Retrieval", True, "Parameters retrieved successfully"))
        else:
            critical_tests.append(("Parameter Retrieval", False, f"HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Parameter Retrieval", False, f"Error: {str(e)}"))
    
    # 3. Basic Text Evaluation
    try:
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": "Hello world"},
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            processing_time = data.get('evaluation', {}).get('processing_time', 0)
            critical_tests.append(("Basic Evaluation", True, f"Processed in {processing_time:.2f}s"))
        else:
            critical_tests.append(("Basic Evaluation", False, f"HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Basic Evaluation", False, f"Error: {str(e)}"))
    
    # 4. Database Operations
    try:
        response = requests.get(f"{API_BASE}/evaluations", timeout=30)
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            critical_tests.append(("Database Access", True, f"Retrieved {count} evaluations"))
        else:
            critical_tests.append(("Database Access", False, f"HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Database Access", False, f"Error: {str(e)}"))
    
    # 5. Error Handling
    try:
        response = requests.post(f"{API_BASE}/evaluate", json={"text": None}, timeout=30)
        if response.status_code == 422:
            critical_tests.append(("Error Handling", True, "Properly rejects invalid input"))
        else:
            critical_tests.append(("Error Handling", False, f"Unexpected response: HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Error Handling", False, f"Error: {str(e)}"))
    
    # 6. Performance Metrics
    try:
        response = requests.get(f"{API_BASE}/performance-metrics", timeout=30)
        if response.status_code == 200:
            critical_tests.append(("Performance Metrics", True, "Metrics endpoint working"))
        else:
            critical_tests.append(("Performance Metrics", False, f"HTTP {response.status_code}"))
    except Exception as e:
        critical_tests.append(("Performance Metrics", False, f"Error: {str(e)}"))
    
    # Print results
    passed = 0
    failed = 0
    
    for test_name, success, message in critical_tests:
        if success:
            print(f"✅ {test_name}: {message}")
            passed += 1
        else:
            print(f"❌ {test_name}: {message}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"🏁 CRITICAL FUNCTIONALITY SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 ALL CRITICAL FUNCTIONALITY WORKING!")
    else:
        print("⚠️  CRITICAL ISSUES FOUND!")
    
    return failed == 0

if __name__ == "__main__":
    test_critical_functionality()