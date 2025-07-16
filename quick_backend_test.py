#!/usr/bin/env python3
"""
Quick Backend Testing for Ethical AI Developer Testbed
Focused testing with longer timeouts
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

def test_health_check():
    """Test /api/health endpoint"""
    try:
        print("Testing health check...")
        response = requests.get(f"{API_BASE}/health", timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: Service is healthy, evaluator_initialized: {data.get('evaluator_initialized', 'unknown')}")
            return True
        else:
            print(f"❌ Health Check: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check: Error: {str(e)}")
        return False

def test_parameters():
    """Test parameter endpoints"""
    try:
        print("Testing parameter retrieval...")
        response = requests.get(f"{API_BASE}/parameters", timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            params = data.get('parameters', {})
            print(f"✅ Get Parameters: Retrieved parameters with thresholds: virtue={params.get('virtue_threshold')}, deontological={params.get('deontological_threshold')}, consequentialist={params.get('consequentialist_threshold')}")
            return True
        else:
            print(f"❌ Get Parameters: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get Parameters: Error: {str(e)}")
        return False

def test_evaluation():
    """Test text evaluation"""
    try:
        print("Testing text evaluation...")
        test_text = "This is a neutral test message for evaluation"
        
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": test_text},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            processing_time = evaluation.get('processing_time', 0)
            overall_ethical = evaluation.get('overall_ethical', False)
            print(f"✅ Text Evaluation: Processed in {processing_time:.2f}s, ethical: {overall_ethical}")
            return True
        else:
            print(f"❌ Text Evaluation: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Text Evaluation: Error: {str(e)}")
        return False

def test_problematic_text():
    """Test with problematic text"""
    try:
        print("Testing problematic text evaluation...")
        test_text = "You are stupid and worthless"
        
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": test_text},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            evaluation = data.get('evaluation', {})
            overall_ethical = evaluation.get('overall_ethical', True)
            minimal_spans = evaluation.get('minimal_spans', [])
            print(f"✅ Problematic Text: Ethical: {overall_ethical}, Violations found: {len(minimal_spans)}")
            return True
        else:
            print(f"❌ Problematic Text: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Problematic Text: Error: {str(e)}")
        return False

def test_database_operations():
    """Test database operations"""
    try:
        print("Testing database operations...")
        response = requests.get(f"{API_BASE}/evaluations", timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            print(f"✅ Database Operations: Retrieved {count} evaluations from database")
            return True
        else:
            print(f"❌ Database Operations: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database Operations: Error: {str(e)}")
        return False

def test_stress_large_text():
    """Test with large text"""
    try:
        print("Testing large text processing...")
        large_text = "This is a test sentence. " * 200  # ~5KB
        
        response = requests.post(
            f"{API_BASE}/evaluate",
            json={"text": large_text},
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            processing_time = data.get('evaluation', {}).get('processing_time', 0)
            print(f"✅ Large Text: Processed 5KB text in {processing_time:.2f}s")
            return True
        else:
            print(f"❌ Large Text: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Large Text: Error: {str(e)}")
        return False

def main():
    """Run focused backend tests"""
    print(f"🚀 Starting focused backend testing for: {API_BASE}")
    print("=" * 80)
    
    results = []
    
    # Core tests
    results.append(test_health_check())
    results.append(test_parameters())
    results.append(test_evaluation())
    results.append(test_problematic_text())
    results.append(test_database_operations())
    
    # Stress test
    results.append(test_stress_large_text())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 80)
    print("🏁 TESTING COMPLETE")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📊 Total: {total}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)