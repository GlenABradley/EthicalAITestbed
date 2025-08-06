#!/usr/bin/env python3
"""
API Endpoint Test Script for Ethical AI Testbed

This script tests all the main API endpoints of the Ethical AI Testbed backend
to verify they are working correctly after refactoring.
"""

import json
import requests
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8001"
ENDPOINTS = [
    # Core endpoints
    "/api/health",
    "/api/evaluate",
    "/api/parameters",
    "/api/learning-stats",
    
    # Visualization endpoints
    "/api/heat-map-mock",
    
    # Ethics analysis endpoints
    "/api/ethics/comprehensive-analysis",
    "/api/ethics/meta-analysis",
    "/api/ethics/normative-analysis",
    "/api/ethics/applied-analysis",
    "/api/ethics/ml-training-guidance"
]

# Test data for POST requests
TEST_DATA = {
    "/api/evaluate": {
        "text": "This is a test text for ethical evaluation.",
        "context": {"domain": "testing"},
        "parameters": {"explanation_level": "detailed"},
        "mode": "development"
    },
    "/api/heat-map-mock": {
        "text": "This is a test text for heat map visualization."
    },
    "/api/ethics/comprehensive-analysis": {
        "text": "This is a test for comprehensive ethics analysis.",
        "context": {"domain": "testing"}
    },
    "/api/ethics/meta-analysis": {
        "text": "This is a test for meta-ethics analysis.",
        "context": {"domain": "testing"}
    },
    "/api/ethics/normative-analysis": {
        "text": "This is a test for normative ethics analysis.",
        "context": {"domain": "testing"}
    },
    "/api/ethics/applied-analysis": {
        "text": "This is a test for applied ethics analysis.",
        "context": {"domain": "testing"}
    },
    "/api/ethics/ml-training-guidance": {
        "content": "This is a test for ML training guidance.",
        "context": {"domain": "testing"}
    }
}

def test_endpoint(endpoint):
    """Test a specific API endpoint and return the result."""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"Testing endpoint: {endpoint}")
    
    # Determine if this is a GET or POST request
    if endpoint in TEST_DATA:
        # POST request with JSON data
        try:
            response = requests.post(url, json=TEST_DATA[endpoint], timeout=10)
        except requests.exceptions.RequestException as e:
            return {
                "endpoint": endpoint,
                "status": "error",
                "error": str(e),
                "status_code": None,
                "response": None
            }
    else:
        # GET request
        try:
            response = requests.get(url, timeout=10)
        except requests.exceptions.RequestException as e:
            return {
                "endpoint": endpoint,
                "status": "error",
                "error": str(e),
                "status_code": None,
                "response": None
            }
    
    # Process response
    result = {
        "endpoint": endpoint,
        "status": "success" if response.status_code == 200 else "failure",
        "status_code": response.status_code,
        "response": None
    }
    
    # Try to parse JSON response
    try:
        if response.text:
            result["response"] = response.json()
    except json.JSONDecodeError:
        result["response"] = response.text[:100] + "..." if len(response.text) > 100 else response.text
    
    return result

def main():
    """Main function to test all endpoints and save results."""
    print(f"Starting API endpoint tests at {datetime.now().isoformat()}")
    print(f"Base URL: {BASE_URL}")
    print(f"Testing {len(ENDPOINTS)} endpoints")
    print("-" * 50)
    
    results = []
    
    for endpoint in ENDPOINTS:
        result = test_endpoint(endpoint)
        results.append(result)
        
        # Print status
        status_symbol = "✅" if result["status"] == "success" else "❌"
        print(f"{status_symbol} {endpoint}: {result['status_code']}")
        
        # Add a small delay between requests
        import time
        time.sleep(0.5)
    
    print("-" * 50)
    
    # Count successes and failures
    successes = sum(1 for r in results if r["status"] == "success")
    failures = len(results) - successes
    
    print(f"Results: {successes} successful, {failures} failed")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_test_results_{timestamp}.json"
    
    with open(filename, "w") as fp:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "summary": {
                "total": len(results),
                "successful": successes,
                "failed": failures
            },
            "results": results
        }, fp, indent=2)
    
    print(f"Results saved to {filename}")
    
    # Return non-zero exit code if any tests failed
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
