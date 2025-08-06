#!/usr/bin/env python3
"""
Test Script for Tracing Evaluation Pipeline

This script helps identify where evaluation responses are coming from by:
1. Sending requests to different endpoints with detailed logging
2. Tracing the request/response flow
3. Identifying caching behavior
4. Logging detailed timing and response information
"""

import asyncio
import aiohttp
import time
import json
import uuid
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:8001"  # Update if your server runs on a different port
TEST_TEXT = "This is a test message for evaluating the ethical considerations of AI systems."

# Test cases with different endpoints and parameters
TEST_CASES = [
    {
        "name": "Basic Evaluation",
        "endpoint": "/api/evaluate",
        "method": "POST",
        "payload": {
            "text": TEST_TEXT,
            "context": {"test": True},
            "parameters": {"test_mode": True}
        }
    },
    {
        "name": "Evaluation with Thresholds",
        "endpoint": "/api/evaluate",
        "method": "POST",
        "payload": {
            "text": TEST_TEXT,
            "tau_slider": 0.5,
            "context": {"test": True},
            "parameters": {"test_mode": True}
        }
    },
    {
        "name": "Update Thresholds",
        "endpoint": "/api/thresholds/update-all",
        "method": "POST",
        "payload": {
            "virtue_threshold": 0.5,
            "deontological_threshold": 0.5,
            "consequentialist_threshold": 0.5,
            "use_exponential": True
        }
    },
    {
        "name": "Get Parameters",
        "endpoint": "/api/parameters",
        "method": "GET"
    }
]

async def send_request(session: aiohttp.ClientSession, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Send a request and return detailed timing and response information."""
    url = f"{BASE_URL}{test_case['endpoint']}"
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4())
    }
    
    start_time = time.time()
    
    try:
        if test_case["method"] == "POST":
            async with session.post(
                url, 
                json=test_case.get("payload", {}), 
                headers=headers
            ) as response:
                response_data = await response.json()
                status = response.status
        else:
            async with session.get(url, headers=headers) as response:
                response_data = await response.json()
                status = response.status
                
        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return {
            "test_name": test_case["name"],
            "endpoint": test_case["endpoint"],
            "status": status,
            "response_time_ms": round(elapsed, 2),
            "response_data": response_data,
            "success": 200 <= status < 300,
            "error": None
        }
        
    except Exception as e:
        return {
            "test_name": test_case["name"],
            "endpoint": test_case["endpoint"],
            "status": 0,
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "response_data": None,
            "success": False,
            "error": str(e)
        }

async def run_tests():
    """Run all test cases and display results."""
    print("🚀 Starting Ethical AI Pipeline Test")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Test Text: {TEST_TEXT[:100]}...")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, test_case) for test_case in TEST_CASES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Test {i + 1} failed with error: {str(result)}")
                continue
                
            print(f"\n📊 Test {i + 1}: {result['test_name']}")
            print(f"   Endpoint: {result['endpoint']}")
            print(f"   Status: {result['status']} ({'✅' if result['success'] else '❌'})")
            print(f"   Response Time: {result['response_time_ms']}ms")
            
            if result['error']:
                print(f"   Error: {result['error']}")
            
            # Print response data summary
            if result['response_data']:
                print("   Response Data:")
                print(f"   - Type: {type(result['response_data']).__name__}")
                if isinstance(result['response_data'], dict):
                    print(f"   - Keys: {', '.join(result['response_data'].keys())}")
                    # Print first few items of the response
                    for key, value in list(result['response_data'].items())[:3]:
                        print(f"   - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                else:
                    print(f"   - {str(result['response_data'])[:200]}...")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(run_tests())
