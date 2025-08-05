"""
Concise endpoint tester for Ethical AI Testbed backend.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8011"

# Define test endpoints with expected status codes and methods
# Note: Updated to match actual implemented endpoints in server.py
TEST_CASES = [
    {"path": "/api/health", "method": "GET", "expected_status": 200, "data": None, "desc": "Health check"},
    {"path": "/api/evaluate", "method": "POST", "expected_status": 200, "data": {"text": "This is a test sentence for ethical evaluation."}, "desc": "Evaluate text"},
    
    # Legacy compatibility endpoints
    {"path": "/api/parameters", "method": "GET", "expected_status": 200, "data": None, "desc": "Get parameters"},
    {"path": "/api/learning-stats", "method": "GET", "expected_status": 200, "data": None, "desc": "Get learning stats"},
    
    # Visualization endpoints
    {"path": "/api/heat-map-mock", "method": "POST", "expected_status": 200, "data": {"text": "This is a test sentence for heat map generation."}, "desc": "Generate heat map"},
    
    # ML Ethics Assistant endpoints
    {"path": "/api/ethics/comprehensive-analysis", "method": "POST", "expected_status": 200, "data": {"text": "This is a test for comprehensive ethical analysis."}, "desc": "Comprehensive analysis"},
    {"path": "/api/ethics/meta-analysis", "method": "POST", "expected_status": 200, "data": {"text": "This is a test for meta-ethical analysis."}, "desc": "Meta-ethical analysis"},
    {"path": "/api/ethics/normative-analysis", "method": "POST", "expected_status": 200, "data": {"text": "This is a test for normative ethical analysis."}, "desc": "Normative analysis"},
    {"path": "/api/ethics/applied-analysis", "method": "POST", "expected_status": 200, "data": {"text": "This is a test for applied ethical analysis."}, "desc": "Applied analysis"},
    {"path": "/api/ethics/ml-training-guidance", "method": "POST", "expected_status": 200, "data": {"content": "This is a test for ML training guidance."}, "desc": "ML training guidance"},
    
    # Real-time streaming
    {"path": "/api/streaming/status", "method": "GET", "expected_status": 200, "data": None, "desc": "Streaming status"}
]

async def test_endpoint(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single endpoint with the given test case."""
    url = f"{BASE_URL}{test_case['path']}"
    method = test_case["method"]
    payload = test_case["data"]
    expected_status = test_case["expected_status"]
    desc = test_case["desc"]
    
    result = {
        "endpoint": test_case['path'],
        "method": method,
        "success": False,
        "status": None,
        "time_ms": 0,
        "error": None
    }
    
    try:
        async with httpx.AsyncClient() as client:
            start = asyncio.get_event_loop().time()
            
            if method == "GET":
                response = await client.get(url, timeout=10.0)
            elif method == "POST":
                response = await client.post(url, json=payload, timeout=10.0)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            result["time_ms"] = round((asyncio.get_event_loop().time() - start) * 1000, 2)
            result["status"] = response.status_code
            result["success"] = response.status_code == expected_status
            
            if not result["success"]:
                result["error"] = f"Expected status {expected_status}, got {response.status_code}"
                try:
                    error_data = response.json()
                    result["error"] = error_data.get("detail", result["error"])
                except:
                    result["error"] = response.text[:200]
            
            status_emoji = "✅" if result["success"] else "❌"
            logger.info(f"{status_emoji} {method} {test_case['path']} - {result['status']} ({result['time_ms']}ms) - {desc}")
            
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ {method} {test_case['path']} - Error: {e}")
    
    return result

async def run_tests():
    """Run all test cases and report results."""
    logger.info("🚀 Starting endpoint tests...")
    
    results = []
    for test_case in TEST_CASES:
        result = await test_endpoint(test_case)
        results.append(result)
        await asyncio.sleep(0.1)  # Small delay between tests
    
    # Print summary
    logger.info("\n📊 Test Summary:")
    logger.info("-" * 50)
    
    success_count = sum(1 for r in results if r["success"])
    total = len(results)
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        logger.info(f"{status} {r['method']} {r['endpoint']} - {r['status']} ({r['time_ms']}ms)" + 
                   (f" - Error: {r['error']}" if r["error"] else ""))
    
    logger.info("-" * 50)
    logger.info(f"🎯 Success: {success_count}/{total} ({success_count/total*100:.1f}%)")
    
    # Save results to file
    timestamp = int(asyncio.get_event_loop().time())
    filename = f"test_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"results": results, "success_rate": success_count/total}, f, indent=2)
    logger.info(f"\n📊 Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_tests())
