import requests
import json
import time
from typing import Dict, Any, List, Optional

BASE_URL = "http://localhost:8001"

def test_health() -> bool:
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"✅ Health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_evaluate() -> bool:
    """Test the evaluation endpoint."""
    try:
        data = {
            "text": "This is a test sentence to evaluate.",
            "context": {
                "domain": "test",
                "purpose": "evaluation",
                "cultural_context": "universal"
            },
            "parameters": {
                "confidence_threshold": 0.8,
                "explanation_level": "detailed"
            },
            "mode": "production",
            "priority": "normal"
        }
        print(f"Sending evaluation request: {json.dumps(data, indent=2)}")
        response = requests.post(
            f"{BASE_URL}/api/evaluate",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Evaluation: {response.status_code}")
        if response.status_code == 200:
            print(f"   Result: {json.dumps(response.json(), indent=2)[:200]}...")
        else:
            print(f"   Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return False

def test_parameters() -> bool:
    """Test the parameters endpoint."""
    try:
        # Test GET
        response = requests.get(f"{BASE_URL}/api/parameters")
        print(f"✅ GET Parameters: {response.status_code}")
        if response.status_code == 200:
            print(f"   Current parameters: {json.dumps(response.json(), indent=2)[:200]}...")
        
        # Test POST with updated parameters - using the correct endpoint
        if response.status_code == 200:
            current_params = response.json()
            updated_params = {**current_params, "virtue_threshold": 0.2}
            response = requests.post(
                f"{BASE_URL}/api/update-parameters",  # Correct endpoint for updating parameters
                json=updated_params,
                headers={"Content-Type": "application/json"}
            )
            print(f"✅ POST Parameters (update): {response.status_code}")
            if response.status_code == 200:
                print(f"   Update response: {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        return False
    except Exception as e:
        print(f"❌ Parameters test failed: {str(e)}")
        return False

def test_learning_stats() -> bool:
    """Test the learning stats endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/learning-stats")
        print(f"✅ Learning Stats: {response.status_code}")
        if response.status_code == 200:
            print(f"   Stats: {json.dumps(response.json(), indent=2)[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Learning stats failed: {str(e)}")
        return False

def test_heat_map() -> bool:
    """Test the heat map endpoint."""
    try:
        # The heat map endpoint expects a POST request with text to analyze
        data = {
            "text": "This is a sample text for heat map visualization.",
            "context": {
                "domain": "test",
                "purpose": "visualization"
            }
        }
        response = requests.post(
            f"{BASE_URL}/api/heat-map-mock",  # Correct endpoint for heat map
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Heat Map: {response.status_code}")
        if response.status_code == 200:
            print(f"   Heat map data received")
            print(f"   Response: {json.dumps(response.json(), indent=2)[:200]}...")
        else:
            print(f"   Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Heat map failed: {str(e)}")
        return False

def test_ethics_analysis(endpoint: str, name: str) -> bool:
    """Test an ethics analysis endpoint."""
    try:
        # Special handling for ML Training Guidance endpoint which expects 'content' field
        if endpoint == "ml-training-guidance":
            data = {
                "content": "This is a sample ML training dataset description for ethical analysis. "
                          "It includes various demographic groups and aims to be fair and unbiased."
            }
        else:
            data = {
                "text": "This is a test sentence for ethical analysis.",
                "context": {
                    "domain": "test",
                    "purpose": "analysis"
                },
                "evaluation_id": f"test_analysis_{int(time.time())}"
            }
            
        print(f"Sending request to {endpoint}: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/api/ethics/{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"✅ {name}: {response.status_code}")
        if response.status_code == 200:
            print(f"   Result: {json.dumps(response.json(), indent=2)[:200]}...")
        else:
            print(f"   Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {name} failed: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return False

def test_streaming_status() -> bool:
    """Test the streaming status endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/streaming/status")
        print(f"✅ Streaming Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Status: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Streaming status failed: {str(e)}")
        return False

def main():
    print("🚀 Starting Ethical AI Testbed API Endpoint Tests")
    print("=" * 60)
    
    tests = [
        (test_health, "Health Check"),
        (test_evaluate, "Evaluation"),
        (test_parameters, "Parameters"),
        (test_learning_stats, "Learning Stats"),
        (test_heat_map, "Heat Map"),
        (lambda: test_ethics_analysis("comprehensive-analysis", "Comprehensive Analysis"), "Comprehensive Analysis"),
        (lambda: test_ethics_analysis("meta-analysis", "Meta Analysis"), "Meta Analysis"),
        (lambda: test_ethics_analysis("normative-analysis", "Normative Analysis"), "Normative Analysis"),
        (lambda: test_ethics_analysis("applied-analysis", "Applied Analysis"), "Applied Analysis"),
        (lambda: test_ethics_analysis("ml-training-guidance", "ML Training Guidance"), "ML Training Guidance"),
        (test_streaming_status, "Streaming Status")
    ]
    
    results = []
    for test_func, test_name in tests:
        print(f"\n🔍 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Add a small delay between tests
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the logs above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
