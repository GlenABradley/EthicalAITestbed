import pytest
import requests
import json
import time

# Define the base URL of the API
BASE_URL = "http://localhost:8001/api"

# Helper function to check if the server is running
def is_server_running():
    """Polls the health endpoint to check if the server is responsive."""
    try:
        # The health endpoint is at /api/health
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Pytest fixture to wait for the server and skip tests if it's not running
@pytest.fixture(scope="module", autouse=True)
def check_server():
    """
    Waits for the server to be available before running tests.
    If the server doesn't start within the timeout, skips the tests.
    """
    max_wait_seconds = 20
    start_time = time.time()
    server_is_up = False
    print("\nWaiting for API server to be available...")
    while time.time() - start_time < max_wait_seconds:
        if is_server_running():
            server_is_up = True
            print("API server is up. Running tests.")
            break
        time.sleep(1)

    if not server_is_up:
        pytest.skip(f"API server did not start within {max_wait_seconds} seconds.")

# Test for the health check endpoint
def test_health_check():
    """
    Tests the /api/health endpoint to ensure the server is healthy.
    """
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["orchestrator_healthy"] is True
    assert data["database_connected"] is True

# Test for the main evaluation endpoint
def test_evaluate_endpoint():
    """
    Tests the /api/evaluate endpoint with a simple text.
    """
    payload = {
        "text": "This is a test sentence for ethical evaluation.",
        "tau_slider": 0.5
    }
    response = requests.post(f"{BASE_URL}/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "overall_ethical" in data
    assert "evaluation" in data
    assert "spans" in data["evaluation"]

# Test for the threshold scaling endpoint
def test_threshold_scaling():
    """
    Tests the /api/threshold-scaling endpoint to ensure dynamic scaling works.
    """
    payload = {
        "slider_value": 0.75,
        "use_exponential": True
    }
    response = requests.post(f"{BASE_URL}/threshold-scaling", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["slider_value"] == 0.75
    assert "scaled_threshold" in data

# Test for getting legacy parameters
def test_get_legacy_parameters():
    """
    Tests the GET /api/parameters legacy endpoint.
    """
    response = requests.get(f"{BASE_URL}/parameters")
    assert response.status_code == 200
    data = response.json()
    assert "virtue_threshold" in data

# Test for updating legacy parameters
def test_update_legacy_parameters():
    """
    Tests the POST /api/update-parameters legacy endpoint.
    """
    payload = {"virtue_threshold": 0.99}
    response = requests.post(f"{BASE_URL}/update-parameters", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Parameters updated successfully"
    assert data["parameters"]["virtue_threshold"] == 0.99

# Test evaluation with a scaled threshold
def test_evaluation_with_scaled_threshold():
    """
    Tests evaluation after applying a new threshold via the scaling endpoint.
    """
    # Set a strict threshold
    eval_payload = {
        "text": "This could be seen as slightly problematic.",
        "tau_slider": 0.01
    }
    response = requests.post(f"{BASE_URL}/evaluate", json=eval_payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_ethical" in data

    # Set a lenient threshold
    eval_payload["tau_slider"] = 0.99
    response_lenient = requests.post(f"{BASE_URL}/evaluate", json=eval_payload)
    assert response_lenient.status_code == 200
    data_lenient = response_lenient.json()
    assert "overall_ethical" in data_lenient

if __name__ == "__main__":
    pytest.main([__file__])

