"""
Test script for the refactored Ethical AI Testbed API.

This script tests all endpoints of the refactored API to ensure
they are working correctly with the new architecture.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001/api")

# Test data
TEST_TEXT = """
The AI system will collect user data to improve its recommendations.
This data includes browsing history, purchase patterns, and demographic information.
The system will use this data to personalize the user experience and provide more relevant content.
"""

TEST_REQUEST = {
    "text": TEST_TEXT,
    "context": {
        "domain": "general",
        "model_type": "recommendation"
    },
    "tau_slider": 0.5
}

async def test_health_endpoint():
    """Test the health endpoint."""
    logger.info("Testing health endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            logger.info(f"Health endpoint response: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
            return True
        except Exception as e:
            logger.error(f"Error testing health endpoint: {str(e)}", exc_info=True)
            return False

async def test_parameters_endpoints():
    """Test the parameters endpoints."""
    logger.info("Testing parameters endpoints...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test get parameters
            get_response = await client.get(f"{API_BASE_URL}/parameters")
            logger.info(f"Get parameters response: {get_response.status_code}")
            logger.info(f"Response content: {get_response.text}")
            
            assert get_response.status_code == 200, f"Expected status code 200, got {get_response.status_code}"
            
            # Test update parameters
            update_data = {
                "virtue_threshold": 0.6,
                "deontological_threshold": 0.6,
                "consequentialist_threshold": 0.6
            }
            update_response = await client.post(
                f"{API_BASE_URL}/parameters/update",
                json=update_data
            )
            logger.info(f"Update parameters response: {update_response.status_code}")
            logger.info(f"Response content: {update_response.text}")
            
            assert update_response.status_code == 200, f"Expected status code 200, got {update_response.status_code}"
            
            # Test update thresholds
            thresholds_data = {
                "tau_slider": 0.7
            }
            thresholds_response = await client.post(
                f"{API_BASE_URL}/thresholds/update-all",
                json=thresholds_data
            )
            logger.info(f"Update thresholds response: {thresholds_response.status_code}")
            logger.info(f"Response content: {thresholds_response.text}")
            
            assert thresholds_response.status_code == 200, f"Expected status code 200, got {thresholds_response.status_code}"
            
            return True
        except Exception as e:
            logger.error(f"Error testing parameters endpoints: {str(e)}", exc_info=True)
            return False

async def test_evaluation_endpoint():
    """Test the evaluation endpoint."""
    logger.info("Testing evaluation endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/evaluate",
                json=TEST_REQUEST
            )
            logger.info(f"Evaluation endpoint response: {response.status_code}")
            logger.info(f"Response content: {response.text[:500]}...")  # Truncate long response
            
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
            
            # Save full response to file for inspection
            with open("evaluation_response.json", "w") as f:
                json.dump(response.json(), f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error testing evaluation endpoint: {str(e)}", exc_info=True)
            return False

async def test_learning_stats_endpoint():
    """Test the learning stats endpoint."""
    logger.info("Testing learning stats endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/learning-stats")
            logger.info(f"Learning stats endpoint response: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
            return True
        except Exception as e:
            logger.error(f"Error testing learning stats endpoint: {str(e)}", exc_info=True)
            return False

async def test_visualization_endpoints():
    """Test the visualization endpoints."""
    logger.info("Testing visualization endpoints...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test heat map endpoint
            heat_map_response = await client.get(f"{API_BASE_URL}/heat-map")
            logger.info(f"Heat map endpoint response: {heat_map_response.status_code}")
            logger.info(f"Response content: {heat_map_response.text[:500]}...")  # Truncate long response
            
            # Test mock heat map endpoint
            mock_response = await client.get(f"{API_BASE_URL}/heat-map-mock")
            logger.info(f"Mock heat map endpoint response: {mock_response.status_code}")
            logger.info(f"Response content: {mock_response.text[:500]}...")  # Truncate long response
            
            return True
        except Exception as e:
            logger.error(f"Error testing visualization endpoints: {str(e)}", exc_info=True)
            return False

async def test_ethics_endpoints():
    """Test the ethics endpoints."""
    logger.info("Testing ethics endpoints...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test meta ethics analysis
            meta_response = await client.post(
                f"{API_BASE_URL}/ethics/meta-analysis",
                json=TEST_REQUEST
            )
            logger.info(f"Meta ethics analysis endpoint response: {meta_response.status_code}")
            logger.info(f"Response content: {meta_response.text[:500]}...")  # Truncate long response
            
            # Test normative ethics analysis
            normative_response = await client.post(
                f"{API_BASE_URL}/ethics/normative-analysis",
                json=TEST_REQUEST
            )
            logger.info(f"Normative ethics analysis endpoint response: {normative_response.status_code}")
            logger.info(f"Response content: {normative_response.text[:500]}...")  # Truncate long response
            
            # Test applied ethics analysis
            applied_response = await client.post(
                f"{API_BASE_URL}/ethics/applied-analysis",
                json=TEST_REQUEST
            )
            logger.info(f"Applied ethics analysis endpoint response: {applied_response.status_code}")
            logger.info(f"Response content: {applied_response.text[:500]}...")  # Truncate long response
            
            # Test ML training guidance
            ml_response = await client.post(
                f"{API_BASE_URL}/ethics/ml-training-guidance",
                json=TEST_REQUEST
            )
            logger.info(f"ML training guidance endpoint response: {ml_response.status_code}")
            logger.info(f"Response content: {ml_response.text[:500]}...")  # Truncate long response
            
            # Test comprehensive analysis
            comprehensive_response = await client.post(
                f"{API_BASE_URL}/ethics/comprehensive-analysis",
                json=TEST_REQUEST
            )
            logger.info(f"Comprehensive ethics analysis endpoint response: {comprehensive_response.status_code}")
            logger.info(f"Response content: {comprehensive_response.text[:500]}...")  # Truncate long response
            
            return True
        except Exception as e:
            logger.error(f"Error testing ethics endpoints: {str(e)}", exc_info=True)
            return False

async def run_tests():
    """Run all API tests."""
    logger.info("Starting API tests...")
    
    # Test results
    results = {
        "health": await test_health_endpoint(),
        "parameters": await test_parameters_endpoints(),
        "evaluation": await test_evaluation_endpoint(),
        "learning_stats": await test_learning_stats_endpoint(),
        "visualization": await test_visualization_endpoints(),
        "ethics": await test_ethics_endpoints()
    }
    
    # Log summary
    logger.info("Test results summary:")
    for test, result in results.items():
        logger.info(f"{test}: {'PASS' if result else 'FAIL'}")
    
    # Overall result
    overall_result = all(results.values())
    logger.info(f"Overall test result: {'PASS' if overall_result else 'FAIL'}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_tests())
