"""
Test script for the logged server with comprehensive component testing.
"""
import asyncio
import httpx
import time
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8011"
TEST_TEXT = "This is a test message for evaluation."
TEST_ITERATIONS = 3
TEST_DELAY = 1.0  # seconds between tests

async def test_health_check():
    """Test the health check endpoint."""
    logger.info("🔍 Testing health check endpoint...")
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        try:
            response = await client.get(f"{BASE_URL}/health")
            response.raise_for_status()
            duration = time.time() - start_time
            logger.info(f"✅ Health check successful (took {duration:.4f}s)")
            logger.debug(f"Response: {response.json()}")
            return True, duration
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False, 0

async def test_evaluation(text: str):
    """Test the evaluation endpoint."""
    logger.info(f"📝 Testing evaluation with text: {text[:30]}...")
    payload = {
        "text": text,
        "context": {"test": True},
        "parameters": {"test_mode": True},
        "mode": "test",
        "priority": "normal"
    }
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        try:
            response = await client.post(
                f"{BASE_URL}/evaluate",
                json=payload,
                timeout=30.0  # 30 second timeout
            )
            response.raise_for_status()
            duration = time.time() - start_time
            logger.info(f"✅ Evaluation successful (took {duration:.4f}s)")
            logger.debug(f"Response: {response.json()}")
            return True, duration
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return False, 0

async def test_component(component_name: str, test_func, *args):
    """Run a test for a specific component."""
    logger.info(f"\n{'='*20} Testing {component_name} {'='*20}")
    
    results = {
        'success': 0,
        'fail': 0,
        'total_duration': 0.0,
        'durations': []
    }
    
    for i in range(TEST_ITERATIONS):
        logger.info(f"\n🔹 Test {i+1}/{TEST_ITERATIONS}")
        success, duration = await test_func(*args)
        
        if success:
            results['success'] += 1
            results['total_duration'] += duration
            results['durations'].append(duration)
        else:
            results['fail'] += 1
            
        if i < TEST_ITERATIONS - 1:  # Don't sleep after last test
            await asyncio.sleep(TEST_DELAY)
    
    # Calculate statistics
    if results['durations']:
        avg_duration = results['total_duration'] / results['success']
        min_duration = min(results['durations'])
        max_duration = max(results['durations'])
    else:
        avg_duration = min_duration = max_duration = 0.0
    
    # Log summary
    logger.info(f"\n📊 {component_name} Test Summary:")
    logger.info(f"   Success: {results['success']}/{TEST_ITERATIONS}")
    logger.info(f"   Failed: {results['fail']}/{TEST_ITERATIONS}")
    if results['durations']:
        logger.info(f"   Avg Duration: {avg_duration:.4f}s")
        logger.info(f"   Min Duration: {min_duration:.4f}s")
        logger.info(f"   Max Duration: {max_duration:.4f}s")
    
    return results

async def run_all_tests():
    """Run all component tests."""
    logger.info("🚀 Starting comprehensive server tests...")
    start_time = time.time()
    
    test_results = {}
    
    # Test 1: Health Check
    test_results['health_check'] = await test_component("Health Check", test_health_check)
    
    # Test 2: Basic Evaluation
    test_results['basic_evaluation'] = await test_component(
        "Basic Evaluation", 
        test_evaluation, 
        TEST_TEXT
    )
    
    # Test 3: Long Text Evaluation
    long_text = " ".join([TEST_TEXT] * 10)  # Create a longer text
    test_results['long_evaluation'] = await test_component(
        "Long Text Evaluation",
        test_evaluation,
        long_text
    )
    
    # Calculate overall statistics
    total_tests = sum(r['success'] + r['fail'] for r in test_results.values())
    passed_tests = sum(r['success'] for r in test_results.values())
    failed_tests = sum(r['fail'] for r in test_results.values())
    total_duration = time.time() - start_time
    
    # Final report
    logger.info("\n" + "="*50)
    logger.info("🏁 TEST EXECUTION COMPLETE")
    logger.info("="*50)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"✅ Passed: {passed_tests}")
    logger.info(f"❌ Failed: {failed_tests}")
    logger.info(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    logger.info("="*50)
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_duration': total_duration,
            'test_results': test_results
        }, f, indent=2)
    
    logger.info(f"\n📊 Detailed results saved to: {results_file}")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(run_all_tests())
