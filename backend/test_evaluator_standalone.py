"""
Standalone test for the TestEthicalEvaluator.
This script tests the test evaluator outside of FastAPI to help isolate the hanging issue.
"""
import asyncio
import time
from test_evaluator import TestEthicalEvaluator, get_test_evaluator

async def test_evaluator():
    """Test the evaluator's basic functionality."""
    print("Starting test evaluator test...")
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = get_test_evaluator()
    
    # Test evaluation
    test_text = "This is a test message for evaluation."
    print(f"Evaluating text: {test_text}")
    
    try:
        # Test synchronous evaluation
        print("\nTesting synchronous evaluation...")
        start_time = time.time()
        result = evaluator.evaluate_text(test_text)
        elapsed = time.time() - start_time
        
        print(f"Synchronous evaluation completed in {elapsed:.4f} seconds")
        print(f"Result: {result}")
        
        # Test async evaluation (wrap in asyncio.to_thread)
        print("\nTesting async evaluation...")
        start_time = time.time()
        result = await asyncio.to_thread(evaluator.evaluate_text, test_text)
        elapsed = time.time() - start_time
        
        print(f"Async evaluation completed in {elapsed:.4f} seconds")
        print(f"Result: {result}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_evaluator())
    exit(0 if success else 1)
