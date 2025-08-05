"""
Test script to run the evaluator in a separate process.
This helps isolate the hanging issue by running the test in a clean environment.
"""
import sys
import os
import time
import logging
import multiprocessing
from test_evaluator import TestEthicalEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_evaluator():
    """Run the evaluator in the current process."""
    logger.info("Starting evaluator in current process...")
    try:
        evaluator = TestEthicalEvaluator()
        result = evaluator.evaluate_text("Test message")
        logger.info(f"Evaluation result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error in evaluator: {e}", exc_info=True)
        return False

def run_in_process():
    """Run the evaluator in a separate process."""
    logger.info("Starting new process...")
    p = multiprocessing.Process(target=run_evaluator)
    p.start()
    p.join(timeout=5)  # Wait up to 5 seconds
    
    if p.is_alive():
        logger.error("Process is hanging! Terminating...")
        p.terminate()
        p.join()
        return False
    return p.exitcode == 0

def main():
    logger.info("=== Starting process test ===")
    
    # Test 1: Run in current process
    logger.info("\n=== TEST 1: Running in current process ===")
    try:
        success = run_evaluator()
        logger.info(f"Current process test {'succeeded' if success else 'failed'}")
    except Exception as e:
        logger.error(f"Current process test failed with error: {e}", exc_info=True)
    
    # Test 2: Run in separate process
    logger.info("\n=== TEST 2: Running in separate process ===")
    try:
        success = run_in_process()
        logger.info(f"Separate process test {'succeeded' if success else 'failed'}")
    except Exception as e:
        logger.error(f"Separate process test failed with error: {e}", exc_info=True)
    
    logger.info("\n=== Process test completed ===")
    return 0

if __name__ == "__main__":
    # Ensure we're running in 'spawn' mode for process creation
    multiprocessing.set_start_method('spawn')
    sys.exit(main())
