"""
Minimal test script to isolate the test evaluator issue.
"""
import logging
import sys
from test_evaluator import get_test_evaluator, TestEthicalEvaluator, EthicalParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def test_evaluator_initialization():
    """Test if we can initialize the test evaluator"""
    logger.info("Starting test evaluator initialization test...")
    
    try:
        # Test 1: Direct instantiation with default parameters
        logger.info("Test 1: Direct instantiation with default parameters")
        evaluator1 = TestEthicalEvaluator()
        logger.info("Successfully created evaluator with default parameters")
        
        # Test 2: Direct instantiation with explicit parameters
        logger.info("Test 2: Direct instantiation with explicit parameters")
        params = EthicalParameters(
            enable_graph_attention=False,
            enable_intent_hierarchy=False,
            enable_causal_analysis=False,
            enable_uncertainty_analysis=False,
            enable_purpose_alignment=False
        )
        evaluator2 = TestEthicalEvaluator(parameters=params)
        logger.info("Successfully created evaluator with explicit parameters")
        
        # Test 3: Using the get_test_evaluator() function
        logger.info("Test 3: Using get_test_evaluator() function")
        evaluator3 = get_test_evaluator()
        logger.info("Successfully got evaluator from get_test_evaluator()")
        
        logger.info("All test evaluator initialization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting test evaluator test script...")
    success = test_evaluator_initialization()
    if success:
        logger.info("Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Test failed!")
        sys.exit(1)
