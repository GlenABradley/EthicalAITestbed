"""
Simplest possible FastAPI server with test evaluator.
"""
import logging
import time
from fastapi import FastAPI
from test_evaluator import get_test_evaluator, TestEthicalEvaluator, EthicalParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")
    # Try to initialize the evaluator at startup
    try:
        logger.info("Initializing test evaluator...")
        app.state.evaluator = get_test_evaluator()
        logger.info("Test evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize test evaluator: {e}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "evaluator_initialized": hasattr(app.state, 'evaluator')
    }

@app.get("/evaluate")
async def evaluate():
    """Test evaluation endpoint"""
    if not hasattr(app.state, 'evaluator'):
        return {"error": "Evaluator not initialized"}, 500
    
    try:
        result = app.state.evaluator.evaluate_text("Test input")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting simple evaluator server...")
    uvicorn.run("simple_evaluator_server:app", host="0.0.0.0", port=8005, log_level="info")
