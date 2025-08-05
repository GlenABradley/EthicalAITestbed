"""
Simple FastAPI server with detailed logging to debug hanging issues.
"""
import logging
import time
import sys
from fastapi import FastAPI, HTTPException, status, Body
from pydantic import BaseModel
from test_evaluator import TestEthicalEvaluator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Initialize evaluator
evaluator = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global evaluator
    logger.info("Starting up server...")
    logger.info("Creating test evaluator...")
    evaluator = TestEthicalEvaluator()
    logger.info("Test evaluator created successfully")
    logger.info("Server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down server...")
    # No cleanup needed for test evaluator
    logger.info("Server shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "test_mode": True,
        "message": "Simple FastAPI server is running"
    }

class EvaluateRequest(BaseModel):
    text: str

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Simple evaluation endpoint."""
    logger.info(f"Evaluation request received for text: {request.text[:50]}...")
    
    if not evaluator:
        logger.error("Evaluator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluator not initialized"
        )
    
    try:
        logger.info("Starting evaluation...")
        start_time = time.time()
        
        # Perform evaluation
        logger.debug("Calling evaluator.evaluate_text()")
        result = evaluator.evaluate_text(request.text)
        
        processing_time = time.time() - start_time
        logger.info(f"Evaluation completed in {processing_time:.4f} seconds")
        
        return {
            "success": True,
            "result": result,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with explicit settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8010,  # Different port to avoid conflicts
        log_level="debug",
        reload=False,
        workers=1,
        timeout_keep_alive=30,
    )
    
    logger.info("Starting server...")
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server stopped")
