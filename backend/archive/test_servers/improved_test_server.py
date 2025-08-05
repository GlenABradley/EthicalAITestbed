"""
Improved test server for the Ethical AI Testbed.
This version follows the same pattern as the working detailed_step_server.py
"""
import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global flag to track server status
server_shutting_down = False

# Add signal handlers for graceful shutdown
def handle_shutdown(signum, frame):
    global server_shutting_down
    if not server_shutting_down:
        logger.info("Shutdown signal received. Starting graceful shutdown...")
        server_shutting_down = True
        # This will trigger the FastAPI shutdown event
        raise KeyboardInterrupt()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Import test evaluator with logging
try:
    logger.info("Importing test_evaluator...")
    from test_evaluator import get_test_evaluator, TestEthicalEvaluator, EthicalParameters
    logger.info("Successfully imported test_evaluator"
                f"\n  - TestEthicalEvaluator: {TestEthicalEvaluator}"
                f"\n  - get_test_evaluator: {get_test_evaluator}"
                f"\n  - EthicalParameters: {EthicalParameters}")
except Exception as e:
    logger.error(f"Failed to import test_evaluator: {e}", exc_info=True)
    raise

# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up improved test server...")
    app.state.startup_time = time.time()
    
    # Initialize test evaluator
    try:
        logger.info("Initializing test evaluator...")
        app.state.evaluator = get_test_evaluator()
        logger.info("Test evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize test evaluator: {e}", exc_info=True)
        raise
    
    logger.info("Improved test server startup complete")
    
    # App is running
    yield  
    
    # Shutdown
    logger.info("Shutting down improved test server...")
    if hasattr(app.state, 'evaluator'):
        logger.info("Cleaning up test evaluator...")
        del app.state.evaluator
    logger.info("Improved test server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Ethical AI Testbed - Improved Test Server",
    description="Improved test server with better lifecycle management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class EvaluationRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = {}
    parameters: Optional[Dict[str, Any]] = {}
    mode: str = "test"
    priority: str = "normal"

class EvaluationResponse(BaseModel):
    request_id: str
    overall_ethical: bool
    processing_time: float
    test_mode: bool
    message: str
    input_text: str
    tokens: List[str]
    spans: List[Dict[str, Any]] = []
    minimal_spans: List[Dict[str, Any]] = []

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    test_mode: bool = True
    message: str = "Improved test server is running"

# API Endpoints
@app.get("/api/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Health check endpoint with detailed status information
    """
    try:
        status = {
            "status": "healthy",
            "timestamp": time.ctime(),
            "uptime_seconds": time.time() - app.state.startup_time,
            "test_mode": True,
            "message": "Improved test server is running"
        }
        logger.info(f"Health check: {status}")
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_text(
    request: EvaluationRequest,
):
    """
    Lightweight evaluation endpoint for testing.
    """
    start_time = time.time()
    request_id = f"test_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"Evaluation request {request_id} started")
        
        if not hasattr(app.state, 'evaluator'):
            logger.error("Evaluator not initialized")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluator not initialized"
            )
        
        # Perform evaluation
        result = app.state.evaluator.evaluate_text(request.text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "request_id": request_id,
            "overall_ethical": result.get("overall_ethical", True),
            "processing_time": processing_time,
            "test_mode": True,
            "message": "Evaluation completed successfully",
            "input_text": result.get("input_text", request.text),
            "tokens": result.get("tokens", []),
            "spans": result.get("spans", []),
            "minimal_spans": result.get("minimal_spans", [])
        }
        
        logger.info(f"Evaluation request {request_id} completed in {processing_time:.4f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with explicit settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8007,  # Different port to avoid conflicts
        log_level="info",
        reload=False,
        workers=1,
        timeout_keep_alive=30,
    )
    
    logger.info("Starting improved test server...")
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server stopped")
