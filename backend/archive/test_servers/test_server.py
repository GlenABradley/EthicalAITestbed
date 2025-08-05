"""
Lightweight test server for the Ethical AI Testbed.
This version uses a simplified evaluator for testing purposes.
"""
"""
Lightweight test server for the Ethical AI Testbed.
This version uses a simplified evaluator for testing purposes.
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
from pydantic import BaseModel

from test_evaluator import get_test_evaluator, TestEthicalEvaluator
from health_check import health_checker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Middleware for request/response logging
async def log_requests(request: Request, call_next):
    request_id = str(id(request))
    logger.info(f"Request started: {request.method} {request.url.path} (ID: {request_id})")
    
    try:
        response = await call_next(request)
        logger.info(f"Request completed: {request.method} {request.url.path} (ID: {request_id}) - Status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} (ID: {request_id}) - Error: {str(e)}")
        raise
    finally:
        logger.debug(f"Request cleanup complete: {request.method} {request.url.path} (ID: {request_id})")

# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up test server...")
    app.state.startup_time = time.time()
    
    # Register middleware
    app.middleware("http")(log_requests)
    
    yield  # App is running
    
    # Shutdown
    logger.info("Shutting down test server...")
    # Add any cleanup code here
    logger.info("Test server shutdown complete")

app = FastAPI(
    title="Ethical AI Testbed - Test Server",
    description="Lightweight test server for the Ethical AI Testbed",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with explicit settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Request/Response Models
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
    message: str = "Test server is running in lightweight mode"

# Dependencies
def get_evaluator() -> TestEthicalEvaluator:
    return get_test_evaluator()

# API Endpoints
@app.get("/api/health")
async def health_check():
    """
    🏥 **Lightweight System Health Check**
    
    Provides basic health status information for the Ethical AI system.
    This is a lightweight endpoint that doesn't depend on external services.
    
    Returns:
        Dict with basic health information
    """
    logger.info("Health check started")
    start_time = time.time()
    
    try:
        # Check if server is shutting down
        if server_shutting_down:
            logger.warning("Health check: Server is shutting down")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is shutting down"
            )
            
        # Get basic health info
        health_info = await health_checker.basic_check()
        
        # Add timing information
        processing_time = time.time() - start_time
        health_info["processing_time"] = processing_time
        
        logger.info(f"Health check completed in {processing_time:.4f} seconds")
        return health_info
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        logger.error(f"Health check failed with HTTP exception", exc_info=True)
        raise
        
    except Exception as e:
        # Log the full exception
        logger.error(f"Health check failed with error: {str(e)}", exc_info=True)
        
        # Return a 500 error with minimal details to avoid leaking sensitive info
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during health check"
        )
    finally:
        logger.debug("Health check cleanup complete")

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_text(
    request: EvaluationRequest,
    evaluator: TestEthicalEvaluator = Depends(get_evaluator)
):
    """
    Lightweight evaluation endpoint for testing.
    
    This endpoint uses a simplified evaluator that doesn't load heavy ML models.
    It's intended for testing the API infrastructure without the full evaluation stack.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Evaluating text (test mode): {request.text[:50]}...")
        
        # Use the test evaluator
        result = evaluator.evaluate_text(request.text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the response
        return {
            "request_id": result.get("evaluation_id", "test_id"),
            "overall_ethical": result.get("overall_ethical", True),
            "processing_time": processing_time,
            "test_mode": True,
            "message": "This is a test response from the lightweight evaluator",
            "input_text": result.get("input_text", request.text),
            "tokens": result.get("tokens", []),
            "spans": result.get("spans", []),
            "minimal_spans": result.get("minimal_spans", [])
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with explicit settings
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
        timeout_keep_alive=30,  # Close idle connections after 30 seconds
        limit_concurrency=100,  # Maximum number of concurrent connections
        limit_max_requests=1000,  # Maximum number of requests before restarting workers
        workers=1,  # Single worker for test server
    )
    
    # Create and run the server
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting uvicorn server...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.critical(f"Server error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")
