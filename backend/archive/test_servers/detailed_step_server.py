"""
Detailed step-by-step server with enhanced logging to diagnose hanging issues.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to track if we're in the middle of a request
request_in_progress = False

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager with detailed logging"""
    logger.info("Application startup: Starting...")
    
    # Initialize app state
    app.state.startup_time = time.time()
    
    try:
        # Initialize test evaluator
        logger.info("Initializing test evaluator...")
        from test_evaluator import get_test_evaluator
        app.state.evaluator = get_test_evaluator()
        logger.info("Test evaluator initialized successfully")
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutdown: Starting...")
        # Cleanup resources
        if hasattr(app.state, 'evaluator'):
            logger.info("Cleaning up test evaluator...")
            del app.state.evaluator
        logger.info("Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log request/response cycle
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_in_progress
    
    # Log request start
    request_id = int(time.time() * 1000) % 10000
    logger.info(f"[{request_id}] Request started: {request.method} {request.url}")
    
    # Set request in progress flag
    request_in_progress = True
    
    try:
        # Process the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log successful completion
        logger.info(f"[{request_id}] Request completed in {process_time:.4f}s - Status: {response.status_code}")
        return response
        
    except Exception as e:
        # Log any exceptions
        logger.error(f"[{request_id}] Request failed: {e}", exc_info=True)
        raise
    finally:
        # Clear request in progress flag
        request_in_progress = False
        logger.info(f"[{request_id}] Request cleanup complete")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint with detailed status"""
    logger.info("Health check started")
    
    try:
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "startup_time": app.state.startup_time,
            "evaluator_initialized": hasattr(app.state, 'evaluator'),
            "request_in_progress": request_in_progress,
            "python_module": __name__,
            "step": 1
        }
        logger.info(f"Health check completed: {status}")
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise

# Evaluation endpoint
@app.get("/api/evaluate")
async def evaluate():
    """Evaluation endpoint"""
    logger.info("Evaluation started")
    
    if not hasattr(app.state, 'evaluator'):
        logger.error("Evaluator not initialized")
        return {"error": "Evaluator not initialized"}, 500
    
    try:
        logger.info("Calling evaluator...")
        result = app.state.evaluator.evaluate_text("Test input")
        logger.info("Evaluation completed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting detailed step server...")
    uvicorn.run(
        "detailed_step_server:app",
        host="0.0.0.0",
        port=8006,
        log_level="info",
        # These settings help with debugging hanging issues
        timeout_keep_alive=5,
        timeout_graceful_shutdown=5,
        limit_concurrency=10,
        limit_max_requests=100
    )
