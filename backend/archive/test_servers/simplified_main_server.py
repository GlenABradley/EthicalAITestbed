"""
Simplified version of the main Ethical AI Testbed server
with reduced complexity for debugging the hanging issue.
"""
import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends, status, Request
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

# Simple test evaluator class for demonstration
class TestEthicalEvaluator:
    def __init__(self):
        self.initialized = True
        self.version = "1.0.0"
        
    async def evaluate(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Simple evaluation that returns a mock response."""
        return {
            "overall_ethical": True,
            "confidence_score": 0.95,
            "explanation": "This is a mock evaluation response.",
            "tokens": text.split(),
            "spans": [],
            "minimal_spans": []
        }

def get_test_evaluator():
    """Factory function to get a test evaluator instance."""
    return TestEthicalEvaluator()

# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up simplified main server...")
    app.state.startup_time = time.time()
    
    # Initialize test evaluator
    try:
        logger.info("Initializing test evaluator...")
        app.state.evaluator = get_test_evaluator()
        logger.info("Test evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize test evaluator: {e}", exc_info=True)
        raise
    
    logger.info("Simplified main server startup complete")
    
    # App is running
    yield  
    
    # Shutdown
    logger.info("Shutting down simplified main server...")
    if hasattr(app.state, 'evaluator'):
        logger.info("Cleaning up test evaluator...")
        del app.state.evaluator
    logger.info("Simplified main server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Ethical AI Testbed - Simplified Main Server",
    description="Simplified version of the main server for debugging",
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
    text: str = Field(..., min_length=1, max_length=50000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mode: str = Field(default="production")
    priority: str = Field(default="normal")

class EvaluationResponse(BaseModel):
    request_id: str
    overall_ethical: bool
    confidence_score: float
    processing_time: float
    timestamp: datetime
    version: str
    explanation: str
    tokens: List[str]
    spans: List[Dict[str, Any]] = []
    minimal_spans: List[Dict[str, Any]] = []

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    message: str

# API Endpoints
@app.get("/api/health", response_model=SystemHealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now(),
            "uptime_seconds": time.time() - app.state.startup_time,
            "version": getattr(app.state.evaluator, "version", "1.0.0"),
            "message": "Simplified main server is running"
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
    """Simplified evaluation endpoint."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    
    try:
        logger.info(f"Evaluation request {request_id} started")
        
        if not hasattr(app.state, 'evaluator'):
            logger.error("Evaluator not initialized")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluator not initialized"
            )
        
        # Perform evaluation
        result = await app.state.evaluator.evaluate(
            text=request.text,
            context=request.context
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "request_id": request_id,
            "overall_ethical": result.get("overall_ethical", True),
            "confidence_score": result.get("confidence_score", 1.0),
            "processing_time": processing_time,
            "timestamp": datetime.now(),
            "version": getattr(app.state.evaluator, "version", "1.0.0"),
            "explanation": result.get("explanation", "Evaluation completed"),
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

# Add a simple root endpoint
@app.get("/")
async def root():
    return {
        "message": "Ethical AI Testbed - Simplified Main Server",
        "status": "running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with explicit settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8008,  # Different port to avoid conflicts
        log_level="info",
        reload=False,
        workers=1,
        timeout_keep_alive=30,
    )
    
    logger.info("Starting simplified main server...")
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server stopped")
