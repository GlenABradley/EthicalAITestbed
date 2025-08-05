"""
Minimal FastAPI server to test basic functionality.
This is the simplest possible FastAPI server to help identify if the issue is with our code or the environment.
"""
import asyncio
import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str = "Minimal FastAPI server is running"

class EvaluationRequest(BaseModel):
    text: str

class EvaluationResponse(BaseModel):
    request_id: str
    text: str
    processed: bool
    timestamp: str

# Simple in-memory storage for testing
request_count = 0

# API Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_text(request: EvaluationRequest):
    """Simple echo endpoint that returns the input text."""
    global request_count
    request_count += 1
    
    logger.info(f"Processing evaluation request #{request_count}")
    
    # Simulate some processing
    await asyncio.sleep(0.1)
    
    return {
        "request_id": f"req_{request_count}",
        "text": f"Processed: {request.text}",
        "processed": True,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/")
async def root():
    return {
        "message": "Minimal FastAPI Server",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "request_count": request_count
    }

if __name__ == "__main__":
    # Run with uvicorn directly
    logger.info("Starting minimal FastAPI server on http://0.0.0.0:8009")
    uvicorn.run(
        "minimal_fastapi_server:app",
        host="0.0.0.0",
        port=8009,
        reload=False,
        workers=1,
        log_level="info"
    )
