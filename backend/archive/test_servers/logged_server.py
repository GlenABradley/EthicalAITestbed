"""
🏛️ LOGGING-ENHANCED ETHICAL AI SERVER

This version of the server includes extensive logging to help diagnose hanging issues.
"""
import asyncio
import logging
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure root logger first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Import FastAPI and other dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our components with error handling
try:
    from unified_ethical_orchestrator import (
        get_unified_orchestrator, 
        initialize_unified_system,
        UnifiedEthicalContext,
        UnifiedEthicalResult,
        EthicalAIMode,
        ProcessingPriority
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import unified_ethical_orchestrator: {e}")
    ORCHESTRATOR_AVAILABLE = False

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from dotenv import load_dotenv
    import os
    DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import database dependencies: {e}")
    DB_AVAILABLE = False

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# --- Models ---
class EvaluationRequest(BaseModel):
    """Simplified request model for testing."""
    text: str = Field(..., min_length=1, max_length=50000)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    mode: str = "production"
    priority: str = "normal"

class EvaluationResponse(BaseModel):
    """Simplified response model for testing."""
    success: bool
    result: Dict[str, Any]
    processing_time: float

# --- Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with detailed logging."""
    logger.info("🚀 Application startup: Beginning lifespan...")
    
    # Initialize components with error handling
    db = None
    try:
        # Initialize database connection if available
        if DB_AVAILABLE:
            logger.info("🔌 Initializing database connection...")
            db = AsyncIOMotorClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
            logger.info("✅ Database connection established")
        else:
            logger.warning("⚠️  Database support not available")
        
        # Initialize orchestrator if available
        if ORCHESTRATOR_AVAILABLE:
            logger.info("⚙️  Initializing unified orchestrator...")
            try:
                # Create a minimal configuration
                config = {
                    'test_mode': True,
                    'log_level': 'DEBUG',
                    'evaluation': {
                        'mode': 'test',
                        'enable_caching': False
                    }
                }
                await initialize_unified_system(config)
                logger.info("✅ Unified orchestrator initialized in test mode")
            except Exception as e:
                logger.error(f"❌ Failed to initialize orchestrator: {e}")
                # Continue without the orchestrator
                logger.warning("⚠️  Continuing without unified orchestrator")
        else:
            logger.warning("⚠️  Unified orchestrator not available")
        
        app.state.db = db
        logger.info("🚀 Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Application shutdown: Cleaning up...")
        if db is not None:
            logger.info("🔌 Closing database connections...")
            db.close()
        logger.info("✅ Cleanup complete")

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Add middleware with logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"⬆️  Request {request_id}: {request.method} {request.url}")
    
    try:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"⬇️  Response {request_id}: {response.status_code} "
            f"(took {process_time:.4f}s)"
        )
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in request {request_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection ---
async def get_db():
    """Get database connection."""
    if not hasattr(app.state, 'db') or not app.state.db:
        logger.warning("⚠️  Database not initialized")
        return None
    return app.state.db

async def get_orchestrator():
    """Get orchestrator instance."""
    if not ORCHESTRATOR_AVAILABLE:
        logger.warning("⚠️  Orchestrator not available")
        return None
    return get_unified_orchestrator()

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    logger.info("🔍 Health check requested")
    
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "available" if DB_AVAILABLE and hasattr(app.state, 'db') else "unavailable",
            "orchestrator": "available" if ORCHESTRATOR_AVAILABLE else "unavailable",
        },
        "test_mode": True
    }
    
    logger.info(f"✅ Health check completed: {status}")
    return status

@app.post("/evaluate")
async def evaluate_text(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db),
    orchestrator = Depends(get_orchestrator)
):
    """Simplified evaluation endpoint with detailed logging."""
    logger.info(f"📥 Evaluation request received: {request.text[:50]}...")
    start_time = time.time()
    
    try:
        # Log request details
        logger.debug(f"🔍 Request details: {request.dict()}")
        
        # Simulate processing
        logger.info("🔄 Processing evaluation...")
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Create a simple response
        result = {
            "input_text": request.text,
            "tokens": request.text.split(),
            "spans": [],
            "minimal_spans": [],
            "overall_ethical": True,
            "processing_time": 0.1,
            "evaluation_id": f"test_eval_{int(time.time() * 1000)}",
            "test_mode": True,
            "message": "This is a test response from the logged server"
        }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {processing_time:.4f} seconds")
        
        return EvaluationResponse(
            success=True,
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting server with enhanced logging...")
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8011,  # Different port to avoid conflicts
        log_level="debug",
        reload=False,
        workers=1,
        timeout_keep_alive=30,
    )
    
    try:
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.critical(f"❌ Server crashed: {e}", exc_info=True)
        sys.exit(1)
