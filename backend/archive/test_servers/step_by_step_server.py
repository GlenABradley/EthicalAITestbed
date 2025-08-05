"""
Step-by-step test server to identify the cause of hanging.
This version adds components incrementally to isolate the issue.
"""
import logging
import time
from fastapi import FastAPI, HTTPException, status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Step 1: Basic FastAPI app with health check ---
app = FastAPI()

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint"""
    logger.info("Health check started")
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "message": "Basic health check",
            "step": 1
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("Health check completed")

# --- Step 2: Add CORS middleware ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 3: Add test evaluator dependency ---
logger.info("Attempting to import test_evaluator...")
try:
    from test_evaluator import get_test_evaluator, TestEthicalEvaluator, EthicalParameters
    logger.info("Successfully imported test_evaluator")
except Exception as e:
    logger.error(f"Error importing test_evaluator: {e}")
    raise

def get_evaluator() -> TestEthicalEvaluator:
    logger.info("Creating test evaluator instance...")
    try:
        # Try with explicit parameters for better control
        params = EthicalParameters(
            enable_graph_attention=False,
            enable_intent_hierarchy=False,
            enable_causal_analysis=False,
            enable_uncertainty_analysis=False,
            enable_purpose_alignment=False
        )
        evaluator = TestEthicalEvaluator(parameters=params)
        logger.info("Successfully created test evaluator instance")
        return evaluator
    except Exception as e:
        logger.error(f"Error creating test evaluator: {e}")
        raise

@app.get("/api/evaluate")
async def evaluate():
    logger.info("/api/evaluate endpoint called")
    try:
        evaluator = get_evaluator()
        logger.info("Test evaluator initialized successfully")
        return {
            "status": "evaluator initialized", 
            "step": 3,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in /api/evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Step 4: Add health check from health_checker ---
# Uncomment to test with health_checker
"""
from health_check import health_checker

@app.get("/api/health2")
async def health_check2():
    return await health_checker.basic_check()
"""

# --- Step 5: Add request logging middleware ---
# Uncomment to test with request logging
"""
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
"""

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting step-by-step server...")
    uvicorn.run("step_by_step_server:app", host="0.0.0.0", port=8004, log_level="info")
