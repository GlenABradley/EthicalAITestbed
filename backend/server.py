from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our ethical evaluation engine
from ethical_engine import EthicalEvaluator, EthicalParameters, EthicalEvaluation

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Ethical AI Developer Testbed")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global evaluator instance
evaluator = None
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for API
class EthicalEvaluationRequest(BaseModel):
    text: str
    parameters: Optional[Dict[str, Any]] = None

class EthicalEvaluationResponse(BaseModel):
    evaluation: Dict[str, Any]
    clean_text: str
    explanation: str
    delta_summary: Dict[str, Any]

class ParameterUpdateRequest(BaseModel):
    parameters: Dict[str, Any]

class CalibrationTest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    expected_result: str
    actual_result: Optional[str] = None
    parameters_used: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    passed: Optional[bool] = None

class CalibrationTestCreate(BaseModel):
    text: str
    expected_result: str

def initialize_evaluator():
    """Initialize the ethical evaluator"""
    global evaluator
    try:
        evaluator = EthicalEvaluator()
        logger.info("Ethical evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ethical evaluator: {e}")
        raise

def run_evaluation(text: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run evaluation in thread pool to avoid blocking"""
    global evaluator
    
    if evaluator is None:
        initialize_evaluator()
    
    # Update parameters if provided
    if parameters:
        evaluator.update_parameters(parameters)
    
    # Run evaluation
    evaluation = evaluator.evaluate_text(text)
    clean_text = evaluator.generate_clean_text(evaluation)
    explanation = evaluator.generate_explanation(evaluation)
    
    # Calculate delta summary
    delta_summary = {
        "original_length": len(text),
        "clean_length": len(clean_text),
        "removed_characters": len(text) - len(clean_text),
        "removed_spans": len(evaluation.minimal_spans),
        "ethical_status": evaluation.overall_ethical
    }
    
    return {
        "evaluation": evaluation.to_dict(),
        "clean_text": clean_text,
        "explanation": explanation,
        "delta_summary": delta_summary
    }

@api_router.get("/")
async def root():
    return {"message": "Ethical AI Developer Testbed API"}

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    global evaluator
    return {
        "status": "healthy",
        "evaluator_initialized": evaluator is not None,
        "timestamp": datetime.utcnow()
    }

@api_router.post("/evaluate", response_model=EthicalEvaluationResponse)
async def evaluate_text(request: EthicalEvaluationRequest):
    """Evaluate text for ethical violations"""
    try:
        # Run evaluation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            run_evaluation, 
            request.text, 
            request.parameters
        )
        
        # Store evaluation in database
        evaluation_record = {
            "id": str(uuid.uuid4()),
            "input_text": request.text,
            "parameters": request.parameters or {},
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
        await db.evaluations.insert_one(evaluation_record)
        
        return EthicalEvaluationResponse(
            evaluation=result["evaluation"],
            clean_text=result["clean_text"],
            explanation=result["explanation"],
            delta_summary=result["delta_summary"]
        )
        
    except Exception as e:
        logger.error(f"Error in evaluate_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/update-parameters")
async def update_parameters(request: ParameterUpdateRequest):
    """Update evaluation parameters for calibration"""
    try:
        global evaluator
        
        if evaluator is None:
            initialize_evaluator()
        
        evaluator.update_parameters(request.parameters)
        
        return {
            "status": "success",
            "updated_parameters": request.parameters,
            "current_parameters": evaluator.parameters.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/parameters")
async def get_parameters():
    """Get current evaluation parameters"""
    try:
        global evaluator
        
        if evaluator is None:
            initialize_evaluator()
        
        return {
            "parameters": evaluator.parameters.to_dict(),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/calibration-test")
async def create_calibration_test(request: CalibrationTestCreate):
    """Create a new calibration test case"""
    try:
        test_case = CalibrationTest(
            text=request.text,
            expected_result=request.expected_result
        )
        
        await db.calibration_tests.insert_one(test_case.dict())
        
        return {
            "status": "success",
            "test_case": test_case.dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating calibration test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/run-calibration-test/{test_id}")
async def run_calibration_test(test_id: str):
    """Run a specific calibration test"""
    try:
        # Get test case from database
        test_case = await db.calibration_tests.find_one({"id": test_id})
        
        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")
        
        # Run evaluation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            run_evaluation, 
            test_case["text"]
        )
        
        # Update test case with results
        passed = result["evaluation"]["overall_ethical"] == (test_case["expected_result"] == "ethical")
        
        await db.calibration_tests.update_one(
            {"id": test_id},
            {
                "$set": {
                    "actual_result": "ethical" if result["evaluation"]["overall_ethical"] else "unethical",
                    "parameters_used": result["evaluation"]["parameters"],
                    "passed": passed,
                    "timestamp": datetime.utcnow()
                }
            }
        )
        
        return {
            "status": "success",
            "test_id": test_id,
            "passed": passed,
            "expected": test_case["expected_result"],
            "actual": "ethical" if result["evaluation"]["overall_ethical"] else "unethical",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error running calibration test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/calibration-tests")
async def get_calibration_tests():
    """Get all calibration test cases"""
    try:
        tests = await db.calibration_tests.find().to_list(1000)
        return {
            "tests": tests,
            "count": len(tests)
        }
        
    except Exception as e:
        logger.error(f"Error getting calibration tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/evaluations")
async def get_evaluations(limit: int = 100):
    """Get recent evaluations"""
    try:
        evaluations = await db.evaluations.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return {
            "evaluations": evaluations,
            "count": len(evaluations)
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics for processing overhead analysis"""
    try:
        # Get recent evaluations
        evaluations = await db.evaluations.find().sort("timestamp", -1).limit(100).to_list(100)
        
        if not evaluations:
            return {"message": "No evaluations found"}
        
        # Calculate metrics
        processing_times = [e["result"]["evaluation"]["processing_time"] for e in evaluations]
        text_lengths = [len(e["input_text"]) for e in evaluations]
        
        metrics = {
            "total_evaluations": len(evaluations),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "average_text_length": sum(text_lengths) / len(text_lengths),
            "throughput_chars_per_second": sum(text_lengths) / sum(processing_times) if sum(processing_times) > 0 else 0
        }
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

# Initialize evaluator on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the ethical evaluator on startup"""
    try:
        # Run initialization in thread pool to avoid blocking startup
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, initialize_evaluator)
        logger.info("Ethical AI Developer Testbed started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize testbed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        client.close()
        executor.shutdown(wait=True)
        logger.info("Ethical AI Developer Testbed shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")