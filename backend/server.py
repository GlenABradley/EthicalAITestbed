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
from ethical_engine import EthicalEvaluator, EthicalParameters, EthicalEvaluation, create_learning_entry_async

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

class FeedbackRequest(BaseModel):
    evaluation_id: str
    feedback_score: float = Field(ge=0.0, le=1.0, description="Dopamine feedback score (0.0-1.0)")
    user_comment: Optional[str] = ""

class ThresholdScalingRequest(BaseModel):
    slider_value: float = Field(ge=0.0, le=1.0, description="Slider value (0.0-1.0)")
    use_exponential: bool = True

class LearningStatsResponse(BaseModel):
    total_learning_entries: int
    average_feedback_score: float
    learning_active: bool

def initialize_evaluator():
    """Initialize the ethical evaluator with learning layer"""
    global evaluator
    try:
        # Get learning collection from database
        learning_collection = db.learning_data
        evaluator = EthicalEvaluator(db_collection=learning_collection)
        logger.info("Ethical evaluator initialized successfully with learning layer")
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
        
        # Store evaluation in database with consistent ID
        evaluation_data = result["evaluation"]
        evaluation_id = evaluation_data.get("evaluation_id")
        
        evaluation_record = {
            "id": str(uuid.uuid4()),
            "evaluation_id": evaluation_id,  # Store both IDs for compatibility
            "input_text": request.text,
            "parameters": request.parameters or {},
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
        # Insert and get the ObjectId for proper serialization
        insert_result = await db.evaluations.insert_one(evaluation_record)
        evaluation_record["_id"] = str(insert_result.inserted_id)
        
        # Create learning entry if learning mode is enabled
        evaluation_data = result["evaluation"]
        if evaluation_data.get("parameters", {}).get("enable_learning_mode", False):
            dynamic_scaling = evaluation_data.get("dynamic_scaling", {})
            if dynamic_scaling.get("used_dynamic_scaling", False):
                await create_learning_entry_async(
                    db.learning_data,
                    evaluation_data.get("evaluation_id"),
                    request.text,
                    dynamic_scaling.get("ambiguity_score", 0.0),
                    dynamic_scaling.get("original_thresholds", {}),
                    dynamic_scaling.get("adjusted_thresholds", {})
                )
        
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
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        logger.error(f"Error running calibration test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/calibration-tests")
async def get_calibration_tests():
    """Get all calibration test cases"""
    try:
        tests = await db.calibration_tests.find().to_list(1000)
        
        # Convert ObjectId to string for JSON serialization
        for test in tests:
            if "_id" in test:
                test["_id"] = str(test["_id"])
        
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
        
        # Convert ObjectId to string for JSON serialization
        for evaluation in evaluations:
            if "_id" in evaluation:
                evaluation["_id"] = str(evaluation["_id"])
        
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

@api_router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit dopamine feedback for learning system"""
    try:
        global evaluator
        if evaluator is None:
            initialize_evaluator()
        
        # Record feedback directly in database to avoid sync/async issues
        learning_collection = db.learning_data
        
        result = await learning_collection.update_one(
            {'evaluation_id': request.evaluation_id},
            {
                '$inc': {
                    'feedback_count': 1,
                    'feedback_score': request.feedback_score
                },
                '$push': {
                    'feedback_history': {
                        'score': request.feedback_score,
                        'comment': request.user_comment,
                        'timestamp': datetime.utcnow()
                    }
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Recorded dopamine feedback {request.feedback_score} for evaluation {request.evaluation_id}")
            message = "Feedback recorded successfully"
        else:
            logger.warning(f"No learning entry found for evaluation {request.evaluation_id}")
            message = "No learning entry found for this evaluation"
        
        return {
            "message": message,
            "evaluation_id": request.evaluation_id,
            "feedback_score": request.feedback_score
        }
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/learning-stats", response_model=LearningStatsResponse)
async def get_learning_stats():
    """Get learning system statistics"""
    try:
        global evaluator
        if evaluator is None:
            initialize_evaluator()
        
        # Get stats directly from database to avoid sync/async issues
        learning_collection = db.learning_data
        
        total_entries = await learning_collection.count_documents({})
        
        # Use async aggregation
        avg_feedback_cursor = learning_collection.aggregate([
            {'$group': {'_id': None, 'avg_feedback': {'$avg': '$feedback_score'}}}
        ])
        
        avg_feedback_result = []
        async for doc in avg_feedback_cursor:
            avg_feedback_result.append(doc)
        
        avg_feedback_score = avg_feedback_result[0]['avg_feedback'] if avg_feedback_result else 0.0
        
        return LearningStatsResponse(
            total_learning_entries=total_entries,
            average_feedback_score=avg_feedback_score,
            learning_active=total_entries > 0
        )
        
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/threshold-scaling")
async def test_threshold_scaling(request: ThresholdScalingRequest):
    """Test threshold scaling conversion (exponential vs linear)"""
    try:
        from ethical_engine import exponential_threshold_scaling, linear_threshold_scaling
        
        if request.use_exponential:
            scaled_value = exponential_threshold_scaling(request.slider_value)
            scaling_type = "exponential"
        else:
            scaled_value = linear_threshold_scaling(request.slider_value)
            scaling_type = "linear"
        
        return {
            "slider_value": request.slider_value,
            "scaled_threshold": scaled_value,
            "scaling_type": scaling_type,
            "formula": "e^(4*x) - 1 / (e^4 - 1) * 0.3" if request.use_exponential else "x"
        }
        
    except Exception as e:
        logger.error(f"Error testing threshold scaling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dynamic-scaling-test/{evaluation_id}")
async def get_dynamic_scaling_details(evaluation_id: str):
    """Get detailed information about dynamic scaling for a specific evaluation"""
    try:
        # Look up evaluation in database using evaluation_id
        evaluation_doc = await db.evaluations.find_one({"evaluation_id": evaluation_id})
        
        if not evaluation_doc:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        # Extract dynamic scaling information from the result
        result = evaluation_doc.get('result', {})
        evaluation_data = result.get('evaluation', {})
        dynamic_scaling = evaluation_data.get('dynamic_scaling', {})
        
        return {
            "evaluation_id": evaluation_id,
            "dynamic_scaling_enabled": dynamic_scaling.get('used_dynamic_scaling', False),
            "cascade_filtering_enabled": dynamic_scaling.get('used_cascade_filtering', False),
            "ambiguity_score": dynamic_scaling.get('ambiguity_score', 0.0),
            "original_thresholds": dynamic_scaling.get('original_thresholds', {}),
            "adjusted_thresholds": dynamic_scaling.get('adjusted_thresholds', {}),
            "processing_stages": dynamic_scaling.get('processing_stages', []),
            "cascade_result": dynamic_scaling.get('cascade_result', None)
        }
        
    except Exception as e:
        logger.error(f"Error getting dynamic scaling details: {e}")
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