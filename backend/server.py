"""
Ethical AI Developer Testbed - Backend API Server

This module provides the FastAPI backend server for the Ethical AI Developer Testbed.
It handles multi-perspective ethical text evaluation, dynamic scaling, learning systems,
and comprehensive parameter management.

Key Features:
- Multi-perspective ethical evaluation (virtue, deontological, consequentialist)
- Dynamic threshold adjustment and cascade filtering
- Machine learning integration with dopamine-based feedback
- Comprehensive API endpoints for evaluation and management
- MongoDB integration for persistent storage
- Real-time performance monitoring

Author: AI Developer Testbed Team
Version: 1.0 - Production Release
"""

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

# Environment configuration
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection setup
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# FastAPI application setup
app = FastAPI(
    title="Ethical AI Developer Testbed",
    description="Version 1.0 - Production-ready multi-perspective ethical text evaluation system",
    version="1.0.0"
)

# API router with versioning
api_router = APIRouter(prefix="/api")

# CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Configure for production environment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
evaluator = None
executor = ThreadPoolExecutor(max_workers=4)  # Optimized for production load

# Pydantic models for API
class EthicalEvaluationRequest(BaseModel):
    """Request model for ethical text evaluation"""
    text: str = Field(..., description="Text to evaluate for ethical violations")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Optional parameter overrides")

class EthicalEvaluationResponse(BaseModel):
    """Response model for ethical text evaluation"""
    evaluation: Dict[str, Any] = Field(..., description="Comprehensive evaluation results")
    clean_text: str = Field(..., description="Text with violations removed")
    explanation: str = Field(..., description="Detailed explanation of evaluation")
    delta_summary: Dict[str, Any] = Field(..., description="Summary of changes made")

class ParameterUpdateRequest(BaseModel):
    """Request model for parameter updates"""
    parameters: Dict[str, Any] = Field(..., description="Parameters to update")

class CalibrationTest(BaseModel):
    """Model for calibration test cases"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique test identifier")
    text: str = Field(..., description="Test text content")
    expected_result: str = Field(..., description="Expected ethical evaluation result")
    actual_result: Optional[str] = Field(None, description="Actual evaluation result")
    parameters_used: Optional[Dict[str, Any]] = Field(None, description="Parameters used in test")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Test creation timestamp")
    passed: Optional[bool] = Field(None, description="Whether test passed")

class CalibrationTestCreate(BaseModel):
    """Request model for creating calibration tests"""
    text: str = Field(..., description="Test text content")
    expected_result: str = Field(..., description="Expected result (ethical/unethical)")

class FeedbackRequest(BaseModel):
    """Request model for learning system feedback"""
    evaluation_id: str = Field(..., description="Evaluation ID to provide feedback for")
    feedback_score: float = Field(..., ge=0.0, le=1.0, description="Dopamine feedback score (0.0-1.0)")
    user_comment: Optional[str] = Field("", description="Optional user comment")

class ThresholdScalingRequest(BaseModel):
    """Request model for threshold scaling testing"""
    slider_value: float = Field(..., ge=0.0, le=1.0, description="Slider value (0.0-1.0)")
    use_exponential: bool = Field(True, description="Use exponential scaling")

class LearningStatsResponse(BaseModel):
    """Response model for learning system statistics"""
    total_learning_entries: int = Field(..., description="Total learning entries in system")
    average_feedback_score: float = Field(..., description="Average feedback score")
    learning_active: bool = Field(..., description="Whether learning system is active")

class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    text: str = Field(..., description="Text to evaluate")

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

@api_router.post("/heat-map-visualization")
async def get_heat_map_visualization(request: EvaluationRequest):
    """Enhanced endpoint for heat-map visualization data (Phase 4)"""
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    try:
        # Get full evaluation
        evaluation_result = evaluator.evaluate_text(request.text)
        
        # Process spans by type (short/medium/long/stochastic)
        all_spans = evaluation_result.spans
        text_length = len(request.text)
        
        def categorize_spans(spans):
            """Categorize spans by length"""
            short_spans = []
            medium_spans = []
            long_spans = []
            stochastic_spans = []
            
            for span in spans:
                span_length = span.end - span.start
                span_data = {
                    'span': [span.start, span.end],
                    'text': span.text,
                    'scores': {
                        'V': span.virtue_score,
                        'A': span.deontological_score,  # Autonomy mapping
                        'C': span.consequentialist_score
                    },
                    'uncertainty': getattr(span, 'uncertainty', 0.0)
                }
                
                if span_length <= 10:
                    short_spans.append(span_data)
                elif span_length <= 50:
                    medium_spans.append(span_data)
                elif span_length <= 200:
                    long_spans.append(span_data)
                else:
                    stochastic_spans.append(span_data)
            
            return short_spans, medium_spans, long_spans, stochastic_spans
        
        short, medium, long_spans, stochastic = categorize_spans(all_spans)
        
        def calculate_avg_score(spans):
            """Calculate average score across all dimensions"""
            if not spans:
                return 0.0
            total_score = 0.0
            total_count = 0
            for span in spans:
                for dim_score in span['scores'].values():
                    total_score += dim_score
                    total_count += 1
            return total_score / total_count if total_count > 0 else 0.0
        
        # Calculate grades for each span type
        def calculate_grade(avg_score):
            percentage = int(avg_score * 100)
            if avg_score >= 0.97: return f"A+ ({percentage}%)"
            elif avg_score >= 0.93: return f"A ({percentage}%)"
            elif avg_score >= 0.90: return f"A- ({percentage}%)"
            elif avg_score >= 0.87: return f"B+ ({percentage}%)"
            elif avg_score >= 0.83: return f"B ({percentage}%)"
            elif avg_score >= 0.80: return f"B- ({percentage}%)"
            elif avg_score >= 0.77: return f"C+ ({percentage}%)"
            elif avg_score >= 0.73: return f"C ({percentage}%)"
            elif avg_score >= 0.70: return f"C- ({percentage}%)"
            elif avg_score >= 0.67: return f"D+ ({percentage}%)"
            elif avg_score >= 0.63: return f"D ({percentage}%)"
            elif avg_score >= 0.60: return f"D- ({percentage}%)"
            else: return f"F ({percentage}%)"
        
        # Structure data for heat-map visualization
        visualization_data = {
            "evaluations": {
                "short": {
                    "spans": short,
                    "averageScore": calculate_avg_score(short),
                    "metadata": {"dataset_source": "ethical_engine_v1.1"}
                },
                "medium": {
                    "spans": medium,
                    "averageScore": calculate_avg_score(medium),
                    "metadata": {"dataset_source": "ethical_engine_v1.1"}
                },
                "long": {
                    "spans": long_spans,
                    "averageScore": calculate_avg_score(long_spans),
                    "metadata": {"dataset_source": "ethical_engine_v1.1"}
                },
                "stochastic": {
                    "spans": stochastic,
                    "averageScore": calculate_avg_score(stochastic),
                    "metadata": {"dataset_source": "ethical_engine_v1.1"}
                }
            },
            "overallGrades": {
                "short": calculate_grade(calculate_avg_score(short)),
                "medium": calculate_grade(calculate_avg_score(medium)),
                "long": calculate_grade(calculate_avg_score(long_spans)),
                "stochastic": calculate_grade(calculate_avg_score(stochastic))
            },
            "textLength": text_length,
            "originalEvaluation": {
                "input_text": evaluation_result.input_text,
                "overall_ethical": evaluation_result.overall_ethical,
                "violation_count": evaluation_result.violation_count,
                "processing_time": evaluation_result.processing_time,
                "evaluation_id": evaluation_result.evaluation_id
            }
        }
        
        return visualization_data
        
    except Exception as e:
        logger.error(f"Heat-map visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@api_router.post("/test-intent")
async def test_intent_classification(request: dict):
    """Test intent classification endpoint for v1.1 debugging"""
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        if evaluator.intent_hierarchy:
            intent_scores = evaluator.intent_hierarchy.classify_intent(text)
            dominant_intent, confidence = evaluator.intent_hierarchy.get_dominant_intent(text)
            return {
                "text": text,
                "intent_scores": intent_scores,
                "dominant_intent": dominant_intent,
                "intent_confidence": confidence,
                "intent_hierarchy_enabled": True
            }
        else:
            return {
                "text": text,
                "intent_hierarchy_enabled": False,
                "message": "Intent hierarchy not initialized"
            }
    except Exception as e:
        return {
            "text": text,
            "error": str(e),
            "intent_hierarchy_enabled": evaluator.intent_hierarchy is not None
        }

@api_router.post("/test-causal")
async def test_causal_analysis(request: dict):
    """Test causal counterfactual analysis endpoint for v1.1 debugging"""
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        if evaluator.causal_analyzer:
            # Create mock harmful spans for testing
            harmful_spans = [{
                'text': text,
                'start': 0,
                'end': len(text),
                'virtue_score': 0.8,
                'deontological_score': 0.7,
                'consequentialist_score': 0.6,
                'dominant_intent': 'fraud',
                'intent_confidence': 0.5
            }]
            
            causal_analysis = evaluator.causal_analyzer.analyze_causal_chain(text, harmful_spans)
            return {
                "text": text,
                "causal_analysis": causal_analysis,
                "causal_analyzer_enabled": True
            }
        else:
            return {
                "text": text,
                "causal_analyzer_enabled": False,
                "message": "Causal analyzer not initialized"
            }
    except Exception as e:
        return {
            "text": text,
            "error": str(e),
            "causal_analyzer_enabled": evaluator.causal_analyzer is not None
        }

@api_router.post("/test-uncertainty")
async def test_uncertainty_analysis(request: dict):
    """Test uncertainty analysis endpoint for v1.1 debugging"""
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        if evaluator.uncertainty_analyzer:
            uncertainty_analysis = evaluator.uncertainty_analyzer.analyze_uncertainty(text)
            return {
                "text": text,
                "uncertainty_analysis": uncertainty_analysis,
                "uncertainty_analyzer_enabled": True
            }
        else:
            return {
                "text": text,
                "uncertainty_analyzer_enabled": False,
                "message": "Uncertainty analyzer not initialized"
            }
    except Exception as e:
        return {
            "text": text,
            "error": str(e),
            "uncertainty_analyzer_enabled": evaluator.uncertainty_analyzer is not None
        }

@api_router.post("/test-purpose")
async def test_purpose_alignment(request: dict):
    """Test purpose alignment analysis endpoint for v1.1 debugging"""
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    text = request.get("text", "")
    context = request.get("context", "")
    declared_purpose = request.get("declared_purpose", "")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        if evaluator.purpose_alignment:
            purpose_alignment_analysis = evaluator.purpose_alignment.analyze_purpose_alignment(
                text=text,
                evaluation_result=None,  # Will compute automatically
                context=context,
                declared_purpose=declared_purpose
            )
            return {
                "text": text,
                "context": context,
                "declared_purpose": declared_purpose,
                "purpose_alignment_analysis": purpose_alignment_analysis,
                "purpose_alignment_enabled": True
            }
        else:
            return {
                "text": text,
                "purpose_alignment_enabled": False,
                "message": "Purpose alignment not initialized"
            }
    except Exception as e:
        return {
            "text": text,
            "error": str(e),
            "purpose_alignment_enabled": evaluator.purpose_alignment is not None
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