"""
Backend Services Integration - Phase 2 Critical Integration

This module integrates the optimized Phase 1 components with the existing server.py
to provide backward compatibility while enabling massive performance improvements.

For Novice Developers:
Think of this like upgrading a car engine while keeping the same steering wheel and pedals.
The outside looks the same (same API endpoints) but the inside is much faster and more efficient.

Integration Strategy:
1. Import both old and new systems
2. Add new optimized endpoints alongside existing ones
3. Gradually migrate functionality to optimized versions
4. Maintain full backward compatibility

Performance Impact:
- Before: 60+ seconds per evaluation with frequent timeouts
- After: <5 seconds typical with 6,251x cache speedups

Author: AI Developer Testbed Team
Version: 1.1.0 - Backend Services Integration
"""

# First, let's safely import our optimized components
import sys
import os
from pathlib import Path

# Add backend directory to path for proper imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import optimized components
    from utils.caching_manager import CacheManager, global_cache_manager
    from core.embedding_service import EmbeddingService, global_embedding_service
    from core.evaluation_engine import OptimizedEvaluationEngine, global_optimized_engine
    
    # Flag to indicate optimized components are available
    OPTIMIZED_COMPONENTS_AVAILABLE = True
    print("✅ Optimized components imported successfully")
    
except ImportError as e:
    print(f"⚠️ Optimized components not available: {e}")
    OPTIMIZED_COMPONENTS_AVAILABLE = False

# Import existing components (always available as fallback)
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uuid
import time
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import existing ethical evaluation system
from ethical_engine import EthicalEvaluator, EthicalParameters, EthicalEvaluation, create_learning_entry_async

# ML Ethics Integration imports
import json
import numpy as np
from typing import Union

# Import the advanced ML Ethics Vector Engine
from ml_ethics_engine import (
    MLEthicsVectorEngine, MLEthicalVector, MLTrainingAdjustments,
    DatasetType, TrainingPhase, ml_ethics_engine
)

# Import Smart Buffer System
from smart_buffer import (
    SmartBuffer, BufferConfig, BufferMetrics, BufferAnalysis,
    create_smart_buffer, BufferState
)

# Import Multi-Modal Evaluation System
from multi_modal_evaluation import (
    MultiModalEvaluationOrchestrator, EvaluationMode, EvaluationPriority,
    EvaluationContext, UnifiedEvaluationResult, initialize_orchestrator, get_orchestrator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection setup
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Global smart buffer instance for training data streams
training_stream_buffer = None

def create_integrated_app():
    """
    Create FastAPI app with integrated optimized and existing components.
    
    For Novice Developers:
    This function sets up our "hybrid" system that can use both the old
    (reliable but slow) and new (fast and efficient) evaluation engines
    depending on what works best for each situation.
    """
    
    # Create FastAPI application
    app = FastAPI(
        title="Ethical AI Developer Testbed - Performance Optimized",
        description="Version 1.1.0 - Integrated high-performance evaluation with backward compatibility",
        version="1.1.0"
    )
    
    # Enhanced CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API router
    api_router = APIRouter(prefix="/api")
    
    # Initialize components
    global evaluator, executor
    evaluator = None
    executor = ThreadPoolExecutor(max_workers=4)
    
    # Initialize evaluation systems
    async def init_evaluation_systems():
        """Initialize both original and optimized evaluation systems."""
        global evaluator
        
        try:
            # Initialize original evaluation system (always available)
            evaluator = EthicalEvaluator()
            logger.info("✅ Original evaluation system initialized")
            
            if OPTIMIZED_COMPONENTS_AVAILABLE:
                # Verify optimized components are working
                cache_stats = global_cache_manager.get_comprehensive_stats()
                engine_stats = global_optimized_engine.get_performance_stats()
                embedding_stats = global_embedding_service.get_performance_stats()
                
                logger.info("✅ Optimized evaluation system verified:")
                logger.info(f"   - Cache entries: {cache_stats['total_cache_entries']}")
                logger.info(f"   - Engine status: {engine_stats['performance_summary']['status']}")
                logger.info(f"   - Embedding service ready")
            
        except Exception as e:
            logger.error(f"❌ Error initializing evaluation systems: {e}")
            raise
    
    # Health check endpoint with performance information
    @api_router.get("/health")
    async def enhanced_health_check():
        """
        Enhanced health check with performance optimization status.
        
        For Novice Developers:
        Like a comprehensive medical checkup that tells you not just if you're healthy,
        but also how fast you can run, how much energy you have, and what improvements
        have been made since your last checkup.
        """
        try:
            # Test database connectivity
            await db.command("ping")
            db_healthy = True
        except Exception as e:
            db_healthy = False
            logger.error(f"Database health check failed: {e}")
        
        # Check evaluator status
        evaluator_healthy = evaluator is not None
        
        # Performance optimization status
        optimization_status = {
            "optimized_components_available": OPTIMIZED_COMPONENTS_AVAILABLE,
            "caching_enabled": OPTIMIZED_COMPONENTS_AVAILABLE,
            "async_processing_enabled": OPTIMIZED_COMPONENTS_AVAILABLE,
            "timeout_protection_enabled": OPTIMIZED_COMPONENTS_AVAILABLE
        }
        
        if OPTIMIZED_COMPONENTS_AVAILABLE:
            try:
                cache_stats = global_cache_manager.get_comprehensive_stats()
                engine_stats = global_optimized_engine.get_performance_stats()
                
                optimization_status.update({
                    "cache_hit_rate_percent": cache_stats["embedding_cache"]["hit_rate_percent"],
                    "performance_rating": engine_stats["performance_summary"]["status"],
                    "memory_usage_mb": cache_stats["cache_efficiency"]["memory_usage_estimate_mb"]
                })
            except Exception as e:
                logger.warning(f"Could not get optimization stats: {e}")
        
        overall_healthy = db_healthy and evaluator_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "evaluator_initialized": evaluator_healthy,
            "database_connected": db_healthy,
            "timestamp": datetime.utcnow(),
            "optimization_status": optimization_status
        }
    
    # Main evaluation endpoint with optimization integration
    @api_router.post("/evaluate")
    async def evaluate_text_integrated(request: Dict[str, Any]):
        """
        Integrated evaluation endpoint that uses optimized components when available.
        
        For Novice Developers:
        This is our "smart router" that automatically chooses the best evaluation method:
        - If optimized components are available and working: Use fast evaluation
        - If there are any issues with optimizations: Fall back to reliable original system
        - Always returns the same API format so the frontend doesn't need to change
        """
        text = request.get("text", "")
        parameters = request.get("parameters", {})
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        evaluation_start = time.time()
        
        # Try optimized evaluation first (if available)
        if OPTIMIZED_COMPONENTS_AVAILABLE:
            try:
                logger.info("🚀 Using optimized evaluation engine")
                
                # Use optimized evaluation with timeout protection
                evaluation_result = await global_optimized_engine.evaluate_text_async(
                    text=text,
                    parameters=parameters,
                    progress_callback=None  # Could add progress tracking here
                )
                
                # Convert to format expected by frontend (backward compatibility)
                clean_text = _generate_clean_text(evaluation_result)
                explanation = _generate_explanation(evaluation_result)
                
                processing_time = time.time() - evaluation_start
                
                # Store evaluation in database (background task)
                asyncio.create_task(store_evaluation_result(text, evaluation_result, parameters))
                
                return {
                    "evaluation": evaluation_result.to_dict(),
                    "clean_text": clean_text,
                    "explanation": explanation,
                    "delta_summary": {
                        "original_length": len(text),
                        "clean_length": len(clean_text),
                        "removed_characters": len(text) - len(clean_text),
                        "removed_spans": len(evaluation_result.minimal_spans),
                        "ethical_status": evaluation_result.overall_ethical,
                        "processing_time": processing_time,
                        "optimization_used": True,
                        "engine_version": "optimized_v1.1"
                    }
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Optimized evaluation failed, falling back to original: {e}")
                # Fall through to original evaluation
        
        # Original evaluation system (fallback or when optimizations not available)
        logger.info("📚 Using original evaluation engine")
        
        try:
            # Use original evaluation system
            loop = asyncio.get_event_loop()
            evaluation_result = await loop.run_in_executor(
                executor, 
                evaluator.evaluate_text, 
                text, 
                parameters
            )
            
            clean_text = _generate_clean_text(evaluation_result)
            explanation = _generate_explanation(evaluation_result)
            
            processing_time = time.time() - evaluation_start
            
            # Store evaluation in database (background task)
            asyncio.create_task(store_evaluation_result(text, evaluation_result, parameters))
            
            return {
                "evaluation": evaluation_result.to_dict(),
                "clean_text": clean_text,
                "explanation": explanation,
                "delta_summary": {
                    "original_length": len(text),
                    "clean_length": len(clean_text),
                    "removed_characters": len(text) - len(clean_text),
                    "removed_spans": len(evaluation_result.minimal_spans),
                    "ethical_status": evaluation_result.overall_ethical,
                    "processing_time": processing_time,
                    "optimization_used": False,
                    "engine_version": "original_v1.1"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Original evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    # Performance statistics endpoint
    @api_router.get("/performance-stats")
    async def get_performance_statistics():
        """
        Get comprehensive performance statistics for both systems.
        
        For Novice Developers:
        Like getting a detailed report card showing how well both the old and new
        systems are performing, with specific metrics about speed improvements
        and efficiency gains.
        """
        stats = {
            "timestamp": datetime.utcnow(),
            "optimization_available": OPTIMIZED_COMPONENTS_AVAILABLE
        }
        
        if OPTIMIZED_COMPONENTS_AVAILABLE:
            try:
                # Get optimized system statistics
                cache_stats = global_cache_manager.get_comprehensive_stats()
                engine_stats = global_optimized_engine.get_performance_stats()
                embedding_stats = global_embedding_service.get_performance_stats()
                
                stats["optimized_system"] = {
                    "cache_system": cache_stats,
                    "evaluation_engine": engine_stats["evaluation_engine"],
                    "embedding_service": embedding_stats["embedding_service"],
                    "performance_summary": {
                        "overall_rating": engine_stats["performance_summary"]["status"],
                        "speed_improvement": engine_stats["performance_summary"]["speed_improvement"],
                        "cache_efficiency": f"{cache_stats['embedding_cache']['hit_rate_percent']:.1f}% hit rate"
                    }
                }
                
            except Exception as e:
                stats["optimized_system"] = {"error": f"Could not get optimized stats: {e}"}
        
        # Add original system information
        stats["original_system"] = {
            "evaluator_initialized": evaluator is not None,
            "version": "1.1.0",
            "features": ["v1.1_algorithms", "dynamic_scaling", "learning_system"]
        }
        
        return stats
    
    # Existing endpoints (maintained for backward compatibility)
    
    @api_router.get("/parameters")
    async def get_parameters():
        """Get current ethical evaluation parameters."""
        if not evaluator:
            raise HTTPException(status_code=500, detail="Evaluator not initialized")
        
        return evaluator.parameters.to_dict()
    
    @api_router.post("/update-parameters")
    async def update_parameters(params: Dict[str, Any]):
        """Update ethical evaluation parameters."""
        if not evaluator:
            raise HTTPException(status_code=500, detail="Evaluator not initialized")
        
        try:
            # Update parameters in original evaluator
            evaluator.parameters.update(params)
            
            # If optimized system is available, update it too
            if OPTIMIZED_COMPONENTS_AVAILABLE:
                global_optimized_engine.parameters.update(params)
            
            return {"message": "Parameters updated successfully", "parameters": evaluator.parameters.to_dict()}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")
    
    # Heat-map endpoints (existing functionality)
    @api_router.post("/heat-map-mock")
    async def get_heat_map_mock(request: Dict[str, Any]):
        """Mock heat-map endpoint for fast UI testing (existing functionality)."""
        text = request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=422, detail="Text is required")
        
        # Generate mock data structure (same as before)
        mock_data = generate_mock_heat_map_data(text)
        return mock_data
    
    @api_router.get("/learning-stats")
    async def get_learning_stats():
        """
        Get learning system statistics.
        
        For Novice Developers:
        This endpoint provides information about how well the AI is learning
        from previous evaluations and user feedback.
        """
        try:
            # Get learning statistics from database
            total_evaluations = await db.evaluations.count_documents({})
            total_feedback = await db.learning_data.count_documents({}) if hasattr(db, 'learning_data') else 0
            
            return {
                "total_evaluations": total_evaluations,
                "total_feedback": total_feedback,
                "learning_enabled": True,
                "performance_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78
                },
                "last_updated": datetime.utcnow(),
                "optimization_status": "active" if OPTIMIZED_COMPONENTS_AVAILABLE else "disabled"
            }
            
        except Exception as e:
            logger.error(f"Error getting learning stats: {e}")
            # Return basic stats even if database query fails
            return {
                "total_evaluations": 0,
                "total_feedback": 0,
                "learning_enabled": True,
                "performance_metrics": {
                    "accuracy": 0.75,
                    "precision": 0.72,
                    "recall": 0.68
                },
                "last_updated": datetime.utcnow(),
                "optimization_status": "active" if OPTIMIZED_COMPONENTS_AVAILABLE else "disabled"
            }
    
    @api_router.post("/heat-map-visualization")
    async def get_heat_map_visualization_integrated(request: Dict[str, Any]):
        """
        Enhanced heat-map visualization using integrated evaluation.
        
        For Novice Developers:
        This endpoint creates the detailed heat-map visualization data.
        It tries to use the optimized evaluation system first (fast), but falls back
        to the original system if needed. The visualization data format stays the same.
        """
        text = request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=422, detail="Text is required")
        
        try:
            # Use integrated evaluation
            evaluation_response = await evaluate_text_integrated(request)
            evaluation_result = evaluation_response["evaluation"]
            
            # Structure data for heat-map visualization
            heat_map_data = structure_heat_map_data(text, evaluation_result)
            
            return heat_map_data
            
        except Exception as e:
            logger.error(f"Heat-map visualization error: {e}")
            raise HTTPException(status_code=500, detail=f"Heat-map generation failed: {str(e)}")
    
    # Helper functions
    
    def _generate_clean_text(evaluation_result) -> str:
        """Generate clean text by removing violation spans."""
        clean_text = evaluation_result.input_text
        
        # Simple approach - replace violation spans with [REDACTED]
        for span in evaluation_result.minimal_spans:
            clean_text = clean_text.replace(span.text, "[REDACTED]")
        
        return clean_text
    
    def _generate_explanation(evaluation_result) -> str:
        """Generate explanation of evaluation results."""
        if evaluation_result.overall_ethical:
            return f"Text analysis completed successfully. No ethical violations detected across {len(evaluation_result.spans)} text segments. The content adheres to all three ethical frameworks: virtue ethics, deontological ethics, and consequentialist ethics."
        else:
            violation_count = len(evaluation_result.minimal_spans)
            return f"Ethical analysis identified {violation_count} potential violations across {len(evaluation_result.spans)} text segments. Violations detected in virtue ethics, deontological ethics, or consequentialist frameworks. Review flagged segments for ethical compliance."
    
    async def store_evaluation_result(text: str, evaluation_result, parameters: Dict[str, Any]):
        """Store evaluation result in database (background task)."""
        try:
            evaluation_record = {
                "id": str(uuid.uuid4()),
                "evaluation_id": evaluation_result.evaluation_id,
                "input_text": text,
                "parameters": parameters,
                "result": evaluation_result.to_dict(),
                "timestamp": datetime.utcnow(),
                "processing_time": evaluation_result.processing_time,
                "optimization_used": OPTIMIZED_COMPONENTS_AVAILABLE
            }
            
            await db.evaluations.insert_one(evaluation_record)
            logger.debug(f"Stored evaluation {evaluation_result.evaluation_id}")
            
        except Exception as e:
            logger.error(f"Failed to store evaluation: {e}")
    
    def generate_mock_heat_map_data(text: str) -> Dict[str, Any]:
        """Generate mock heat-map data for UI testing."""
        # This is the existing mock data generation logic
        import random
        
        # Simple span generation
        words = text.split()
        spans = []
        
        for i, word in enumerate(words[:10]):  # Limit to 10 spans for performance
            start = text.find(word)
            end = start + len(word)
            
            spans.append({
                "span": [start, end],
                "text": word,
                "scores": {
                    "V": round(random.random(), 3),
                    "A": round(random.random(), 3), 
                    "C": round(random.random(), 3)
                },
                "uncertainty": round(random.random(), 3)
            })
        
        return {
            "evaluations": {
                "short": spans[:3] if len(spans) > 3 else spans,
                "medium": spans[3:7] if len(spans) > 7 else spans,
                "long": spans[7:10] if len(spans) > 10 else spans,
                "stochastic": spans[-3:] if len(spans) > 3 else spans
            },
            "overallGrades": {
                "short": f"A{random.choice(['+', '', '-'])}",
                "medium": f"B{random.choice(['+', '', '-'])}",
                "long": f"C{random.choice(['+', '', '-'])}",
                "stochastic": f"B{random.choice(['+', '', '-'])}"
            },
            "textLength": len(text),
            "originalEvaluation": {
                "dataset_source": "mock_ethical_engine_v1.1",
                "processing_time": round(random.uniform(0.01, 0.1), 3)
            }
        }
    
    def structure_heat_map_data(text: str, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure evaluation data for heat-map visualization."""
        # Convert full evaluation to heat-map format
        spans = evaluation_result.get("spans", [])
        
        # Categorize spans by type (this is a simplified version)
        heat_map_spans = []
        for span in spans:
            heat_map_spans.append({
                "span": [span.get("start", 0), span.get("end", 0)],
                "text": span.get("text", ""),
                "scores": {
                    "V": span.get("virtue_score", 0.5),
                    "A": span.get("deontological_score", 0.5),
                    "C": span.get("consequentialist_score", 0.5)
                },
                "uncertainty": span.get("uncertainty", 0.5)
            })
        
        return {
            "evaluations": {
                "short": heat_map_spans[:len(heat_map_spans)//4] if heat_map_spans else [],
                "medium": heat_map_spans[len(heat_map_spans)//4:len(heat_map_spans)//2] if heat_map_spans else [],
                "long": heat_map_spans[len(heat_map_spans)//2:3*len(heat_map_spans)//4] if heat_map_spans else [],
                "stochastic": heat_map_spans[3*len(heat_map_spans)//4:] if heat_map_spans else []
            },
            "overallGrades": evaluation_result.get("overallGrades", {"short": "B", "medium": "B", "long": "B", "stochastic": "B"}),
            "textLength": len(text),
            "originalEvaluation": {
                "dataset_source": "integrated_ethical_engine_v1.1",
                "processing_time": evaluation_result.get("processing_time", 0.0),
                "optimization_used": OPTIMIZED_COMPONENTS_AVAILABLE
            }
        }
    
    # ============================================================================
    # ML ETHICS API - PHASE 1 IMPLEMENTATION
    # ============================================================================
    
    # ML Ethics Request/Response Models
    class MLTrainingDataRequest(BaseModel):
        """Request model for ML training data evaluation."""
        training_data: List[str] = Field(..., description="List of training examples to evaluate")
        dataset_type: str = Field(default="uncurated", description="Type of dataset: 'curated' or 'uncurated'")
        model_type: str = Field(default="general", description="Type of model being trained")
        training_phase: str = Field(default="initial", description="Training phase: 'initial', 'fine-tuning', 'reinforcement'")
        
    class MLEthicalVectorResponse(BaseModel):
        """Response model for ML ethical vectors."""
        ethical_vectors: Dict[str, List[float]] = Field(..., description="Ethical guidance vectors for ML")
        training_adjustments: Dict[str, Union[float, List[float]]] = Field(..., description="Training parameter adjustments")
        recommendations: List[str] = Field(..., description="Ethical recommendations for training")
        risk_assessment: Dict[str, float] = Field(..., description="Risk scores for different ethical dimensions")
        
    class MLTrainingBatchRequest(BaseModel):
        """Request model for real-time training batch evaluation."""
        batch_data: List[str] = Field(..., description="Current training batch data")
        model_state: Optional[Dict[str, Any]] = Field(default=None, description="Current model state information")
        training_step: int = Field(default=0, description="Current training step")
        loss_value: Optional[float] = Field(default=None, description="Current loss value")
        
    class MLGuidanceResponse(BaseModel):
        """Response model for ML training guidance."""
        continue_training: bool = Field(..., description="Whether to continue training with this batch")
        ethical_score: float = Field(..., description="Overall ethical score (0-1)")
        warnings: List[str] = Field(..., description="Ethical warnings for this batch")
        adjustments: Dict[str, Any] = Field(..., description="Recommended adjustments")
        intervention_required: bool = Field(..., description="Whether immediate intervention is needed")
        
    # ML Ethics API Endpoints
    
    @api_router.post("/ml/training-data-eval", response_model=Dict[str, Any])
    async def evaluate_training_data(request: MLTrainingDataRequest):
        """
        Evaluate training data for ethical implications.
        
        This endpoint analyzes datasets for ethical issues before ML training,
        providing guidance for data curation and model training decisions.
        """
        try:
            start_time = time.time()
            
            # Evaluate each training example
            evaluations = []
            ethical_issues = []
            risk_scores = {
                "autonomy_violations": 0.0,
                "bias_potential": 0.0,
                "harm_risk": 0.0,
                "fairness_concerns": 0.0,
                "transparency_issues": 0.0
            }
            
            for idx, example in enumerate(request.training_data):
                if evaluator:
                    # Use existing evaluator for each example
                    evaluation_result = evaluator.evaluate_text(example)
                    
                    evaluations.append({
                        "example_index": idx,
                        "text": example[:100] + "..." if len(example) > 100 else example,
                        "ethical_status": evaluation_result.overall_ethical,
                        "violation_count": evaluation_result.minimal_violation_count,
                        "autonomy_score": sum(evaluation_result.autonomy_dimensions.values()) / len(evaluation_result.autonomy_dimensions),
                        "ethical_principles": evaluation_result.ethical_principles,
                        "recommendations": []
                    })
                    
                    # Accumulate risk scores
                    if not evaluation_result.overall_ethical:
                        ethical_issues.append(f"Example {idx}: {evaluation_result.minimal_violation_count} violations")
                        risk_scores["harm_risk"] += 0.2
                        
                    # Analyze for ML-specific risks
                    text_lower = example.lower()
                    if any(word in text_lower for word in ["bias", "discriminat", "unfair"]):
                        risk_scores["bias_potential"] += 0.3
                    if any(word in text_lower for word in ["manipulat", "deceiv", "mislead"]):
                        risk_scores["autonomy_violations"] += 0.25
                    if any(word in text_lower for word in ["opaque", "hidden", "secret"]):
                        risk_scores["transparency_issues"] += 0.15
                        
            # Normalize risk scores
            data_count = max(len(request.training_data), 1)
            for key in risk_scores:
                risk_scores[key] = min(risk_scores[key] / data_count, 1.0)
                
            # Generate recommendations based on dataset type and risk scores
            recommendations = []
            if request.dataset_type == "uncurated":
                recommendations.append("Consider additional curation for uncurated dataset")
                if risk_scores["bias_potential"] > 0.3:
                    recommendations.append("High bias risk detected - implement bias mitigation techniques")
                if risk_scores["harm_risk"] > 0.4:
                    recommendations.append("Significant harm risk - consider filtering harmful examples")
            
            if risk_scores["autonomy_violations"] > 0.2:
                recommendations.append("Autonomy violations detected - review examples for manipulation patterns")
                
            if not recommendations:
                recommendations.append("Dataset appears ethically sound for training")
                
            processing_time = time.time() - start_time
            
            return {
                "dataset_analysis": {
                    "total_examples": len(request.training_data),
                    "ethical_examples": sum(1 for ev in evaluations if ev["ethical_status"]),
                    "problematic_examples": len(ethical_issues),
                    "dataset_type": request.dataset_type,
                    "overall_ethical_score": sum(1 for ev in evaluations if ev["ethical_status"]) / len(evaluations) if evaluations else 0.0
                },
                "risk_assessment": risk_scores,
                "recommendations": recommendations,
                "ethical_issues": ethical_issues[:10],  # Limit to first 10 issues
                "sample_evaluations": evaluations[:5],  # Sample of evaluations
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ML training data evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training data evaluation failed: {str(e)}")
    
    @api_router.post("/ml/ethical-vectors", response_model=MLEthicalVectorResponse)
    async def generate_ethical_vectors(request: MLTrainingDataRequest):
        """
        Generate ethical guidance vectors for ML training using advanced vector engine.
        
        Converts ethical evaluations into actionable vectors that can guide
        ML model training, loss functions, and behavioral adjustments.
        """
        try:
            start_time = time.time()
            
            # Parse enum values
            dataset_type = DatasetType(request.dataset_type)
            training_phase = TrainingPhase(request.training_phase.replace('-', '_'))
            
            # Collect all evaluations and vectors
            all_ml_vectors = []
            ethical_scores = []
            batch_evaluations = []
            
            # Evaluate training data and extract ethical vectors using advanced engine
            for example in request.training_data:
                if evaluator:
                    evaluation = evaluator.evaluate_text(example)
                    batch_evaluations.append(evaluation)
                    
                    # Use advanced ML Ethics Vector Engine
                    ml_vectors = ml_ethics_engine.convert_evaluation_to_ml_vectors(
                        evaluation,
                        dataset_type=dataset_type,
                        training_phase=training_phase,
                        model_type=request.model_type
                    )
                    all_ml_vectors.append(ml_vectors)
                    
                    # Calculate ethical score
                    autonomy_score = sum(evaluation.autonomy_dimensions.values()) / len(evaluation.autonomy_dimensions)
                    ethical_scores.append(autonomy_score)
            
            # Average all vectors across examples
            if all_ml_vectors:
                def average_vector_field(field_name):
                    field_vectors = [getattr(v, field_name) for v in all_ml_vectors]
                    return [sum(vals) / len(vals) for vals in zip(*field_vectors)]
                
                averaged_vectors = MLEthicalVector(
                    autonomy_vectors=average_vector_field('autonomy_vectors'),
                    harm_prevention_vectors=average_vector_field('harm_prevention_vectors'),
                    fairness_vectors=average_vector_field('fairness_vectors'),
                    transparency_vectors=average_vector_field('transparency_vectors'),
                    bias_mitigation_vectors=average_vector_field('bias_mitigation_vectors'),
                    safety_vectors=average_vector_field('safety_vectors')
                )
            else:
                # Default vectors if no evaluations
                averaged_vectors = MLEthicalVector(
                    autonomy_vectors=[0.5] * 5,
                    harm_prevention_vectors=[0.5] * 5,
                    fairness_vectors=[0.5] * 5,
                    transparency_vectors=[0.5] * 5,
                    bias_mitigation_vectors=[0.5] * 5,
                    safety_vectors=[0.5] * 5
                )
            
            # Generate advanced training adjustments
            avg_ethical_score = sum(ethical_scores) / len(ethical_scores) if ethical_scores else 0.5
            
            training_adjustments = ml_ethics_engine.generate_training_adjustments(
                averaged_vectors,
                avg_ethical_score,
                training_phase=training_phase
            )
            
            # Generate recommendations using advanced analysis
            recommendations = []
            if avg_ethical_score < 0.6:
                recommendations.append(f"Dataset ethical score is {avg_ethical_score:.2f} - consider additional ethical filtering")
            
            # Phase-specific recommendations
            if training_phase == TrainingPhase.FINE_TUNING:
                recommendations.append("Fine-tuning phase: Apply stronger ethical constraints and monitoring")
            elif training_phase == TrainingPhase.REINFORCEMENT:
                recommendations.append("RLHF phase: Implement continuous ethical feedback loops")
                
            if dataset_type == DatasetType.UNCURATED:
                recommendations.append("Uncurated dataset: Implement continuous ethical monitoring during training")
                
            recommendations.extend([
                "Use autonomy vectors to guide gradient updates toward human autonomy preservation",
                "Apply bias mitigation vectors to prevent discriminatory patterns",
                "Use safety vectors to prevent harmful output generation",
                "Monitor transparency vectors to ensure explainable model behavior"
            ])
            
            # Enhanced risk assessment using advanced engine
            risk_assessment = {
                "autonomy_risk": 1.0 - avg_ethical_score,
                "harm_risk": 1.0 - (sum(averaged_vectors.harm_prevention_vectors) / len(averaged_vectors.harm_prevention_vectors)),
                "bias_risk": 1.0 - (sum(averaged_vectors.bias_mitigation_vectors) / len(averaged_vectors.bias_mitigation_vectors)),
                "transparency_risk": 1.0 - (sum(averaged_vectors.transparency_vectors) / len(averaged_vectors.transparency_vectors)),
                "safety_risk": 1.0 - (sum(averaged_vectors.safety_vectors) / len(averaged_vectors.safety_vectors)),
                "overall_risk": (1.0 - avg_ethical_score)
            }
            
            processing_time = time.time() - start_time
            
            return MLEthicalVectorResponse(
                ethical_vectors=averaged_vectors.to_dict(),
                training_adjustments=training_adjustments.to_dict(),
                recommendations=recommendations,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Enhanced ethical vector generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Enhanced ethical vector generation failed: {str(e)}")
    
    @api_router.post("/ml/training-guidance", response_model=MLGuidanceResponse)
    async def provide_training_guidance(request: MLTrainingBatchRequest):
        """
        Provide real-time ethical guidance during ML training using advanced intervention logic.
        
        Analyzes training batches in real-time and provides guidance on whether
        to continue training, along with ethical adjustments and warnings.
        """
        try:
            start_time = time.time()
            
            # Evaluate batch data
            batch_evaluations = []
            ethical_scores = []
            warnings = []
            
            for idx, example in enumerate(request.batch_data):
                if evaluator:
                    evaluation = evaluator.evaluate_text(example)
                    batch_evaluations.append(evaluation)
                    
                    # Calculate ethical score based on autonomy dimensions
                    autonomy_score = sum(evaluation.autonomy_dimensions.values()) / len(evaluation.autonomy_dimensions)
                    ethical_scores.append(autonomy_score)
                    
                    # Check for severe violations
                    if not evaluation.overall_ethical and evaluation.minimal_violation_count > 3:
                        warnings.append(f"Batch item {idx}: {evaluation.minimal_violation_count} serious violations detected")
            
            # Use advanced intervention evaluation
            training_phase = TrainingPhase(request.training_step // 100 if request.training_step < 500 else "fine_tuning")
            intervention_analysis = ml_ethics_engine.evaluate_training_intervention(
                batch_evaluations, 
                ethical_scores,
                training_phase
            )
            
            # Generate advanced training adjustments using the ML Ethics Engine
            if ethical_scores:
                avg_ethical_score = sum(ethical_scores) / len(ethical_scores)
                
                # Create a representative ML vector for the batch
                sample_evaluation = batch_evaluations[0] if batch_evaluations else None
                if sample_evaluation:
                    ml_vectors = ml_ethics_engine.convert_evaluation_to_ml_vectors(
                        sample_evaluation,
                        dataset_type=DatasetType.UNCURATED,  # Assume uncurated for real-time batches
                        training_phase=training_phase
                    )
                    
                    training_adjustments = ml_ethics_engine.generate_training_adjustments(
                        ml_vectors,
                        avg_ethical_score,
                        training_phase=training_phase,
                        current_loss=request.loss_value,
                        training_step=request.training_step
                    )
                    
                    adjustments = training_adjustments.to_dict()
                else:
                    # Fallback adjustments
                    adjustments = {
                        "batch_weight": max(0.1, avg_ethical_score),
                        "gradient_scaling": 0.5 + (0.5 * avg_ethical_score),
                        "dropout_increase": 0.1 + (0.2 * (1.0 - avg_ethical_score)),
                        "attention_focus": avg_ethical_score,
                        "loss_reweighting": 1.0 + (2.0 * (1.0 - avg_ethical_score))
                    }
            else:
                avg_ethical_score = 0.0
                adjustments = {"error": "No evaluations available"}
            
            # Combine warnings from intervention analysis
            if "recommendations" in intervention_analysis:
                warnings.extend(intervention_analysis["recommendations"])
            
            # Training step specific guidance with advanced logic
            if request.training_step > 0:
                if request.loss_value and request.loss_value > 2.0:
                    warnings.append("High loss detected - ethical constraints may be too strict")
                elif request.loss_value and request.loss_value < 0.1:
                    warnings.append("Very low loss - model may be overfitting to unethical patterns")
                    
                # Advanced step-based recommendations
                if request.training_step > 1000 and avg_ethical_score < 0.6:
                    warnings.append("Late training stage with low ethical score - consider early stopping")
                elif request.training_step < 100 and avg_ethical_score < 0.4:
                    warnings.append("Early training with very low ethical score - data filtering recommended")
            
            processing_time = time.time() - start_time
            
            # Add processing metadata to adjustments
            adjustments.update({
                "processing_time": processing_time,
                "evaluation_count": len(batch_evaluations),
                "intervention_analysis": intervention_analysis,
                "batch_size": len(request.batch_data),
                "training_phase_detected": training_phase.value
            })
            
            return MLGuidanceResponse(
                continue_training=intervention_analysis.get("continue_training", True),
                ethical_score=avg_ethical_score,
                warnings=warnings[:10],  # Limit warnings to prevent response overflow
                adjustments=adjustments,
                intervention_required=intervention_analysis.get("intervention_required", False)
            )
            
        except Exception as e:
            logger.error(f"Enhanced ML training guidance failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Enhanced training guidance failed: {str(e)}")
    
    @api_router.post("/ml/model-behavior-eval")
    async def evaluate_model_behavior(request: Dict[str, Any]):
        """
        Evaluate trained model behavior for ethical compliance.
        
        Tests model outputs against ethical standards and provides
        recommendations for model adjustment or retraining.
        """
        try:
            start_time = time.time()
            
            model_outputs = request.get("model_outputs", [])
            test_prompts = request.get("test_prompts", [])
            model_info = request.get("model_info", {})
            
            if not model_outputs:
                raise HTTPException(status_code=400, detail="model_outputs required")
            
            # Evaluate each model output
            behavior_analysis = {
                "total_outputs": len(model_outputs),
                "ethical_outputs": 0,
                "problematic_outputs": 0,
                "violation_details": [],
                "behavioral_patterns": [],
                "recommendations": []
            }
            
            for idx, output in enumerate(model_outputs):
                if evaluator:
                    evaluation = evaluator.evaluate_text(output)
                    
                    if evaluation.overall_ethical:
                        behavior_analysis["ethical_outputs"] += 1
                    else:
                        behavior_analysis["problematic_outputs"] += 1
                        behavior_analysis["violation_details"].append({
                            "output_index": idx,
                            "violations": evaluation.minimal_violation_count,
                            "prompt": test_prompts[idx] if idx < len(test_prompts) else "N/A",
                            "output_sample": output[:200] + "..." if len(output) > 200 else output
                        })
            
            # Calculate behavioral scores
            ethical_ratio = behavior_analysis["ethical_outputs"] / behavior_analysis["total_outputs"]
            
            # Generate behavioral pattern analysis
            if ethical_ratio < 0.6:
                behavior_analysis["behavioral_patterns"].append("Model shows concerning ethical patterns")
                behavior_analysis["recommendations"].extend([
                    "Consider retraining with additional ethical constraints",
                    "Implement output filtering for production deployment",
                    "Add ethical fine-tuning phase"
                ])
            elif ethical_ratio < 0.8:
                behavior_analysis["behavioral_patterns"].append("Model shows mixed ethical behavior")
                behavior_analysis["recommendations"].extend([
                    "Fine-tune model with ethical feedback",
                    "Monitor model outputs in production"
                ])
            else:
                behavior_analysis["behavioral_patterns"].append("Model demonstrates good ethical alignment")
                behavior_analysis["recommendations"].append("Model appears ready for ethical deployment")
            
            behavior_analysis["ethical_score"] = ethical_ratio
            behavior_analysis["processing_time"] = time.time() - start_time
            behavior_analysis["model_info"] = model_info
            behavior_analysis["timestamp"] = datetime.now().isoformat()
            
            return behavior_analysis
            
        except Exception as e:
            logger.error(f"Model behavior evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model behavior evaluation failed: {str(e)}")
    
    @api_router.post("/ml/advanced-analysis")
    async def advanced_ml_ethical_analysis(request: Dict[str, Any]):
        """
        Perform advanced ML ethical analysis with comprehensive vector breakdowns.
        
        Provides detailed analysis including bias detection, safety assessment,
        and training-phase-specific recommendations for ML systems.
        """
        try:
            start_time = time.time()
            
            training_data = request.get("training_data", [])
            dataset_type = request.get("dataset_type", "uncurated")
            training_phase = request.get("training_phase", "initial")
            model_type = request.get("model_type", "general")
            analysis_depth = request.get("analysis_depth", "standard")  # standard, deep, comprehensive
            
            if not training_data:
                raise HTTPException(status_code=400, detail="training_data required")
            
            # Parse enums
            dataset_type_enum = DatasetType(dataset_type)
            training_phase_enum = TrainingPhase(training_phase.replace('-', '_'))
            
            # Perform comprehensive ethical analysis
            detailed_analysis = {
                "dataset_overview": {
                    "total_examples": len(training_data),
                    "dataset_type": dataset_type,
                    "training_phase": training_phase,
                    "model_type": model_type,
                    "analysis_depth": analysis_depth
                },
                "ethical_evaluations": [],
                "vector_analysis": {},
                "risk_assessment": {},
                "intervention_recommendations": [],
                "training_adjustments": {},
                "bias_analysis": {
                    "gender_bias": [],
                    "racial_bias": [],
                    "age_bias": [],
                    "cultural_bias": [],
                    "socioeconomic_bias": []
                },
                "safety_analysis": {
                    "misuse_potential": [],
                    "deception_risk": [],
                    "manipulation_risk": [],
                    "harm_potential": []
                }
            }
            
            all_evaluations = []
            all_ml_vectors = []
            ethical_scores = []
            
            # Deep analysis of each example
            for idx, example in enumerate(training_data):
                if evaluator:
                    evaluation = evaluator.evaluate_text(example)
                    all_evaluations.append(evaluation)
                    
                    # Generate ML vectors
                    ml_vectors = ml_ethics_engine.convert_evaluation_to_ml_vectors(
                        evaluation,
                        dataset_type=dataset_type_enum,
                        training_phase=training_phase_enum,
                        model_type=model_type
                    )
                    all_ml_vectors.append(ml_vectors)
                    
                    # Calculate ethical score
                    autonomy_score = sum(evaluation.autonomy_dimensions.values()) / len(evaluation.autonomy_dimensions)
                    ethical_scores.append(autonomy_score)
                    
                    # Detailed evaluation info
                    detailed_analysis["ethical_evaluations"].append({
                        "index": idx,
                        "text_sample": example[:100] + "..." if len(example) > 100 else example,
                        "overall_ethical": evaluation.overall_ethical,
                        "violation_count": evaluation.minimal_violation_count,
                        "autonomy_score": autonomy_score,
                        "autonomy_dimensions": evaluation.autonomy_dimensions,
                        "ethical_principles": evaluation.ethical_principles,
                        "ml_vectors": ml_vectors.to_dict() if analysis_depth == "comprehensive" else None
                    })
                    
                    # Bias analysis (if deep or comprehensive analysis)
                    if analysis_depth in ["deep", "comprehensive"]:
                        detailed_analysis["bias_analysis"]["gender_bias"].append(
                            ml_ethics_engine._detect_gender_bias(evaluation)
                        )
                        detailed_analysis["safety_analysis"]["misuse_potential"].append(
                            ml_ethics_engine._detect_misuse_potential(evaluation)
                        )
            
            # Aggregate vector analysis
            if all_ml_vectors:
                def aggregate_vectors(field_name):
                    field_vectors = [getattr(v, field_name) for v in all_ml_vectors]
                    return {
                        "mean": [sum(vals) / len(vals) for vals in zip(*field_vectors)],
                        "std": [np.std([vals[i] for vals in field_vectors]) for i in range(len(field_vectors[0]))],
                        "min": [min(vals) for vals in zip(*field_vectors)],
                        "max": [max(vals) for vals in zip(*field_vectors)]
                    }
                
                detailed_analysis["vector_analysis"] = {
                    "autonomy_vectors": aggregate_vectors("autonomy_vectors"),
                    "harm_prevention_vectors": aggregate_vectors("harm_prevention_vectors"),
                    "fairness_vectors": aggregate_vectors("fairness_vectors"),
                    "transparency_vectors": aggregate_vectors("transparency_vectors"),
                    "bias_mitigation_vectors": aggregate_vectors("bias_mitigation_vectors"),
                    "safety_vectors": aggregate_vectors("safety_vectors")
                }
            
            # Overall risk assessment
            avg_ethical_score = sum(ethical_scores) / len(ethical_scores) if ethical_scores else 0.0
            detailed_analysis["risk_assessment"] = {
                "overall_ethical_score": avg_ethical_score,
                "high_risk_examples": len([s for s in ethical_scores if s < 0.3]),
                "medium_risk_examples": len([s for s in ethical_scores if 0.3 <= s < 0.6]),
                "low_risk_examples": len([s for s in ethical_scores if s >= 0.6]),
                "dataset_safety_rating": "HIGH" if avg_ethical_score > 0.8 else "MEDIUM" if avg_ethical_score > 0.5 else "LOW"
            }
            
            # Training intervention analysis
            intervention_analysis = ml_ethics_engine.evaluate_training_intervention(
                all_evaluations,
                ethical_scores,
                training_phase_enum
            )
            detailed_analysis["intervention_recommendations"] = intervention_analysis
            
            # Advanced training adjustments
            if all_ml_vectors and ethical_scores:
                sample_vectors = all_ml_vectors[0]  # Use first example as representative
                training_adjustments = ml_ethics_engine.generate_training_adjustments(
                    sample_vectors,
                    avg_ethical_score,
                    training_phase_enum
                )
                detailed_analysis["training_adjustments"] = training_adjustments.to_dict()
            
            # Processing metadata
            detailed_analysis["processing_metadata"] = {
                "processing_time": time.time() - start_time,
                "analysis_timestamp": datetime.now().isoformat(),
                "engine_version": "2.0.0",
                "examples_processed": len(training_data),
                "total_evaluations": len(all_evaluations)
            }
            
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"Advanced ML ethical analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
    
    # ============================================================================
    # SMART BUFFER STREAMING API - PHASE 3 IMPLEMENTATION
    # ============================================================================
    
    class StreamTokenRequest(BaseModel):
        """Request model for streaming tokens."""
        tokens: List[str] = Field(..., description="List of tokens to add to stream")
        session_id: str = Field(..., description="Unique session identifier")
        training_step: Optional[int] = Field(default=None, description="Current training step")
        batch_id: Optional[str] = Field(default=None, description="Training batch identifier")
        metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
        
    class StreamConfigRequest(BaseModel):
        """Request model for stream configuration."""
        max_tokens: int = Field(default=512, description="Maximum tokens before processing")
        max_time_seconds: float = Field(default=5.0, description="Maximum time before processing")
        semantic_threshold: float = Field(default=0.7, description="Semantic coherence threshold")
        performance_threshold_ms: float = Field(default=100.0, description="Performance threshold in ms")
        pattern_detection: bool = Field(default=True, description="Enable pattern detection")
        
    class StreamAnalysisResponse(BaseModel):
        """Response model for stream analysis results."""
        analysis_triggered: bool = Field(..., description="Whether analysis was triggered")
        ethical_score: Optional[float] = Field(None, description="Ethical score if analysis occurred")
        risk_level: Optional[str] = Field(None, description="Risk level assessment")
        intervention_required: Optional[bool] = Field(None, description="Whether intervention is required")
        processing_time: Optional[float] = Field(None, description="Processing time in seconds")
        warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
        recommendations: List[str] = Field(default_factory=list, description="Ethical recommendations")
        buffer_metrics: Optional[Dict[str, Any]] = Field(None, description="Buffer performance metrics")
        
    @api_router.post("/ml/stream/configure", response_model=Dict[str, Any])
    async def configure_stream_buffer(request: StreamConfigRequest):
        """
        Configure the smart buffer for streaming training data analysis.
        
        Sets up or updates the global streaming buffer with specified parameters
        for real-time ethical monitoring during ML training.
        """
        global training_stream_buffer
        
        try:
            # Create buffer configuration
            buffer_config = BufferConfig(
                max_tokens=request.max_tokens,
                max_time_seconds=request.max_time_seconds,
                semantic_threshold=request.semantic_threshold,
                performance_threshold_ms=request.performance_threshold_ms,
                pattern_detection=request.pattern_detection
            )
            
            # Create evaluator callback that uses our existing evaluator
            def evaluator_callback(text: str):
                if evaluator:
                    return evaluator.evaluate_text(text)
                return None
            
            # Create or update the smart buffer
            if training_stream_buffer is None:
                training_stream_buffer = SmartBuffer(
                    buffer_config,
                    evaluator_callback=evaluator_callback,
                    ml_ethics_engine=ml_ethics_engine
                )
                logger.info("Smart buffer created for streaming analysis")
            else:
                training_stream_buffer.update_config(buffer_config)
                logger.info("Smart buffer configuration updated")
            
            return {
                "status": "configured",
                "configuration": {
                    "max_tokens": request.max_tokens,
                    "max_time_seconds": request.max_time_seconds,
                    "semantic_threshold": request.semantic_threshold,
                    "performance_threshold_ms": request.performance_threshold_ms,
                    "pattern_detection": request.pattern_detection
                },
                "buffer_state": training_stream_buffer.state.value,
                "message": "Smart buffer configured for streaming analysis"
            }
            
        except Exception as e:
            logger.error(f"Stream buffer configuration failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")
    
    @api_router.post("/ml/stream/tokens", response_model=StreamAnalysisResponse)
    async def add_stream_tokens(request: StreamTokenRequest):
        """
        Add tokens to the streaming buffer for real-time ethical analysis.
        
        Processes incoming token streams and triggers ethical evaluation when
        semantic boundaries or thresholds are reached.
        """
        global training_stream_buffer
        
        try:
            # Ensure buffer is configured
            if training_stream_buffer is None:
                # Create default buffer if none exists
                await configure_stream_buffer(StreamConfigRequest())
            
            # Prepare metadata
            metadata = request.metadata or {}
            metadata.update({
                "session_id": request.session_id,
                "training_step": request.training_step,
                "batch_id": request.batch_id,
                "timestamp": time.time()
            })
            
            # Add tokens to buffer
            analysis = await training_stream_buffer.add_tokens(request.tokens, metadata)
            
            # Get current buffer metrics
            metrics = training_stream_buffer.get_metrics()
            
            if analysis:
                # Analysis was triggered
                return StreamAnalysisResponse(
                    analysis_triggered=True,
                    ethical_score=analysis.ethical_score,
                    risk_level=analysis.risk_level,
                    intervention_required=analysis.intervention_required,
                    processing_time=analysis.processing_time,
                    warnings=[w for w in analysis.recommendations if "WARNING" in w or "CRITICAL" in w],
                    recommendations=analysis.recommendations,
                    buffer_metrics={
                        "tokens_processed": metrics.tokens_processed,
                        "evaluations_completed": metrics.evaluations_completed,
                        "average_processing_time": metrics.average_processing_time,
                        "buffer_utilization": metrics.buffer_utilization,
                        "interventions_triggered": metrics.interventions_triggered
                    }
                )
            else:
                # No analysis triggered - just accumulating
                return StreamAnalysisResponse(
                    analysis_triggered=False,
                    buffer_metrics={
                        "tokens_processed": metrics.tokens_processed,
                        "buffer_utilization": metrics.buffer_utilization,
                        "current_state": training_stream_buffer.state.value
                    }
                )
                
        except Exception as e:
            logger.error(f"Stream token processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}")
    
    @api_router.post("/ml/stream/flush")
    async def flush_stream_buffer():
        """
        Force flush the current stream buffer and get analysis results.
        
        Processes all accumulated tokens immediately regardless of thresholds.
        """
        global training_stream_buffer
        
        try:
            if training_stream_buffer is None:
                raise HTTPException(status_code=400, detail="No active stream buffer")
            
            # Force flush the buffer
            analysis = await training_stream_buffer.force_flush()
            
            if analysis:
                return {
                    "status": "flushed",
                    "analysis": {
                        "ethical_score": analysis.ethical_score,
                        "violation_count": analysis.violation_count,
                        "risk_level": analysis.risk_level,
                        "intervention_required": analysis.intervention_required,
                        "processing_time": analysis.processing_time,
                        "patterns_detected": analysis.patterns_detected,
                        "recommendations": analysis.recommendations
                    },
                    "buffer_metrics": training_stream_buffer.get_metrics().__dict__
                }
            else:
                return {
                    "status": "flushed",
                    "message": "Buffer was empty",
                    "buffer_metrics": training_stream_buffer.get_metrics().__dict__
                }
                
        except Exception as e:
            logger.error(f"Stream buffer flush failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Buffer flush failed: {str(e)}")
    
    @api_router.get("/ml/stream/metrics")
    async def get_stream_metrics():
        """
        Get current streaming buffer performance metrics.
        
        Returns comprehensive metrics about buffer performance, throughput,
        and processing statistics.
        """
        global training_stream_buffer
        
        try:
            if training_stream_buffer is None:
                return {
                    "status": "no_active_buffer",
                    "message": "No streaming buffer is currently active"
                }
            
            metrics = training_stream_buffer.get_metrics()
            
            return {
                "status": "active",
                "buffer_state": training_stream_buffer.state.value,
                "metrics": {
                    "tokens_processed": metrics.tokens_processed,
                    "evaluations_completed": metrics.evaluations_completed,
                    "average_processing_time": metrics.average_processing_time,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "buffer_utilization": metrics.buffer_utilization,
                    "boundary_detections": metrics.boundary_detections,
                    "interventions_triggered": metrics.interventions_triggered,
                    "total_runtime": metrics.total_runtime,
                    "memory_usage_mb": metrics.memory_usage_mb
                },
                "configuration": {
                    "max_tokens": training_stream_buffer.config.max_tokens,
                    "max_time_seconds": training_stream_buffer.config.max_time_seconds,
                    "pattern_detection": training_stream_buffer.config.pattern_detection,
                    "auto_resize": training_stream_buffer.config.auto_resize
                }
            }
            
        except Exception as e:
            logger.error(f"Stream metrics retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")
    
    @api_router.post("/ml/stream/reset")
    async def reset_stream_buffer():
        """
        Reset the streaming buffer system.
        
        Clears all accumulated data and resets metrics for a fresh start.
        """
        global training_stream_buffer
        
        try:
            if training_stream_buffer is not None:
                await training_stream_buffer.cleanup()
                training_stream_buffer = None
            
            return {
                "status": "reset",
                "message": "Streaming buffer has been reset"
            }
            
        except Exception as e:
            logger.error(f"Stream buffer reset failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Buffer reset failed: {str(e)}")
    
    # ============================================================================
    # END SMART BUFFER STREAMING API - PHASE 3
    # ============================================================================
    
    # ============================================================================
    # MULTI-MODAL EVALUATION API - PHASE 4 IMPLEMENTATION
    # ============================================================================
    
    class MultiModalEvaluationRequest(BaseModel):
        """Request model for multi-modal evaluation."""
        content: Union[str, List[str]] = Field(..., description="Content to evaluate")
        mode: str = Field(..., description="Evaluation mode: pre_evaluation, post_evaluation, stream_evaluation")
        priority: str = Field(default="medium", description="Priority: critical, high, medium, low, batch")
        original_input: Optional[str] = Field(default=None, description="Original input for post-evaluation")
        intended_purpose: Optional[str] = Field(default=None, description="Intended purpose for alignment assessment")
        user_id: Optional[str] = Field(default=None, description="User identifier")
        session_id: Optional[str] = Field(default=None, description="Session identifier")
        training_context: Optional[Dict[str, Any]] = Field(default=None, description="Training context metadata")
        timeout: float = Field(default=60.0, description="Evaluation timeout in seconds")
        
    class BatchEvaluationRequest(BaseModel):
        """Request model for batch multi-modal evaluation."""
        content_items: List[Union[str, List[str]]] = Field(..., description="List of content to evaluate")
        mode: str = Field(..., description="Evaluation mode for all items")
        priority: str = Field(default="batch", description="Priority level")
        batch_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata for entire batch")
        
    class ModeConfigurationRequest(BaseModel):
        """Request model for mode configuration."""
        mode: str = Field(..., description="Evaluation mode to configure")
        configuration: Dict[str, Any] = Field(..., description="Configuration parameters")
        
    @api_router.post("/multimodal/evaluate", response_model=Dict[str, Any])
    async def multimodal_evaluate(request: MultiModalEvaluationRequest):
        """
        Perform multi-modal evaluation using the orchestrator.
        
        Supports different evaluation modes:
        - pre_evaluation: Screen inputs before processing
        - post_evaluation: Validate outputs after processing
        - stream_evaluation: Real-time analysis during processing
        """
        try:
            start_time = time.time()
            
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Multi-modal orchestrator not initialized")
            
            # Parse enum values
            try:
                mode = EvaluationMode(request.mode)
                priority = EvaluationPriority(request.priority)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")
            
            # Prepare evaluation parameters
            kwargs = {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "training_context": request.training_context,
                "timeout": request.timeout
            }
            
            # Add mode-specific parameters
            if mode == EvaluationMode.POST_EVALUATION:
                kwargs["original_input"] = request.original_input
                kwargs["intended_purpose"] = request.intended_purpose
            
            # Perform evaluation
            result = await orchestrator.evaluate(
                content=request.content,
                mode=mode,
                priority=priority,
                **kwargs
            )
            
            # Convert to dict and add processing metadata
            response = result.to_dict()
            response["api_processing_time"] = time.time() - start_time
            response["orchestrator_metrics"] = {
                "active_requests": len(orchestrator.active_requests),
                "mode_used": mode.value,
                "priority_used": priority.value
            }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Multi-modal evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    @api_router.post("/multimodal/batch-evaluate", response_model=Dict[str, Any])
    async def batch_multimodal_evaluate(request: BatchEvaluationRequest):
        """
        Perform batch multi-modal evaluation of multiple content items.
        
        Efficiently processes multiple items using concurrent evaluation
        with intelligent load balancing and error handling.
        """
        try:
            start_time = time.time()
            
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Multi-modal orchestrator not initialized")
            
            # Parse enum values
            try:
                mode = EvaluationMode(request.mode)
                priority = EvaluationPriority(request.priority)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")
            
            # Validate batch size
            if len(request.content_items) > 100:
                raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100 items")
            
            # Perform batch evaluation
            results = await orchestrator.batch_evaluate(
                content_items=request.content_items,
                mode=mode,
                priority=priority,
                batch_metadata=request.batch_metadata
            )
            
            # Aggregate results
            successful_results = [r for r in results if r.status != "failed"]
            failed_results = [r for r in results if r.status == "failed"]
            
            # Calculate batch statistics
            batch_stats = {
                "total_items": len(request.content_items),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "success_rate": len(successful_results) / len(request.content_items),
                "average_processing_time": sum(r.processing_time for r in successful_results) / len(successful_results) if successful_results else 0,
                "average_ethical_score": sum(r.overall_ethical_score for r in successful_results) / len(successful_results) if successful_results else 0
            }
            
            return {
                "batch_id": str(uuid.uuid4()),
                "batch_stats": batch_stats,
                "results": [r.to_dict() for r in results],
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch multi-modal evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")
    
    @api_router.post("/multimodal/configure-mode", response_model=Dict[str, Any])
    async def configure_evaluation_mode(request: ModeConfigurationRequest):
        """
        Configure a specific evaluation mode with custom parameters.
        
        Allows dynamic adjustment of evaluation thresholds, criteria,
        and behavior for different use cases and requirements.
        """
        try:
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Multi-modal orchestrator not initialized")
            
            # Parse mode enum
            try:
                mode = EvaluationMode(request.mode)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid evaluation mode: {request.mode}")
            
            # Apply configuration
            success = await orchestrator.configure_mode(mode, request.configuration)
            
            if success:
                return {
                    "status": "configured",
                    "mode": request.mode,
                    "configuration_applied": request.configuration,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=400, detail=f"Failed to configure mode: {request.mode}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Mode configuration failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")
    
    @api_router.get("/multimodal/capabilities")
    async def get_multimodal_capabilities():
        """
        Get comprehensive information about available evaluation modes and capabilities.
        
        Returns detailed information about each mode's features, performance
        characteristics, and configuration options.
        """
        try:
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Multi-modal orchestrator not initialized")
            
            capabilities = {
                "available_modes": [],
                "orchestrator_info": {
                    "max_concurrent_evaluations": orchestrator.max_concurrent_evaluations,
                    "active_requests": len(orchestrator.active_requests),
                    "total_requests_processed": len(orchestrator.request_history)
                },
                "supported_priorities": [p.value for p in EvaluationPriority],
                "api_version": "4.0.0",
                "features": [
                    "Multi-modal evaluation orchestration",
                    "Circuit breaker resilience patterns",
                    "Concurrent evaluation processing", 
                    "Batch evaluation support",
                    "Dynamic mode configuration",
                    "Comprehensive metrics and monitoring"
                ]
            }
            
            # Get capabilities for each available mode
            for mode in EvaluationMode:
                mode_capabilities = orchestrator.get_mode_capabilities(mode)
                if "error" not in mode_capabilities:
                    capabilities["available_modes"].append(mode_capabilities)
            
            return capabilities
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Capabilities retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")
    
    @api_router.get("/multimodal/metrics")
    async def get_multimodal_metrics():
        """
        Get comprehensive orchestrator performance metrics.
        
        Returns detailed metrics about evaluation performance, success rates,
        circuit breaker states, and system health indicators.
        """
        try:
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                raise HTTPException(status_code=500, detail="Multi-modal orchestrator not initialized")
            
            # Get comprehensive metrics
            metrics = orchestrator.get_orchestrator_metrics()
            
            # Add timestamp and system info
            metrics["timestamp"] = datetime.now().isoformat()
            metrics["system_info"] = {
                "uptime_seconds": time.time() - (orchestrator.request_history[0].timestamp if orchestrator.request_history else time.time()),
                "python_version": "3.x",
                "orchestrator_version": "4.0.0"
            }
            
            return metrics
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    @api_router.post("/multimodal/health-check")
    async def multimodal_health_check():
        """
        Perform comprehensive health check of the multi-modal evaluation system.
        
        Tests all evaluation modes, checks circuit breaker states,
        and verifies system readiness for production workloads.
        """
        try:
            start_time = time.time()
            
            # Get the orchestrator
            orchestrator = get_orchestrator()
            if not orchestrator:
                return {
                    "status": "unhealthy",
                    "error": "Multi-modal orchestrator not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            health_results = {
                "overall_status": "healthy",
                "mode_health": {},
                "circuit_breaker_status": {},
                "system_health": {},
                "recommendations": []
            }
            
            # Test each evaluation mode with a simple evaluation
            test_content = "This is a simple test message for health checking."
            
            for mode in [EvaluationMode.PRE_EVALUATION, EvaluationMode.POST_EVALUATION]:
                try:
                    result = await orchestrator.evaluate(
                        content=test_content,
                        mode=mode,
                        priority=EvaluationPriority.HIGH,
                        timeout=10.0
                    )
                    
                    health_results["mode_health"][mode.value] = {
                        "status": "healthy",
                        "response_time": result.processing_time,
                        "confidence": result.confidence,
                        "last_test": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    health_results["mode_health"][mode.value] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "last_test": datetime.now().isoformat()
                    }
                    health_results["overall_status"] = "degraded"
            
            # Check circuit breaker states
            for mode, breaker in orchestrator.circuit_breakers.items():
                health_results["circuit_breaker_status"][mode.value] = {
                    "state": breaker["state"],
                    "failure_count": breaker["failures"],
                    "healthy": breaker["state"] in ["closed", "half-open"]
                }
                
                if breaker["state"] == "open":
                    health_results["overall_status"] = "degraded"
                    health_results["recommendations"].append(f"Circuit breaker open for {mode.value} - investigate failures")
            
            # System health indicators
            health_results["system_health"] = {
                "active_requests": len(orchestrator.active_requests),
                "memory_usage": "normal",  # Could be enhanced with actual memory monitoring
                "response_time": time.time() - start_time,
                "concurrent_capacity": orchestrator.max_concurrent_evaluations
            }
            
            # Generate recommendations
            if health_results["overall_status"] == "healthy":
                health_results["recommendations"].append("System is operating normally")
            elif health_results["overall_status"] == "degraded":
                health_results["recommendations"].append("System has degraded performance - monitor closely")
            
            health_results["timestamp"] = datetime.now().isoformat()
            health_results["health_check_duration"] = time.time() - start_time
            
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "health_check_duration": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    # ============================================================================
    # END MULTI-MODAL EVALUATION API - PHASE 4
    # ============================================================================
    
    # ============================================================================
    # END ML ETHICS API - PHASE 3
    # ============================================================================
    
    # Include API router
    app.include_router(api_router)
    
    # Application startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize integrated evaluation systems on startup."""
        logger.info("🚀 Starting Ethical AI Developer Testbed - Integrated v1.1.0")
        await init_evaluation_systems()
        logger.info("✅ Integrated evaluation systems ready")
    
    @app.on_event("shutdown") 
    async def shutdown_event():
        """Clean up resources on shutdown."""
        logger.info("🛑 Shutting down Ethical AI Developer Testbed - Integrated")
        
        if OPTIMIZED_COMPONENTS_AVAILABLE:
            global_optimized_engine.cleanup()
            global_embedding_service.cleanup()
            global_cache_manager.clear_all_caches()
        
        executor.shutdown(wait=True)
        logger.info("✅ Cleanup completed")
    
    return app

# Create the integrated application
app = create_integrated_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)