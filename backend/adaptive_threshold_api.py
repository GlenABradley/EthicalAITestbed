"""
Adaptive Threshold Learning API Endpoints

This module provides FastAPI endpoints for the new adaptive threshold learning system,
replacing the obsolete manual threshold tuning interface.

Features:
- Perceptron model training and management
- Training data generation and management
- Adaptive prediction with confidence scores
- Model performance monitoring and audit logs
- User override and transparency features

Author: Ethical AI Testbed Team
Date: 2025-08-06
Version: 1.0
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.adaptive_threshold_learner import IntentNormalizedFeatureExtractor
from backend.perceptron_threshold_learner import PerceptronThresholdLearner, TrainingExample
from backend.training_data_pipeline import TrainingDataPipeline, DataGenerationConfig

logger = logging.getLogger(__name__)

# Create router for adaptive threshold endpoints
router = APIRouter(prefix="/api/adaptive", tags=["Adaptive Threshold Learning"])

# Global instances (will be initialized by the main server)
_evaluation_engine = None
_feature_extractor = None
_threshold_learner = None
_training_pipeline = None

def initialize_adaptive_system():
    """Initialize the adaptive threshold learning system."""
    global _evaluation_engine, _feature_extractor, _threshold_learner, _training_pipeline
    
    if _evaluation_engine is None:
        _evaluation_engine = OptimizedEvaluationEngine()
        _feature_extractor = IntentNormalizedFeatureExtractor(
            alpha=0.2,
            evaluation_engine=_evaluation_engine
        )
        _threshold_learner = PerceptronThresholdLearner(
            evaluation_engine=_evaluation_engine,
            feature_extractor=_feature_extractor,
            learning_rate=0.01,
            max_epochs=50,
            convergence_threshold=0.85
        )
        _training_pipeline = TrainingDataPipeline(_threshold_learner)
        
        logger.info("Adaptive threshold learning system initialized")

def get_threshold_learner():
    """Dependency injection for threshold learner."""
    if _threshold_learner is None:
        initialize_adaptive_system()
    return _threshold_learner

def get_training_pipeline():
    """Dependency injection for training pipeline."""
    if _training_pipeline is None:
        initialize_adaptive_system()
    return _training_pipeline

# Pydantic models for API requests/responses

class AdaptivePredictionRequest(BaseModel):
    """Request model for adaptive violation prediction."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to analyze for ethical violations")
    model_preference: Optional[str] = Field("best", description="Preferred model (classic, averaged, voted, or best)")
    include_metadata: bool = Field(True, description="Include detailed metadata in response")

class AdaptivePredictionResponse(BaseModel):
    """Response model for adaptive violation prediction."""
    text: str = Field(..., description="Original input text")
    is_violation: bool = Field(..., description="Whether a violation was detected")
    confidence: float = Field(..., description="Prediction confidence (0.0 to 1.0)")
    model_used: str = Field(..., description="Model that made the prediction")
    model_accuracy: float = Field(..., description="Accuracy of the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional prediction metadata")

class TrainingDataRequest(BaseModel):
    """Request model for training data generation."""
    synthetic_examples: int = Field(50, ge=10, le=500, description="Number of synthetic examples to generate")
    violation_ratio: float = Field(0.3, ge=0.1, le=0.9, description="Ratio of violation examples")
    domains: List[str] = Field(["healthcare", "finance", "ai_systems"], description="Domains for data generation")
    complexity_levels: List[str] = Field(["simple", "moderate", "complex"], description="Complexity levels")

class TrainingDataResponse(BaseModel):
    """Response model for training data generation."""
    total_examples: int = Field(..., description="Total number of examples generated")
    violation_examples: int = Field(..., description="Number of violation examples")
    non_violation_examples: int = Field(..., description="Number of non-violation examples")
    domains_covered: List[str] = Field(..., description="Domains included in the dataset")
    quality_score: float = Field(..., description="Overall quality score (0.0 to 1.0)")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")

class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    force_retrain: bool = Field(False, description="Force retraining even if models exist")
    learning_rate: Optional[float] = Field(None, ge=0.001, le=0.1, description="Learning rate override")
    max_epochs: Optional[int] = Field(None, ge=10, le=200, description="Maximum epochs override")
    convergence_threshold: Optional[float] = Field(None, ge=0.5, le=0.99, description="Convergence threshold override")

class ModelTrainingResponse(BaseModel):
    """Response model for model training."""
    best_model: str = Field(..., description="Best performing model")
    training_accuracy: float = Field(..., description="Training accuracy")
    validation_accuracy: float = Field(..., description="Validation accuracy")
    convergence_epochs: int = Field(..., description="Epochs to convergence")
    training_time: float = Field(..., description="Training time in seconds")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")

class SystemStatusResponse(BaseModel):
    """Response model for adaptive system status."""
    system_initialized: bool = Field(..., description="Whether the system is initialized")
    models_trained: bool = Field(..., description="Whether models are trained")
    training_examples: int = Field(..., description="Number of training examples")
    best_model: Optional[str] = Field(None, description="Best performing model")
    model_accuracies: Dict[str, float] = Field(..., description="Model accuracy scores")
    last_training: Optional[datetime] = Field(None, description="Last training timestamp")
    audit_events: int = Field(..., description="Number of audit log events")

class ManualAnnotationRequest(BaseModel):
    """Request model for manual annotation."""
    texts: List[str] = Field(..., min_items=1, max_items=50, description="Texts to annotate")
    annotator_id: Optional[str] = Field(None, description="ID of the annotator")

class ManualAnnotationResponse(BaseModel):
    """Response model for manual annotation."""
    annotation_batch_id: str = Field(..., description="Unique ID for this annotation batch")
    texts_to_annotate: List[Dict[str, Any]] = Field(..., description="Texts prepared for annotation")
    instructions: str = Field(..., description="Annotation instructions")

class AnnotationSubmissionRequest(BaseModel):
    """Request model for submitting annotations."""
    batch_id: str = Field(..., description="Annotation batch ID")
    annotations: List[Dict[str, Any]] = Field(..., description="Completed annotations")

# API Endpoints

@router.get("/status", response_model=SystemStatusResponse)
async def get_adaptive_system_status(
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Get the current status of the adaptive threshold learning system."""
    try:
        # Check if models are trained
        models_trained = len(learner.models) > 0
        
        # Get model accuracies
        model_accuracies = {}
        best_model = None
        if models_trained:
            for name, model in learner.models.items():
                model_accuracies[name] = getattr(model, 'accuracy', 0.0)
            best_model = learner.best_model_name
        
        return SystemStatusResponse(
            system_initialized=True,
            models_trained=models_trained,
            training_examples=len(learner.training_examples),
            best_model=best_model,
            model_accuracies=model_accuracies,
            last_training=learner.last_training_time,
            audit_events=len(learner.audit_log)
        )
    except Exception as e:
        logger.error(f"Error getting adaptive system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=AdaptivePredictionResponse)
async def predict_violation(
    request: AdaptivePredictionRequest,
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Predict whether text contains ethical violations using adaptive thresholds."""
    try:
        start_time = datetime.now()
        
        # Make prediction
        is_violation, confidence, metadata = await learner.predict_violation(
            text=request.text,
            model_name=request.model_preference if request.model_preference != "best" else None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = AdaptivePredictionResponse(
            text=request.text,
            is_violation=is_violation,
            confidence=confidence,
            model_used=metadata.get("model_used", "unknown"),
            model_accuracy=metadata.get("model_accuracy", 0.0),
            processing_time=processing_time,
            metadata=metadata if request.include_metadata else None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in adaptive prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training-data/generate", response_model=TrainingDataResponse)
async def generate_training_data(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    pipeline: TrainingDataPipeline = Depends(get_training_pipeline)
):
    """Generate synthetic training data for model training."""
    try:
        # Create configuration
        config = DataGenerationConfig(
            synthetic_examples=request.synthetic_examples,
            violation_ratio=request.violation_ratio,
            domains=request.domains,
            complexity_levels=request.complexity_levels
        )
        
        # Generate dataset
        dataset = await pipeline.create_comprehensive_dataset(config)
        
        # Validate dataset quality
        quality_report = pipeline.validate_dataset_quality(dataset)
        
        # Add to learner's training examples
        learner = get_threshold_learner()
        learner.training_examples.extend(dataset)
        
        return TrainingDataResponse(
            total_examples=quality_report["total_examples"],
            violation_examples=quality_report["violation_examples"],
            non_violation_examples=quality_report["non_violation_examples"],
            domains_covered=request.domains,
            quality_score=min(quality_report["average_confidence"], 1.0),
            recommendations=quality_report.get("recommendations", [])
        )
        
    except Exception as e:
        logger.error(f"Error generating training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train", response_model=ModelTrainingResponse)
async def train_models(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Train or retrain the adaptive threshold models."""
    try:
        # Update learner parameters if provided
        if request.learning_rate is not None:
            learner.learning_rate = request.learning_rate
        if request.max_epochs is not None:
            learner.max_epochs = request.max_epochs
        if request.convergence_threshold is not None:
            learner.convergence_threshold = request.convergence_threshold
        
        # Check if we need to train
        if not request.force_retrain and len(learner.models) > 0:
            raise HTTPException(
                status_code=400, 
                detail="Models already trained. Use force_retrain=true to retrain."
            )
        
        # Ensure we have training data
        if len(learner.training_examples) == 0:
            raise HTTPException(
                status_code=400,
                detail="No training data available. Generate training data first."
            )
        
        # Train models
        results = await learner.train_all_models()
        
        return ModelTrainingResponse(
            best_model=results.best_model,
            training_accuracy=results.training_accuracy,
            validation_accuracy=results.validation_accuracy,
            convergence_epochs=results.convergence_epochs,
            training_time=results.training_time,
            feature_importance=results.feature_importance
        )
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/annotation/create", response_model=ManualAnnotationResponse)
async def create_annotation_batch(
    request: ManualAnnotationRequest,
    pipeline: TrainingDataPipeline = Depends(get_training_pipeline)
):
    """Create a batch of texts for manual annotation."""
    try:
        # Create annotation batch
        annotation_batch = pipeline.create_manual_annotation_batch(request.texts)
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store batch for later retrieval (in production, use database)
        # For now, we'll return the batch directly
        
        instructions = """
        Please annotate each text as either:
        - 'violation': Contains ethical violations (discrimination, harm, bias, etc.)
        - 'no_violation': Ethically acceptable content
        
        For each annotation, also provide:
        - confidence: Your confidence in the annotation (0.0 to 1.0)
        - notes: Optional explanation of your decision
        """
        
        return ManualAnnotationResponse(
            annotation_batch_id=batch_id,
            texts_to_annotate=annotation_batch,
            instructions=instructions
        )
        
    except Exception as e:
        logger.error(f"Error creating annotation batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/annotation/submit")
async def submit_annotations(
    request: AnnotationSubmissionRequest,
    pipeline: TrainingDataPipeline = Depends(get_training_pipeline),
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Submit completed manual annotations."""
    try:
        # Process annotations
        training_examples = await pipeline.process_manual_annotations(request.annotations)
        
        # Add to learner's training data
        learner.training_examples.extend(training_examples)
        
        return {
            "status": "success",
            "message": f"Added {len(training_examples)} manually annotated examples",
            "batch_id": request.batch_id,
            "total_training_examples": len(learner.training_examples)
        }
        
    except Exception as e:
        logger.error(f"Error submitting annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit/logs")
async def get_audit_logs(
    limit: int = 100,
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Get audit logs for transparency and monitoring."""
    try:
        # Get recent audit logs
        recent_logs = learner.audit_log[-limit:] if len(learner.audit_log) > limit else learner.audit_log
        
        return {
            "total_events": len(learner.audit_log),
            "recent_events": recent_logs,
            "event_types": list(set(log.get("event_type", "unknown") for log in learner.audit_log))
        }
        
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/performance")
async def get_model_performance(
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Get detailed performance metrics for all models."""
    try:
        if len(learner.models) == 0:
            return {"message": "No models trained yet"}
        
        performance_data = {}
        for name, model in learner.models.items():
            performance_data[name] = {
                "accuracy": getattr(model, 'accuracy', 0.0),
                "weights": getattr(model, 'weights', []).tolist() if hasattr(model, 'weights') else [],
                "training_iterations": getattr(model, 'iterations', 0),
                "is_best_model": name == learner.best_model_name
            }
        
        return {
            "models": performance_data,
            "best_model": learner.best_model_name,
            "training_examples": len(learner.training_examples),
            "last_training": learner.last_training_time
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/save")
async def save_models(
    filepath: Optional[str] = None,
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Save trained models to disk."""
    try:
        if len(learner.models) == 0:
            raise HTTPException(status_code=400, detail="No models to save")
        
        if filepath is None:
            filepath = f"/tmp/adaptive_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        learner.save_models(filepath)
        
        return {
            "status": "success",
            "message": f"Models saved to {filepath}",
            "models_saved": list(learner.models.keys())
        }
        
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load")
async def load_models(
    filepath: str,
    learner: PerceptronThresholdLearner = Depends(get_threshold_learner)
):
    """Load trained models from disk."""
    try:
        learner.load_models(filepath)
        
        return {
            "status": "success",
            "message": f"Models loaded from {filepath}",
            "models_loaded": list(learner.models.keys()),
            "best_model": learner.best_model_name
        }
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize the system when module is imported
initialize_adaptive_system()
