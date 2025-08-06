"""
Ethical AI Testbed API Server (Refactored)

This module implements the FastAPI server with a refactored architecture
using controllers and dependency injection. It follows clean architecture principles
with proper separation of concerns, domain-driven design, and SOLID principles.

Architecture Overview:
    - Controllers: Handle HTTP requests and responses
    - Use Cases: Implement business logic
    - Domain: Contains business entities and value objects
    - Infrastructure: Provides external service implementations

Endpoints:
    - /api/health: System health check
    - /api/evaluate: Evaluate text for ethical considerations
    - /api/parameters: Get/update ethical parameters
    - /api/learning-stats: Get learning system statistics
    - /api/heat-map-mock: Generate heat map data for visualization
    - /api/ethics/*: Various ethics analysis endpoints

Dependency Injection:
    The server uses FastAPI's dependency injection system to provide
    controllers with their required dependencies, promoting testability
    and maintainability.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from application.controllers.evaluation_controller import EvaluationController
from application.controllers.health_controller import HealthController
from application.controllers.parameters_controller import ParametersController
from application.controllers.learning_controller import LearningController
from application.controllers.visualization_controller import VisualizationController
from application.controllers.ethics_controller_class import EthicsController

from application.use_cases.evaluate_text_use_case import EvaluateTextUseCase
from application.use_cases.get_parameters_use_case import GetParametersUseCase
from application.use_cases.update_parameters_use_case import UpdateParametersUseCase
from application.use_cases.learning_stats_use_case import LearningStatsUseCase
from application.use_cases.heat_map_use_case import HeatMapUseCase
from application.use_cases.meta_ethics_analysis_use_case import MetaEthicsAnalysisUseCase
from application.use_cases.normative_ethics_analysis_use_case import NormativeEthicsAnalysisUseCase
from application.use_cases.applied_ethics_analysis_use_case import AppliedEthicsAnalysisUseCase
from application.use_cases.ml_training_guidance_use_case import MLTrainingGuidanceUseCase

from domain.unified_ethical_orchestrator import UnifiedEthicalOrchestrator
from infrastructure.database.mongodb_client import MongoDBClient
from domain.configuration import Configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles initialization and cleanup of resources.
    """
    global orchestrator
    
    logger.info("Starting server initialization...")
    
    # Initialize configuration
    config = Configuration()
    
    # Initialize database client
    try:
        logger.info("Initializing MongoDB client...")
        mongodb_client = MongoDBClient(
            connection_string=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"),
            database_name=os.getenv("MONGODB_DATABASE", "ethical_ai_testbed"),
            max_retries=3,
            retry_delay=1.0
        )
        await mongodb_client.connect()
        logger.info("MongoDB client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize MongoDB client: {str(e)}")
        logger.warning("Continuing in degraded mode with in-memory storage")
        mongodb_client = None
    
    # Initialize orchestrator
    try:
        logger.info("Initializing Unified Ethical Orchestrator...")
        orchestrator = UnifiedEthicalOrchestrator(
            configuration=config,
            database_client=mongodb_client
        )
        await orchestrator.initialize()
        logger.info("Unified Ethical Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {str(e)}", exc_info=True)
        raise
    
    # Yield control back to FastAPI
    logger.info("Server initialization complete")
    yield
    
    # Cleanup resources
    logger.info("Shutting down server...")
    
    # Close database connection
    if mongodb_client:
        logger.info("Closing MongoDB connection...")
        await mongodb_client.close()
        logger.info("MongoDB connection closed")
    
    # Cleanup orchestrator resources
    if orchestrator:
        logger.info("Cleaning up orchestrator resources...")
        await orchestrator.cleanup()
        logger.info("Orchestrator resources cleaned up")
    
    logger.info("Server shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Ethical AI Testbed API",
    description="API for the Ethical AI Testbed",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses."""
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} - {request.method} {request.url.path}")
    
    # Create request-specific logger
    request_logger = logging.LoggerAdapter(
        logger, {"request_id": request_id}
    )
    
    try:
        response = await call_next(request)
        request_logger.info(f"Response {response.status_code}")
        return response
    except Exception as e:
        request_logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)}
        )

# Dependency for getting the orchestrator
async def get_orchestrator():
    """Dependency for getting the orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Orchestrator not initialized")
    return orchestrator

# Initialize controllers
def get_health_controller():
    """Get health controller instance."""
    return HealthController()

def get_evaluation_controller(orchestrator=Depends(get_orchestrator)):
    """Get evaluation controller instance."""
    use_case = EvaluateTextUseCase(orchestrator)
    return EvaluationController(use_case)

def get_parameters_controller(orchestrator=Depends(get_orchestrator)):
    """Get parameters controller instance."""
    get_parameters_use_case = GetParametersUseCase(orchestrator)
    update_parameters_use_case = UpdateParametersUseCase(orchestrator)
    return ParametersController(get_parameters_use_case, update_parameters_use_case)

def get_learning_controller(orchestrator=Depends(get_orchestrator)):
    """Get learning controller instance."""
    use_case = LearningStatsUseCase(orchestrator)
    return LearningController(use_case)

def get_visualization_controller(orchestrator=Depends(get_orchestrator)):
    """Get visualization controller instance."""
    use_case = HeatMapUseCase(orchestrator)
    return VisualizationController(use_case)

def get_ethics_controller(orchestrator=Depends(get_orchestrator)):
    """Get ethics controller instance."""
    meta_ethics_use_case = MetaEthicsAnalysisUseCase(orchestrator)
    normative_ethics_use_case = NormativeEthicsAnalysisUseCase(orchestrator)
    applied_ethics_use_case = AppliedEthicsAnalysisUseCase(orchestrator)
    ml_training_guidance_use_case = MLTrainingGuidanceUseCase(orchestrator)
    return EthicsController(
        meta_ethics_use_case,
        normative_ethics_use_case,
        applied_ethics_use_case,
        ml_training_guidance_use_case
    )

# Health endpoints
@app.get("/api/health", tags=["Health"])
async def health_check(controller: HealthController = Depends(get_health_controller)):
    """Health check endpoint."""
    return await controller.health_check()

# Evaluation endpoints
@app.post("/api/evaluate", tags=["Evaluation"])
async def evaluate_text(
    request: Request,
    controller: EvaluationController = Depends(get_evaluation_controller)
):
    """Evaluate text for ethical considerations."""
    data = await request.json()
    return await controller.evaluate_text(data)

# Parameters endpoints
@app.get("/api/parameters", tags=["Parameters"])
async def get_parameters(controller: ParametersController = Depends(get_parameters_controller)):
    """Get current ethical parameters."""
    return await controller.get_parameters()

@app.post("/api/parameters/update", tags=["Parameters"])
async def update_parameters(
    request: Request,
    controller: ParametersController = Depends(get_parameters_controller)
):
    """Update ethical parameters."""
    data = await request.json()
    return await controller.update_parameters(data)

@app.post("/api/thresholds/update-all", tags=["Parameters"])
async def update_thresholds(
    request: Request,
    controller: ParametersController = Depends(get_parameters_controller)
):
    """Update all ethical thresholds."""
    data = await request.json()
    return await controller.update_thresholds(data)

# Learning endpoints
@app.get("/api/learning-stats", tags=["Learning"])
async def get_learning_stats(controller: LearningController = Depends(get_learning_controller)):
    """Get learning statistics."""
    return await controller.get_learning_stats()

# Visualization endpoints
@app.get("/api/heat-map", tags=["Visualization"])
async def get_heat_map(controller: VisualizationController = Depends(get_visualization_controller)):
    """Get heat map data."""
    return await controller.get_heat_map()

@app.get("/api/heat-map-mock", tags=["Visualization"])
async def get_heat_map_mock(controller: VisualizationController = Depends(get_visualization_controller)):
    """Get mock heat map data."""
    return await controller.get_heat_map_mock()

# Ethics endpoints
@app.post("/api/ethics/meta-analysis", tags=["Ethics"])
async def meta_ethics_analysis(
    request: Request,
    controller: EthicsController = Depends(get_ethics_controller)
):
    """Perform meta-ethics analysis."""
    data = await request.json()
    return await controller.meta_ethics_analysis(data)

@app.post("/api/ethics/normative-analysis", tags=["Ethics"])
async def normative_ethics_analysis(
    request: Request,
    controller: EthicsController = Depends(get_ethics_controller)
):
    """Perform normative ethics analysis."""
    data = await request.json()
    return await controller.normative_ethics_analysis(data)

@app.post("/api/ethics/applied-analysis", tags=["Ethics"])
async def applied_ethics_analysis(
    request: Request,
    controller: EthicsController = Depends(get_ethics_controller)
):
    """Perform applied ethics analysis."""
    data = await request.json()
    return await controller.applied_ethics_analysis(data)

@app.post("/api/ethics/ml-training-guidance", tags=["Ethics"])
async def ml_training_guidance(
    request: Request,
    controller: EthicsController = Depends(get_ethics_controller)
):
    """Get ML training guidance."""
    data = await request.json()
    return await controller.ml_training_guidance(data)

@app.post("/api/ethics/comprehensive-analysis", tags=["Ethics"])
async def comprehensive_ethics_analysis(
    request: Request,
    controller: EthicsController = Depends(get_ethics_controller)
):
    """Perform comprehensive ethics analysis."""
    data = await request.json()
    return await controller.comprehensive_analysis(data)

# Run server if executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server_refactored:app", host="0.0.0.0", port=port, reload=True)
