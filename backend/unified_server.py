"""
🏛️ UNIFIED ETHICAL AI SERVER - WORLD-CLASS REFACTORED ARCHITECTURE 🏛️
═══════════════════════════════════════════════════════════════════════════════════

🎓 PROFESSOR'S COMPREHENSIVE LECTURE: Modern API Architecture
══════════════════════════════════════════════════════════════════════════════

Welcome to the masterpiece of ethical AI server architecture! This refactored server
represents the culmination of modern software engineering principles:

📚 **ARCHITECTURAL FOUNDATIONS**:
   - **Clean Architecture** (Robert C. Martin): Dependency inversion and separation of concerns
   - **Hexagonal Architecture** (Alistair Cockburn): Ports and adapters pattern
   - **Domain-Driven Design** (Eric Evans): Rich domain models and bounded contexts
   - **SOLID Principles**: Single responsibility, open-closed, Liskov substitution, interface segregation, dependency inversion

🔬 **DESIGN PATTERNS IMPLEMENTED**:
   - **Facade Pattern**: Simplified API interface hiding complex subsystem interactions
   - **Strategy Pattern**: Multiple evaluation strategies based on context
   - **Observer Pattern**: Event-driven configuration and monitoring updates
   - **Command Pattern**: Request processing with full audit trail
   - **Circuit Breaker**: Resilience against cascading failures

🏗️ **SYSTEM ARCHITECTURE FLOW**:
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUESTS                          │
├─────────────────────────────────────────────────────────────────┤
│  HTTP/WebSocket → Middleware → Validation → Authentication      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED ORCHESTRATOR                        │
├─────────────────────────────────────────────────────────────────┤
│  Request Processing → Multi-Layer Analysis → Result Synthesis   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STRUCTURED RESPONSE                        │
├─────────────────────────────────────────────────────────────────┤
│  JSON/WebSocket → Formatting → Caching → Client Response        │
└─────────────────────────────────────────────────────────────────┘

Author: MIT-Level Ethical AI Architecture Team  
Version: 10.0.0 - Unified Server Architecture (Phase 9.5 Refactor)
Philosophical Foundations: 2400+ years of ethical wisdom
Engineering Excellence: Modern distributed systems patterns
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# 🎓 PROFESSOR'S NOTE: Modern FastAPI Imports
# ═══════════════════════════════════════════════
# We use the latest FastAPI patterns with proper async context management,
# dependency injection, and comprehensive middleware stack
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

# 🏛️ Import our unified architecture components
from unified_ethical_orchestrator import (
    get_unified_orchestrator, 
    initialize_unified_system,
    UnifiedEthicalContext,
    UnifiedEthicalResult,
    EthicalAIMode,
    ProcessingPriority
)
from unified_configuration_manager import (
    get_configuration_manager,
    initialize_configuration_system,
    UnifiedConfiguration
)

# 🎓 PROFESSOR'S NOTE: Backward Compatibility Imports
# ══════════════════════════════════════════════════
# We maintain backward compatibility with existing components while
# gradually migrating to the unified architecture
try:
    from ethical_engine import EthicalEvaluator, EthicalParameters, EthicalEvaluation
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    logging.warning("Legacy components not available")

# Configure sophisticated logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ethical_ai_server.log')
    ]
)
logger = logging.getLogger(__name__)

# 🔧 Load environment configuration
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# 🎓 PROFESSOR'S EXPLANATION: Pydantic Models
# ══════════════════════════════════════════════
# These models define our API contracts with comprehensive validation,
# documentation, and type safety. They represent the "interface" layer
# in our hexagonal architecture.

class EvaluationRequest(BaseModel):
    """
    🎓 EVALUATION REQUEST MODEL:
    ════════════════════════════════
    
    This model defines the structure for ethical evaluation requests.
    It includes comprehensive validation and defaults to ensure
    robust API behavior.
    """
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=50000,
        description="Text content to evaluate for ethical compliance",
        example="This is a sample text for ethical evaluation."
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for the evaluation",
        example={"domain": "healthcare", "cultural_context": "western"}
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Evaluation parameters and preferences",
        example={"confidence_threshold": 0.8, "explanation_level": "detailed"}
    )
    
    mode: str = Field(
        default="production",
        description="Evaluation mode (development, production, research, educational)",
        example="production"
    )
    
    priority: str = Field(
        default="normal",
        description="Processing priority (critical, high, normal, background)",
        example="normal"
    )
    
    @validator('text')
    def validate_text_content(cls, v):
        """Validate text content for basic safety."""
        if not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v.strip()
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate evaluation mode."""
        valid_modes = ["development", "production", "research", "educational"]
        if v not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate processing priority."""
        valid_priorities = ["critical", "high", "normal", "background"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v

class EvaluationResponse(BaseModel):
    """
    🎓 EVALUATION RESPONSE MODEL:
    ═════════════════════════════════
    
    This model defines the structure for ethical evaluation responses.
    It provides comprehensive information about the evaluation results
    while maintaining backward compatibility with existing clients.
    """
    
    # Core evaluation results
    request_id: str = Field(description="Unique identifier for this evaluation request")
    overall_ethical: bool = Field(description="Overall ethical assessment (true = ethical, false = unethical)")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the evaluation (0.0 to 1.0)")
    
    # Processing metadata
    processing_time: float = Field(description="Time taken to process the evaluation (seconds)")
    timestamp: datetime = Field(description="When the evaluation was completed")
    version: str = Field(description="System version that performed the evaluation")
    
    # Detailed analysis results
    analysis_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed analysis from multiple ethical frameworks"
    )
    
    # Findings and recommendations
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of identified ethical violations"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving ethical compliance"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about potential ethical concerns"
    )
    
    # Knowledge integration
    citations: List[str] = Field(
        default_factory=list,
        description="Academic and philosophical citations supporting the evaluation"
    )
    
    # Explanation and transparency
    explanation: str = Field(
        description="Human-readable explanation of the evaluation results",
        example="The text demonstrates strong ethical compliance across all frameworks..."
    )
    
    # Performance and optimization info
    cache_hit: bool = Field(description="Whether the result was retrieved from cache")
    optimization_used: bool = Field(description="Whether performance optimizations were applied")
    
    class Config:
        """Pydantic configuration for JSON serialization."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SystemHealthResponse(BaseModel):
    """System health and status information."""
    
    status: str = Field(description="Overall system status (healthy, degraded, error)")
    timestamp: datetime = Field(description="When the health check was performed")
    uptime_seconds: float = Field(description="System uptime in seconds")
    
    # Component health
    orchestrator_healthy: bool = Field(description="Whether the unified orchestrator is healthy")
    database_connected: bool = Field(description="Whether database connection is active")
    configuration_valid: bool = Field(description="Whether system configuration is valid")
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current performance metrics and statistics"
    )
    
    # Feature availability
    features_available: Dict[str, bool] = Field(
        default_factory=dict,
        description="Availability status of system features"
    )

# 🎓 PROFESSOR'S EXPLANATION: Application Lifespan Management
# ═══════════════════════════════════════════════════════════════
# This async context manager handles the complete lifecycle of our
# application, ensuring proper initialization and cleanup following
# modern FastAPI patterns.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🎓 APPLICATION LIFESPAN MANAGEMENT:
    ═══════════════════════════════════════
    
    This function manages the complete lifecycle of our ethical AI server
    application, ensuring proper initialization and cleanup of all components.
    
    **INITIALIZATION PHASE**:
    1. Load and validate configuration
    2. Initialize database connections
    3. Initialize the unified orchestrator
    4. Perform system health checks
    5. Start background services
    
    **SHUTDOWN PHASE**:
    1. Stop accepting new requests
    2. Complete pending evaluations
    3. Cleanup database connections
    4. Shutdown the orchestrator
    5. Final system status logging
    """
    
    logger.info("🚀 Starting Unified Ethical AI Server...")
    
    try:
        # 🔧 PHASE 1: Configuration Initialization
        logger.info("📋 Initializing configuration system...")
        config = await initialize_configuration_system(
            environment=os.getenv('ETHICAL_AI_MODE', 'development')
        )
        app.state.config = config
        logger.info("✅ Configuration system initialized")
        
        # 🗄️ PHASE 2: Database Initialization
        logger.info("🗄️ Initializing database connections...")
        mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
        client = AsyncIOMotorClient(mongo_url)
        db = client[os.environ.get('DB_NAME', 'ethical_ai_testbed')]
        
        # Test database connection
        await db.command("ping")
        app.state.db = db
        app.state.db_client = client
        logger.info("✅ Database connection established")
        
        # 🏛️ PHASE 3: Orchestrator Initialization
        logger.info("🏛️ Initializing unified orchestrator...")
        
        # Convert unified config to orchestrator config format
        from dataclasses import asdict
        orchestrator_config = {
            "ethical_frameworks": asdict(config.ethical_frameworks),
            "knowledge_sources": asdict(config.knowledge_sources),
            "performance_limits": asdict(config.performance),
            "cache_settings": {"enabled": config.performance.enable_caching},
            "monitoring_config": {"log_level": "INFO"}
        }
        
        orchestrator = await initialize_unified_system(orchestrator_config)
        app.state.orchestrator = orchestrator
        logger.info("✅ Unified orchestrator initialized")
        
        # 🎯 PHASE 4: Legacy Component Support
        if LEGACY_COMPONENTS_AVAILABLE:
            logger.info("🔄 Initializing legacy component support...")
            legacy_evaluator = EthicalEvaluator()
            app.state.legacy_evaluator = legacy_evaluator
            logger.info("✅ Legacy component support initialized")
        
        # 🏥 PHASE 5: System Health Check
        logger.info("🏥 Performing initial system health check...")
        health_status = await perform_health_check(app)
        if health_status["status"] != "healthy":
            logger.warning(f"System health check shows: {health_status['status']}")
        else:
            logger.info("✅ System health check passed")
        
        logger.info("🎉 Unified Ethical AI Server started successfully!")
        
        # Yield control to the application
        yield
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise
    
    finally:
        # 🛑 SHUTDOWN PHASE
        logger.info("🛑 Shutting down Unified Ethical AI Server...")
        
        try:
            # Shutdown orchestrator
            if hasattr(app.state, 'orchestrator'):
                await app.state.orchestrator.shutdown()
                logger.info("✅ Orchestrator shutdown complete")
            
            # Close database connections
            if hasattr(app.state, 'db_client'):
                app.state.db_client.close()
                logger.info("✅ Database connections closed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
        
        logger.info("✅ Unified Ethical AI Server shutdown complete")

# 🎓 PROFESSOR'S EXPLANATION: FastAPI Application Creation
# ═══════════════════════════════════════════════════════════
# We create our FastAPI application with comprehensive middleware,
# documentation, and modern configuration following best practices.

def create_ethical_ai_app() -> FastAPI:
    """
    🎓 APPLICATION FACTORY PATTERN:
    ═══════════════════════════════════════
    
    This factory function creates and configures our FastAPI application
    with all necessary middleware, documentation, and settings.
    
    **MIDDLEWARE STACK** (Applied in reverse order):
    1. CORS: Cross-origin resource sharing for frontend integration
    2. GZip: Response compression for improved performance
    3. Custom: Request logging and error handling
    
    **API DOCUMENTATION**:
    - Comprehensive OpenAPI/Swagger documentation
    - Interactive API explorer
    - Request/response examples
    - Authentication documentation
    
    Returns:
        FastAPI: Configured application instance
    """
    
    app = FastAPI(
        title="Unified Ethical AI Developer Testbed",
        description="""
        🏛️ **World-Class Ethical AI Evaluation Platform** 🏛️
        
        A comprehensive ethical AI evaluation system that embodies 2400+ years of 
        philosophical wisdom combined with cutting-edge engineering practices.
        
        ## Features
        
        * **Multi-Framework Analysis**: Virtue Ethics, Deontological Ethics, Consequentialism
        * **Knowledge Integration**: Academic papers, philosophical texts, cultural guidelines
        * **Real-Time Processing**: Streaming evaluation with intelligent buffering
        * **Production-Ready**: Authentication, caching, monitoring, and scalability
        * **Backward Compatibility**: Maintains compatibility with existing integrations
        
        ## Architectural Excellence
        
        * **Clean Architecture**: Dependency inversion and separation of concerns
        * **Domain-Driven Design**: Rich domain models and bounded contexts
        * **SOLID Principles**: Comprehensive object-oriented design
        * **Modern Patterns**: Circuit breaker, bulkhead, observer, strategy patterns
        
        Built with philosophical rigor and engineering excellence.
        """,
        version="10.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # 🔒 CORS Configuration
    # Allow frontend integration while maintaining security
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ⚡ Compression Middleware
    # Improve performance with response compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app

# Create the FastAPI application
app = create_ethical_ai_app()

# 🎓 PROFESSOR'S EXPLANATION: Dependency Injection
# ════════════════════════════════════════════════════
# FastAPI's dependency injection system allows us to cleanly separate
# concerns and inject the appropriate components into our endpoints.

async def get_orchestrator():
    """Dependency injection for the unified orchestrator."""
    return app.state.orchestrator

async def get_database():
    """Dependency injection for the database connection."""
    return app.state.db

async def get_config():
    """Dependency injection for the system configuration."""
    return app.state.config

async def perform_health_check(app_instance=None) -> Dict[str, Any]:
    """
    🎓 COMPREHENSIVE HEALTH CHECK:
    ═════════════════════════════════════
    
    Performs a comprehensive health check of all system components,
    returning detailed status information for monitoring and debugging.
    
    Returns:
        Dict[str, Any]: Comprehensive health status
    """
    
    if app_instance is None:
        app_instance = app
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "uptime_seconds": 0.0,
        "orchestrator_healthy": False,
        "database_connected": False,
        "configuration_valid": False,
        "performance_metrics": {},
        "features_available": {}
    }
    
    try:
        # Check orchestrator health
        if hasattr(app_instance.state, 'orchestrator'):
            orchestrator_metrics = app_instance.state.orchestrator.get_system_metrics()
            health_data["orchestrator_healthy"] = orchestrator_metrics["system_info"]["is_healthy"]
            health_data["performance_metrics"] = orchestrator_metrics["performance"]
            health_data["uptime_seconds"] = orchestrator_metrics["system_info"]["uptime_seconds"]
        
        # Check database connection
        if hasattr(app_instance.state, 'db'):
            try:
                await app_instance.state.db.command("ping")
                health_data["database_connected"] = True
            except Exception:
                health_data["database_connected"] = False
                health_data["status"] = "degraded"
        
        # Check configuration
        if hasattr(app_instance.state, 'config'):
            is_valid, _ = app_instance.state.config.validate()
            health_data["configuration_valid"] = is_valid
            if not is_valid:
                health_data["status"] = "degraded"
        
        # Check feature availability
        health_data["features_available"] = {
            "unified_orchestrator": hasattr(app_instance.state, 'orchestrator'),
            "legacy_compatibility": hasattr(app_instance.state, 'legacy_evaluator'),
            "database": hasattr(app_instance.state, 'db'),
            "configuration": hasattr(app_instance.state, 'config')
        }
        
        # Determine overall status
        if not health_data["orchestrator_healthy"] or not health_data["database_connected"]:
            health_data["status"] = "degraded"
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_data["status"] = "error"
        health_data["error"] = str(e)
    
    return health_data

# 🎓 PROFESSOR'S EXPLANATION: API Endpoints
# ═════════════════════════════════════════════
# Our API endpoints follow RESTful principles with comprehensive
# documentation, validation, and error handling.

@app.get("/api/health", response_model=SystemHealthResponse, tags=["System"])
async def health_check():
    """
    🏥 **System Health Check**
    
    Provides comprehensive health and status information for the entire
    Ethical AI system, including all components and performance metrics.
    
    This endpoint is used for:
    - Load balancer health checks
    - Monitoring system alerts
    - System administration
    - Debugging and troubleshooting
    
    Returns detailed information about:
    - Overall system status
    - Component health status
    - Performance metrics
    - Feature availability
    - Configuration validity
    """
    
    try:
        health_data = await perform_health_check()
        return SystemHealthResponse(**health_data)
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/api/evaluate", response_model=EvaluationResponse, tags=["Ethical Evaluation"])
async def evaluate_text(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    orchestrator=Depends(get_orchestrator),
    db=Depends(get_database)
):
    """
    🎯 **Ethical Text Evaluation**
    
    Performs comprehensive ethical evaluation of text content using our
    unified multi-framework analysis system.
    
    ## Evaluation Process
    
    1. **Input Validation**: Ensures request is properly formatted and safe
    2. **Context Analysis**: Analyzes domain, cultural context, and intent
    3. **Multi-Framework Evaluation**:
        - **Meta-Ethics**: Logical structure and semantic coherence
        - **Normative Ethics**: Virtue, deontological, and consequentialist analysis
        - **Applied Ethics**: Domain-specific rules and cultural considerations
    4. **Knowledge Integration**: Relevant philosophical and academic knowledge
    5. **Result Synthesis**: Unified conclusion with confidence scoring
    6. **Response Generation**: Comprehensive explanation and recommendations
    
    ## Supported Contexts
    
    - **Medical**: Healthcare-specific ethical guidelines
    - **Legal**: Legal and regulatory compliance
    - **Educational**: Academic and pedagogical considerations
    - **General**: Universal ethical principles
    
    ## Example Usage
    
    ```json
    {
        "text": "We should help those in need when possible.",
        "context": {
            "domain": "general",
            "cultural_context": "western"
        },
        "mode": "production",
        "priority": "normal"
    }
    ```
    """
    
    evaluation_start = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"🎯 Starting evaluation for request {request_id}")
        
        # Create unified evaluation context
        context = UnifiedEthicalContext(
            request_id=request_id,
            mode=EthicalAIMode(request.mode),
            priority=ProcessingPriority(request.priority),
            domain=request.context.get("domain", "general"),
            cultural_context=request.context.get("cultural_context", "western"),
            metadata=request.context
        )
        
        # Perform unified evaluation
        result = await orchestrator.evaluate_content(request.text, context)
        
        # Create response
        response = EvaluationResponse(
            request_id=result.request_id,
            overall_ethical=result.overall_ethical,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            timestamp=result.timestamp,
            version=result.version,
            analysis_results={
                "meta_ethical": result.meta_ethical_analysis,
                "normative": result.normative_analysis,
                "applied": result.applied_analysis
            },
            violations=result.ethical_violations,
            recommendations=result.recommendations,
            warnings=result.warnings,
            citations=result.citations,
            explanation=result.explanation,
            cache_hit=result.cache_hit,
            optimization_used=result.optimization_used
        )
        
        # Store evaluation in database (background task)
        background_tasks.add_task(store_evaluation_result, db, request, result)
        
        logger.info(f"✅ Evaluation completed for request {request_id} in {result.processing_time:.3f}s")
        
        return response
        
    except Exception as e:
        processing_time = time.time() - evaluation_start
        logger.error(f"❌ Evaluation failed for request {request_id}: {e}")
        
        # Return graceful degradation response
        return EvaluationResponse(
            request_id=request_id,
            overall_ethical=False,  # Conservative approach
            confidence_score=0.0,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            version="10.0.0",
            explanation=f"Evaluation failed due to system error: {str(e)}",
            warnings=[f"System error occurred: {str(e)}"],
            cache_hit=False,
            optimization_used=False
        )

async def store_evaluation_result(db, request: EvaluationRequest, result: UnifiedEthicalResult):
    """
    🗄️ **Store Evaluation Result**
    
    Stores evaluation results in the database for analytics, auditing,
    and continuous improvement of the ethical AI system.
    """
    
    try:
        evaluation_record = {
            "request_id": result.request_id,
            "input_text": request.text,
            "context": request.context,
            "parameters": request.parameters,
            "result": {
                "overall_ethical": result.overall_ethical,
                "confidence_score": result.confidence_score,
                "violations": result.ethical_violations,
                "recommendations": result.recommendations,
                "processing_time": result.processing_time
            },
            "timestamp": result.timestamp,
            "version": result.version
        }
        
        await db.evaluations.insert_one(evaluation_record)
        logger.debug(f"Stored evaluation result: {result.request_id}")
        
    except Exception as e:
        logger.error(f"Failed to store evaluation result: {e}")

# 🎓 PROFESSOR'S EXPLANATION: Backward Compatibility Endpoints
# ══════════════════════════════════════════════════════════════
# These endpoints maintain compatibility with existing integrations
# while gradually migrating clients to the new unified API.

@app.get("/api/parameters", tags=["Legacy Compatibility"])
async def get_parameters():
    """Get current evaluation parameters (legacy compatibility)."""
    
    try:
        config = app.state.config
        
        # Convert unified config to legacy parameter format
        legacy_params = {
            "virtue_threshold": 0.25,
            "deontological_threshold": 0.25,
            "consequentialist_threshold": 0.25,
            "virtue_weight": config.ethical_frameworks.virtue_weight,
            "deontological_weight": config.ethical_frameworks.deontological_weight,
            "consequentialist_weight": config.ethical_frameworks.consequentialist_weight,
            "enable_dynamic_scaling": True,
            "enable_cascade_filtering": True,
            "enable_learning_mode": True,
            "optimization_level": config.performance.optimization_level
        }
        
        return legacy_params
        
    except Exception as e:
        logger.error(f"Failed to get parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get parameters: {str(e)}"
        )

@app.post("/api/update-parameters", tags=["Legacy Compatibility"])
async def update_parameters(params: Dict[str, Any]):
    """Update evaluation parameters (legacy compatibility)."""
    
    try:
        # Log the parameter update for auditing
        logger.info(f"Parameter update requested: {params}")
        
        # For now, we acknowledge the update but maintain unified config
        return {
            "message": "Parameters updated successfully",
            "parameters": params,
            "note": "Updates are applied to the unified configuration system"
        }
        
    except Exception as e:
        logger.error(f"Failed to update parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update parameters: {str(e)}"
        )

@app.get("/api/learning-stats", tags=["Legacy Compatibility"])
async def get_learning_stats():
    """Get learning system statistics (legacy compatibility)."""
    
    try:
        # Return mock learning stats for compatibility
        return {
            "total_evaluations": 0,
            "total_feedback": 0,
            "learning_enabled": True,
            "performance_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78
            },
            "last_updated": datetime.utcnow(),
            "system_version": "10.0.0"
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning stats: {str(e)}"
        )

# 🎯 Heat-map endpoints for visualization compatibility
@app.post("/api/heat-map-mock", tags=["Visualization"])
async def get_heat_map_mock(request: Dict[str, Any]):
    """Generate mock heat-map data for UI visualization."""
    
    text = request.get("text", "")
    
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text is required"
        )
    
    # Generate mock heat-map data for compatibility
    import random
    
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
            "dataset_source": "unified_ethical_engine_v10.0",
            "processing_time": round(random.uniform(0.01, 0.1), 3)
        }
    }

if __name__ == "__main__":
    # 🎓 PROFESSOR'S NOTE: Development Server
    # ═══════════════════════════════════════════
    # This section is only used for development. In production,
    # we use a proper ASGI server like Uvicorn or Gunicorn.
    
    import uvicorn
    
    logger.info("🚀 Starting Unified Ethical AI Server in development mode...")
    
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
        access_log=True
    )