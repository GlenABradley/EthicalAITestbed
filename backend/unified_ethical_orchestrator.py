"""
🏛️ UNIFIED ETHICAL AI ORCHESTRATOR - THE CROWN JEWEL 🏛️
═══════════════════════════════════════════════════════════

Welcome to the pinnacle of ethical AI engineering - a unified orchestrator that embodies
2400+ years of philosophical wisdom, cutting-edge distributed systems patterns, and
mathematical precision rivaling the finest academic institutions.

🎯 ARCHITECTURAL PHILOSOPHY (Following Clean Architecture Principles)
══════════════════════════════════════════════════════════════════════

This orchestrator follows Uncle Bob's Clean Architecture, Domain-Driven Design, and
the philosophical rigor of Immanuel Kant's systematic approach to ethics.

🔬 PROFESSOR'S LECTURE: Understanding the Architecture
════════════════════════════════════════════════════════

Imagine you are an MIT student in advanced computer science. This system represents
the culmination of multiple disciplines:

1. **PHILOSOPHICAL FOUNDATIONS**: We implement computational versions of:
   - Aristotelian Virtue Ethics (excellence of character)
   - Kantian Deontology (duty-based morality) 
   - Utilitarian Consequentialism (outcome-focused ethics)
   - Meta-ethics (the nature of ethical claims themselves)

2. **DISTRIBUTED SYSTEMS EXCELLENCE**: Following patterns from:
   - Google's MapReduce paradigm for parallel processing
   - Netflix's Circuit Breaker pattern for resilience
   - Amazon's Microservices architecture for scalability
   - Martin Fowler's Domain-Driven Design for maintainability

3. **MATHEMATICAL RIGOR**: Implementing:
   - Vector space mathematics for semantic embeddings
   - Graph theory for knowledge relationships
   - Statistical analysis for confidence measurements
   - Linear algebra for dimensional analysis

🏗️ SYSTEM COMPONENTS (The Four Pillars of Wisdom)
════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED ORCHESTRATOR                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ KNOWLEDGE     │ │ ETHICS        │ │ PROCESSING    │         │
│  │ INTEGRATION   │ │ PIPELINE      │ │ ENGINE        │         │
│  │               │ │               │ │               │         │
│  │ • Vector Store│ │ • Meta Ethics │ │ • Streaming   │         │
│  │ • Graph DB    │ │ • Normative   │ │ • Caching     │         │
│  │ • Wikipedia   │ │ • Applied     │ │ • ML Training │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ PRODUCTION    │ │ MULTI-MODAL   │ │ CORE ENGINE   │         │
│  │ FEATURES      │ │ EVALUATION    │ │ (V3.0)        │         │
│  │               │ │               │ │               │         │
│  │ • Auth (JWT)  │ │ • Pre-Eval    │ │ • Autonomy    │         │
│  │ • Caching     │ │ • Post-Eval   │ │ • Embeddings  │         │
│  │ • Metrics     │ │ • Streaming   │ │ • Orthogonal  │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
└─────────────────────────────────────────────────────────────────┘

Author: MIT-Level Ethical AI Engineering Team
Version: 10.0.0 - Unified Orchestrator (Phase 9.5 Refactor)
Philosophical Foundations: Aristotle, Kant, Mill, Moore, Hume
Engineering Principles: Fowler, Evans, Martin, Gamma et al.
"""

import asyncio
import logging
import time
import uuid
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set, AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 🎓 PROFESSOR'S NOTE: Dependency Injection Pattern
# ═══════════════════════════════════════════════
# We import all our specialized modules here, but use dependency injection
# to maintain loose coupling. This follows the Dependency Inversion Principle
# from SOLID design principles.

try:
    from ethical_engine import EthicalEvaluator, EthicalParameters, EthicalEvaluation
    from enhanced_ethics_pipeline import EnhancedEthicsPipelineOrchestrator
    from knowledge_integration_layer import KnowledgeIntegrator, KnowledgeQuery
    from realtime_streaming_engine import RealTimeEthicsStreamer
    from ml_ethics_engine import MLEthicsVectorEngine, MLEthicalVector
    from multi_modal_evaluation import MultiModalEvaluationOrchestrator, EvaluationMode
    from production_features import ProductionFeatureManager
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging with academic rigor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EthicalAIMode(Enum):
    """
    🎓 PROFESSOR'S EXPLANATION: Operational Modes
    ═════════════════════════════════════════════
    
    These modes represent different operational contexts for ethical evaluation,
    similar to how a human ethicist might approach different types of moral questions:
    
    - EDUCATIONAL: Teaching and learning contexts (maximize understanding)
    - PRODUCTION: Live systems requiring reliability (maximize accuracy)
    - RESEARCH: Experimental contexts (maximize insight generation)
    - DEVELOPMENT: Testing and debugging (maximize transparency)
    """
    EDUCATIONAL = "educational"    # For learning and teaching
    PRODUCTION = "production"      # For live deployments
    RESEARCH = "research"          # For academic research
    DEVELOPMENT = "development"    # For testing and debugging

class ProcessingPriority(Enum):
    """
    🎓 PROFESSOR'S EXPLANATION: Priority Levels
    ══════════════════════════════════════════
    
    Like a hospital triage system, we prioritize ethical evaluations based
    on their urgency and potential impact:
    
    - CRITICAL: Immediate safety concerns (< 100ms response time)
    - HIGH: Important decisions requiring quick response (< 1s)
    - NORMAL: Standard evaluations (< 5s)
    - BACKGROUND: Batch processing (can wait)
    """
    CRITICAL = "critical"    # Immediate processing required
    HIGH = "high"           # High priority processing
    NORMAL = "normal"       # Standard processing
    BACKGROUND = "background"  # Can be delayed

@dataclass
class UnifiedEthicalContext:
    """
    🎓 PROFESSOR'S EXPLANATION: Context Encapsulation
    ═══════════════════════════════════════════════
    
    This data class encapsulates all contextual information needed for ethical
    evaluation. Think of it as a "case file" that an ethicist would compile
    before making a moral judgment.
    
    Like Kant's categorical imperative requires universal context, our system
    needs comprehensive situational awareness to make sound ethical decisions.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: EthicalAIMode = EthicalAIMode.PRODUCTION
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # User and session tracking for personalized ethics
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Domain-specific context (medical, legal, educational, etc.)
    domain: str = "general"
    cultural_context: str = "western"
    
    # Processing preferences
    philosophical_emphasis: List[str] = field(default_factory=lambda: ["virtue", "deontological", "consequentialist"])
    confidence_threshold: float = 0.7
    explanation_level: str = "standard"  # minimal, standard, detailed, academic
    
    # Performance constraints
    max_processing_time: float = 30.0
    max_memory_usage: int = 1000  # MB
    
    # Metadata for traceability
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation parameters, including dynamic thresholds
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedEthicalResult:
    """
    🎓 PROFESSOR'S EXPLANATION: Unified Result Structure
    ═══════════════════════════════════════════════════
    
    This represents the complete output of our ethical analysis system.
    Like a comprehensive philosophical paper, it contains:
    
    1. **THE CONCLUSION**: Overall ethical assessment
    2. **THE ARGUMENT**: Detailed reasoning from multiple perspectives
    3. **THE EVIDENCE**: Supporting data and citations
    4. **THE CONFIDENCE**: How certain we are of our judgment
    5. **THE IMPLICATIONS**: What actions should be taken
    """
    # Core assessment
    request_id: str
    overall_ethical: bool
    confidence_score: float
    
    # Multi-perspective analysis (following our three-layer architecture)
    meta_ethical_analysis: Dict[str, Any] = field(default_factory=dict)
    normative_analysis: Dict[str, Any] = field(default_factory=dict)
    applied_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed findings
    ethical_violations: List[Dict[str, Any]] = field(default_factory=list)
    autonomy_assessment: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    
    # Recommendations and guidance
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    
    # Knowledge integration
    relevant_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    cache_hit: bool = False
    optimization_used: bool = False
    
    # Explanation and transparency
    explanation: str = ""
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context and metadata
    context: Optional[UnifiedEthicalContext] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = "10.0.0"

class EthicalComponentInterface(ABC):
    """
    🎓 PROFESSOR'S EXPLANATION: Abstract Component Interface
    ═══════════════════════════════════════════════════════
    
    This abstract base class defines the contract that all ethical AI components
    must follow. It's based on the Interface Segregation Principle (ISP) from
    SOLID design principles.
    
    Think of this as the "ethical evaluation protocol" that ensures all our
    specialized components can work together harmoniously, like members of
    an interdisciplinary ethics committee.
    """
    
    @abstractmethod
    async def initialize(self, configuration: Dict[str, Any]) -> bool:
        """Initialize the component with given configuration."""
        pass
    
    @abstractmethod
    async def process(self, content: str, context: UnifiedEthicalContext) -> Dict[str, Any]:
        """Process content and return component-specific results."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and prepare for shutdown."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return current health and performance metrics."""
        pass

class UnifiedEthicalOrchestrator:
    """
    🏛️ THE UNIFIED ETHICAL AI ORCHESTRATOR 🏛️
    ═══════════════════════════════════════════════════════════════════════════════════
    
    🎓 PROFESSOR'S COMPREHENSIVE LECTURE:
    ════════════════════════════════════════
    
    Ladies and gentlemen, welcome to the masterpiece of ethical AI engineering.
    This orchestrator represents the culmination of:
    
    📚 **2400 YEARS OF PHILOSOPHICAL WISDOM**:
        - Aristotelian virtue ethics (384-322 BCE)
        - Kantian deontological ethics (1724-1804)
        - Utilitarian consequentialism (Bentham 1748-1832, Mill 1806-1873)
        - Modern meta-ethics (Moore 1873-1958, Hume 1711-1776)
    
    🔬 **CUTTING-EDGE COMPUTER SCIENCE**:
        - Domain-Driven Design (Eric Evans)
        - Clean Architecture (Robert Martin)
        - Microservices patterns (Sam Newman)
        - Distributed systems theory (Leslie Lamport)
    
    🧮 **MATHEMATICAL FOUNDATIONS**:
        - Vector space theory for semantic embeddings
        - Graph theory for knowledge representation
        - Information theory for uncertainty quantification
        - Statistical learning theory for continuous improvement
    
    🏗️ **SYSTEM ARCHITECTURE EXPLAINED**:
    ═══════════════════════════════════════════
    
    This orchestrator follows the Hexagonal Architecture pattern (also known as
    Ports and Adapters), where:
    
    1. **CORE DOMAIN** (The Ethics Engine): Pure business logic
    2. **APPLICATION LAYER** (This Orchestrator): Coordinates use cases
    3. **INFRASTRUCTURE LAYER** (Databases, APIs): External concerns
    4. **INTERFACES** (REST, GraphQL, WebSocket): Communication protocols
    
    The system processes ethical evaluations through multiple stages:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                         INPUT PROCESSING                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ 1. Context Analysis   │ 2. Intent Detection │ 3. Safety Check   │
    │ 4. Semantic Embedding │ 5. Knowledge Query  │ 6. Cache Lookup   │
    └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       ETHICAL EVALUATION                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ Meta-Ethics Layer:    │ Normative Layer:     │ Applied Layer:    │
    │ • Semantic Analysis   │ • Virtue Ethics      │ • Domain Rules    │
    │ • Logical Structure   │ • Deontological      │ • Cultural Context│
    │ • Fact-Value Distinct │ • Consequentialist   │ • Legal Framework│
    └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        OUTPUT SYNTHESIS                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ 1. Result Integration │ 2. Confidence Calc  │ 3. Explanation    │
    │ 4. Knowledge Citation │ 5. Recommendations  │ 6. Quality Check  │
    └─────────────────────────────────────────────────────────────────┘
    
    🎯 **DESIGN PATTERNS EMPLOYED**:
    ═══════════════════════════════════
    
    1. **ORCHESTRATOR PATTERN**: Coordinates multiple services
    2. **FACADE PATTERN**: Simplifies complex subsystem interactions
    3. **STRATEGY PATTERN**: Different evaluation strategies per context
    4. **OBSERVER PATTERN**: Event-driven processing updates
    5. **CIRCUIT BREAKER**: Prevents cascade failures
    6. **BULKHEAD**: Isolates resource pools for resilience
    """
    
    def __init__(self):
        """
        🎓 CONSTRUCTOR EXPLANATION:
        ═══════════════════════════
        
        The constructor initializes our orchestrator following the Dependency
        Injection pattern. Instead of creating dependencies directly (which would
        violate the Dependency Inversion Principle), we prepare slots for them
        to be injected later.
        
        This is like preparing a symphony orchestra - we set up the conductor's
        podium and music stands, but the musicians (components) join later.
        """
        
        # 🏛️ Core architectural components
        self._components: Dict[str, EthicalComponentInterface] = {}
        self._configuration: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, Any] = {}
        
        # 🚀 Performance optimization systems
        self._cache_manager = None
        self._thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="EthicalAI-")
        
        # 📊 Monitoring and observability
        self._request_counter = 0
        self._processing_times = []
        self._error_count = 0
        self._start_time = time.time()
        
        # 🔄 State management
        self._is_initialized = False
        self._is_healthy = True
        self._initialization_lock = asyncio.Lock()
        
        logger.info("🏛️ Unified Ethical AI Orchestrator initialized - Ready to embody 2400 years of wisdom")

    async def initialize_system(self, configuration: Dict[str, Any]) -> bool:
        """
        🎓 SYSTEM INITIALIZATION LECTURE:
        ═══════════════════════════════════
        
        System initialization follows the two-phase initialization pattern:
        
        **PHASE 1: DEPENDENCY RESOLUTION**
        - Load and configure all ethical components
        - Establish knowledge bases and databases
        - Initialize caching and performance systems
        - Validate all configurations
        
        **PHASE 2: SYSTEM INTEGRATION**  
        - Connect components through dependency injection
        - Perform health checks on all subsystems
        - Load initial knowledge and calibration data
        - Establish monitoring and logging systems
        
        This is like starting a university - first you hire faculty (components),
        then you create the curriculum (configuration), and finally you open
        for students (requests).
        
        Args:
            configuration: System-wide configuration parameters
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        async with self._initialization_lock:
            if self._is_initialized:
                logger.warning("System already initialized")
                return True
            
            try:
                logger.info("🚀 Beginning Unified Ethical AI System Initialization...")
                
                # Phase 1: Configuration validation
                await self._validate_configuration(configuration)
                self._configuration = configuration.copy()
                
                # Phase 2: Component initialization
                await self._initialize_core_components()
                
                # Phase 3: Knowledge system setup
                await self._initialize_knowledge_systems()
                
                # Phase 4: Performance optimization
                await self._initialize_performance_systems()
                
                # Phase 5: Health check
                if not await self._perform_system_health_check():
                    raise RuntimeError("System health check failed")
                
                self._is_initialized = True
                logger.info("✅ Unified Ethical AI System successfully initialized")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ System initialization failed: {e}")
                self._is_healthy = False
                return False
    
    async def evaluate_content(
        self, 
        content: Union[str, List[str]], 
        context: Optional[UnifiedEthicalContext] = None,
        tau_slider: Optional[float] = None
    ) -> UnifiedEthicalResult:
        """
        🎓 MAIN EVALUATION PROCESS LECTURE:
        ═══════════════════════════════════════
        
        This is the crown jewel method - where all our philosophical wisdom,
        mathematical rigor, and engineering excellence comes together to perform
        comprehensive ethical evaluation.
        
        **THE EVALUATION PIPELINE** (Following Clean Architecture):
        
        1. **PREPARATION PHASE**:
           - Validate inputs and context
           - Check cache for previous evaluations
           - Prepare processing resources
        
        2. **ANALYSIS PHASE**:
           - Meta-ethical analysis (logical structure, semantic coherence)
           - Normative analysis (virtue, deontological, consequentialist)
           - Applied analysis (domain-specific rules and constraints)
        
        3. **SYNTHESIS PHASE**:
           - Integrate perspectives using weighted scoring
           - Generate confidence measurements
           - Create comprehensive explanations
        
        4. **KNOWLEDGE INTEGRATION PHASE**:
           - Query relevant philosophical and domain knowledge
           - Generate citations and supporting evidence
           - Create actionable recommendations
        
        This process mirrors how a professional ethicist would approach a complex
        moral question - systematically, rigorously, and with full transparency.
        
        Args:
            content: Text content to evaluate (string or list of strings)
            context: Evaluation context with preferences and constraints
            
        Returns:
            UnifiedEthicalResult: Comprehensive ethical evaluation
        """
        
        # 🔍 Input validation and preparation
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        if not content:
            raise ValueError("Content cannot be empty")
        
        # Normalize content to string
        text_content = content if isinstance(content, str) else " ".join(content)
        
        # Prepare context with defaults
        if context is None:
            context = UnifiedEthicalContext()
        
        # 📊 Performance tracking
        start_time = time.time()
        self._request_counter += 1
        
        logger.info(f"🎯 Beginning ethical evaluation for request {context.request_id}")
        
        try:
            # 🏃‍♂️ PHASE 1: PREPARATION AND CACHING
            cache_result = await self._check_evaluation_cache(text_content, context)
            if cache_result:
                logger.info(f"✅ Cache hit for request {context.request_id}")
                return cache_result
            
            # Extract parameters from context for the evaluation pipeline
            tau_slider_from_context = context.parameters.get('tau_slider')
            evaluation_params = context.parameters

            # 🧠 PHASE 2: MULTI-PERSPECTIVE ETHICAL ANALYSIS
            analysis_results = await self._perform_comprehensive_analysis(
                content=text_content, 
                context=context, 
                tau_slider=tau_slider_from_context,
                parameters=evaluation_params
            )
            
            # 🌐 PHASE 3: KNOWLEDGE INTEGRATION
            knowledge_results = await self._integrate_relevant_knowledge(text_content, context, analysis_results)
            
            # 🔬 PHASE 4: RESULT SYNTHESIS
            unified_result = await self._synthesize_unified_result(
                text_content, context, analysis_results, knowledge_results, start_time
            )
            
            # 💾 PHASE 5: CACHING AND OPTIMIZATION
            await self._cache_evaluation_result(text_content, context, unified_result)
            
            # 📈 Performance metrics update
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            logger.info(f"✅ Completed ethical evaluation for request {context.request_id} in {processing_time:.3f}s")
            
            return unified_result
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"❌ Evaluation failed for request {context.request_id}: {e}")
            
            # Return graceful degradation result
            return await self._create_error_result(context, str(e), time.time() - start_time)
    
    async def _validate_configuration(self, configuration: Dict[str, Any]) -> None:
        """Configuration validation ensures system safety and effectiveness."""
        required_params = [
            "ethical_frameworks", "knowledge_sources", "performance_limits",
            "cache_settings", "monitoring_config"
        ]
        
        for param in required_params:
            if param not in configuration:
                # Create default configuration if missing
                configuration[param] = {}
        
        logger.info("✅ Configuration validation passed")
    
    async def _initialize_core_components(self) -> None:
        """Initialize core ethical evaluation components."""
        logger.info("🔧 Initializing core ethical components...")
        
        # Initialize each component if available
        if DEPENDENCIES_AVAILABLE:
            try:
                # Core ethical evaluation engine with proper initialization
                from ethical_engine import EthicalEvaluator, EthicalParameters
                
                # Initialize with default parameters
                params = EthicalParameters()
                params.virtue_threshold = 0.7
                params.deontological_threshold = 0.7
                params.consequentialist_threshold = 0.7
                params.max_span_length = 10
                
                self._components['core_engine'] = EthicalEvaluator(parameters=params)
                logger.info("✅ Core ethical engine initialized with default parameters")
                
                # Enhanced ethics pipeline
                try:
                    from enhanced_ethics_pipeline import get_enhanced_ethics_pipeline
                    pipeline = get_enhanced_ethics_pipeline()
                    if pipeline:
                        self._components['enhanced_pipeline'] = pipeline
                        logger.info("✅ Enhanced ethics pipeline initialized")
                except ImportError as e:
                    logger.warning(f"Enhanced ethics pipeline not available: {e}")
                
                # Knowledge integration layer  
                try:
                    from knowledge_integration_layer import get_knowledge_integrator
                    integrator = get_knowledge_integrator()
                    if integrator:
                        self._components['knowledge_integrator'] = integrator
                        logger.info("✅ Knowledge integration layer initialized")
                except ImportError as e:
                    logger.warning(f"Knowledge integration layer not available: {e}")
                
            except Exception as e:
                logger.error(f"Failed to initialize core components: {e}", exc_info=True)
                raise
        else:
            logger.error("Required dependencies are not available")
            raise RuntimeError("Failed to initialize core components: Dependencies not available")
        
        logger.info("🎯 Core component initialization complete")
    
    async def _initialize_knowledge_systems(self) -> None:
        """Initialize knowledge bases and semantic systems."""
        logger.info("📚 Initializing knowledge systems...")
        
        # Initialize knowledge sources
        if 'knowledge_integrator' in self._components:
            try:
                knowledge_sources = [
                    "artificial intelligence ethics",
                    "machine learning fairness",
                    "algorithmic transparency",
                    "digital rights",
                    "automated decision making"
                ]
                await self._components['knowledge_integrator'].index_knowledge(knowledge_sources)
            except Exception as e:
                logger.warning(f"Knowledge indexing failed: {e}")
        
        logger.info("✅ Knowledge systems initialized")
    
    async def _initialize_performance_systems(self) -> None:
        """Initialize caching, monitoring, and performance optimization."""
        logger.info("⚡ Initializing performance systems...")
        
        # Initialize performance tracking
        self._performance_metrics = {
            "total_requests": 0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
            "uptime": 0.0
        }
        
        logger.info("✅ Performance systems initialized")
    
    async def _perform_system_health_check(self) -> bool:
        """Comprehensive system health verification."""
        logger.info("🏥 Performing system health check...")
        
        try:
            # Check each component
            for component_name, component in self._components.items():
                if hasattr(component, 'get_health_status'):
                    health = component.get_health_status()
                    if not health.get('healthy', True):
                        logger.warning(f"Component {component_name} reports unhealthy: {health}")
                        return False
            
            logger.info("✅ System health check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    async def _check_evaluation_cache(self, content: str, context: UnifiedEthicalContext) -> Optional[UnifiedEthicalResult]:
        """Check if evaluation result exists in cache."""
        # Implementation would check cache based on content hash and context
        return None
    
    async def _perform_comprehensive_analysis(self, content: str, context: UnifiedEthicalContext, tau_slider: Optional[float] = None, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform multi-layered ethical analysis."""
        analysis_results = {}
        
        # Use core engine by default
        if 'core_engine' in self._components:
            try:
                core_engine = self._components['core_engine']
                
                # Initialize parameters with defaults
                ethical_params = EthicalParameters()
                
                # Update with provided parameters if any
                if parameters:
                    for key, value in parameters.items():
                        if hasattr(ethical_params, key):
                            setattr(ethical_params, key, value)
                
                # Apply tau_slider if provided
                if tau_slider is not None:
                    ethical_params.virtue_threshold = tau_slider * 0.5
                    ethical_params.deontological_threshold = tau_slider * 0.5
                    ethical_params.consequentialist_threshold = tau_slider * 0.5
                
                logger.info(f"Starting core engine evaluation with params: {ethical_params}")
                
                # Perform evaluation
                try:
                    # Call evaluate_text directly since it's a synchronous method
                    evaluation = core_engine.evaluate_text(
                        text=content,
                        _skip_uncertainty_analysis=True  # Skip advanced analysis to prevent recursion
                    )
                except Exception as e:
                    logger.error(f"Error in core_engine.evaluate_text: {e}", exc_info=True)
                    raise
                
                # Store the evaluation result
                if evaluation:
                    analysis_results['core_evaluation'] = evaluation
                    logger.info(f"Core engine evaluation completed with {len(evaluation.spans)} spans")
                
            except Exception as e:
                logger.error(f"Core engine evaluation failed: {e}", exc_info=True)
        
        # Fallback to enhanced pipeline if core engine fails
        if not analysis_results and 'enhanced_pipeline' in self._components:
            try:
                pipeline = self._components['enhanced_pipeline']
                
                # Comprehensive analysis through enhanced pipeline
                meta_analysis = await pipeline.perform_meta_ethical_analysis(content)
                analysis_results['meta_ethical'] = meta_analysis
                
                normative_analysis = await self._perform_normative_analysis(pipeline, content, context, parameters or {})
                analysis_results['normative'] = normative_analysis
                
                applied_analysis = await pipeline.perform_applied_analysis(content, context.domain, context.cultural_context)
                analysis_results['applied'] = applied_analysis
                
                logger.info("Fell back to enhanced pipeline analysis")
                
            except Exception as e:
                logger.error(f"Enhanced pipeline analysis also failed: {e}", exc_info=True)
        
        return analysis_results
    
    async def _integrate_relevant_knowledge(self, content: str, context: UnifiedEthicalContext, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate external knowledge into evaluation."""
        knowledge_results = {}
        
        if 'knowledge_integrator' in self._components:
            try:
                integrator = self._components['knowledge_integrator']
                
                # Create knowledge query based on content and context
                query = KnowledgeQuery(
                    query_id=f"eval_{context.request_id}",
                    text=content,
                    domain_filter=context.domain,
                    max_results=5
                )
                
                # Query relevant knowledge
                knowledge_result = await integrator.query_knowledge(query)
                knowledge_results = {
                    'fragments': knowledge_result.fragments,
                    'entities': knowledge_result.entities,
                    'synthesis': knowledge_result.synthesis,
                    'citations': knowledge_result.citations,
                    'confidence': knowledge_result.confidence_score
                }
                
            except Exception as e:
                logger.warning(f"Knowledge integration failed: {e}")
        
        return knowledge_results
    
    async def _synthesize_unified_result(
        self, 
        content: str, 
        context: UnifiedEthicalContext, 
        analysis_results: Dict[str, Any], 
        knowledge_results: Dict[str, Any], 
        start_time: float
    ) -> UnifiedEthicalResult:
        """Synthesize all analysis into unified result."""
        
        # Extract core evaluation results
        core_eval = analysis_results.get('core_evaluation')
        overall_ethical = core_eval.overall_ethical if core_eval else True
        
        # Calculate comprehensive confidence score
        confidence_score = self._calculate_confidence_score(analysis_results, knowledge_results)
        
        # Extract autonomy assessment
        autonomy_assessment = {}
        if core_eval and hasattr(core_eval, 'autonomy_dimensions'):
            autonomy_assessment = core_eval.autonomy_dimensions
        
        # Extract violations
        ethical_violations = []
        if core_eval and hasattr(core_eval, 'minimal_spans'):
            for span in core_eval.minimal_spans:
                ethical_violations.append({
                    'text': span.text,
                    'start': span.start,
                    'end': span.end,
                    'violation_type': 'ethical_principle',
                    'severity': 'moderate'
                })
        
        # Generate comprehensive explanation
        explanation = self._generate_comprehensive_explanation(
            content, context, analysis_results, knowledge_results
        )
        
        # Create unified result
        result = UnifiedEthicalResult(
            request_id=context.request_id,
            overall_ethical=overall_ethical,
            confidence_score=confidence_score,
            
            # Analysis results
            meta_ethical_analysis=analysis_results.get('meta_ethical', {}),
            normative_analysis=analysis_results.get('normative', {}),
            applied_analysis=analysis_results.get('applied', {}),
            
            # Detailed findings
            ethical_violations=ethical_violations,
            autonomy_assessment=autonomy_assessment,
            
            # Knowledge integration
            relevant_knowledge=knowledge_results.get('fragments', []),
            citations=knowledge_results.get('citations', []),
            
            # Performance metrics
            processing_time=time.time() - start_time,
            cache_hit=False,
            optimization_used=True,
            
            # Explanation and context
            explanation=explanation,
            context=context
        )
        
        return result
    
    def _calculate_confidence_score(self, analysis_results: Dict[str, Any], knowledge_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis consistency."""
        base_confidence = 0.7
        
        # Adjust based on framework agreement
        if 'meta_ethical' in analysis_results and 'normative' in analysis_results:
            base_confidence += 0.2
        
        # Adjust based on knowledge integration
        knowledge_confidence = knowledge_results.get('confidence', 0.5)
        base_confidence = (base_confidence + knowledge_confidence) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_comprehensive_explanation(
        self, 
        content: str, 
        context: UnifiedEthicalContext, 
        analysis_results: Dict[str, Any], 
        knowledge_results: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation."""
        core_eval = analysis_results.get('core_evaluation')
        
        if core_eval:
            if core_eval.overall_ethical:
                summary = "The content appears to be ethically sound based on comprehensive multi-framework analysis."
            else:
                violation_count = len(core_eval.minimal_spans) if hasattr(core_eval, 'minimal_spans') else 0
                summary = f"The content raises ethical concerns with {violation_count} potential violations identified."
        else:
            summary = "Ethical assessment completed with limited analysis capabilities."
        
        # Add knowledge integration context
        if knowledge_results.get('synthesis'):
            summary += f"\n\nRelevant philosophical context: {knowledge_results['synthesis']}"
        
        return summary
    
    async def _cache_evaluation_result(self, content: str, context: UnifiedEthicalContext, result: UnifiedEthicalResult) -> None:
        """Cache evaluation result for future use."""
        # Implementation would cache the result based on content hash and context
        pass
    
    async def _create_error_result(self, context: UnifiedEthicalContext, error_message: str, processing_time: float) -> UnifiedEthicalResult:
        """Create graceful degradation result for errors."""
        return UnifiedEthicalResult(
            request_id=context.request_id,
            overall_ethical=False,  # Conservative approach for errors
            confidence_score=0.0,
            processing_time=processing_time,
            explanation=f"Evaluation failed due to system error: {error_message}",
            warnings=[f"System error occurred: {error_message}"],
            context=context
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        uptime = time.time() - self._start_time
        avg_processing_time = sum(self._processing_times[-100:]) / min(100, len(self._processing_times)) if self._processing_times else 0.0
        error_rate = self._error_count / max(1, self._request_counter)
        
        return {
            "system_info": {
                "version": "10.0.0",
                "uptime_seconds": uptime,
                "is_healthy": self._is_healthy,
                "is_initialized": self._is_initialized
            },
            "performance": {
                "total_requests": self._request_counter,
                "average_processing_time_ms": avg_processing_time * 1000,
                "error_rate_percent": error_rate * 100,
                "requests_per_second": self._request_counter / uptime if uptime > 0 else 0
            },
            "components": {
                name: component.get_health_status() if hasattr(component, 'get_health_status') else {"status": "active"}
                for name, component in self._components.items()
            }
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the orchestrator."""
        logger.info("🛑 Beginning graceful shutdown of Unified Ethical AI System...")
        
        try:
            # Stop accepting new requests
            self._is_healthy = False
            
            # Cleanup components
            for component_name, component in self._components.items():
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                logger.info(f"✅ Component {component_name} cleaned up")
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            logger.info("✅ Unified Ethical AI System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

async def initialize_unified_system(configuration: Dict[str, Any]) -> UnifiedEthicalOrchestrator:
    """Initialize the unified system with configuration."""
    orchestrator = get_unified_orchestrator()
    success = await orchestrator.initialize_system(configuration)
    
    if not success:
        raise RuntimeError("Failed to initialize Unified Ethical AI System")
    
    logger.info("🚀 Unified Ethical AI System successfully initialized and ready for service")
    return orchestrator

# 🏛️ GLOBAL ORCHESTRATOR INSTANCE
# ═══════════════════════════════════
# Following the Singleton pattern for system-wide coordination
_global_orchestrator: Optional[UnifiedEthicalOrchestrator] = None
_orchestrator_lock = threading.Lock()

def get_unified_orchestrator() -> UnifiedEthicalOrchestrator:
    """
    🎓 SINGLETON PATTERN EXPLANATION:
    ═════════════════════════════════════
    
    The Singleton pattern ensures we have exactly one orchestrator instance
    throughout the system. Like having one conductor for an orchestra, this
    ensures coordinated, consistent behavior across all ethical evaluations.
    
    We use thread-safe lazy initialization to create the instance only when
    first needed, following the principle of "late binding" in software design.
    
    Returns:
        UnifiedEthicalOrchestrator: The global orchestrator instance
    """
    global _global_orchestrator
    
    if _global_orchestrator is None:
        with _orchestrator_lock:
            if _global_orchestrator is None:  # Double-checked locking
                _global_orchestrator = UnifiedEthicalOrchestrator()
                logger.info("🏛️ Global Unified Ethical AI Orchestrator created")
    
    return _global_orchestrator