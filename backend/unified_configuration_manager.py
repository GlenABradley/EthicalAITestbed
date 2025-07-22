"""
🔧 UNIFIED CONFIGURATION MANAGER 🔧
═══════════════════════════════════════════════════════════════════════════════════

🎓 PROFESSOR'S LECTURE: Configuration Management in Large-Scale Systems
══════════════════════════════════════════════════════════════════════════════

Welcome to the Configuration Management masterclass! This module represents the
centralized nervous system of our Ethical AI platform - the single source of truth
for all system parameters, preferences, and operational settings.

📚 THEORETICAL FOUNDATIONS:
═══════════════════════════════════════════════════════════════════════════════

**CONFIGURATION MANAGEMENT PRINCIPLES** (Based on "The Twelve-Factor App"):
1. **Separation of Concerns**: Configuration separate from code
2. **Environment Parity**: Same config structure across dev/staging/production  
3. **Explicit Dependencies**: All dependencies declared and isolated
4. **Stateless Processes**: Configuration enables stateless operation
5. **Administrative Processes**: Config changes through admin interfaces

**DESIGN PATTERNS EMPLOYED**:
- **Strategy Pattern**: Different configurations for different deployment modes
- **Template Method**: Common configuration loading with customizable steps
- **Observer Pattern**: Components notified of configuration changes
- **Singleton Pattern**: Single source of configuration truth
- **Factory Pattern**: Configuration factories for different environments

🏗️ ARCHITECTURAL OVERVIEW:
═══════════════════════════════════════════════════════════════════════════════

Our configuration system follows a hierarchical structure:

┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION HIERARCHY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   DEFAULTS      │    │   ENVIRONMENT   │    │     USER        │  │
│  │                 │    │                 │    │                 │  │
│  │ • Base values   │ ←→ │ • Dev/Prod      │ ←→ │ • Preferences   │  │
│  │ • Fallbacks     │    │ • Overrides     │    │ • Runtime       │  │
│  │ • Constraints   │    │ • Secrets       │    │ • Customization │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │         │
│           └───────────────────────┼───────────────────────┘         │
│                                   ▼                                 │
│                    ┌─────────────────────────────────────┐           │
│                    │       UNIFIED CONFIG                │           │
│                    │   (Runtime Configuration)           │           │
│                    └─────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Author: MIT-Level Configuration Engineering Team
Version: 10.0.0 - Unified Configuration Management (Phase 9.5 Refactor)
Inspired by: Martin Fowler's Configuration patterns, Netflix's Archaius,
             Spring Boot's configuration management
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
import threading
from datetime import datetime
import hashlib
import copy

# Configure logging
logger = logging.getLogger(__name__)

class ConfigurationMode(Enum):
    """
    🎓 CONFIGURATION MODES EXPLANATION:
    ═══════════════════════════════════════
    
    Different operational modes require different configuration approaches:
    
    - **DEVELOPMENT**: Optimized for debugging and rapid iteration
    - **TESTING**: Isolated and reproducible for automated testing
    - **STAGING**: Production-like for final validation
    - **PRODUCTION**: Optimized for performance and reliability
    - **RESEARCH**: Flexible for experimental configurations
    """
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"

@dataclass
class EthicalFrameworkConfig:
    """
    🎓 ETHICAL FRAMEWORK CONFIGURATION:
    ═══════════════════════════════════════
    
    This configuration defines how our three primary ethical frameworks
    (Virtue, Deontological, Consequentialist) are weighted and applied.
    
    Like a judicial system with different courts having different jurisdictions,
    our ethical frameworks have different strengths and appropriate contexts.
    """
    
    # Framework weights (must sum to 1.0)
    virtue_weight: float = 0.33
    deontological_weight: float = 0.33
    consequentialist_weight: float = 0.34
    
    # Individual framework configurations
    virtue_config: Dict[str, Any] = field(default_factory=lambda: {
        "cardinal_virtues_weight": 0.8,  # Aristotelian cardinal virtues
        "practical_wisdom_weight": 0.2,  # Phronesis
        "cultural_adaptation": True
    })
    
    deontological_config: Dict[str, Any] = field(default_factory=lambda: {
        "categorical_imperative_weight": 0.7,  # Kantian universalizability
        "duty_based_weight": 0.2,              # Ross's prima facie duties
        "rights_based_weight": 0.1             # Human rights framework
    })
    
    consequentialist_config: Dict[str, Any] = field(default_factory=lambda: {
        "utilitarian_weight": 0.6,     # Classical utilitarianism
        "welfare_weight": 0.3,         # Welfare economics
        "preference_weight": 0.1       # Preference satisfaction
    })
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0 and configurations are consistent."""
        total_weight = self.virtue_weight + self.deontological_weight + self.consequentialist_weight
        return abs(total_weight - 1.0) < 0.001

@dataclass
class PerformanceConfig:
    """
    🎓 PERFORMANCE CONFIGURATION EXPLANATION:
    ═════════════════════════════════════════════
    
    Performance configuration balances speed, accuracy, and resource usage.
    These parameters directly affect the system's operational characteristics.
    
    Like tuning a high-performance engine, each parameter affects overall
    system behavior and must be carefully calibrated.
    """
    
    # Processing limits
    max_processing_time_seconds: float = 30.0
    max_memory_usage_mb: int = 1000
    max_concurrent_evaluations: int = 10
    
    # Cache configuration
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size_mb: int = 500
    cache_hit_target_percent: float = 80.0
    
    # Thread pool configuration
    core_thread_pool_size: int = 4
    max_thread_pool_size: int = 16
    thread_keepalive_seconds: int = 60
    
    # Streaming configuration
    enable_streaming: bool = True
    max_streaming_connections: int = 100
    streaming_buffer_size: int = 1000
    streaming_timeout_seconds: float = 5.0
    
    # Optimization settings
    enable_gpu_acceleration: bool = False
    enable_parallel_processing: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive

@dataclass
class KnowledgeSourceConfig:
    """
    🎓 KNOWLEDGE SOURCE CONFIGURATION:
    ══════════════════════════════════════════
    
    Knowledge source configuration defines how we integrate external wisdom
    into our ethical evaluations. This includes philosophical databases,
    legal frameworks, cultural guidelines, and academic research.
    
    Like a research library with different sections and access policies,
    we configure how to access and weight different knowledge sources.
    """
    
    # Enable/disable knowledge sources
    enable_wikipedia: bool = True
    enable_philosophy_database: bool = True
    enable_legal_database: bool = False
    enable_cultural_database: bool = True
    
    # Source prioritization
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "academic_papers": 0.4,
        "philosophical_texts": 0.3,
        "legal_documents": 0.2,
        "cultural_guidelines": 0.1
    })
    
    # Query configuration
    max_knowledge_results: int = 10
    knowledge_confidence_threshold: float = 0.6
    enable_knowledge_caching: bool = True
    knowledge_refresh_hours: int = 24
    
    # Language and localization
    primary_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    cultural_context: str = "western"

@dataclass
class SecurityConfig:
    """
    🎓 SECURITY CONFIGURATION EXPLANATION:
    ══════════════════════════════════════════
    
    Security configuration ensures our ethical AI system operates safely
    and protects sensitive information. This follows defense-in-depth
    principles with multiple layers of security controls.
    
    Like a multi-layered security system for a research facility, we
    implement authentication, authorization, encryption, and monitoring.
    """
    
    # Authentication
    enable_authentication: bool = True
    auth_method: str = "jwt"  # jwt, oauth, ldap
    jwt_secret_key: Optional[str] = None
    token_expiry_hours: int = 24
    
    # Authorization  
    enable_authorization: bool = True
    default_user_role: str = "user"
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "superuser"])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    burst_allowance: int = 10
    
    # Encryption
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    sensitive_data_masking: bool = True

@dataclass
class UnifiedConfiguration:
    """
    🎓 UNIFIED CONFIGURATION STRUCTURE:
    ═══════════════════════════════════════
    
    This is the master configuration structure that unifies all system
    settings into a single, coherent, type-safe configuration object.
    
    Like a constitution that defines the fundamental principles and
    operational structure of a government, this configuration defines
    how our ethical AI system operates.
    """
    
    # System metadata
    version: str = "10.0.0"
    mode: ConfigurationMode = ConfigurationMode.DEVELOPMENT
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    # Core configuration sections
    ethical_frameworks: EthicalFrameworkConfig = field(default_factory=EthicalFrameworkConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    knowledge_sources: KnowledgeSourceConfig = field(default_factory=KnowledgeSourceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Feature toggles
    features: Dict[str, bool] = field(default_factory=lambda: {
        "enhanced_ethics_pipeline": True,
        "knowledge_integration": True,
        "real_time_streaming": True,
        "ml_training_guidance": True,
        "multi_modal_evaluation": True,
        "production_monitoring": True,
        "advanced_caching": True,
        "distributed_processing": False
    })
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Custom extensions
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        🎓 CONFIGURATION VALIDATION:
        ════════════════════════════════
        
        Comprehensive validation ensures our configuration is internally
        consistent and operationally viable. Like a compiler checking
        syntax and semantics, we verify all constraints and dependencies.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate ethical framework weights
        if not self.ethical_frameworks.validate():
            errors.append("Ethical framework weights must sum to 1.0")
        
        # Validate performance constraints
        if self.performance.max_processing_time_seconds <= 0:
            errors.append("Max processing time must be positive")
        
        if self.performance.max_memory_usage_mb <= 0:
            errors.append("Max memory usage must be positive")
        
        # Validate knowledge source weights
        source_weight_sum = sum(self.knowledge_sources.source_weights.values())
        if abs(source_weight_sum - 1.0) > 0.001:
            errors.append("Knowledge source weights must sum to 1.0")
        
        # Validate security settings
        if self.security.enable_authentication and not self.security.jwt_secret_key:
            # Generate a default secret key for development
            if self.mode == ConfigurationMode.DEVELOPMENT:
                self.security.jwt_secret_key = "dev_secret_key_please_change_in_production"
            else:
                errors.append("JWT secret key required when authentication is enabled")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return asdict(self)
    
    def get_config_hash(self) -> str:
        """Generate hash for configuration change detection."""
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class UnifiedConfigurationManager:
    """
    🏛️ UNIFIED CONFIGURATION MANAGER 🏛️
    ═══════════════════════════════════════════════════════════════════════════════════
    
    🎓 PROFESSOR'S COMPREHENSIVE EXPLANATION:
    ═════════════════════════════════════════════════════════════════════════════════════
    
    The Unified Configuration Manager is the central nervous system for our
    entire Ethical AI platform. It orchestrates configuration from multiple
    sources, validates consistency, and provides a unified interface for
    all system components.
    
    **ARCHITECTURAL PRINCIPLES**:
    
    1. **Single Source of Truth**: All configuration flows through this manager
    2. **Layered Configuration**: Multiple sources with defined precedence
    3. **Type Safety**: Strong typing prevents configuration errors
    4. **Validation**: Comprehensive validation prevents invalid states
    5. **Hot Reloading**: Dynamic configuration updates without restarts
    6. **Observability**: Full visibility into configuration changes
    
    **DESIGN PATTERNS**:
    
    - **Facade Pattern**: Simplifies complex configuration subsystem
    - **Observer Pattern**: Notifies components of configuration changes
    - **Strategy Pattern**: Different providers for different sources
    - **Template Method**: Common configuration loading workflow
    - **Singleton Pattern**: Global configuration access point
    
    This manager follows the principle of "Configuration as Code" where all
    system behavior is explicitly defined through declarative configuration.
    """
    
    def __init__(self):
        """Initialize the Unified Configuration Manager."""
        self._configuration: Optional[UnifiedConfiguration] = None
        self._observers: List[Callable[[UnifiedConfiguration], None]] = []
        self._config_lock = threading.RLock()  # Reentrant lock for nested access
        self._last_config_hash: Optional[str] = None
        
        # Configuration change tracking
        self._config_history: List[Tuple[datetime, str]] = []  # (timestamp, config_hash)
        self._load_count = 0
        self._validation_errors: List[str] = []
        
        logger.info("🔧 Unified Configuration Manager initialized")
    
    async def load_configuration(self, force_reload: bool = False) -> UnifiedConfiguration:
        """
        Load configuration with defaults and environment overrides.
        
        🎓 CONFIGURATION LOADING PROCESS:
        ════════════════════════════════════════════
        
        The loading process follows a carefully orchestrated sequence:
        
        1. **Default Configuration**: Start with sensible defaults
        2. **Environment Variables**: Apply environment overrides
        3. **Validation**: Comprehensive validation of the merged configuration
        4. **Change Detection**: Detect if configuration has changed since last load
        5. **Notification**: Notify observers of configuration changes
        6. **Persistence**: Cache the validated configuration for future use
        
        Args:
            force_reload: Force reload even if configuration hasn't changed
            
        Returns:
            UnifiedConfiguration: The loaded and validated configuration
        """
        with self._config_lock:
            start_time = datetime.utcnow()
            self._load_count += 1
            
            try:
                logger.info(f"🔄 Loading configuration (attempt #{self._load_count})")
                
                # Initialize with default configuration
                config = UnifiedConfiguration()
                
                # Apply environment variable overrides
                config = self._apply_environment_overrides(config)
                
                # Validate configuration
                is_valid, validation_errors = config.validate()
                if not is_valid:
                    self._validation_errors = validation_errors
                    logger.error(f"Configuration validation failed: {validation_errors}")
                    raise ValueError(f"Configuration validation failed: {validation_errors}")
                
                # Check if configuration has changed
                config_hash = config.get_config_hash()
                has_changed = force_reload or config_hash != self._last_config_hash
                
                if has_changed:
                    # Update configuration
                    self._configuration = config
                    self._last_config_hash = config_hash
                    self._config_history.append((start_time, config_hash))
                    
                    # Notify observers
                    await self._notify_configuration_change(config)
                    
                    logger.info(f"✅ Configuration loaded successfully (hash: {config_hash[:8]})")
                else:
                    logger.info("📋 Configuration unchanged, using cached version")
                
                return self._configuration
                
            except Exception as e:
                logger.error(f"❌ Configuration loading failed: {e}")
                
                # Return existing configuration if available, otherwise raise
                if self._configuration is not None:
                    logger.warning("Using previous configuration due to load failure")
                    return self._configuration
                else:
                    # Create minimal default configuration
                    self._configuration = UnifiedConfiguration()
                    return self._configuration
    
    def get_configuration(self) -> UnifiedConfiguration:
        """
        Get the current configuration.
        
        🎓 CONFIGURATION ACCESS PATTERN:
        ═══════════════════════════════════════
        
        This method provides thread-safe access to the current configuration.
        If no configuration has been loaded, it creates a default one to prevent
        components from operating with undefined behavior.
        
        Returns:
            UnifiedConfiguration: Current configuration
        """
        with self._config_lock:
            if self._configuration is None:
                # Create default configuration
                self._configuration = UnifiedConfiguration()
                logger.info("Created default configuration")
            
            # Return a deep copy to prevent accidental modifications
            return copy.deepcopy(self._configuration)
    
    def subscribe_to_changes(self, observer: Callable[[UnifiedConfiguration], None]) -> None:
        """Subscribe to configuration changes."""
        with self._config_lock:
            self._observers.append(observer)
        
        logger.info(f"Added configuration change observer: {observer.__name__}")
    
    def unsubscribe_from_changes(self, observer: Callable[[UnifiedConfiguration], None]) -> None:
        """Remove a configuration change observer."""
        with self._config_lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.info(f"Removed configuration change observer: {observer.__name__}")
    
    def get_configuration_metrics(self) -> Dict[str, Any]:
        """Get configuration system metrics and health information."""
        with self._config_lock:
            current_time = datetime.utcnow()
            
            return {
                "system_info": {
                    "configuration_loaded": self._configuration is not None,
                    "last_config_hash": self._last_config_hash,
                    "load_count": self._load_count,
                    "validation_errors": len(self._validation_errors)
                },
                "observers": {
                    "total_observers": len(self._observers),
                    "observer_types": [type(obs).__name__ for obs in self._observers]
                },
                "history": {
                    "total_changes": len(self._config_history),
                    "recent_changes": self._config_history[-5:] if self._config_history else []
                }
            }
    
    def _apply_environment_overrides(self, config: UnifiedConfiguration) -> UnifiedConfiguration:
        """Apply environment variable overrides to configuration."""
        
        # Check for common environment variables
        if os.getenv('ETHICAL_AI_MODE'):
            try:
                config.mode = ConfigurationMode(os.getenv('ETHICAL_AI_MODE'))
            except ValueError:
                logger.warning(f"Invalid mode in environment: {os.getenv('ETHICAL_AI_MODE')}")
        
        # Performance overrides
        if os.getenv('ETHICAL_AI_MAX_PROCESSING_TIME'):
            try:
                config.performance.max_processing_time_seconds = float(os.getenv('ETHICAL_AI_MAX_PROCESSING_TIME'))
            except ValueError:
                logger.warning("Invalid max processing time in environment")
        
        if os.getenv('ETHICAL_AI_MAX_MEMORY_MB'):
            try:
                config.performance.max_memory_usage_mb = int(os.getenv('ETHICAL_AI_MAX_MEMORY_MB'))
            except ValueError:
                logger.warning("Invalid max memory in environment")
        
        # Feature toggles
        if os.getenv('ETHICAL_AI_ENABLE_CACHING'):
            config.performance.enable_caching = os.getenv('ETHICAL_AI_ENABLE_CACHING').lower() == 'true'
        
        if os.getenv('ETHICAL_AI_ENABLE_STREAMING'):
            config.performance.enable_streaming = os.getenv('ETHICAL_AI_ENABLE_STREAMING').lower() == 'true'
        
        # Security overrides
        if os.getenv('ETHICAL_AI_JWT_SECRET'):
            config.security.jwt_secret_key = os.getenv('ETHICAL_AI_JWT_SECRET')
        
        if os.getenv('ETHICAL_AI_ENABLE_AUTH'):
            config.security.enable_authentication = os.getenv('ETHICAL_AI_ENABLE_AUTH').lower() == 'true'
        
        return config
    
    async def _notify_configuration_change(self, config: UnifiedConfiguration) -> None:
        """Notify all observers of configuration change."""
        for observer in self._observers.copy():  # Copy to avoid modification during iteration
            try:
                observer(config)
                logger.debug(f"Notified observer: {observer.__name__}")
            except Exception as e:
                logger.error(f"Observer notification failed for {observer.__name__}: {e}")

# 🏛️ GLOBAL CONFIGURATION MANAGER INSTANCE
# ═════════════════════════════════════════════
_global_config_manager: Optional[UnifiedConfigurationManager] = None
_config_manager_lock = threading.Lock()

def get_configuration_manager() -> UnifiedConfigurationManager:
    """
    Get the global configuration manager instance.
    
    🎓 SINGLETON PATTERN FOR CONFIGURATION:
    ═══════════════════════════════════════════════
    
    Configuration management requires a single, consistent source of truth
    throughout the application. This singleton ensures all components
    access the same configuration manager instance.
    
    Returns:
        UnifiedConfigurationManager: Global configuration manager
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        with _config_manager_lock:
            if _global_config_manager is None:  # Double-checked locking
                _global_config_manager = UnifiedConfigurationManager()
                logger.info("🔧 Global Configuration Manager created")
    
    return _global_config_manager

async def initialize_configuration_system(environment: str = "development") -> UnifiedConfiguration:
    """
    Initialize the configuration system.
    
    🎓 STANDARD CONFIGURATION INITIALIZATION:
    ═══════════════════════════════════════════════════
    
    This function sets up the configuration system with environment-based
    settings and loads the initial configuration.
    
    Args:
        environment: Environment name (development, production, etc.)
        
    Returns:
        UnifiedConfiguration: Loaded and validated configuration
    """
    manager = get_configuration_manager()
    
    # Load configuration with environment context
    config = await manager.load_configuration()
    config.mode = ConfigurationMode(environment)
    
    logger.info(f"🚀 Configuration system initialized for environment: {environment}")
    return config