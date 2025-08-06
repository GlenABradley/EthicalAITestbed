"""
Perform Health Check Use Case for the Ethical AI Testbed.

This module defines the use case for performing a system health check.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from application.dtos.system_health_response import SystemHealthResponse

logger = logging.getLogger(__name__)

# Global start time for uptime calculation
START_TIME = time.time()

class PerformHealthCheckUseCase:
    """
    Use case for performing a system health check.
    
    This class implements the use case for checking the health of the
    entire Ethical AI system. It follows the Clean Architecture
    pattern for use cases.
    """
    
    def __init__(self, orchestrator, db):
        """
        Initialize the use case with dependencies.
        
        Args:
            orchestrator: The unified ethical orchestrator
            db: The database connection
        """
        self.orchestrator = orchestrator
        self.db = db
        
    async def execute(self) -> SystemHealthResponse:
        """
        Execute the use case to perform a health check.
        
        Returns:
            SystemHealthResponse: The health check results
        """
        logger.info("Performing system health check")
        
        # Calculate uptime
        uptime_seconds = time.time() - START_TIME
        
        # Check orchestrator health
        orchestrator_healthy = self._check_orchestrator_health()
        
        # Check database connection
        database_connected = self._check_database_connection()
        
        # Check configuration validity
        configuration_valid = self._check_configuration_validity()
        
        # Collect performance metrics
        performance_metrics = await self._collect_performance_metrics()
        
        # Check feature availability
        features_available = self._check_feature_availability()
        
        # Determine overall status
        status = self._determine_overall_status(
            orchestrator_healthy,
            database_connected,
            configuration_valid
        )
        
        # Create response
        return SystemHealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime_seconds,
            orchestrator_healthy=orchestrator_healthy,
            database_connected=database_connected,
            configuration_valid=configuration_valid,
            performance_metrics=performance_metrics,
            features_available=features_available
        )
        
    def _check_orchestrator_health(self) -> bool:
        """Check if the orchestrator is healthy."""
        try:
            return self.orchestrator is not None and hasattr(self.orchestrator, 'evaluate_content')
        except Exception as e:
            logger.error(f"Error checking orchestrator health: {str(e)}", exc_info=True)
            return False
            
    def _check_database_connection(self) -> bool:
        """Check if the database connection is active."""
        try:
            return self.db is not None
        except Exception as e:
            logger.error(f"Error checking database connection: {str(e)}", exc_info=True)
            return False
            
    def _check_configuration_validity(self) -> bool:
        """Check if the system configuration is valid."""
        try:
            return self.orchestrator is not None and hasattr(self.orchestrator, 'config')
        except Exception as e:
            logger.error(f"Error checking configuration validity: {str(e)}", exc_info=True)
            return False
            
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "average_evaluation_time_ms": 0,
            "evaluations_per_second": 0,
            "cache_hit_ratio": 0
        }
        
        try:
            # Try to get metrics from orchestrator if available
            if hasattr(self.orchestrator, 'get_performance_metrics'):
                orchestrator_metrics = await self.orchestrator.get_performance_metrics()
                metrics.update(orchestrator_metrics)
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}", exc_info=True)
            
        return metrics
        
    def _check_feature_availability(self) -> Dict[str, bool]:
        """Check which features are available."""
        features = {
            "ethical_evaluation": True,
            "learning_system": False,
            "graph_attention": False,
            "intent_hierarchy": False,
            "causal_counterfactual": False,
            "uncertainty_analysis": False,
            "irl_purpose_alignment": False
        }
        
        try:
            # Check if advanced features are available
            if self.orchestrator is not None:
                if hasattr(self.orchestrator, 'learning_layer'):
                    features["learning_system"] = self.orchestrator.learning_layer is not None
                    
                if hasattr(self.orchestrator, 'evaluator'):
                    evaluator = self.orchestrator.evaluator
                    features["graph_attention"] = hasattr(evaluator, 'graph_attention') and evaluator.graph_attention is not None
                    features["intent_hierarchy"] = hasattr(evaluator, 'intent_hierarchy') and evaluator.intent_hierarchy is not None
                    features["causal_counterfactual"] = hasattr(evaluator, 'causal_counterfactual') and evaluator.causal_counterfactual is not None
                    features["uncertainty_analyzer"] = hasattr(evaluator, 'uncertainty_analyzer') and evaluator.uncertainty_analyzer is not None
                    features["irl_purpose_alignment"] = hasattr(evaluator, 'irl_purpose_alignment') and evaluator.irl_purpose_alignment is not None
        except Exception as e:
            logger.error(f"Error checking feature availability: {str(e)}", exc_info=True)
            
        return features
        
    def _determine_overall_status(self, orchestrator_healthy: bool, database_connected: bool, configuration_valid: bool) -> str:
        """Determine the overall system status."""
        if orchestrator_healthy and configuration_valid:
            if database_connected:
                return "healthy"
            else:
                return "degraded"
        else:
            return "error"
