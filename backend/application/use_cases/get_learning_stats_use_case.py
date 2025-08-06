"""
Get Learning Stats Use Case for the Ethical AI Testbed.

This module defines the use case for retrieving learning system statistics.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GetLearningStatsUseCase:
    """
    Use case for retrieving learning system statistics.
    
    This class implements the use case for retrieving statistics about
    the learning system. It follows the Clean Architecture pattern for use cases.
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
        
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the use case to retrieve learning statistics.
        
        Returns:
            Dict[str, Any]: Learning system statistics
        """
        logger.info("Retrieving learning system statistics")
        
        try:
            # Default response if learning layer is not available
            default_response = {
                "status": "degraded",
                "message": "Learning system not fully available",
                "entries_count": 0,
                "recent_entries": [],
                "performance_metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                }
            }
            
            # Check if learning layer is available
            if not hasattr(self.orchestrator, 'learning_layer') or self.orchestrator.learning_layer is None:
                logger.warning("Learning layer not available")
                return default_response
                
            # Get learning layer
            learning_layer = self.orchestrator.learning_layer
            
            # Get statistics
            stats = {}
            
            # Get entries count
            entries_count = await learning_layer.get_entries_count()
            stats["entries_count"] = entries_count
            
            # Get recent entries
            recent_entries = await learning_layer.get_recent_entries(limit=10)
            stats["recent_entries"] = recent_entries
            
            # Get performance metrics
            performance_metrics = await learning_layer.get_performance_metrics()
            stats["performance_metrics"] = performance_metrics
            
            # Get feedback distribution
            feedback_distribution = await learning_layer.get_feedback_distribution()
            stats["feedback_distribution"] = feedback_distribution
            
            # Add timestamp
            stats["timestamp"] = datetime.utcnow().isoformat()
            stats["status"] = "success"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving learning statistics: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to retrieve learning statistics: {str(e)}",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
