"""
Store Evaluation Result Use Case for the Ethical AI Testbed.

This module defines the use case for storing evaluation results in the database.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any

from application.dtos.evaluation_request import EvaluationRequest
from application.dtos.evaluation_response import EvaluationResponse

logger = logging.getLogger(__name__)

class StoreEvaluationResultUseCase:
    """
    Use case for storing evaluation results in the database.
    
    This class implements the use case for storing evaluation results
    in the database. It follows the Clean Architecture pattern for use cases.
    """
    
    def __init__(self, db):
        """
        Initialize the use case with dependencies.
        
        Args:
            db: The database connection
        """
        self.db = db
        
    async def execute(
        self,
        request: EvaluationRequest,
        response: EvaluationResponse,
        request_id: str
    ) -> None:
        """
        Execute the use case to store evaluation results.
        
        Args:
            request: The evaluation request
            response: The evaluation response
            request_id: The unique request ID
            
        Returns:
            None
        """
        logger.info(f"Storing evaluation result for request {request_id}")
        
        try:
            # Skip if database is not available
            if not self.db:
                logger.warning("Database not available, skipping result storage")
                return
                
            # Prepare document for storage
            document = {
                "request_id": request_id,
                "timestamp": datetime.utcnow(),
                "text": request.text,
                "context": request.context,
                "parameters": request.parameters,
                "mode": request.mode,
                "priority": request.priority,
                "tau_slider": request.tau_slider,
                "evaluation_id": response.evaluation.evaluation_id,
                "spans_count": len(response.evaluation.spans),
                "violations_count": len(response.evaluation.violations),
                "clean_text": response.clean_text,
                "delta_summary": response.delta_summary
            }
            
            # Store in database
            collection = self.db.get_collection("evaluation_results")
            await collection.insert_one(document)
            
            logger.info(f"Successfully stored evaluation result for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error storing evaluation result: {str(e)}", exc_info=True)
