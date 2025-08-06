"""
Evaluation Response DTO for the Ethical AI Testbed.

This module defines the data transfer object for evaluation responses.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field

from core.domain.entities.ethical_evaluation import EthicalEvaluation

class EvaluationResponse(BaseModel):
    """
    Defines the structure for ethical evaluation responses, 
    directly mirroring the EthicalEvaluation class.
    """
    evaluation: EthicalEvaluation = Field(description="The detailed ethical evaluation results.")
    clean_text: str = Field(description="The processed, ethically compliant text.")
    delta_summary: Dict[str, int] = Field(description="A summary of the changes made to the text.")
