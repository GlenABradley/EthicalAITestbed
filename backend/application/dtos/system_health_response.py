"""
System Health Response DTO for the Ethical AI Testbed.

This module defines the data transfer object for system health responses.
"""

from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class SystemHealthResponse(BaseModel):
    """
    System health and status information.
    """
    status: str = Field(description="Overall system status (healthy, degraded, error)")
    timestamp: datetime = Field(description="When the health check was performed")
    uptime_seconds: float = Field(description="System uptime in seconds")
    orchestrator_healthy: bool = Field(description="Whether the unified orchestrator is healthy")
    database_connected: bool = Field(description="Whether database connection is active")
    configuration_valid: bool = Field(description="Whether system configuration is valid")
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current performance metrics and statistics"
    )
    features_available: Dict[str, bool] = Field(
        default_factory=dict,
        description="Availability status of system features"
    )
