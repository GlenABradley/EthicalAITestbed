"""
Lightweight health check implementation for the Ethical AI Testbed.
This version provides basic status information without deep system checks.
"""
from datetime import datetime
import time
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self):
        self.start_time = time.time()
        self.status = "initializing"
        self.checks = {}
        
    async def basic_check(self) -> Dict[str, Any]:
        """
        Perform a basic health check that doesn't depend on external services.
        
        Returns:
            Dict containing basic health information
        """
        try:
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": round(uptime, 2),
                "version": "1.0.0",
                "checks": {
                    "basic": {"status": "ok", "message": "Basic health check passed"}
                },
                "test_mode": True,
                "message": "Basic health check successful"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "test_mode": True,
                "message": "Health check failed due to an error"
            }

# Create a singleton instance
health_checker = HealthCheck()

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        result = await health_checker.basic_check()
        print("Health check result:", result)
    
    asyncio.run(test())
