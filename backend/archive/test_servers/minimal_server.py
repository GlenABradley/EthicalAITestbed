"""
Minimal test server for debugging the hanging issue.
This version only includes a basic health check endpoint.
"""
import logging
import time
from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint"""
    logger.info("Health check started")
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "message": "Basic health check"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("Health check completed")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting minimal server...")
    uvicorn.run("minimal_server:app", host="0.0.0.0", port=8003, log_level="info")
