"""
Debug script to isolate the hanging issue.
This runs a simple FastAPI app with minimal dependencies.
"""
import sys
import logging

# Configure logging to show timestamps and be more verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Starting debug script ===")
    
    # Test basic Python functionality
    logger.info("Testing basic Python functionality...")
    try:
        import os
        import time
        import asyncio
        logger.info("Basic imports successful")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return 1
    
    # Test FastAPI import
    logger.info("\nTesting FastAPI import...")
    try:
        from fastapi import FastAPI
        logger.info("FastAPI import successful")
    except ImportError as e:
        logger.error(f"Failed to import FastAPI: {e}")
        return 1
    
    # Run a simple FastAPI app
    logger.info("\nCreating minimal FastAPI app...")
    try:
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
            
        @app.get("/test")
        async def test():
            return {"status": "ok"}
            
        logger.info("FastAPI app created successfully")
        
        # Don't actually run the server, just verify we can create it
        logger.info("FastAPI app creation test passed")
        
    except Exception as e:
        logger.error(f"Error creating FastAPI app: {e}", exc_info=True)
        return 1
    
    logger.info("\n=== Debug script completed successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
