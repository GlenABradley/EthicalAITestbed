"""
Runner script for the refactored Ethical AI Testbed API server.

This script runs the refactored server implementation using the server_refactored.py module.
The refactored server implements a clean architecture with proper separation of concerns,
following domain-driven design principles and dependency injection patterns.

Usage:
    python run_refactored_server.py

Environment Variables:
    PORT: The port to run the server on (default: 8001)

Note:
    This is the preferred method for running the server in production environments.
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Set environment variables if needed
    os.environ["PORT"] = os.environ.get("PORT", "8001")
    
    # Run the server
    uvicorn.run(
        "server_refactored:app",
        host="0.0.0.0",
        port=int(os.environ["PORT"]),
        reload=True,
        log_level="info"
    )
