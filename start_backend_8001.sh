#!/bin/bash

# Stop any running Python processes
echo "Stopping any running Python processes..."
pkill -f "python"

# Kill any process using port 8001
echo "Clearing port 8001..."
lsof -ti :8001 | xargs kill -9 2>/dev/null || true

# Start the backend server on port 8001
echo "Starting backend server on port 8001..."
cd backend
python3 -u server.py --port 8001 --api-prefix /api 2>&1 | tee server_8001.log
