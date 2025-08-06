#!/bin/bash

# Stop any running Python processes
echo "Stopping any running Python processes..."
pkill -f "python"

# Kill any process using port 8000
echo "Killing processes using port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# Verify port is free
echo "Checking if port 8000 is available..."
if lsof -i :8000 >/dev/null; then
    echo "❌ Port 8000 is still in use. Please close the application using it and try again."
    exit 1
else
    echo "✅ Port 8000 is available"
fi

# Start the backend server
echo "Starting backend server..."
cd backend
python3 -u server.py --port 8000 --api-prefix /api 2>&1 | tee server.log
