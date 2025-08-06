#!/bin/bash

# Stop any running backend servers
echo "Stopping any existing backend servers..."
pkill -f "python.*server.py"

# Change to backend directory
cd "$(dirname "$0")/backend"

# Set environment variables
export PORT=8000
export API_PREFIX=/api

# Start the server
echo "Starting backend server on port $PORT with API prefix $API_PREFIX..."
python3 server.py --port $PORT --api-prefix $API_PREFIX

# Check if server started successfully
if [ $? -ne 0 ]; then
    echo "Failed to start backend server"
    exit 1
fi
