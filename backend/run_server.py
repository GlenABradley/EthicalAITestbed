import os
import sys
import socket
import subprocess
import time

PORT = 8000
API_PREFIX = '/api'

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_server():
    # Check if port is in use
    if is_port_in_use(PORT):
        print(f"Port {PORT} is already in use. Checking for existing server...")
        try:
            # Try to connect to the health endpoint
            import requests
            response = requests.get(f"http://localhost:{PORT}{API_PREFIX}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is already running on port {PORT}")
                return
        except:
            # If we can't connect, kill any Python processes using the port
            print("No healthy server found. Cleaning up...")
            subprocess.run(["pkill", "-f", f"python.*server.py"])
            time.sleep(1)  # Give it a moment to clean up

    print(f"🚀 Starting server on port {PORT} with API prefix '{API_PREFIX}'...")
    
    # Set environment variables
    os.environ["PORT"] = str(PORT)
    os.environ["API_PREFIX"] = API_PREFIX
    
    # Start the server
    try:
        subprocess.run(["python3", "server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
