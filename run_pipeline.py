#!/usr/bin/env python3
"""
Complete pipeline runner for dress code detection system
"""
import subprocess
import sys
import os
import time

def run_inference_server():
    """Start the dress code inference server"""
    print("🚀 Starting dress code inference server...")
    return subprocess.Popen([
        "./venv/bin/python", 
        "scripts/inference_yoloe.py"
    ], cwd="/home/devi-1202324/Azure/pazofunc")

def run_dustbin_server():
    """Start the dustbin detection server"""
    print("🗑️ Starting dustbin detection server...")
    return subprocess.Popen([
        "./venv/bin/python", 
        "scripts/dustbin_detection.py"
    ], cwd="/home/devi-1202324/Azure/pazofunc")

def run_streamlit_app():
    """Start the Streamlit web app"""
    print("🌐 Starting Streamlit web app...")
    return subprocess.Popen([
        "streamlit", "run", "app.py", 
        "--server.port=8501", 
        "--server.address=0.0.0.0"
    ], cwd="/home/devi-1202324/Azure/pazofunc")

def main():
    # Set environment variables
    os.environ["DRESS_CODE_URL"] = "http://localhost:5000/check_dresscode"
    os.environ["DUSTBIN_DETECTION_URL"] = "http://localhost:5001/detect_dustbin"
    
    # Start inference servers
    inference_process = run_inference_server()
    dustbin_process = run_dustbin_server()
    time.sleep(3)  # Wait for servers to start
    
    # Start Streamlit app
    streamlit_process = run_streamlit_app()
    
    print("\n✅ Pipeline started successfully!")
    print("📱 Streamlit app: http://localhost:8501")
    print("🧠 Dress Code API: http://localhost:5000")
    print("🗑️ Dustbin Detection API: http://localhost:5001")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Wait for processes
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        inference_process.terminate()
        dustbin_process.terminate()
        streamlit_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()