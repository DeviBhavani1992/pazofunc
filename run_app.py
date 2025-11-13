#!/usr/bin/env python3
import os
import subprocess
import sys

# Set environment variable
os.environ["AZURE_FUNCTION_URL"] = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

# Run streamlit
subprocess.run([
    "./venv/bin/python", "-m", "streamlit", "run", "app.py",
    "--server.port", "8502",
    "--server.address", "0.0.0.0"
])