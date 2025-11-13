#!/usr/bin/env python3
import requests
import os

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

if os.path.exists("test.jpg"):
    print("Testing different file parameter names...")
    
    # Test 1: 'files' (current)
    with open("test.jpg", "rb") as f:
        files = {'files': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(url, files=files)
        print(f"Test 1 - 'files': {response.status_code} - {response.text[:100]}")
    
    # Test 2: 'file' (singular)
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(url, files=files)
        print(f"Test 2 - 'file': {response.status_code} - {response.text[:100]}")
    
    # Test 3: Multiple files with same name
    with open("test.jpg", "rb") as f1, open("test.jpg", "rb") as f2:
        files = [
            ('files', ('test1.jpg', f1, 'image/jpeg')),
            ('files', ('test2.jpg', f2, 'image/jpeg'))
        ]
        response = requests.post(url, files=files)
        print(f"Test 3 - multiple 'files': {response.status_code} - {response.text[:100]}")

else:
    print("❌ test.jpg not found")