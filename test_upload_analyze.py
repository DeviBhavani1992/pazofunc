#!/usr/bin/env python3
import requests
import os

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

if os.path.exists("test.jpg"):
    print("Testing upload + analyze...")
    
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(f"{url}&action=dresscode", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if 'results' in result:
                print("✅ Upload + analyze works!")
            else:
                print("❌ No results in response")
else:
    print("❌ test.jpg not found")