#!/usr/bin/env python3
import requests
import os

base_url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image"
code = "G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

if os.path.exists("test.jpg"):
    # Test the exact URL format from the app
    action_url = f"{base_url}?code={code}&action=dresscode"
    print(f"Testing URL: {action_url}")
    
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(action_url, files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if 'results' in result and result['results']:
                print("✅ SUCCESS - Analysis working!")
            else:
                print("❌ FAIL - No analysis results")
        else:
            print("❌ FAIL - Request failed")
else:
    print("❌ test.jpg not found")