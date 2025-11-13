#!/usr/bin/env python3
import requests
import os

base_url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image"
code = "G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

if os.path.exists("test.jpg"):
    print("Testing action parameter...")
    
    # Test with action in URL
    url_with_action = f"{base_url}?code={code}&action=dresscode"
    print(f"URL: {url_with_action}")
    
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(url_with_action, files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if 'results' in result:
                print("✅ Action parameter works!")
                for res in result['results']:
                    print(f"Analysis: {res}")
            else:
                print("❌ No results - just upload")
else:
    print("❌ test.jpg not found")