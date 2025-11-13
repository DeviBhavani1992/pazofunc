#!/usr/bin/env python3
import requests
import os

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

if os.path.exists("test.jpg"):
    print("📸 Testing upload with corrected parameter...")
    
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        
        response = requests.post(url, files=files)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"Response: {result}")
            
            if 'uploaded' in result:
                print("📋 Blob URLs:")
                for i, blob_url in enumerate(result['uploaded'], 1):
                    print(f"  {i}. {blob_url}")
            elif 'blob_url' in result:
                print(f"🌐 Blob URL: {result['blob_url']}")
        else:
            print(f"❌ Upload failed: {response.text}")
else:
    print("❌ test.jpg not found")