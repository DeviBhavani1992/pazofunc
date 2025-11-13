#!/usr/bin/env python3
import requests
import os

# Test the Azure Function upload
url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

# Check if test image exists
if os.path.exists("test.jpg"):
    print("📸 Found test.jpg, uploading...")
    
    with open("test.jpg", "rb") as f:
        files = {'files': ('test.jpg', f, 'image/jpeg')}
        
        try:
            response = requests.post(url, files=files)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Upload successful!")
                if 'uploaded' in result:
                    for blob_url in result['uploaded']:
                        print(f"🌐 Blob URL: {blob_url}")
            else:
                print(f"❌ Upload failed: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
else:
    print("❌ test.jpg not found")
    print("Available images:")
    for f in os.listdir("."):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"  - {f}")