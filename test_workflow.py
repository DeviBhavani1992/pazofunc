#!/usr/bin/env python3
import requests
import os

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

print("🔄 Testing complete workflow...")

# Step 1: Upload image
if os.path.exists("test.jpg"):
    print("\n1️⃣ Uploading image...")
    with open("test.jpg", "rb") as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            blob_url = result['blob_url']
            print(f"✅ Upload successful: {blob_url}")
            
            # Step 2: Analyze dress code using blob URL
            print("\n2️⃣ Analyzing dress code...")
            response = requests.post(f"{url}&action=dresscode&blob_url={blob_url}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Dress code analysis: {result}")
            else:
                print(f"❌ Dress code failed: {response.text}")
            
            # Step 3: Analyze dustbin using blob URL
            print("\n3️⃣ Analyzing dustbin...")
            response = requests.post(f"{url}&action=dustbin&blob_url={blob_url}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Dustbin analysis: {result}")
            else:
                print(f"❌ Dustbin failed: {response.text}")
        else:
            print(f"❌ Upload failed: {response.text}")
else:
    print("❌ test.jpg not found")