#!/usr/bin/env python3
import requests
import os

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="

# Test multiple uploads
images = ["test.jpg", "Ganapathi.png"]
uploaded_urls = []

print("📸 Testing multiple image uploads...")

for i, img_name in enumerate(images):
    if os.path.exists(img_name):
        print(f"\nUploading {img_name}...")
        
        with open(img_name, "rb") as f:
            files = {'file': (img_name, f, 'image/jpeg' if img_name.endswith('.jpg') else 'image/png')}
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                if 'blob_url' in result:
                    uploaded_urls.append(result['blob_url'])
                    print(f"✅ Uploaded: {result['blob_url']}")
            else:
                print(f"❌ Failed: {response.text}")

print(f"\n📋 Total uploaded: {len(uploaded_urls)} URLs")
for i, url in enumerate(uploaded_urls, 1):
    print(f"  {i}. {url}")

# Test dress code analysis
if uploaded_urls:
    print(f"\n👔 Testing dress code analysis on first image...")
    with open(images[0], "rb") as f:
        files = {'file': (images[0], f, 'image/jpeg')}
        response = requests.post(f"{url}?action=dresscode", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis result: {result}")
        else:
            print(f"❌ Analysis failed: {response.text}")