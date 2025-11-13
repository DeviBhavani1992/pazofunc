#!/usr/bin/env python3
import requests

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
blob_url = "https://pazouploadetest.blob.core.windows.net/images/test.jpg"

print("🔍 Testing fixed analysis...")

response = requests.post(url, params={'action': 'dresscode', 'blob_url': blob_url})
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    result = response.json()
    if 'results' in result:
        print("✅ Analysis working!")
    else:
        print("❌ No results in response")