#!/usr/bin/env python3
import requests
from urllib.parse import quote

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
blob_url = "https://pazouploadetest.blob.core.windows.net/images/test.jpg"

print("🔍 Testing analysis requests...")

# Test 1: Direct URL
print(f"\n1️⃣ Testing direct URL:")
response = requests.post(f"{url}&action=dresscode&blob_url={blob_url}")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 2: Encoded URL
print(f"\n2️⃣ Testing encoded URL:")
encoded_url = quote(blob_url, safe='')
response = requests.post(f"{url}&action=dresscode&blob_url={encoded_url}")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test 3: Using params
print(f"\n3️⃣ Testing with params:")
response = requests.post(url, params={'action': 'dresscode', 'blob_url': blob_url})
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")