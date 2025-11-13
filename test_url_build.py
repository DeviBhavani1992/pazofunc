#!/usr/bin/env python3
import requests
from urllib.parse import quote

base_url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
blob_url = "https://pazouploadetest.blob.core.windows.net/images/test.jpg"

# Test URL building approach
analysis_url = f"{base_url}&action=dresscode&blob_url={blob_url}"
print(f"Built URL: {analysis_url}")

response = requests.post(analysis_url)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Test with URL encoding
encoded_blob_url = quote(blob_url, safe='')
analysis_url_encoded = f"{base_url}&action=dresscode&blob_url={encoded_blob_url}"
print(f"\nEncoded URL: {analysis_url_encoded}")

response = requests.post(analysis_url_encoded)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")