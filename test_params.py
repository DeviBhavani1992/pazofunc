#!/usr/bin/env python3
import requests

base_url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image"
code = "G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
blob_url = "https://pazouploadetest.blob.core.windows.net/images/test.jpg"

print("Testing parameter formats...")

# Test with URL params
url = f"{base_url}?code={code}&action=dresscode&blob_url={blob_url}"
response = requests.post(url)
print(f"URL params: {response.status_code} - {response.text[:100]}")

# Test with requests params
response = requests.post(f"{base_url}?code={code}", params={'action': 'dresscode', 'blob_url': blob_url})
print(f"Requests params: {response.status_code} - {response.text[:100]}")

# Test with JSON body
response = requests.post(f"{base_url}?code={code}", json={'action': 'dresscode', 'blob_url': blob_url})
print(f"JSON body: {response.status_code} - {response.text[:100]}")