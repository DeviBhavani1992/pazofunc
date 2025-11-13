#!/usr/bin/env python3
import requests

url = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
blob_url = "https://pazouploadetest.blob.core.windows.net/images/test.jpg"

response = requests.get(url, params={'action': 'dresscode', 'blob_url': blob_url})
print(f"GET request: {response.status_code} - {response.text}")