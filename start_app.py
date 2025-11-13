import os
os.environ["AZURE_FUNCTION_URL"] = "https://cavin-pazzo-20251015.azurewebsites.net/api/upload_image?code=G6Q_rqSNGitH6rfMhNgk4CzJnuizzAps5oBoYC-Gld24AzFusNduWg=="
os.system("./venv/bin/python -m streamlit run app.py --server.port 8503")