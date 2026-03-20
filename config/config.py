import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = "vectors.db"
UPLOAD_DIR = "uploaded_pdfs"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)