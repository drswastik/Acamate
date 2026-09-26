import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory (AcaMate/)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables explicitly from root .env
load_dotenv(dotenv_path=ENV_PATH)

# Gemini API 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database 
DB_PATH = os.path.join(BASE_DIR, "data", "acamate.db")

# Safety Check 
if not GEMINI_API_KEY:
    raise ValueError("❌ Gemini API key not found! Please check your root .env file.")
