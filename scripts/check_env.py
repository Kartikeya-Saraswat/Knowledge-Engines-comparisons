import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("GROQ_API_KEY is configured.")
else:
    print("GROQ_API_KEY is missing. Check your .env file.")