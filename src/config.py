"""
Configuration settings for Instagram Reel Extractor
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Video Processing Settings
DEFAULT_FRAME_COUNT = 8
MAX_FRAME_COUNT = 20
FRAME_FORMAT = "jpg"
FRAME_QUALITY = 95

# Audio Settings
AUDIO_FORMAT = "wav"
AUDIO_SAMPLE_RATE = 16000

# AI Analysis Settings
AI_MODEL_VISION = "gpt-4o"
AI_MODEL_TEXT = "gpt-4o"
MAX_TOKENS = 2000
TEMPERATURE = 0.3

# Download Settings
DOWNLOAD_TIMEOUT = 300  # seconds
VIDEO_QUALITY = "best"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Instagram Settings
INSTAGRAM_BASE_URL = "https://www.instagram.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# File naming
FILENAME_TEMPLATE = "%(id)s.%(ext)s"


def ensure_directories():
    """Create necessary directories if they don't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")
    
    if not OUTPUT_DIR.exists():
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    return errors