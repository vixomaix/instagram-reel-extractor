"""
Enhanced Configuration for Instagram Reel Extractor v2.0

Supports multi-provider AI, caching, and advanced features.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
load_dotenv()


class AIProvider(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"


class TaskType(str, Enum):
    """Task types for AI processing"""
    RECIPE_EXTRACTION = "recipe_extraction"
    SCENE_ANALYSIS = "scene_analysis"
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    SUMMARIZATION = "summarization"
    BRAND_DETECTION = "brand_detection"


# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / ".cache"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / ".temp"))

# API Keys for Multi-Provider AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Local AI Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
LM_STUDIO_HOST = os.getenv("LM_STUDIO_HOST", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# AI Provider Configuration
DEFAULT_AI_PROVIDER = AIProvider(os.getenv("DEFAULT_AI_PROVIDER", "openai"))

# Provider per task type (allows fine-grained control)
AI_PROVIDER_CONFIG: Dict[TaskType, List[AIProvider]] = {
    TaskType.RECIPE_EXTRACTION: [
        AIProvider(os.getenv("RECIPE_PROVIDER_1", "openai")),
        AIProvider(os.getenv("RECIPE_PROVIDER_2", "anthropic")),
    ],
    TaskType.SCENE_ANALYSIS: [
        AIProvider(os.getenv("SCENE_PROVIDER_1", "openai")),
        AIProvider(os.getenv("SCENE_PROVIDER_2", "gemini")),
    ],
    TaskType.OBJECT_DETECTION: [
        AIProvider(os.getenv("OBJECT_PROVIDER_1", "openai")),
        AIProvider(os.getenv("OBJECT_PROVIDER_2", "gemini")),
    ],
    TaskType.OCR: [
        AIProvider(os.getenv("OCR_PROVIDER_1", "openai")),
        AIProvider(os.getenv("OCR_PROVIDER_2", "gemini")),
    ],
    TaskType.SENTIMENT_ANALYSIS: [
        AIProvider(os.getenv("SENTIMENT_PROVIDER_1", "anthropic")),
        AIProvider(os.getenv("SENTIMENT_PROVIDER_2", "openai")),
    ],
    TaskType.ENGAGEMENT_PREDICTION: [
        AIProvider(os.getenv("ENGAGEMENT_PROVIDER_1", "openai")),
        AIProvider(os.getenv("ENGAGEMENT_PROVIDER_2", "anthropic")),
    ],
    TaskType.SUMMARIZATION: [
        AIProvider(os.getenv("SUMMARY_PROVIDER_1", "anthropic")),
        AIProvider(os.getenv("SUMMARY_PROVIDER_2", "openai")),
    ],
    TaskType.BRAND_DETECTION: [
        AIProvider(os.getenv("BRAND_PROVIDER_1", "openai")),
        AIProvider(os.getenv("BRAND_PROVIDER_2", "gemini")),
    ],
}

# AI Model Configuration
AI_MODELS: Dict[AIProvider, Dict[str, str]] = {
    AIProvider.OPENAI: {
        "vision": os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
        "text": os.getenv("OPENAI_TEXT_MODEL", "gpt-4o"),
        "embedding": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    },
    AIProvider.ANTHROPIC: {
        "vision": os.getenv("ANTHROPIC_VISION_MODEL", "claude-3-opus-20240229"),
        "text": os.getenv("ANTHROPIC_TEXT_MODEL", "claude-3-sonnet-20240229"),
    },
    AIProvider.GEMINI: {
        "vision": os.getenv("GEMINI_VISION_MODEL", "gemini-pro-vision"),
        "text": os.getenv("GEMINI_TEXT_MODEL", "gemini-pro"),
    },
    AIProvider.OLLAMA: {
        "vision": OLLAMA_MODEL,
        "text": OLLAMA_MODEL,
    },
    AIProvider.LM_STUDIO: {
        "vision": LM_STUDIO_MODEL,
        "text": LM_STUDIO_MODEL,
    },
}

# AI Settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Video Processing Settings
DEFAULT_FRAME_COUNT = int(os.getenv("DEFAULT_FRAME_COUNT", "8"))
MAX_FRAME_COUNT = int(os.getenv("MAX_FRAME_COUNT", "30"))
FRAME_FORMAT = os.getenv("FRAME_FORMAT", "jpg")
FRAME_QUALITY = int(os.getenv("FRAME_QUALITY", "95"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1920"))  # Max width for processing
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "1080"))  # Max height for processing

# Scene Detection Settings
SCENE_DETECTION_THRESHOLD = float(os.getenv("SCENE_DETECTION_THRESHOLD", "30.0"))
MIN_SCENE_LENGTH_FRAMES = int(os.getenv("MIN_SCENE_LENGTH_FRAMES", "10"))
SMART_KEYFRAME_ENABLED = os.getenv("SMART_KEYFRAME_ENABLED", "true").lower() == "true"

# Video Compression Settings
VIDEO_COMPRESSION_ENABLED = os.getenv("VIDEO_COMPRESSION_ENABLED", "true").lower() == "true"
VIDEO_COMPRESSION_CRF = int(os.getenv("VIDEO_COMPRESSION_CRF", "28"))  # 0-51, lower is better quality
VIDEO_COMPRESSION_PRESET = os.getenv("VIDEO_COMPRESSION_PRESET", "fast")
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "50"))

# Audio Settings
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "wav")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# OCR Settings
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
OCR_GPU_ENABLED = os.getenv("OCR_GPU_ENABLED", "false").lower() == "true"

# Caching Settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TYPE = os.getenv("CACHE_TYPE", "disk")  # 'disk', 'redis', or 'hybrid'
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours
CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", "1024"))

# Batch Processing Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
BATCH_PARALLEL_LIMIT = int(os.getenv("BATCH_PARALLEL_LIMIT", "3"))
BATCH_DELAY_SECONDS = float(os.getenv("BATCH_DELAY_SECONDS", "1.0"))

# Parallel Processing
PARALLEL_FRAME_ANALYSIS = os.getenv("PARALLEL_FRAME_ANALYSIS", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Async Processing
ASYNC_ENABLED = os.getenv("ASYNC_ENABLED", "true").lower() == "true"
ASYNC_WORKERS = int(os.getenv("ASYNC_WORKERS", "5"))

# Face Detection Settings
FACE_DETECTION_ENABLED = os.getenv("FACE_DETECTION_ENABLED", "true").lower() == "true"
FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))

# Object Detection Settings
OBJECT_DETECTION_ENABLED = os.getenv("OBJECT_DETECTION_ENABLED", "true").lower() == "true"
OBJECT_DETECTION_CONFIDENCE = float(os.getenv("OBJECT_DETECTION_CONFIDENCE", "0.4"))
OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "yolov8n.pt")

# Brand/Logo Detection
BRAND_DETECTION_ENABLED = os.getenv("BRAND_DETECTION_ENABLED", "true").lower() == "true"
BRAND_DETECTION_CONFIDENCE = float(os.getenv("BRAND_DETECTION_CONFIDENCE", "0.6"))

# Music Recognition
MUSIC_RECOGNITION_ENABLED = os.getenv("MUSIC_RECOGNITION_ENABLED", "false").lower() == "true"
ACR_CLOUD_HOST = os.getenv("ACR_CLOUD_HOST", "")
ACR_CLOUD_ACCESS_KEY = os.getenv("ACR_CLOUD_ACCESS_KEY", "")
ACR_CLOUD_ACCESS_SECRET = os.getenv("ACR_CLOUD_ACCESS_SECRET", "")

# Output Formats
OUTPUT_JSON_LD = os.getenv("OUTPUT_JSON_LD", "true").lower() == "true"
OUTPUT_SRT = os.getenv("OUTPUT_SRT", "true").lower() == "true"
OUTPUT_THUMBNAIL = os.getenv("OUTPUT_THUMBNAIL", "true").lower() == "true"
THUMBNAIL_WIDTH = int(os.getenv("THUMBNAIL_WIDTH", "640"))
THUMBNAIL_HEIGHT = int(os.getenv("THUMBNAIL_HEIGHT", "360"))

# Download Settings
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "300"))
VIDEO_QUALITY = os.getenv("VIDEO_QUALITY", "best")
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "180"))  # 3 minutes

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", None)

# Instagram Settings
INSTAGRAM_BASE_URL = "https://www.instagram.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# File naming
FILENAME_TEMPLATE = "%(id)s.%(ext)s"


@dataclass
class ProviderConfig:
    """Configuration for an AI provider"""
    provider: AIProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_vision: Optional[str] = None
    model_text: Optional[str] = None
    timeout: int = REQUEST_TIMEOUT
    max_retries: int = 3


def get_provider_config(provider: AIProvider) -> ProviderConfig:
    """Get configuration for a specific provider"""
    configs = {
        AIProvider.OPENAI: ProviderConfig(
            provider=AIProvider.OPENAI,
            api_key=OPENAI_API_KEY,
            model_vision=AI_MODELS[AIProvider.OPENAI]["vision"],
            model_text=AI_MODELS[AIProvider.OPENAI]["text"],
        ),
        AIProvider.ANTHROPIC: ProviderConfig(
            provider=AIProvider.ANTHROPIC,
            api_key=ANTHROPIC_API_KEY,
            model_vision=AI_MODELS[AIProvider.ANTHROPIC]["vision"],
            model_text=AI_MODELS[AIProvider.ANTHROPIC]["text"],
        ),
        AIProvider.GEMINI: ProviderConfig(
            provider=AIProvider.GEMINI,
            api_key=GOOGLE_API_KEY,
            model_vision=AI_MODELS[AIProvider.GEMINI]["vision"],
            model_text=AI_MODELS[AIProvider.GEMINI]["text"],
        ),
        AIProvider.OLLAMA: ProviderConfig(
            provider=AIProvider.OLLAMA,
            base_url=OLLAMA_HOST,
            model_vision=AI_MODELS[AIProvider.OLLAMA]["vision"],
            model_text=AI_MODELS[AIProvider.OLLAMA]["text"],
        ),
        AIProvider.LM_STUDIO: ProviderConfig(
            provider=AIProvider.LM_STUDIO,
            base_url=LM_STUDIO_HOST,
            model_vision=AI_MODELS[AIProvider.LM_STUDIO]["vision"],
            model_text=AI_MODELS[AIProvider.LM_STUDIO]["text"],
        ),
    }
    return configs.get(provider, ProviderConfig(provider=provider))


def get_providers_for_task(task_type: TaskType) -> List[AIProvider]:
    """Get list of providers for a task type (for fallback)"""
    providers = AI_PROVIDER_CONFIG.get(task_type, [DEFAULT_AI_PROVIDER])
    # Filter out providers that don't have credentials
    available = []
    for provider in providers:
        config = get_provider_config(provider)
        if config.api_key or provider in (AIProvider.OLLAMA, AIProvider.LM_STUDIO):
            available.append(provider)
    return available if available else [DEFAULT_AI_PROVIDER]


def ensure_directories():
    """Create necessary directories if they don't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> List[str]:
    """Validate configuration settings"""
    errors = []
    ensure_directories()
    
    # Check at least one AI provider is configured
    has_ai_provider = any([
        OPENAI_API_KEY,
        ANTHROPIC_API_KEY,
        GOOGLE_API_KEY,
    ])
    
    if not has_ai_provider:
        errors.append("No AI provider API key configured. Please set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
    
    # Check Redis if using Redis cache
    if CACHE_TYPE in ("redis", "hybrid"):
        try:
            import redis
        except ImportError:
            errors.append("Redis cache type selected but redis package not installed")
    
    # Check video processing dependencies
    try:
        import cv2
    except ImportError:
        errors.append("OpenCV (cv2) not installed. Required for video processing.")
    
    return errors
