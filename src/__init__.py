"""
Instagram Reel Extractor and Analysis Tool v2

A comprehensive tool for downloading Instagram Reels, extracting metadata,
analyzing video content, recipe extraction, and providing AI-powered insights.

Supports multiple AI providers: OpenAI, Anthropic, Gemini, Ollama, LM Studio
"""

__version__ = "2.0.0"
__author__ = "Instagram Reel Extractor Team"

from .main import ReelExtractor
from .main_v2 import ReelExtractorV2
from .downloader import ReelDownloader
from .metadata import MetadataExtractor
from .video_processor import VideoProcessor
from .video_processor_v2 import EnhancedVideoProcessor
from .ai_analyzer import AIAnalyzer
from .ai_providers import AIProviderManager, AIProvider, AIResponse
from .cache import CacheManager
from .recipe_extractor import RecipeExtractor, Recipe, Ingredient, CookingStep

__all__ = [
    "ReelExtractor",
    "ReelExtractorV2",
    "ReelDownloader", 
    "MetadataExtractor",
    "VideoProcessor",
    "EnhancedVideoProcessor",
    "AIAnalyzer",
    "AIProviderManager",
    "AIProvider",
    "AIResponse",
    "CacheManager",
    "RecipeExtractor",
    "Recipe",
    "Ingredient",
    "CookingStep",
]