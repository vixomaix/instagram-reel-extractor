"""
Instagram Reel Extractor and Analysis Tool

A comprehensive tool for downloading Instagram Reels, extracting metadata,
analyzing video content, and providing AI-powered insights.
"""

__version__ = "1.0.0"
__author__ = "Instagram Reel Extractor Team"

from .main import ReelExtractor
from .downloader import ReelDownloader
from .metadata import MetadataExtractor
from .video_processor import VideoProcessor
from .ai_analyzer import AIAnalyzer

__all__ = [
    "ReelExtractor",
    "ReelDownloader", 
    "MetadataExtractor",
    "VideoProcessor",
    "AIAnalyzer",
]