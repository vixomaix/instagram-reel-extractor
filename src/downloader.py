"""
Instagram Reel Downloader Module

Handles downloading Instagram Reels using yt-dlp
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import yt_dlp

from . import config

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a reel download"""
    success: bool
    video_path: Optional[Path]
    metadata: Dict
    reel_id: str
    error: Optional[str] = None


class ReelDownloader:
    """Downloads Instagram Reels using yt-dlp"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_reel_id(self, url: str) -> Optional[str]:
        """
        Extract reel ID from various Instagram URL formats
        
        Supports:
        - https://www.instagram.com/reel/ABC123/
        - https://instagram.com/reel/ABC123/
        - https://www.instagram.com/reels/ABC123/
        """
        patterns = [
            r'instagram\.com/reel[s]?/([A-Za-z0-9_-]+)',
            r'instagram\.com/p/([A-Za-z0-9_-]+)',  # Shortcode format
            r'reel[s]?/([A-Za-z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _get_ydl_options(self, reel_dir: Path) -> Dict:
        """Get yt-dlp options for downloading"""
        return {
            'format': 'best',
            'outtmpl': str(reel_dir / 'video.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'writeinfojson': True,
            'writethumbnail': True,
            'user_agent': config.USER_AGENT,
            'cookiesfrombrowser': None,  # Can be set to ('chrome',) if needed
            'http_headers': {
                'User-Agent': config.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            },
            'socket_timeout': config.DOWNLOAD_TIMEOUT,
        }
    
    def download(self, url: str, reel_id: Optional[str] = None) -> DownloadResult:
        """
        Download an Instagram reel
        
        Args:
            url: Instagram reel URL
            reel_id: Optional reel ID (will be extracted from URL if not provided)
            
        Returns:
            DownloadResult with video path and metadata
        """
        # Extract reel ID
        if not reel_id:
            reel_id = self.extract_reel_id(url)
        
        if not reel_id:
            return DownloadResult(
                success=False,
                video_path=None,
                metadata={},
                reel_id="",
                error="Could not extract reel ID from URL"
            )
        
        logger.info(f"Downloading reel: {reel_id}")
        
        # Create reel-specific directory
        reel_dir = self.output_dir / reel_id
        reel_dir.mkdir(parents=True, exist_ok=True)
        
        ydl_opts = self._get_ydl_options(reel_dir)
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=True)
                
                if not info:
                    return DownloadResult(
                        success=False,
                        video_path=None,
                        metadata={},
                        reel_id=reel_id,
                        error="Could not extract video information"
                    )
                
                # Find the downloaded video file
                video_path = None
                for ext in ['mp4', 'webm', 'mkv']:
                    potential_path = reel_dir / f"video.{ext}"
                    if potential_path.exists():
                        video_path = potential_path
                        break
                
                if not video_path:
                    # Try to find any video file in the directory
                    for file in reel_dir.iterdir():
                        if file.suffix.lower() in ['.mp4', '.webm', '.mkv', '.mov']:
                            video_path = file
                            break
                
                # Load metadata from info json if available
                metadata = dict(info)
                info_json_path = reel_dir / "video.info.json"
                if info_json_path.exists():
                    try:
                        with open(info_json_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load info.json: {e}")
                
                logger.info(f"Successfully downloaded reel: {reel_id}")
                
                return DownloadResult(
                    success=True,
                    video_path=video_path,
                    metadata=metadata,
                    reel_id=reel_id,
                    error=None
                )
                
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Download error: {e}")
            return DownloadResult(
                success=False,
                video_path=None,
                metadata={},
                reel_id=reel_id,
                error=f"Download failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return DownloadResult(
                success=False,
                video_path=None,
                metadata={},
                reel_id=reel_id,
                error=f"Unexpected error: {str(e)}"
            )
    
    def get_video_info(self, url: str) -> Dict:
        """
        Get video information without downloading
        
        Args:
            url: Instagram reel URL
            
        Returns:
            Dictionary with video metadata
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'user_agent': config.USER_AGENT,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return dict(info) if info else {}
        except Exception as e:
            logger.error(f"Error extracting info: {e}")
            return {}


if __name__ == "__main__":
    # Test the downloader
    logging.basicConfig(level=logging.INFO)
    
    downloader = ReelDownloader()
    
    # Test URL extraction
    test_urls = [
        "https://www.instagram.com/reel/ABC123/",
        "https://instagram.com/reel/ABC123/",
        "https://www.instagram.com/reels/ABC123/",
    ]
    
    for url in test_urls:
        reel_id = downloader.extract_reel_id(url)
        print(f"URL: {url}")
        print(f"Reel ID: {reel_id}")
        print()