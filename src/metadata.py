"""
Metadata Extractor Module

Extracts and processes metadata from Instagram reels including:
- Caption and hashtags
- Engagement metrics (likes, comments, views)
- Author information
- Audio information
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AuthorInfo:
    """Author information"""
    username: str = ""
    full_name: str = ""
    user_id: str = ""
    followers: Optional[int] = None
    verified: bool = False
    profile_url: str = ""


@dataclass
class AudioInfo:
    """Audio information"""
    title: str = ""
    artist: str = ""
    audio_id: str = ""
    is_original: bool = False
    is_trending: bool = False
    duration: Optional[float] = None


@dataclass
class ReelMetadata:
    """Complete reel metadata"""
    reel_id: str = ""
    url: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    likes: Optional[int] = None
    comments: Optional[int] = None
    views: Optional[int] = None
    shares: Optional[int] = None
    author: AuthorInfo = field(default_factory=AuthorInfo)
    audio: AudioInfo = field(default_factory=AudioInfo)
    duration: Optional[float] = None
    upload_date: Optional[str] = None
    thumbnail_url: str = ""
    video_url: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    raw_data: Dict = field(default_factory=dict)


class MetadataExtractor:
    """Extracts metadata from yt-dlp info dictionary"""
    
    HASHTAG_PATTERN = re.compile(r'#(\w+)')
    MENTION_PATTERN = re.compile(r'@(\w+)')
    
    def extract(self, info_dict: Dict, reel_id: str = "") -> ReelMetadata:
        """Extract metadata from yt-dlp info dictionary"""
        metadata = ReelMetadata()
        metadata.reel_id = reel_id or info_dict.get('id', '')
        metadata.raw_data = info_dict
        
        metadata.url = info_dict.get('webpage_url', info_dict.get('url', ''))
        metadata.duration = info_dict.get('duration')
        metadata.upload_date = self._parse_upload_date(info_dict.get('upload_date'))
        metadata.thumbnail_url = info_dict.get('thumbnail', '')
        metadata.video_url = info_dict.get('url', '')
        metadata.width = info_dict.get('width')
        metadata.height = info_dict.get('height')
        
        title = info_dict.get('title', '')
        description = info_dict.get('description', '')
        metadata.caption = description or title
        
        text_to_parse = metadata.caption or title
        metadata.hashtags = self._extract_hashtags(text_to_parse)
        metadata.mentions = self._extract_mentions(text_to_parse)
        
        metadata.likes = self._extract_likes(info_dict)
        metadata.comments = self._extract_comments(info_dict)
        metadata.views = info_dict.get('view_count')
        metadata.shares = info_dict.get('repost_count')
        
        metadata.author = self._extract_author(info_dict)
        metadata.audio = self._extract_audio(info_dict)
        
        logger.info(f"Extracted metadata for reel: {metadata.reel_id}")
        return metadata
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        if not text:
            return []
        hashtags = self.HASHTAG_PATTERN.findall(text)
        return [f"#{tag}" for tag in hashtags]
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        if not text:
            return []
        mentions = self.MENTION_PATTERN.findall(text)
        return [f"@{mention}" for mention in mentions]
    
    def _parse_upload_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse upload date string to ISO format"""
        if not date_str:
            return None
        try:
            if len(date_str) == 8:
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            return date_str
        except:
            return date_str
    
    def _extract_likes(self, info_dict: Dict) -> Optional[int]:
        """Extract like count from various possible fields"""
        for key in ['like_count', 'likes', 'favorite_count', 'engagement_count']:
            if key in info_dict:
                return info_dict[key]
        return None
    
    def _extract_comments(self, info_dict: Dict) -> Optional[int]:
        """Extract comment count from various possible fields"""
        for key in ['comment_count', 'comments', 'nb_comments']:
            if key in info_dict:
                return info_dict[key]
        return None
    
    def _extract_author(self, info_dict: Dict) -> AuthorInfo:
        """Extract author information"""
        author = AuthorInfo()
        author.username = (info_dict.get('uploader') or 
                          info_dict.get('uploader_id') or 
                          info_dict.get('channel') or 
                          info_dict.get('creator', ''))
        author.full_name = (info_dict.get('uploader') or 
                           info_dict.get('channel', ''))
        author.user_id = info_dict.get('uploader_id', '')
        author.followers = info_dict.get('channel_follower_count')
        author.verified = info_dict.get('uploader_is_verified', False)
        if author.username:
            author.profile_url = f"https://www.instagram.com/{author.username}/"
        return author
    
    def _extract_audio(self, info_dict: Dict) -> AudioInfo:
        """Extract audio information"""
        audio = AudioInfo()
        track = info_dict.get('track', '')
        artist = info_dict.get('artist', '')
        if track:
            audio.title = track
            audio.artist = artist
        else:
            audio.title = "Original Audio"
            audio.artist = info_dict.get('uploader', '')
            audio.is_original = True
        audio.duration = info_dict.get('duration')
        return audio
    
    def to_dict(self, metadata: ReelMetadata) -> Dict:
        """Convert ReelMetadata to dictionary"""
        return {
            'reel_id': metadata.reel_id,
            'url': metadata.url,
            'caption': metadata.caption,
            'hashtags': metadata.hashtags,
            'mentions': metadata.mentions,
            'likes': metadata.likes,
            'comments': metadata.comments,
            'views': metadata.views,
            'shares': metadata.shares,
            'author': {
                'username': metadata.author.username,
                'full_name': metadata.author.full_name,
                'user_id': metadata.author.user_id,
                'followers': metadata.author.followers,
                'verified': metadata.author.verified,
                'profile_url': metadata.author.profile_url,
            },
            'audio': {
                'title': metadata.audio.title,
                'artist': metadata.audio.artist,
                'audio_id': metadata.audio.audio_id,
                'is_original': metadata.audio.is_original,
                'is_trending': metadata.audio.is_trending,
                'duration': metadata.audio.duration,
            },
            'duration': metadata.duration,
            'upload_date': metadata.upload_date,
            'thumbnail_url': metadata.thumbnail_url,
            'video_url': metadata.video_url,
            'width': metadata.width,
            'height': metadata.height,
        }