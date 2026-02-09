"""
AI Analyzer Module

Uses OpenAI's GPT-4 Vision API to analyze Instagram reel content.
"""

import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)


@dataclass
class AIAnalysisResult:
    """Result of AI analysis"""
    summary: str = ""
    objects_detected: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    mood: str = ""
    content_category: str = ""
    engagement_prediction: str = ""
    hashtags_suggestions: List[str] = field(default_factory=list)
    target_audience: str = ""
    visual_style: str = ""
    audio_description: str = ""
    raw_response: Dict = field(default_factory=dict)


class AIAnalyzer:
    """Analyzes reel content using OpenAI Vision API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        elif not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. AI analysis disabled.")
        elif not self.api_key:
            logger.warning("OpenAI API key not set. AI analysis disabled.")
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _prepare_image_messages(self, frame_paths: List[Path]) -> List[Dict]:
        """Prepare image messages for OpenAI Vision API"""
        messages = []
        
        for frame_path in frame_paths[:8]:
            try:
                base64_image = self._encode_image(frame_path)
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    }
                })
            except Exception as e:
                logger.warning(f"Could not encode image {frame_path}: {e}")
        
        return messages
    
    def analyze_frames(
        self,
        frame_paths: List[Path],
        caption: str = "",
        transcription: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> AIAnalysisResult:
        """Analyze reel frames using GPT-4 Vision"""
        if not self.client:
            logger.warning("OpenAI client not available. Skipping AI analysis.")
            return AIAnalysisResult(summary="AI analysis not available")
        
        if not frame_paths:
            logger.warning("No frames provided for analysis")
            return AIAnalysisResult(summary="No frames available")
        
        logger.info(f"Analyzing {len(frame_paths)} frames with AI...")
        
        system_prompt = """You are an expert social media content analyst specializing in Instagram Reels.
Analyze the provided video frames and provide insights in JSON format:
{
    "summary": "Brief summary of what the video shows",
    "objects_detected": ["list", "of", "key", "objects"],
    "activities": ["list", "of", "activities", "shown"],
    "mood": "Overall mood/atmosphere",
    "content_category": "Category (comedy, education, fashion, etc)",
    "engagement_prediction": "low/medium/high with reasoning",
    "hashtags_suggestions": ["relevant", "hashtag", "suggestions"],
    "target_audience": "Who this content appeals to",
    "visual_style": "Description of visual aesthetic",
    "audio_description": "What type of audio would fit"
}"""

        user_content = []
        user_content.extend(self._prepare_image_messages(frame_paths))
        
        context_text = "Analyze this Instagram Reel:\n\n"
        if caption:
            context_text += f"Caption: {caption}\n\n"
        if transcription:
            context_text += f"Audio Transcription: {transcription}\n\n"
        if metadata:
            context_text += f"Duration: {metadata.get('duration', 'unknown')}s\n"
        
        context_text += "\nProvide JSON response as specified."
        
        user_content.append({"type": "text", "text": context_text})
        
        try:
            response = self.client.chat.completions.create(
                model=config.AI_MODEL_VISION,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result_dict = json.loads(content)
            
            return AIAnalysisResult(
                summary=result_dict.get('summary', ''),
                objects_detected=result_dict.get('objects_detected', []),
                activities=result_dict.get('activities', []),
                mood=result_dict.get('mood', ''),
                content_category=result_dict.get('content_category', ''),
                engagement_prediction=result_dict.get('engagement_prediction', ''),
                hashtags_suggestions=result_dict.get('hashtags_suggestions', []),
                target_audience=result_dict.get('target_audience', ''),
                visual_style=result_dict.get('visual_style', ''),
                audio_description=result_dict.get('audio_description', ''),
                raw_response=result_dict
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return AIAnalysisResult(summary="Error parsing AI response")
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            return AIAnalysisResult(summary=f"Error: {str(e)}")
    
    def to_dict(self, result: AIAnalysisResult) -> Dict:
        """Convert AIAnalysisResult to dictionary"""
        return {
            "summary": result.summary,
            "objects_detected": result.objects_detected,
            "activities": result.activities,
            "mood": result.mood,
            "content_category": result.content_category,
            "engagement_prediction": result.engagement_prediction,
            "hashtags_suggestions": result.hashtags_suggestions,
            "target_audience": result.target_audience,
            "visual_style": result.visual_style,
            "audio_description": result.audio_description,
        }