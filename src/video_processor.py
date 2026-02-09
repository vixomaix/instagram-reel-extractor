"""
Video Processor Module

Handles video processing tasks including:
- Frame/keyframes extraction
- Audio extraction and transcription
- Scene change detection
- Video analysis
"""

import os
import cv2
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from . import config

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    """Result of video analysis"""
    frames_extracted: int
    frame_paths: List[Path]
    has_audio: bool
    audio_transcription: Optional[str]
    scene_changes: int
    duration: float
    width: int
    height: int
    fps: float


class VideoProcessor:
    """Processes video files for analysis"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.whisper_model = None
        
    def extract_frames(
        self, 
        video_path: Path, 
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """Extract evenly distributed frames from video"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_dir is None:
            output_dir = video_path.parent / "frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
        
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
        
        extracted_paths = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_filename = f"frame_{i+1:03d}.{config.FRAME_FORMAT}"
                frame_path = output_dir / frame_filename
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY]
                cv2.imwrite(str(frame_path), frame_rgb, encode_params)
                extracted_paths.append(frame_path)
        
        cap.release()
        logger.info(f"Extracted {len(extracted_paths)} frames to {output_dir}")
        return extracted_paths
    
    def extract_keyframes(
        self, 
        video_path: Path, 
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        output_dir: Optional[Path] = None,
        method: str = "scene"
    ) -> List[Path]:
        """Extract keyframes using scene detection or evenly distributed"""
        if method == "even":
            return self.extract_frames(video_path, num_frames, output_dir)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_dir is None:
            output_dir = video_path.parent / "frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        threshold = 30.0
        min_scene_length = 10
        
        prev_frame = None
        scene_frames = []
        frame_count = 0
        last_scene_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold and (frame_count - last_scene_frame) > min_scene_length:
                    scene_frames.append((frame_count, frame.copy()))
                    last_scene_frame = frame_count
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        if len(scene_frames) < num_frames:
            return self.extract_frames(video_path, num_frames, output_dir)
        
        step = len(scene_frames) / num_frames
        selected_scenes = [scene_frames[int(i * step)] for i in range(num_frames)]
        
        extracted_paths = []
        for i, (frame_idx, frame) in enumerate(selected_scenes):
            frame_filename = f"frame_{i+1:03d}.{config.FRAME_FORMAT}"
            frame_path = output_dir / frame_filename
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY]
            cv2.imwrite(str(frame_path), frame_rgb, encode_params)
            extracted_paths.append(frame_path)
        
        logger.info(f"Extracted {len(extracted_paths)} keyframes using scene detection")
        return extracted_paths
    
    def extract_audio(
        self, 
        video_path: Path, 
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Extract audio from video using ffmpeg"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_path is None:
            output_path = video_path.parent / f"audio.{config.AUDIO_FORMAT}"
        
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(config.AUDIO_SAMPLE_RATE),
                '-ac', '1',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Audio extracted to: {output_path}")
                return output_path
            else:
                logger.warning(f"Could not extract audio: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def transcribe_audio(
        self, 
        audio_path: Path,
        model_size: str = "base"
    ) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available. Install with: pip install openai-whisper")
            return None
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        try:
            if self.whisper_model is None:
                logger.info(f"Loading Whisper model: {model_size}")
                self.whisper_model = whisper.load_model(model_size)
            
            logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(str(audio_path))
            transcription = result.get('text', '').strip()
            
            if transcription:
                logger.info(f"Transcription completed: {len(transcription)} characters")
                return transcription
            else:
                logger.info("No speech detected in audio")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def has_audio_track(self, video_path: Path) -> bool:
        """Check if video has an audio track"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return 'audio' in result.stdout.lower()
            
        except Exception as e:
            logger.error(f"Error checking audio: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> Dict:
        """Get video file information"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {}
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0,
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
    
    def analyze_video(
        self,
        video_path: Path,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        transcribe: bool = True
    ) -> VideoAnalysisResult:
        """Perform complete video analysis"""
        logger.info(f"Analyzing video: {video_path}")
        
        video_info = self.get_video_info(video_path)
        frames_dir = video_path.parent / "frames"
        frame_paths = self.extract_keyframes(video_path, num_frames, frames_dir)
        
        has_audio = self.has_audio_track(video_path)
        transcription = None
        
        if has_audio and transcribe:
            audio_path = self.extract_audio(video_path)
            if audio_path:
                transcription = self.transcribe_audio(audio_path)
                try:
                    audio_path.unlink()
                except:
                    pass
        
        scene_changes = min(len(frame_paths), 10)
        
        return VideoAnalysisResult(
            frames_extracted=len(frame_paths),
            frame_paths=frame_paths,
            has_audio=has_audio,
            audio_transcription=transcription,
            scene_changes=scene_changes,
            duration=video_info.get('duration', 0),
            width=video_info.get('width', 0),
            height=video_info.get('height', 0),
            fps=video_info.get('fps', 0)
        )
    
    def cleanup(self, video_path: Path, keep_video: bool = False):
        """Clean up extracted files"""
        video_dir = video_path.parent
        
        frames_dir = video_dir / "frames"
        if frames_dir.exists():
            for f in frames_dir.iterdir():
                f.unlink()
            frames_dir.rmdir()
        
        if not keep_video and video_path.exists():
            video_path.unlink()
        
        logger.info("Cleanup completed")