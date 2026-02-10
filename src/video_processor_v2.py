"""
Enhanced Video Processor Module

Advanced video processing including:
- Scene change detection with content-aware analysis
- Smart keyframe extraction
- Motion analysis
- Face detection and emotion recognition
- Video compression
- Parallel frame processing
"""

import cv2
import numpy as np
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from functools import partial

try:
    from scenedetect import detect, ContentDetector, ThresholdDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from . import config
from .cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class SceneInfo:
    """Information about a detected scene"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    keyframe_path: Optional[Path] = None
    motion_score: float = 0.0
    complexity_score: float = 0.0


@dataclass
class FaceInfo:
    """Information about detected faces"""
    frame_idx: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    emotion: Optional[str] = None
    age_estimate: Optional[str] = None
    gender_estimate: Optional[str] = None


@dataclass
class MotionInfo:
    """Motion analysis information"""
    frame_idx: int
    timestamp: float
    motion_score: float
    dominant_direction: Optional[str] = None
    object_speeds: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnhancedVideoAnalysis:
    """Enhanced video analysis results"""
    duration: float
    width: int
    height: int
    fps: float
    total_frames: int
    
    # Scene information
    scenes: List[SceneInfo] = field(default_factory=list)
    keyframes: List[Path] = field(default_factory=list)
    
    # Analysis results
    faces: List[FaceInfo] = field(default_factory=list)
    motion_data: List[MotionInfo] = field(default_factory=list)
    has_audio: bool = False
    audio_transcription: Optional[str] = None
    
    # Quality metrics
    brightness_histogram: Optional[List[float]] = None
    color_histogram: Optional[Dict[str, List[float]]] = None
    
    # Text overlays detected via OCR
    text_overlays: List[Dict[str, Any]] = field(default_factory=list)


class SceneDetector:
    """Detects scene changes in video"""
    
    def __init__(self, threshold: float = config.SCENE_DETECTION_THRESHOLD):
        self.threshold = threshold
    
    def detect_scenes(self, video_path: Path) -> List[SceneInfo]:
        """Detect scenes using pyscenedetect if available, fallback to simple method"""
        if SCENEDETECT_AVAILABLE:
            return self._detect_with_pyscenedetect(video_path)
        else:
            return self._detect_with_opencv(video_path)
    
    def _detect_with_pyscenedetect(self, video_path: Path) -> List[SceneInfo]:
        """Use pyscenedetect for professional scene detection"""
        try:
            scene_list = detect(str(video_path), ContentDetector(threshold=self.threshold))
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            scenes = []
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                
                scenes.append(SceneInfo(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    duration=(end_frame - start_frame) / fps
                ))
            
            logger.info(f"Detected {len(scenes)} scenes using pyscenedetect")
            return scenes
            
        except Exception as e:
            logger.warning(f"PySceneDetect failed, falling back to OpenCV: {e}")
            return self._detect_with_opencv(video_path)
    
    def _detect_with_opencv(self, video_path: Path) -> List[SceneInfo]:
        """Fallback scene detection using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_frame = None
        scenes = []
        scene_start = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > self.threshold:
                    # Scene change detected
                    if frame_idx - scene_start > config.MIN_SCENE_LENGTH_FRAMES:
                        scenes.append(SceneInfo(
                            start_frame=scene_start,
                            end_frame=frame_idx,
                            start_time=scene_start / fps,
                            end_time=frame_idx / fps,
                            duration=(frame_idx - scene_start) / fps
                        ))
                    scene_start = frame_idx
            
            prev_frame = gray
            frame_idx += 1
        
        # Add final scene
        if frame_idx - scene_start > config.MIN_SCENE_LENGTH_FRAMES:
            scenes.append(SceneInfo(
                start_frame=scene_start,
                end_frame=frame_idx,
                start_time=scene_start / fps,
                end_time=frame_idx / fps,
                duration=(frame_idx - scene_start) / fps
            ))
        
        cap.release()
        logger.info(f"Detected {len(scenes)} scenes using OpenCV")
        return scenes


class MotionAnalyzer:
    """Analyzes motion in video"""
    
    def __init__(self):
        self.motion_threshold = 5.0
    
    def analyze_motion(self, video_path: Path, sample_interval: int = 5) -> List[MotionInfo]:
        """Analyze motion throughout the video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        prev_frame = None
        motion_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    frame_diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(frame_diff)
                    
                    # Calculate optical flow for direction
                    direction = self._calculate_optical_flow(prev_frame, gray)
                    
                    motion_data.append(MotionInfo(
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        motion_score=motion_score,
                        dominant_direction=direction
                    ))
                
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        return motion_data
    
    def _calculate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[str]:
        """Calculate dominant motion direction using optical flow"""
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate mean flow
            mean_flow_x = np.mean(flow[..., 0])
            mean_flow_y = np.mean(flow[..., 1])
            
            # Determine direction
            if abs(mean_flow_x) > abs(mean_flow_y):
                return "right" if mean_flow_x > 0 else "left"
            else:
                return "down" if mean_flow_y > 0 else "up"
                
        except Exception:
            return None


class FaceDetector:
    """Detects faces and analyzes emotions"""
    
    def __init__(self):
        self.mp_face_detection = None
        self.mp_face_mesh = None
        
        if MEDIAPIPE_AVAILABLE and config.FACE_DETECTION_ENABLED:
            try:
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
                )
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
                )
                logger.info("Initialized MediaPipe face detection")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")
    
    def detect_faces(self, frame: np.ndarray, frame_idx: int, fps: float) -> List[FaceInfo]:
        """Detect faces in a frame"""
        if not self.mp_face_detection:
            return []
        
        faces = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.mp_face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Estimate emotion based on face mesh
                    emotion = self._estimate_emotion(rgb_frame, (x, y, width, height))
                    
                    faces.append(FaceInfo(
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        bbox=(x, y, width, height),
                        confidence=detection.score[0],
                        emotion=emotion
                    ))
                    
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
        
        return faces
    
    def _estimate_emotion(self, rgb_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Estimate emotion from face (simplified)"""
        # This is a placeholder - for production, use a dedicated emotion model
        try:
            x, y, w, h = bbox
            face_roi = rgb_frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # Simple heuristic: analyze face brightness/contrast
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Very basic emotion estimation
            if brightness > 150 and contrast > 40:
                return "happy"
            elif brightness < 80:
                return "sad"
            elif contrast > 60:
                return "surprised"
            else:
                return "neutral"
                
        except Exception:
            return None


class VideoCompressor:
    """Compress videos for efficient processing"""
    
    def compress(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        max_width: int = config.FRAME_WIDTH,
        max_height: int = config.FRAME_HEIGHT,
        crf: int = config.VIDEO_COMPRESSION_CRF
    ) -> Path:
        """Compress video using ffmpeg"""
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_compressed.mp4"
        
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'scale={max_width}:{max_height}:force_original_aspect_ratio=decrease',
                '-c:v', 'libx264',
                '-preset', config.VIDEO_COMPRESSION_PRESET,
                '-crf', str(crf),
                '-c:a', 'copy',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and output_path.exists():
                original_size = video_path.stat().st_size
                compressed_size = output_path.stat().st_size
                ratio = (1 - compressed_size / original_size) * 100
                logger.info(f"Video compressed: {original_size / 1024 / 1024:.1f}MB -> {compressed_size / 1024 / 1024:.1f}MB ({ratio:.1f}% reduction)")
                return output_path
            else:
                logger.warning(f"Compression failed: {result.stderr}")
                return video_path
                
        except Exception as e:
            logger.error(f"Video compression error: {e}")
            return video_path


class EnhancedVideoProcessor:
    """Enhanced video processor with advanced features"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scene_detector = SceneDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.face_detector = FaceDetector()
        self.compressor = VideoCompressor()
        self.cache = get_cache()
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get comprehensive video information"""
        cache_key = self.cache.get_video_cache_key(video_path, "video_info")
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0,
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'bitrate': int(cap.get(cv2.CAP_PROP_BITRATE)),
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        
        self.cache.set(cache_key, info)
        return info
    
    def compress_video(self, video_path: Path) -> Path:
        """Compress video if enabled and needed"""
        if not config.VIDEO_COMPRESSION_ENABLED:
            return video_path
        
        # Check file size
        file_size_mb = video_path.stat().st_size / 1024 / 1024
        if file_size_mb <= config.MAX_VIDEO_SIZE_MB:
            return video_path
        
        return self.compressor.compress(video_path)
    
    def extract_smart_keyframes(
        self,
        video_path: Path,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[Path], List[SceneInfo]]:
        """Extract smart keyframes based on scene detection"""
        cache_key = self.cache.get_video_cache_key(video_path, f"keyframes_{num_frames}")
        cached = self.cache.get(cache_key)
        if cached:
            return cached['keyframes'], cached['scenes']
        
        if output_dir is None:
            output_dir = video_path.parent / "keyframes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get scenes
        scenes = self.scene_detector.detect_scenes(video_path)
        
        if not scenes or len(scenes) < num_frames:
            # Fall back to even distribution
            keyframes = self._extract_even_frames(video_path, num_frames, output_dir)
            scenes = []
        else:
            # Select representative frames from scenes
            keyframes = self._extract_from_scenes(video_path, scenes, num_frames, output_dir)
        
        result = {'keyframes': keyframes, 'scenes': scenes}
        self.cache.set(cache_key, result)
        
        return keyframes, scenes
    
    def _extract_even_frames(
        self,
        video_path: Path,
        num_frames: int,
        output_dir: Path
    ) -> List[Path]:
        """Extract evenly distributed frames"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
        
        keyframes = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = output_dir / f"keyframe_{i+1:03d}.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(frame_path), frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY])
                keyframes.append(frame_path)
        
        cap.release()
        return keyframes
    
    def _extract_from_scenes(
        self,
        video_path: Path,
        scenes: List[SceneInfo],
        num_frames: int,
        output_dir: Path
    ) -> List[Path]:
        """Extract keyframes from detected scenes"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Select scenes to sample from
        if len(scenes) <= num_frames:
            selected_scenes = scenes
        else:
            step = len(scenes) / num_frames
            selected_scenes = [scenes[int(i * step)] for i in range(num_frames)]
        
        keyframes = []
        
        for i, scene in enumerate(selected_scenes):
            # Get frame from middle of scene
            mid_frame = (scene.start_frame + scene.end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            
            if ret:
                frame_path = output_dir / f"keyframe_{i+1:03d}_scene.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(frame_path), frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY])
                keyframes.append(frame_path)
                scene.keyframe_path = frame_path
        
        cap.release()
        return keyframes
    
    def analyze_faces_in_video(
        self,
        video_path: Path,
        sample_interval: int = 10
    ) -> List[FaceInfo]:
        """Detect faces throughout the video"""
        if not config.FACE_DETECTION_ENABLED:
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_faces = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                faces = self.face_detector.detect_faces(frame, frame_idx, fps)
                all_faces.extend(faces)
            
            frame_idx += 1
        
        cap.release()
        return all_faces
    
    def analyze_video_comprehensive(
        self,
        video_path: Path,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        analyze_faces: bool = True,
        analyze_motion: bool = True,
        transcribe: bool = True
    ) -> EnhancedVideoAnalysis:
        """Perform comprehensive video analysis"""
        logger.info(f"Starting comprehensive analysis of {video_path}")
        
        # Get basic info
        video_info = self.get_video_info(video_path)
        
        # Extract smart keyframes
        keyframes, scenes = self.extract_smart_keyframes(video_path, num_frames)
        
        # Analyze faces
        faces = []
        if analyze_faces and config.FACE_DETECTION_ENABLED:
            faces = self.analyze_faces_in_video(video_path)
        
        # Analyze motion
        motion_data = []
        if analyze_motion:
            motion_data = self.motion_analyzer.analyze_motion(video_path)
        
        # Check for audio
        has_audio = self._has_audio_track(video_path)
        
        return EnhancedVideoAnalysis(
            duration=video_info['duration'],
            width=video_info['width'],
            height=video_info['height'],
            fps=video_info['fps'],
            total_frames=video_info['frame_count'],
            scenes=scenes,
            keyframes=keyframes,
            faces=faces,
            motion_data=motion_data,
            has_audio=has_audio
        )
    
    def _has_audio_track(self, video_path: Path) -> bool:
        """Check if video has audio track"""
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
    
    def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        time_offset: float = 0.5,  # Get frame at 50% of video
        width: int = config.THUMBNAIL_WIDTH,
        height: int = config.THUMBNAIL_HEIGHT
    ) -> Optional[Path]:
        """Generate video thumbnail"""
        if output_path is None:
            output_path = video_path.parent / "thumbnail.jpg"
        
        try:
            # Get video duration
            video_info = self.get_video_info(video_path)
            duration = video_info['duration']
            
            # Calculate time
            time_seconds = duration * time_offset
            
            cmd = [
                'ffmpeg',
                '-ss', str(time_seconds),
                '-i', str(video_path),
                '-vframes', '1',
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Generated thumbnail: {output_path}")
                return output_path
            else:
                logger.warning(f"Thumbnail generation failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Thumbnail generation error: {e}")
            return None
