"""
Main Module - CLI Interface and Orchestrator
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from . import config
from .downloader import ReelDownloader
from .metadata import MetadataExtractor
from .video_processor import VideoProcessor
from .ai_analyzer import AIAnalyzer


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=config.LOG_FORMAT)


class ReelExtractor:
    """Main class for extracting and analyzing Instagram Reels"""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        verbose: bool = False
    ):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        self.downloader = ReelDownloader(self.output_dir)
        self.metadata_extractor = MetadataExtractor()
        self.video_processor = VideoProcessor(self.output_dir)
        self.ai_analyzer = AIAnalyzer(api_key)
    
    def extract_reel(
        self,
        url: str,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        use_ai: bool = True,
        transcribe: bool = True,
        keep_video: bool = False
    ) -> Dict[str, Any]:
        """Complete reel extraction and analysis pipeline"""
        self.logger.info(f"Starting extraction for: {url}")
        
        result = {
            "reel_id": None,
            "url": url,
            "downloaded_at": datetime.now().isoformat(),
            "success": False,
            "metadata": {},
            "video_analysis": {},
            "ai_analysis": {},
            "errors": []
        }
        
        # Step 1: Download
        self.logger.info("Step 1/4: Downloading reel...")
        download_result = self.downloader.download(url)
        
        if not download_result.success:
            error_msg = f"Download failed: {download_result.error}"
            self.logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
        
        result["reel_id"] = download_result.reel_id
        
        try:
            # Step 2: Metadata
            self.logger.info("Step 2/4: Extracting metadata...")
            metadata = self.metadata_extractor.extract(
                download_result.metadata,
                download_result.reel_id
            )
            result["metadata"] = self.metadata_extractor.to_dict(metadata)
            
            # Step 3: Video processing
            self.logger.info("Step 3/4: Processing video...")
            if download_result.video_path:
                video_analysis = self.video_processor.analyze_video(
                    download_result.video_path,
                    num_frames=num_frames,
                    transcribe=transcribe
                )
                
                result["video_analysis"] = {
                    "frames_extracted": video_analysis.frames_extracted,
                    "frame_paths": [str(p) for p in video_analysis.frame_paths],
                    "has_audio": video_analysis.has_audio,
                    "audio_transcription": video_analysis.audio_transcription,
                    "scene_changes": video_analysis.scene_changes,
                    "duration": video_analysis.duration,
                    "width": video_analysis.width,
                    "height": video_analysis.height,
                    "fps": video_analysis.fps
                }
            else:
                result["video_analysis"] = {"error": "No video file"}
                video_analysis = None
            
            # Step 4: AI Analysis
            if use_ai and video_analysis and video_analysis.frame_paths:
                self.logger.info("Step 4/4: Performing AI analysis...")
                ai_result = self.ai_analyzer.analyze_frames(
                    frame_paths=video_analysis.frame_paths,
                    caption=result["metadata"].get("caption", ""),
                    transcription=video_analysis.audio_transcription,
                    metadata=result["metadata"]
                )
                result["ai_analysis"] = self.ai_analyzer.to_dict(ai_result)
            elif not use_ai:
                result["ai_analysis"] = {"skipped": True}
            else:
                result["ai_analysis"] = {"skipped": True, "reason": "No frames"}
            
            self._save_results(result, download_result.reel_id)
            result["success"] = True
            self.logger.info(f"Extraction complete: {download_result.reel_id}")
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.logger.error(error_msg)
            result["errors"].append(error_msg)
        
        finally:
            if download_result.video_path and not keep_video:
                self.video_processor.cleanup(download_result.video_path, keep_video=False)
        
        return result
    
    def _save_results(self, result: Dict, reel_id: str):
        """Save results to JSON file"""
        output_file = self.output_dir / reel_id / "analysis_report.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to: {output_file}")
    
    def batch_process(
        self,
        urls: list,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        use_ai: bool = True
    ) -> list:
        """Process multiple reels"""
        results = []
        for i, url in enumerate(urls, 1):
            self.logger.info(f"Processing {i}/{len(urls)}: {url}")
            result = self.extract_reel(url, num_frames, use_ai)
            results.append(result)
        return results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Instagram Reel Extractor and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://www.instagram.com/reel/ABC123/"
  %(prog)s --url "https://www.instagram.com/reel/ABC123/" --no-ai
  %(prog)s --batch urls.txt --frames 10
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--url', help='Instagram reel URL')
    input_group.add_argument('-b', '--batch', help='File with URLs (one per line)')
    
    parser.add_argument('-o', '--output', default='./output', help='Output directory')
    parser.add_argument('-f', '--frames', type=int, default=config.DEFAULT_FRAME_COUNT,
                       help='Number of frames to extract')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI analysis')
    parser.add_argument('--no-transcribe', action='store_true', help='Skip transcription')
    parser.add_argument('-k', '--keep-video', action='store_true', help='Keep video file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--api-key', help='OpenAI API key')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    errors = config.validate_config()
    if errors:
        for error in errors:
            print(f"Configuration Error: {error}", file=sys.stderr)
        sys.exit(1)
    
    api_key = args.api_key or config.OPENAI_API_KEY
    
    extractor = ReelExtractor(
        output_dir=Path(args.output),
        api_key=api_key,
        verbose=args.verbose
    )
    
    if args.url:
        result = extractor.extract_reel(
            url=args.url,
            num_frames=args.frames,
            use_ai=not args.no_ai,
            transcribe=not args.no_transcribe,
            keep_video=args.keep_video
        )
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result['success'] else 1)
        
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"Batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        
        with open(batch_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Processing {len(urls)} URLs...")
        results = extractor.batch_process(urls=urls, num_frames=args.frames, use_ai=not args.no_ai)
        
        batch_output = Path(args.output) / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        successful = sum(1 for r in results if r['success'])
        print(f"Success: {successful}/{len(urls)}")
        print(f"Results saved to: {batch_output}")
        sys.exit(0 if successful == len(urls) else 1)


if __name__ == "__main__":
    main()