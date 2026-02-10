"""
Main Module v2 - Advanced CLI Interface and Orchestrator

Enhanced with:
- Recipe extraction
- Batch processing
- Multi-provider AI support
- Caching
- Async processing
"""

import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm

from . import config
from .config import validate_config, TaskType
from .downloader import ReelDownloader
from .metadata import MetadataExtractor
from .video_processor import VideoProcessor
from .video_processor_v2 import EnhancedVideoProcessor
from .ai_analyzer import AIAnalyzer
from .ai_providers import MultiProviderAI
from .config import AIProvider
from .cache import CacheManager
from .recipe_extractor import RecipeExtractor, Recipe


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=config.LOG_FORMAT)


class ReelExtractorV2:
    """Advanced reel extractor with recipe support and batch processing"""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        provider: Optional[AIProvider] = None,
        verbose: bool = False
    ):
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.use_cache = use_cache
        
        setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        # Validate config
        errors = validate_config()
        if errors:
            for error in errors:
                self.logger.error(error)
            raise ValueError("Configuration validation failed")
        
        # Initialize components
        self.downloader = ReelDownloader(self.output_dir)
        self.metadata_extractor = MetadataExtractor()
        self.video_processor = VideoProcessor(self.output_dir)
        self.video_processor_v2 = EnhancedVideoProcessor(self.output_dir)
        self.ai_manager = MultiProviderAI()
        self.cache = CacheManager() if use_cache else None
        self.recipe_extractor = RecipeExtractor(self.ai_manager)
        
        if provider:
            self.ai_manager.set_default_provider(provider)
    
    def extract_reel(
        self,
        url: str,
        num_frames: int = config.DEFAULT_FRAME_COUNT,
        use_ai: bool = True,
        transcribe: bool = True,
        extract_recipe: bool = True,
        keep_video: bool = False,
        skip_if_cached: bool = True
    ) -> Dict[str, Any]:
        """
        Complete reel extraction with all features
        
        Args:
            url: Instagram reel URL
            num_frames: Number of frames to extract
            use_ai: Whether to perform AI analysis
            transcribe: Whether to transcribe audio
            extract_recipe: Whether to extract recipe if cooking content
            keep_video: Whether to keep downloaded video
            skip_if_cached: Skip if already in cache
        """
        self.logger.info(f"Starting extraction for: {url}")
        
        reel_id = self.downloader.extract_reel_id(url)
        
        # Check cache
        if skip_if_cached and self.cache:
            cached = self.cache.get_reel(reel_id)
            if cached:
                self.logger.info(f"Found cached result for {reel_id}")
                return cached
        
        result = {
            "reel_id": reel_id,
            "url": url,
            "downloaded_at": datetime.now().isoformat(),
            "success": False,
            "metadata": {},
            "video_analysis": {},
            "ai_analysis": {},
            "recipe": None,
            "errors": []
        }
        
        # Step 1: Download
        self.logger.info("Step 1/5: Downloading reel...")
        download_result = self.downloader.download(url, reel_id)
        
        if not download_result.success:
            error_msg = f"Download failed: {download_result.error}"
            self.logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
        
        try:
            # Step 2: Metadata
            self.logger.info("Step 2/5: Extracting metadata...")
            metadata = self.metadata_extractor.extract(
                download_result.metadata,
                download_result.reel_id
            )
            result["metadata"] = self.metadata_extractor.to_dict(metadata)
            
            # Step 3: Video processing (enhanced)
            self.logger.info("Step 3/5: Processing video...")
            video_analysis = {}
            transcription = None
            
            if download_result.video_path:
                # Use enhanced processor for scene detection
                scenes = self.video_processor_v2.extract_scenes(
                    download_result.video_path,
                    max_scenes=num_frames
                )
                
                # Extract audio and transcribe
                if transcribe:
                    audio_path = self.video_processor_v2.extract_audio(
                        download_result.video_path
                    )
                    if audio_path:
                        transcription = self.video_processor_v2.transcribe_audio(
                            audio_path
                        )
                
                video_analysis = {
                    "scenes_extracted": len(scenes),
                    "has_transcription": transcription is not None,
                    "transcription": transcription,
                    "scenes": [
                        {
                            "timestamp": s.timestamp,
                            "description": s.description,
                            "frame_path": str(s.frame_path) if s.frame_path else None
                        }
                        for s in scenes
                    ]
                }
            
            result["video_analysis"] = video_analysis
            
            # Step 4: AI Analysis (multi-provider)
            if use_ai and download_result.video_path:
                self.logger.info("Step 4/5: AI analysis...")
                try:
                    ai_analysis = self._analyze_with_ai(
                        download_result.video_path,
                        scenes,
                        transcription,
                        result["metadata"]
                    )
                    result["ai_analysis"] = ai_analysis
                except Exception as e:
                    self.logger.error(f"AI analysis failed: {e}")
                    result["errors"].append(f"AI analysis: {str(e)}")
            
            # Step 5: Recipe extraction
            if extract_recipe and download_result.video_path:
                self.logger.info("Step 5/5: Checking for recipe content...")
                try:
                    recipe = self.recipe_extractor.extract_recipe(
                        video_path=download_result.video_path,
                        transcription=transcription,
                        metadata=result["metadata"]
                    )
                    
                    if recipe:
                        result["recipe"] = recipe.to_dict()
                        result["recipe_markdown"] = recipe.format_markdown()
                        self.logger.info(f"Recipe extracted: {recipe.title}")
                    else:
                        self.logger.info("No recipe content detected")
                        
                except Exception as e:
                    self.logger.error(f"Recipe extraction failed: {e}")
                    result["errors"].append(f"Recipe extraction: {str(e)}")
            
            result["success"] = len(result["errors"]) == 0
            
            # Save to cache
            if self.cache:
                self.cache.save_reel(reel_id, result)
            
            # Cleanup video if not keeping
            if not keep_video and download_result.video_path:
                try:
                    download_result.video_path.unlink()
                    self.logger.info("Cleaned up video file")
                except Exception as e:
                    self.logger.warning(f"Could not remove video: {e}")
            
        except Exception as e:
            self.logger.exception("Extraction failed")
            result["errors"].append(str(e))
        
        return result
    
    def _analyze_with_ai(
        self,
        video_path: Path,
        scenes: List,
        transcription: Optional[str],
        metadata: Dict
    ) -> Dict:
        """Perform AI analysis using multiple providers"""
        
        # Build prompt
        prompt = f"""Analyze this Instagram Reel and provide comprehensive insights.

Context:
- Caption: {metadata.get('caption', 'N/A')}
- Hashtags: {', '.join(metadata.get('hashtags', []))}
- Author: {metadata.get('author', {}).get('username', 'N/A')}

Audio Transcription:
{transcription or 'No transcription available'}

Provide analysis in JSON format:
{{
    "summary": "brief description of content",
    "content_category": "recipe/fashion/travel/etc",
    "mood": "happy/calm/energetic/etc",
    "objects": ["list of visible objects"],
    "activities": ["what's happening in the video"],
    "engagement_prediction": "high/medium/low with reasoning",
    "target_audience": "who would engage with this",
    "hashtag_suggestions": ["relevant hashtags"],
    "viral_potential": 0-100 score
}}"""
        
        from .ai_providers import VisionRequest
        
        # Get frame paths
        frame_paths = [s.frame_path for s in scenes if s.frame_path][:8]
        
        request = VisionRequest(
            images=frame_paths,
            prompt=prompt,
            max_tokens=2000,
            response_format="json"
        )
        
        response = asyncio.run(self.ai_manager.analyze_with_fallback(TaskType.SCENE_ANALYSIS, vision_request=request))
        
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"error": "Failed to parse AI response", "raw": response.content}
        else:
            return {"error": response.error or "AI analysis failed"}
    
    def batch_extract(
        self,
        urls: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple reels in batch
        
        Args:
            urls: List of Instagram reel URLs
            **kwargs: Additional arguments passed to extract_reel
            
        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction of {len(urls)} reels")
        
        results = []
        
        with tqdm(total=len(urls), desc="Processing reels") as pbar:
            for url in urls:
                try:
                    result = self.extract_reel(url, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {url}: {e}")
                    results.append({
                        "url": url,
                        "success": False,
                        "errors": [str(e)]
                    })
                pbar.update(1)
        
        # Save batch results
        batch_file = self.output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch results saved to {batch_file}")
        
        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Instagram Reel Extractor v2 - Advanced Content Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python -m src.main --url "https://www.instagram.com/reel/ABC123/"
  
  # Extract with recipe detection
  python -m src.main --url "URL" --recipe
  
  # Batch processing
  python -m src.main --batch urls.txt --output ./results
  
  # Use specific AI provider
  python -m src.main --url "URL" --provider anthropic
  
  # Skip AI analysis (faster)
  python -m src.main --url "URL" --no-ai
        """
    )
    
    parser.add_argument(
        "--url",
        help="Instagram reel URL to analyze"
    )
    
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="File containing URLs (one per line) for batch processing"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=config.DEFAULT_FRAME_COUNT,
        help=f"Number of frames to extract (default: {config.DEFAULT_FRAME_COUNT})"
    )
    
    parser.add_argument(
        "--recipe", "-r",
        action="store_true",
        default=True,
        help="Extract recipe if cooking content detected (default: True)"
    )
    
    parser.add_argument(
        "--no-recipe",
        action="store_true",
        help="Skip recipe extraction"
    )
    
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI analysis (faster)"
    )
    
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip audio transcription"
    )
    
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video file"
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "ollama"],
        help="AI provider to use (default: auto-fallback)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "json-ld"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.url and not args.batch:
        parser.error("Either --url or --batch must be provided")
    
    # Initialize extractor
    provider = AIProvider(args.provider) if args.provider else None
    
    extractor = ReelExtractorV2(
        output_dir=Path(args.output),
        use_cache=not args.no_cache,
        provider=provider,
        verbose=args.verbose
    )
    
    # Process
    try:
        if args.batch:
            # Batch mode
            with open(args.batch, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            results = extractor.batch_extract(
                urls,
                num_frames=args.frames,
                use_ai=not args.no_ai,
                transcribe=not args.no_transcribe,
                extract_recipe=not args.no_recipe,
                keep_video=args.keep_video
            )
            
            # Summary
            success_count = sum(1 for r in results if r.get("success"))
            recipe_count = sum(1 for r in results if r.get("recipe"))
            print(f"\n✓ Processed {len(results)} reels")
            print(f"✓ Successful: {success_count}")
            print(f"✓ Recipes found: {recipe_count}")
            
        else:
            # Single URL mode
            result = extractor.extract_reel(
                url=args.url,
                num_frames=args.frames,
                use_ai=not args.no_ai,
                transcribe=not args.no_transcribe,
                extract_recipe=not args.no_recipe,
                keep_video=args.keep_video
            )
            
            # Output
            if args.format == "markdown" and result.get("recipe"):
                # Output recipe as markdown
                from .recipe_extractor import Recipe
                recipe = Recipe(**result["recipe"])
                print(recipe.format_markdown())
            elif args.format == "json-ld" and result.get("recipe"):
                # Output recipe as JSON-LD
                from .recipe_extractor import Recipe
                recipe = Recipe(**result["recipe"])
                print(json.dumps(recipe.to_json_ld(), indent=2))
            else:
                # Standard JSON output
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            
            # Summary
            if result.get("success"):
                print(f"\n✓ Extraction successful")
                if result.get("recipe"):
                    recipe = result["recipe"]
                    print(f"✓ Recipe detected: {recipe.get('title', 'Unknown')}")
                    print(f"  Cuisine: {recipe.get('cuisine_type', 'Unknown')}")
                    print(f"  Time: {recipe.get('total_time', 'N/A')}")
            else:
                print(f"\n✗ Extraction failed")
                for error in result.get("errors", []):
                    print(f"  - {error}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
