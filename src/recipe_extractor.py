"""
Recipe Extraction Module

Specialized extraction for cooking/recipe content from Instagram Reels.
Extracts ingredients, steps, cooking time, cuisine type, etc.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import timedelta

from .ai_providers import AIProviderManager, VisionRequest
from .video_processor_v2 import EnhancedVideoProcessor, Scene

logger = logging.getLogger(__name__)


@dataclass
class Ingredient:
    """Recipe ingredient with quantity and notes"""
    name: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None
    alternative: Optional[str] = None


@dataclass
class CookingStep:
    """Single cooking step"""
    step_number: int
    instruction: str
    duration: Optional[str] = None  # e.g., "5 minutes", "until golden"
    temperature: Optional[str] = None  # e.g., "medium heat", "180°C"
    visual_cues: List[str] = field(default_factory=list)  # e.g., ["golden brown", "bubbling"]


@dataclass
class Recipe:
    """Complete recipe extracted from video"""
    # Basic info
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Classification
    cuisine_type: Optional[str] = None  # Indian, Italian, etc.
    meal_type: Optional[str] = None  # Breakfast, Lunch, Dinner, Snack
    dish_type: Optional[str] = None  # Curry, Bread, Dessert, etc.
    diet_type: List[str] = field(default_factory=list)  # Vegetarian, Vegan, Gluten-free, etc.
    difficulty: Optional[str] = None  # Easy, Medium, Hard
    
    # Time & servings
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    total_time: Optional[str] = None
    servings: Optional[int] = None
    
    # Content
    ingredients: List[Ingredient] = field(default_factory=list)
    steps: List[CookingStep] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)
    
    # Visual
    key_scenes: List[Dict] = field(default_factory=list)
    final_dish_image: Optional[str] = None
    
    # Source
    video_url: Optional[str] = None
    creator: Optional[str] = None
    extracted_at: Optional[str] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_json_ld(self) -> Dict:
        """Convert to JSON-LD format for SEO/web"""
        return {
            "@context": "https://schema.org",
            "@type": "Recipe",
            "name": self.title,
            "description": self.description,
            "recipeCuisine": self.cuisine_type,
            "recipeCategory": self.dish_type,
            "prepTime": self._iso_duration(self.prep_time),
            "cookTime": self._iso_duration(self.cook_time),
            "totalTime": self._iso_duration(self.total_time),
            "recipeYield": f"{self.servings} servings" if self.servings else None,
            "recipeIngredient": [
                f"{i.quantity or ''} {i.unit or ''} {i.name}".strip()
                for i in self.ingredients
            ],
            "recipeInstructions": [
                {
                    "@type": "HowToStep",
                    "text": step.instruction,
                    "url": f"#step-{step.step_number}"
                }
                for step in self.steps
            ],
            "suitableForDiet": [
                f"https://schema.org/{d}Diet" 
                for d in self.diet_type
            ] if self.diet_type else None
        }
    
    def _iso_duration(self, time_str: Optional[str]) -> Optional[str]:
        """Convert time string to ISO 8601 duration format"""
        if not time_str:
            return None
        
        # Parse "30 minutes", "1 hour", "5-10 minutes"
        patterns = [
            (r'(\d+)\s*min(?:ute)?s?', 'M'),
            (r'(\d+)\s*hour?s?', 'H'),
        ]
        
        duration = "PT"
        for pattern, unit in patterns:
            match = re.search(pattern, time_str.lower())
            if match:
                duration += f"{match.group(1)}{unit}"
        
        return duration if duration != "PT" else None
    
    def format_markdown(self) -> str:
        """Format recipe as Markdown"""
        lines = [
            f"# {self.title or 'Recipe'}",
            "",
            f"**Cuisine:** {self.cuisine_type or 'Unknown'} | **Type:** {self.dish_type or 'Unknown'}",
            f"**Difficulty:** {self.difficulty or 'Unknown'} | **Servings:** {self.servings or 'Unknown'}",
            f"**Prep:** {self.prep_time or 'N/A'} | **Cook:** {self.cook_time or 'N/A'} | **Total:** {self.total_time or 'N/A'}",
            "",
            "## Description",
            self.description or "",
            "",
            "## Ingredients",
            "",
        ]
        
        for ing in self.ingredients:
            qty = f"{ing.quantity} {ing.unit}" if ing.quantity and ing.unit else (ing.quantity or "")
            line = f"- {qty} {ing.name}".strip()
            if ing.notes:
                line += f" ({ing.notes})"
            lines.append(line)
        
        lines.extend(["", "## Instructions", ""])
        
        for step in self.steps:
            lines.append(f"{step.step_number}. {step.instruction}")
            if step.duration:
                lines.append(f"   *Duration: {step.duration}*")
        
        if self.tips:
            lines.extend(["", "## Tips", ""])
            for tip in self.tips:
                lines.append(f"- {tip}")
        
        return "\n".join(lines)


class RecipeExtractor:
    """Extract recipe information from Instagram Reels"""
    
    RECIPE_SYSTEM_PROMPT = """You are an expert recipe extraction system. Analyze the provided video frames and transcribed audio to extract complete recipe information.

Extract the following in structured format:
1. Recipe title and description
2. Cuisine type (Indian, Italian, Chinese, etc.)
3. Meal type (Breakfast, Lunch, Dinner, Snack)
4. Dish type (Curry, Bread, Dessert, Beverage, etc.)
5. Dietary information (Vegetarian, Vegan, Gluten-free, Jain, etc.)
6. Preparation time, cooking time, total time
7. Number of servings
8. Complete ingredient list with quantities
9. Step-by-step cooking instructions
10. Chef's tips and variations

Be precise with measurements and cooking times. If information is unclear, indicate uncertainty.

Respond in JSON format with this structure:
{
    "title": "...",
    "description": "...",
    "cuisine_type": "...",
    "meal_type": "...",
    "dish_type": "...",
    "diet_type": ["..."],
    "difficulty": "Easy/Medium/Hard",
    "prep_time": "...",
    "cook_time": "...",
    "total_time": "...",
    "servings": number,
    "ingredients": [
        {"name": "...", "quantity": "...", "unit": "...", "notes": "..."}
    ],
    "steps": [
        {"step_number": 1, "instruction": "...", "duration": "...", "temperature": "..."}
    ],
    "tips": ["..."],
    "variations": ["..."],
    "confidence_score": 0.0-1.0
}"""
    
    def __init__(self, ai_manager: Optional[AIProviderManager] = None):
        self.ai_manager = ai_manager or AIProviderManager()
        self.video_processor = EnhancedVideoProcessor()
        
    def is_recipe_content(self, video_path: Path, transcription: str) -> Tuple[bool, float]:
        """
        Determine if the video contains recipe/cooking content
        
        Returns:
            (is_recipe, confidence_score)
        """
        prompt = """Analyze this video content and determine if it's a cooking/recipe video.

Look for:
- Cooking actions (chopping, stirring, frying, baking)
- Ingredients being shown or mentioned
- Kitchen utensils and appliances
- Food preparation steps
- Recipe instructions in audio

Respond with JSON:
{
    "is_recipe": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
        
        try:
            # Get a few key frames for analysis
            scenes = self.video_processor.extract_scenes(video_path, max_scenes=3)
            if not scenes:
                return False, 0.0
            
            request = VisionRequest(
                images=[s.frame_path for s in scenes if s.frame_path],
                prompt=prompt,
                system_prompt="You are a content classification system. Be precise.",
                response_format="json"
            )
            
            response = self.ai_manager.analyze_with_fallback(request)
            
            if response.success:
                result = json.loads(response.content)
                return result.get("is_recipe", False), result.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error in recipe detection: {e}")
        
        return False, 0.0
    
    def extract_recipe(
        self,
        video_path: Path,
        transcription: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Recipe]:
        """
        Extract complete recipe from video
        
        Args:
            video_path: Path to downloaded video
            transcription: Optional audio transcription
            metadata: Optional video metadata (caption, etc.)
            
        Returns:
            Recipe object or None if not a recipe
        """
        logger.info(f"Extracting recipe from {video_path}")
        
        # First check if it's recipe content
        is_recipe, confidence = self.is_recipe_content(video_path, transcription or "")
        if not is_recipe or confidence < 0.5:
            logger.info(f"Not recipe content (confidence: {confidence})")
            return None
        
        # Extract scenes
        scenes = self.video_processor.extract_scenes(video_path)
        
        # Build context from metadata
        context = ""
        if metadata:
            caption = metadata.get("caption", "")
            hashtags = metadata.get("hashtags", [])
            context = f"Caption: {caption}\nHashtags: {', '.join(hashtags)}\n\n"
        
        if transcription:
            context += f"Audio Transcription:\n{transcription}\n\n"
        
        # Prepare vision request
        scene_images = [s.frame_path for s in scenes if s.frame_path][:8]  # Max 8 frames
        
        prompt = f"{context}Extract the complete recipe information from this cooking video."
        
        request = VisionRequest(
            images=scene_images,
            prompt=prompt,
            system_prompt=self.RECIPE_SYSTEM_PROMPT,
            response_format="json",
            max_tokens=4000
        )
        
        # Get AI response
        response = self.ai_manager.analyze_with_fallback(request)
        
        if not response.success:
            logger.error(f"Recipe extraction failed: {response.error}")
            return None
        
        try:
            data = json.loads(response.content)
            return self._parse_recipe_json(data, video_path, metadata, scenes)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse recipe JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing recipe: {e}")
            return None
    
    def _parse_recipe_json(
        self,
        data: Dict,
        video_path: Path,
        metadata: Optional[Dict],
        scenes: List[Scene]
    ) -> Recipe:
        """Parse recipe JSON into Recipe object"""
        
        # Parse ingredients
        ingredients = []
        for ing_data in data.get("ingredients", []):
            ingredients.append(Ingredient(
                name=ing_data.get("name", ""),
                quantity=ing_data.get("quantity"),
                unit=ing_data.get("unit"),
                notes=ing_data.get("notes"),
                alternative=ing_data.get("alternative")
            ))
        
        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            steps.append(CookingStep(
                step_number=step_data.get("step_number", len(steps) + 1),
                instruction=step_data.get("instruction", ""),
                duration=step_data.get("duration"),
                temperature=step_data.get("temperature"),
                visual_cues=step_data.get("visual_cues", [])
            ))
        
        # Build key scenes info
        key_scenes = []
        for scene in scenes:
            key_scenes.append({
                "timestamp": scene.timestamp,
                "description": scene.description,
                "frame_path": str(scene.frame_path) if scene.frame_path else None
            })
        
        # Get creator info from metadata
        creator = None
        if metadata and "author" in metadata:
            creator = metadata["author"].get("username") if isinstance(metadata["author"], dict) else None
        
        return Recipe(
            title=data.get("title"),
            description=data.get("description"),
            cuisine_type=data.get("cuisine_type"),
            meal_type=data.get("meal_type"),
            dish_type=data.get("dish_type"),
            diet_type=data.get("diet_type", []),
            difficulty=data.get("difficulty"),
            prep_time=data.get("prep_time"),
            cook_time=data.get("cook_time"),
            total_time=data.get("total_time"),
            servings=data.get("servings"),
            ingredients=ingredients,
            steps=steps,
            tips=data.get("tips", []),
            variations=data.get("variations", []),
            key_scenes=key_scenes,
            video_url=metadata.get("url") if metadata else None,
            creator=creator,
            extracted_at=datetime.now().isoformat(),
            confidence_score=data.get("confidence_score", 0.0)
        )
    
    def batch_extract(
        self,
        video_paths: List[Path],
        progress_callback=None
    ) -> List[Tuple[Path, Optional[Recipe]]]:
        """
        Extract recipes from multiple videos
        
        Args:
            video_paths: List of video file paths
            progress_callback: Optional callback(progress, current, total)
            
        Returns:
            List of (video_path, recipe) tuples
        """
        results = []
        total = len(video_paths)
        
        for i, path in enumerate(video_paths):
            try:
                recipe = self.extract_recipe(path)
                results.append((path, recipe))
                
                if progress_callback:
                    progress_callback((i + 1) / total, i + 1, total)
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append((path, None))
        
        return results


# Convenience function
def extract_recipe_from_reel(
    video_path: Path,
    transcription: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Optional[Recipe]:
    """Quick function to extract recipe from a reel video"""
    extractor = RecipeExtractor()
    return extractor.extract_recipe(video_path, transcription, metadata)
