# Instagram Reel Extractor v2 🎬

A comprehensive, AI-powered tool for extracting and analyzing Instagram Reels. Now with **recipe extraction**, **multi-provider AI support**, and **batch processing**.

## ✨ What's New in v2

### 🔥 Major Features
- **Recipe Extraction**: Automatically detect and extract cooking recipes from reels
- **Multi-Provider AI**: Support for OpenAI, Anthropic Claude, Google Gemini, Ollama, LM Studio
- **Batch Processing**: Process multiple reels efficiently
- **Smart Caching**: Redis/disk caching to avoid re-processing
- **Enhanced Video Analysis**: Scene detection, OCR, face detection

### 🎯 Specialized for Recipes
When a cooking video is detected, the tool extracts:
- Complete ingredient list with quantities
- Step-by-step cooking instructions
- Cooking times (prep, cook, total)
- Cuisine type and dietary tags
- Chef tips and variations
- Structured output (JSON, Markdown, JSON-LD)

## Features

### Core Capabilities
- ✅ **Download Instagram Reels**: Fast downloading via yt-dlp
- ✅ **Extract Metadata**: Caption, hashtags, likes, author info
- ✅ **Video Analysis**: Scene detection, keyframe extraction
- ✅ **Audio Transcription**: Speech-to-text with Whisper
- ✅ **Multi-Provider AI**: OpenAI, Anthropic, Gemini, Local models
- ✅ **Recipe Detection**: AI-powered cooking content identification
- ✅ **Recipe Extraction**: Structured recipe data from videos
- ✅ **Batch Processing**: Process multiple URLs efficiently
- ✅ **Smart Caching**: Avoid re-processing same content
- ✅ **Multiple Output Formats**: JSON, Markdown, JSON-LD

### AI Providers Supported
| Provider | Models | Best For |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4o-mini | General purpose, vision |
| **Anthropic** | Claude 3 Sonnet/Opus | Complex reasoning |
| **Google** | Gemini Pro Vision | Cost-effective |
| **Ollama** | Local models | Privacy, no API costs |
| **LM Studio** | Local models | Custom local models |

## Installation

### Prerequisites
- Python 3.8+
- ffmpeg (system dependency)
- (Optional) Redis for caching

```bash
# Install ffmpeg
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg
```

### Install Package

```bash
# Clone repository
git clone https://github.com/vixomaix/instagram-reel-extractor.git
cd instagram-reel-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Basic Extraction

```bash
# Extract a single reel
python -m src.main --url "https://www.instagram.com/reel/ABC123/"

# With recipe detection
python -m src.main --url "URL" --recipe

# Save video file
python -m src.main --url "URL" --keep-video
```

### Recipe Extraction (v2 Highlight)

```bash
# Extract recipe from cooking reel
python -m src.main --url "https://www.instagram.com/reel/COOKING/" --recipe

# Output as Markdown recipe
python -m src.main --url "URL" --format markdown

# Output as JSON-LD (for SEO/web)
python -m src.main --url "URL" --format json-ld
```

### Batch Processing

```bash
# Create file with URLs
echo "https://instagram.com/reel/ABC
https://instagram.com/reel/DEF
https://instagram.com/reel/GHI" > urls.txt

# Process all
python -m src.main --batch urls.txt --output ./results
```

### Multi-Provider AI

```bash
# Use specific provider
python -m src.main --url "URL" --provider anthropic

# Use local model (Ollama)
python -m src.main --url "URL" --provider ollama
```

### Advanced Options

```bash
# Skip AI analysis (faster)
python -m src.main --url "URL" --no-ai

# Disable caching
python -m src.main --url "URL" --no-cache

# Extract more frames
python -m src.main --url "URL" --frames 12

# Verbose logging
python -m src.main --url "URL" --verbose
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
# Required: At least one AI provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379/0

# Optional: Default settings
OUTPUT_DIR=./output
LOG_LEVEL=INFO
DEFAULT_PROVIDER=openai
CACHE_TTL=86400
```

### Provider Configuration

Configure providers in `src/config.py`:

```python
AI_PROVIDERS = {
    "openai": ProviderConfig(
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        fallback_models=["gpt-4o-mini"],
        cost_per_1k_tokens=0.01
    ),
    "anthropic": ProviderConfig(...),
    # ... more providers
}
```

## Output Format

### Standard JSON Output

```json
{
  "reel_id": "ABC123",
  "url": "https://instagram.com/reel/ABC123",
  "success": true,
  "metadata": {
    "caption": "Amazing pasta recipe! 🍝",
    "hashtags": ["#pasta", "#recipe", "#italian"],
    "author": {
      "username": "chef_mario",
      "followers": 50000
    }
  },
  "video_analysis": {
    "scenes_extracted": 8,
    "has_transcription": true,
    "transcription": "Today I'm making carbonara..."
  },
  "recipe": {
    "title": "Authentic Italian Carbonara",
    "cuisine_type": "Italian",
    "prep_time": "10 minutes",
    "cook_time": "15 minutes",
    "servings": 4,
    "ingredients": [
      {"name": "spaghetti", "quantity": "400", "unit": "g"},
      {"name": "eggs", "quantity": "4", "unit": "large"}
    ],
    "steps": [
      {"step_number": 1, "instruction": "Boil water..."}
    ]
  },
  "ai_analysis": {
    "content_category": "recipe",
    "viral_potential": 85,
    "engagement_prediction": "high"
  }
}
```

### Recipe Markdown Output

```markdown
# Authentic Italian Carbonara

**Cuisine:** Italian | **Type:** Pasta
**Difficulty:** Medium | **Servings:** 4
**Prep:** 10 minutes | **Cook:** 15 minutes | **Total:** 25 minutes

## Description
Classic Roman pasta dish made with eggs, cheese, and pancetta...

## Ingredients

- 400 g spaghetti
- 4 large eggs
- 100 g pancetta
- 50 g Pecorino Romano cheese

## Instructions

1. Boil a large pot of salted water...
2. In a separate bowl, whisk eggs...
3. Cook pancetta until crispy...

## Tips

- Use room temperature eggs
- Save pasta water for sauce
```

## Python API

### Basic Usage

```python
from src.main_v2 import ReelExtractorV2

# Initialize
extractor = ReelExtractorV2()

# Extract reel
result = extractor.extract_reel(
    url="https://instagram.com/reel/ABC123",
    extract_recipe=True
)

# Access recipe
if result.get("recipe"):
    print(result["recipe"]["title"])
    print(result["recipe_markdown"])
```

### Recipe Extraction

```python
from src.recipe_extractor import RecipeExtractor

extractor = RecipeExtractor()

recipe = extractor.extract_recipe(
    video_path="./video.mp4",
    transcription="Today we're making...",
    metadata={"caption": "Recipe video"}
)

if recipe:
    print(recipe.to_json())
    print(recipe.format_markdown())
    print(recipe.to_json_ld())  # For web/SEO
```

### Batch Processing

```python
urls = [
    "https://instagram.com/reel/1",
    "https://instagram.com/reel/2",
    "https://instagram.com/reel/3"
]

results = extractor.batch_extract(urls)

for result in results:
    if result.get("recipe"):
        print(f"Found recipe: {result['recipe']['title']}")
```

### Multi-Provider AI

```python
from src.ai_providers import AIProviderManager, AIProvider

# Initialize with fallback
manager = AIProviderManager()

# Set default provider
manager.set_default_provider(AIProvider.ANTHROPIC)

# Analyze with automatic fallback
response = manager.analyze_with_fallback(vision_request)
```

## Project Structure

```
instagram-reel-extractor/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── cache.py               # Caching layer (Redis/disk)
│   ├── downloader.py          # Reel downloading (yt-dlp)
│   ├── metadata.py            # Metadata extraction
│   ├── video_processor.py     # Basic video processing
│   ├── video_processor_v2.py  # Enhanced (scene detection, OCR)
│   ├── ai_analyzer.py         # Basic AI analysis
│   ├── ai_providers.py        # Multi-provider AI support
│   ├── recipe_extractor.py    # Recipe extraction (v2)
│   ├── main.py                # v1 CLI (legacy)
│   └── main_v2.py             # v2 CLI with all features
├── tests/
├── requirements.txt
├── README.md
└── .env.example
```

## Performance

### Optimization Features
- **Scene Detection**: Extracts key moments, not uniform frames
- **Smart Caching**: Avoids re-processing identical content
- **Async Processing**: Parallel frame analysis
- **Video Compression**: Smaller files for faster AI processing
- **Provider Fallback**: Switches providers if one fails

### Benchmarks
| Operation | Time | Notes |
|-----------|------|-------|
| Download (30s reel) | 3-5s | Depends on connection |
| Scene Extraction | 1-2s | 8 scenes |
| Audio Transcription | 5-10s | Whisper local |
| AI Analysis | 3-8s | Depends on provider |
| Recipe Extraction | 5-15s | Complex recipes take longer |
| **Total** | **15-30s** | End-to-end with AI |

## Troubleshooting

### Common Issues

**"Video unavailable"**
- Ensure the reel is public
- Instagram may have rate-limited the request
- Try again after a few minutes

**"No API key found"**
- Set at least one provider API key in `.env`
- Or use local models (Ollama) without API keys

**"Out of memory"**
- Reduce `--frames` count
- Process shorter videos
- Use disk caching instead of Redis

**Recipe not detected**
- Not all cooking videos have clear recipes
- Try with `--verbose` to see detection confidence
- Some content may be ambiguous

## Rate Limits & Ethics

- Respect Instagram's rate limits
- Use responsibly and ethically
- Don't use for unauthorized data scraping
- Comply with Instagram's Terms of Service
- Consider creator rights when extracting recipes

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Roadmap

### v2.1 (Planned)
- [ ] Instagram API integration (official)
- [ ] Real-time processing webhooks
- [ ] Browser extension
- [ ] Recipe import to popular apps (Paprika, etc.)

### v3.0 (Planned)
- [ ] Multi-platform support (TikTok, YouTube Shorts)
- [ ] AI-powered recipe video generation
- [ ] Nutritional analysis integration
- [ ] Community recipe database

## License

MIT License - see LICENSE file

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [OpenCV](https://opencv.org/) for video processing

---

**Made with ❤️ for content creators and food enthusiasts**
