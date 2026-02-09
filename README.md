# Instagram Reel Extractor and Analysis Tool

A comprehensive Python tool for downloading Instagram Reels, extracting metadata, analyzing video content, and providing AI-powered insights.

## Features

- **Download Instagram Reels**: Fast and reliable reel downloading using yt-dlp
- **Extract Metadata**: Caption, hashtags, likes, comments, author info, audio info
- **Video Analysis**: 
  - Extract keyframes from videos
  - Transcribe audio if speech is present
  - Analyze visual content (objects, scenes, actions)
- **AI Content Understanding**: GPT-4 Vision API for comprehensive content analysis
- **Structured Output**: JSON format with all extracted data

## Prerequisites

- Python 3.8+
- ffmpeg (system dependency)
- OpenAI API key

### Installing ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/instagram-reel-extractor.git
cd instagram-reel-extractor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python -m src.main --url "https://www.instagram.com/reel/ABC123/"
```

**With custom output directory:**
```bash
python -m src.main --url "https://www.instagram.com/reel/ABC123/" --output ./my_output
```

**Skip AI analysis (faster):**
```bash
python -m src.main --url "https://www.instagram.com/reel/ABC123/" --no-ai
```

**Extract specific number of frames:**
```bash
python -m src.main --url "https://www.instagram.com/reel/ABC123/" --frames 10
```

**Full options:**
```bash
python -m src.main --url "https://www.instagram.com/reel/ABC123/" \
                   --output ./output \
                   --frames 8 \
                   --keep-video \
                   --verbose
```

### As a Python Module

```python
from src.main import ReelExtractor

# Initialize extractor
extractor = ReelExtractor(output_dir="./output")

# Extract reel
result = extractor.extract_reel("https://www.instagram.com/reel/ABC123/")

# Access results
print(result['metadata']['caption'])
print(result['analysis']['ai_summary'])
```

## Output Structure

```
output/
├── reel_id/
│   ├── video.mp4              # Downloaded video (optional)
│   ├── metadata.json          # Extracted metadata
│   ├── frames/
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── audio_transcription.txt # Speech-to-text output
│   └── analysis_report.json   # Complete analysis
```

## JSON Output Format

```json
{
  "reel_id": "ABC123",
  "url": "https://www.instagram.com/reel/ABC123/",
  "downloaded_at": "2025-01-15T10:30:00",
  "metadata": {
    "caption": "Amazing sunset! 🌅",
    "hashtags": ["#sunset", "#nature", "#photography"],
    "likes": 1500,
    "comments": 45,
    "views": 10000,
    "author": {
      "username": "photographer_jane",
      "full_name": "Jane Doe",
      "followers": 50000
    },
    "audio": {
      "title": "Original Audio",
      "artist": "photographer_jane",
      "trending": false
    },
    "duration": 15.5,
    "upload_date": "2025-01-10"
  },
  "video_analysis": {
    "frames_extracted": 8,
    "frame_paths": [...],
    "has_audio": true,
    "audio_transcription": "Check out this amazing sunset...",
    "scene_changes": 3
  },
  "ai_analysis": {
    "summary": "A cinematic reel showcasing a vibrant orange sunset...",
    "objects_detected": ["sun", "ocean", "clouds", "palm trees"],
    "activities": ["sunset viewing"],
    "mood": "peaceful, inspirational",
    "content_category": "nature/photography",
    "engagement_prediction": "high",
    "hashtags_suggestions": ["#goldenhour", "#seascape"]
  }
}
```

## Project Structure

```
instagram-reel-extractor/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── downloader.py       # Reel downloading logic
│   ├── metadata.py         # Metadata extraction
│   ├── video_processor.py  # Video processing (frames, audio)
│   ├── ai_analyzer.py      # AI analysis with OpenAI
│   └── main.py             # CLI interface
├── tests/
│   └── test_extractor.py
├── requirements.txt
├── README.md
└── .env.example
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI analysis | Yes |
| `OUTPUT_DIR` | Default output directory | No (default: ./output) |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING) | No (default: INFO) |

## Rate Limits and Ethics

- Respect Instagram's rate limits
- Use responsibly and ethically
- Do not use for unauthorized data scraping
- Comply with Instagram's Terms of Service

## Troubleshooting

**Error: "Video unavailable"**
- Ensure the reel is public
- Check the URL is correct
- Instagram may have restricted access

**Error: "OpenAI API key invalid"**
- Verify your API key is set correctly
- Check your OpenAI account has available credits

**ffmpeg not found**
- Ensure ffmpeg is installed and in your PATH
- Try: `which ffmpeg` (Linux/Mac) or `where ffmpeg` (Windows)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for educational and research purposes only. Users are responsible for complying with Instagram's Terms of Service and applicable laws. The authors are not responsible for misuse of this tool.