# AI Cameraman

Automatically crop and edit water polo footage using AI to focus on the main action throughout the video.

## Overview

This system takes raw water polo MP4 footage and uses Gemini 2.5 Pro to analyze the video and identify bounding boxes around the main action. It then creates a smoothly cropped video that zooms in on the action throughout the game.

## Architecture

```
          ┌────────────┐  Files API      ┌─────────────────┐
raw.mp4 → │ google‑genai│ ─────────────► │ Gemini‑2.5 Pro  │
          └────────────┘  boxes JSON     └─────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  spline + Kalman    │  (SciPy, FilterPy)
        └─────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ FFmpeg crop+encode  │  (VideoToolbox)
        └─────────────────────┘
                 │
                 ▼
           smart‑crop.mp4
```

## Quick Start

### Prerequisites

1. **Gemini API key**: Get from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. **Python 3.12**: `python -m venv .venv && source .venv/bin/activate`
3. **FFmpeg with VideoToolbox**: `brew install ffmpeg`

### Installation

```bash
# Clone and setup
git clone <this-repo>
cd cameraman
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install google-genai pandas numpy scipy filterpy av fastapi uvicorn python-multipart

# Set API key
export GOOGLE_API_KEY="your-api-key-here"
```

### Usage

```bash
# Start the FastAPI server
uvicorn app.main:app --reload

# Upload a video via curl
curl -X POST "http://localhost:8000/render" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-video.mp4"

# Check status
curl "http://localhost:8000/status/{job_id}"

# Download result
curl "http://localhost:8000/download/{job_id}" -o processed-video.mp4
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| AI Analysis | **Gemini 2.5 Pro via Files API** | Video analysis and bounding box detection |
| Math Processing | **SciPy CubicSpline + FilterPy Kalman** | Smooth interpolation and filtering |
| Video Processing | **FFmpeg + VideoToolbox** | Hardware-accelerated cropping and encoding |
| API | **FastAPI** | Upload handling and background processing |

## Performance

- **File upload**: Up to 2GB via Gemini Files API
- **Analysis**: 20-40 seconds for 10-minute video
- **Processing**: 2000 fps interpolation on M1
- **Encoding**: 60-90 fps at 1080p with VideoToolbox
- **Total**: ~4 minutes for 10-minute match

## File Structure

```
cameraman/
├── app/
│   └── main.py           # FastAPI application
├── pipelines/
│   └── render_job.py     # Background processing
├── videos/               # Uploaded files (temporary)
├── outputs/              # Processed videos
├── requirements.txt      # Python dependencies
└── README.md
```

## Development

```bash
# Run with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Check FFmpeg VideoToolbox support
ffmpeg -encoders | grep videotoolbox
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Gemini API key from Google AI Studio | Yes |

## Limitations

- Maximum file size: 2GB (Gemini Files API limit)
- File retention: 48 hours (automatic cleanup)
- Supported formats: MP4, MOV, AVI
- Optimal performance: Apple Silicon Macs with VideoToolbox

## License

MIT License
