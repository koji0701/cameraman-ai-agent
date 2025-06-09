# AI Cameraman Python Backend

This module implements Phase 1 of the desktop GUI plan: the Python backend integration for the AI Cameraman desktop application.

## Overview

The Python backend provides a python-shell bridge that enables Electron to communicate with the existing AI Cameraman video processing pipeline via JSON messages over stdin/stdout.

## Architecture

```
Electron Main Process ←→ python-shell ←→ Python Backend ←→ AI Cameraman Pipeline
                     JSON over stdio              Direct function calls
```

## Components

### 1. `video_processor.py` - Main Bridge
- **Purpose**: Main python-shell bridge for Electron communication
- **Key Features**:
  - JSON command handling (process_video, cancel, ping, get_video_info)
  - Progress reporting with real-time updates
  - Threading support for non-blocking processing
  - Error handling and graceful cancellation
  - Integration with existing AICameramanPipeline

### 2. `progress_tracker.py` - Progress Management
- **Purpose**: Standardized progress tracking and reporting
- **Key Features**:
  - Real-time progress percentage calculation
  - Stage-based processing tracking
  - Time estimation for remaining work
  - Thread-safe progress updates
  - Configurable stage definitions

### 3. `file_manager.py` - File Operations
- **Purpose**: File validation, management, and statistics
- **Key Features**:
  - Video file format validation
  - Directory creation and management
  - File size statistics and compression ratios
  - Temporary directory management
  - Processing time estimation

### 4. `test_bridge.py` - Test Suite
- **Purpose**: Comprehensive testing of the Python backend
- **Key Features**:
  - Bridge communication testing
  - File manager validation
  - Progress tracker verification
  - Mock processing simulation

## Usage

### As Python-Shell Bridge (for Electron)
```bash
python python-backend/video_processor.py
```

Then send JSON commands via stdin:
```json
{"type": "process_video", "input_path": "/path/to/input.mp4", "output_path": "/path/to/output.mp4", "options": {"padding_factor": 1.1}}
```

### Via Desktop Launcher
```bash
# Start Python bridge
python app/desktop_launcher.py bridge

# Process video via CLI with progress
python app/desktop_launcher.py cli input.mp4 output.mp4

# Check system status
python app/desktop_launcher.py status
```

### Direct Import
```python
from python_backend import VideoCameramanBridge, FileManager, ProgressTracker

# Create and use the bridge
bridge = VideoCameramanBridge()
bridge.handle_command({
    "type": "process_video",
    "input_path": "input.mp4",
    "output_path": "output.mp4"
})
```

## Message Protocol

### Commands (Electron → Python)
```json
{
  "type": "process_video",
  "input_path": "/path/to/input.mp4",
  "output_path": "/path/to/output.mp4",
  "options": {
    "padding_factor": 1.1,
    "smoothing_strength": "balanced",
    "interpolation_method": "cubic"
  }
}
```

### Responses (Python → Electron)
```json
{
  "type": "progress",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "percentage": 45.2,
    "stage": "processing_video",
    "details": "Processing frame 1234/2000"
  }
}
```

### Message Types

**Commands:**
- `process_video`: Start video processing
- `cancel`: Cancel current processing
- `ping`: Test connection
- `get_video_info`: Get video file information

**Responses:**
- `ready`: Bridge is ready for commands
- `progress`: Processing progress update
- `status`: Status change notification
- `completed`: Processing completed successfully
- `cancelled`: Processing was cancelled
- `error`: Error occurred
- `pong`: Response to ping

## Integration with Existing Pipeline

The backend seamlessly integrates with the existing AI Cameraman components:

- **AICameramanPipeline**: Main processing pipeline
- **Gemini API Client**: Action coordinate extraction
- **Kalman Smoother**: Coordinate smoothing
- **OpenCV Processor**: Video processing engine

## Testing

Run the test suite:
```bash
python python-backend/test_bridge.py
```

This will test:
- Bridge communication
- File manager functionality
- Progress tracking
- Mock video processing

## Next Steps (Phase 2)

1. Create Electron desktop GUI
2. Implement python-shell integration in Electron
3. Build React frontend components
4. Add real-time progress visualization
5. Implement drag-and-drop file handling

## Dependencies

- Python 3.8+
- opencv-python
- numpy
- pandas
- scipy
- Existing AI Cameraman pipeline components

## File Structure

```
python-backend/
├── __init__.py           # Package initialization
├── video_processor.py    # Main python-shell bridge
├── progress_tracker.py   # Progress tracking utilities
├── file_manager.py       # File operations and validation
├── test_bridge.py        # Test suite
└── README.md            # This documentation
```

## Error Handling

The backend includes comprehensive error handling:
- File validation before processing
- Graceful cancellation support
- Thread-safe operations
- Detailed error messages
- Automatic cleanup of temporary files

## Performance

- Non-blocking processing via threading
- Real-time progress updates
- Efficient JSON communication
- Memory-conscious file operations
- Optimized for 60-90 FPS processing speeds 