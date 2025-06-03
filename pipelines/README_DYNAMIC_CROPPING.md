# Dynamic Cropping Implementation

## Overview

The dynamic cropping feature has been completely rewritten to fix issues with the previous `sendcmd`-based approach that was producing fake/corrupted MP4 files.

**NEW: Resolution-Aware Processing** - The system now automatically detects your video resolution and adapts all cropping and scaling accordingly, fixing the zoom bias issue that occurred when processing 4K videos with hardcoded 1920x1080 settings.

## The Problem

The previous implementation had two major issues:

1. **sendcmd approach**: Used FFmpeg's `sendcmd` filter which was unreliable and produced corrupted outputs
2. **Resolution bias**: Hardcoded 1920x1080 resolution caused zoom bias towards top-left when processing higher resolution videos (e.g., 4K 3840x2160)

## The Solution

Following the approach outlined in `ffmpeg-crop.mdc`, the new implementation uses:

1. **Frame-by-frame processing**: Extract frames → Crop individually → Re-encode
2. **Resolution-aware normalization**: Auto-detects video resolution and adapts all coordinates
3. **Adaptive scaling**: Uses original crop dimensions rather than forcing to fixed resolution

## Key Features

### Resolution Independence
- ✅ **Auto-detection**: Automatically detects input video resolution
- ✅ **Adaptive coordinates**: Normalizes bounding boxes for actual video dimensions
- ✅ **Preserves aspect ratio**: Maintains original video aspect ratio by default
- ✅ **No zoom bias**: Correctly centers crops regardless of resolution

### Dynamic Cropping
- ✅ **Reliable per-frame cropping**: No more fake/corrupted MP4 files
- ✅ **Proper audio preservation**: Audio extracted and re-muxed correctly
- ✅ **Progress reporting**: Shows extraction, cropping, and encoding progress
- ✅ **Error handling**: Comprehensive validation and debugging

### Performance
- ✅ **OpenCV support**: Faster image processing when available
- ✅ **Hardware encoding**: Apple Silicon VideoToolbox support
- ✅ **Temporary file cleanup**: Automatic cleanup of intermediate files

## Usage

### Basic Dynamic Cropping (Auto-Resolution)

```python
from render_video import render_cropped_video

success = render_cropped_video(
    input_video_path="input.mp4",
    output_video_path="output.mp4", 
    smoothed_coords_df=your_coordinates_df,
    rendering_mode="dynamic",  # Uses new frame-by-frame approach
    scale_resolution="original"  # NEW: Adaptive resolution (default)
)
```

### Advanced Options

```python
success = render_cropped_video_dynamic(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    smoothed_coords_df=coords_df,
    video_codec="h264_videotoolbox",  # Hardware encoding
    scale_resolution="original",      # NEW: Uses crop dimensions
    enable_stabilization=True,
    color_correction=True,
    verbose=True
)
```

### Complete Pipeline with Auto-Detection

```python
from genai_client import process_video_complete_pipeline

# Auto-detects resolution and preserves aspect ratio
original_boxes, normalized_crops, smoothed_coords, quality_metrics, motion_metrics, ffmpeg_filter = process_video_complete_pipeline(
    "input_4k.mp4",  # Works with any resolution
    preserve_aspect_ratio=True  # NEW: Preserves original aspect ratio
)
```

## Resolution Handling

### Automatic Detection
The system automatically detects video resolution using ffprobe:
- **4K (3840x2160)**: Full 4K processing with correct coordinate scaling
- **1080p (1920x1080)**: Standard HD processing
- **Custom resolutions**: Adapts to any input resolution

### Scaling Options
- `scale_resolution="original"` (default): Uses optimal crop dimensions
- `scale_resolution="1920:1080"`: Forces specific output resolution
- `scale_resolution="3840:2160"`: Forces 4K output

### Aspect Ratio Preservation
- `preserve_aspect_ratio=True` (default): Maintains original video aspect ratio
- `preserve_aspect_ratio=False`: Uses standard 16:9 aspect ratio

## Coordinate Data Format

The `smoothed_coords_df` DataFrame must contain these columns:

- `t_ms`: Timestamp in milliseconds
- `crop_x`: X coordinate of crop rectangle (in original video coordinates)
- `crop_y`: Y coordinate of crop rectangle (in original video coordinates)
- `crop_w`: Width of crop rectangle (will be made even)
- `crop_h`: Height of crop rectangle (will be made even)

**Note**: All coordinates are now properly scaled for the actual video resolution.

## Migration from Old Implementation

### From Hardcoded Resolution
**Old approach (hardcoded 1920x1080):**
```python
normalized_crops = normalize_bounding_boxes_to_1080p(
    df, original_width=1920, original_height=1080
)  # ⚠️ Caused zoom bias for 4K videos
```

**New approach (auto-detected resolution):**
```python
normalized_crops = normalize_bounding_boxes_to_video_resolution(
    df, original_width, original_height, target_aspect_ratio=16/9
)  # ✅ Works correctly with any resolution
```

### From sendcmd Approach
**Old approach (deprecated):**
```python
sendcmd = generate_sendcmd_filter(coords_df)  # ⚠️ Unreliable
```

**New approach:**
```python
success = render_cropped_video_dynamic(input_video, output_video, coords_df)  # ✅ Reliable
```

## Troubleshooting

### Zoom Bias Issues
- ✅ **Fixed**: No longer biased towards top-left
- ✅ **Auto-detection**: System detects and adapts to video resolution
- ✅ **Validation**: Coordinates are validated against actual frame bounds

### Resolution Mismatch
- Use `scale_resolution="original"` for adaptive resolution
- Check that bounding box coordinates match your video resolution
- Enable verbose mode to see detected resolution: `verbose=True`

### Performance with 4K Videos
- Install OpenCV for faster processing: `pip install opencv-python`
- Use hardware encoding: `video_codec="h264_videotoolbox"`
- Consider using `rendering_mode="simple"` for faster processing

## Example: 4K Video Processing

```python
# Process a 4K video with automatic resolution detection
success = render_cropped_video(
    input_video_path="4k_video.mp4",          # 3840x2160
    output_video_path="4k_cropped.mp4",
    smoothed_coords_df=coords_df,
    rendering_mode="dynamic",
    scale_resolution="original",              # Adapts to 4K
    preserve_aspect_ratio=True,               # Maintains 16:9 ratio
    video_codec="h264_videotoolbox",         # Hardware encoding
    verbose=True
)

# Output will be correctly cropped without zoom bias
```

This validates coordinate structure and generates test data without actual video processing. 