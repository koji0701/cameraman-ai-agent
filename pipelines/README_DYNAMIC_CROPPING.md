# Dynamic Cropping Implementation

## Overview

The dynamic cropping feature has been completely rewritten to fix issues with the previous `sendcmd`-based approach that was producing fake/corrupted MP4 files.

## The Problem

The previous implementation used FFmpeg's `sendcmd` filter to dynamically change crop parameters during playback. This approach had several issues:

- Unreliable execution leading to corrupted outputs
- Complex syntax prone to errors
- Poor error handling
- Inconsistent results across different FFmpeg versions

## The Solution

Following the approach outlined in `ffmpeg-crop.mdc`, the new implementation uses a three-step process:

1. **Extract frames**: Convert video to individual image frames
2. **Crop frames**: Process each frame individually with precise coordinates
3. **Re-encode**: Combine cropped frames back into video with original audio

## Usage

### Basic Dynamic Cropping

```python
from render_video import render_cropped_video

success = render_cropped_video(
    input_video_path="input.mp4",
    output_video_path="output.mp4", 
    smoothed_coords_df=your_coordinates_df,
    rendering_mode="dynamic"  # This uses the new implementation
)
```

### Advanced Options

```python
success = render_cropped_video_dynamic(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    smoothed_coords_df=coords_df,
    video_codec="h264_videotoolbox",  # Hardware encoding on Apple Silicon
    enable_stabilization=True,        # Optional video stabilization
    color_correction=True,            # Optional color enhancement
    scale_resolution="1920:1080",     # Target resolution
    verbose=True
)
```

## Coordinate Data Format

The `smoothed_coords_df` DataFrame must contain these columns:

- `t_ms`: Timestamp in milliseconds
- `crop_x`: X coordinate of crop rectangle
- `crop_y`: Y coordinate of crop rectangle  
- `crop_w`: Width of crop rectangle (will be made even)
- `crop_h`: Height of crop rectangle (will be made even)

## Performance

### With OpenCV (Recommended)
- Install: `pip install opencv-python`
- Faster image processing using native libraries
- Better memory management

### FFmpeg Only (Fallback)
- Works without additional dependencies
- Slower but more compatible
- Uses subprocess calls for each frame

## Features

- ✅ Reliable per-frame cropping
- ✅ Proper audio preservation
- ✅ Error handling and progress reporting
- ✅ Support for video stabilization
- ✅ Color correction options
- ✅ Hardware encoding support
- ✅ Temporary file cleanup
- ✅ Backward compatibility

## Migration from Old Implementation

The old `generate_sendcmd_filter()` function is now deprecated but still available for backward compatibility. It will show a warning and return an empty string.

**Old approach (deprecated):**
```python
sendcmd = generate_sendcmd_filter(coords_df)  # ⚠️ Deprecated
```

**New approach:**
```python
success = render_cropped_video_dynamic(input_video, output_video, coords_df)  # ✅ Recommended
```

## Troubleshooting

### Small/Empty Output Files
- Check coordinate bounds are within video dimensions
- Verify coordinate data is valid (positive dimensions)
- Enable verbose mode for detailed error messages

### Performance Issues
- Install OpenCV for faster processing: `pip install opencv-python`
- Reduce video resolution before processing
- Use hardware encoding if available

### Audio Issues
- The implementation automatically preserves original audio
- If audio extraction fails, video will be rendered without audio
- Check FFmpeg installation supports your audio codec

## Testing

Use the built-in test function to verify your coordinate data:

```python
from render_video import test_dynamic_cropping

success = test_dynamic_cropping(your_coords_df, "test_output.mp4")
```

This validates coordinate structure and generates test data without actual video processing. 