# AI Cameraman Pipeline Debug Outputs

This document explains the comprehensive debug output system that saves data at each step of the AI Cameraman pipeline for dynamic testing and analysis.

## Overview

The pipeline now saves detailed debugging information at each major step:

1. **Gemini AI Analysis** - Raw responses and extracted bounding boxes
2. **Coordinate Normalization** - Transformed coordinates for target resolution
3. **Kalman Filtering** - Interpolation and smoothing details
4. **Quality/Motion Analysis** - Performance metrics
5. **FFmpeg Rendering** - Commands and frame processing logs

## Debug Output Structure

When debug outputs are enabled, the system creates a timestamped directory in `/outputs/` with the following structure:

```
outputs/
└── debug_YYYYMMDD_HHMMSS_pipeline/
    ├── step_00_pipeline_config.json          # Pipeline configuration
    ├── step_01_gemini_raw_response.txt       # Raw Gemini AI response
    ├── step_01_video_dimensions.json         # Auto-detected video info
    ├── step_02_gemini_bounding_boxes.csv     # Extracted bounding boxes
    ├── step_02_gemini_bounding_boxes_raw.json # Raw JSON before DataFrame
    ├── step_03_normalized_coordinates.csv    # Normalized crop coordinates
    ├── step_03_normalization_config.json     # Normalization settings
    ├── step_04a_interpolated_coordinates.csv # Frame-by-frame interpolation
    ├── step_04a_interpolation_config.json    # Interpolation settings
    ├── step_04b_kalman_config.json           # Kalman filter configuration
    ├── step_04b_kalman_detailed_debug.json   # Detailed Kalman data
    ├── step_04b_smoothing_comparison.csv     # Before/after smoothing
    ├── step_04c_validated_final_coords.csv   # Final validated coordinates
    ├── step_04c_coordinate_summary.json      # Coordinate processing summary
    ├── step_05_quality_metrics.json          # Quality analysis results
    ├── step_05_motion_metrics.json           # Motion analysis results
    ├── step_06_ffmpeg_crop_filter.txt        # FFmpeg crop filter
    ├── step_06_sample_ffmpeg_command.txt     # Sample FFmpeg command
    ├── step_07_render_config.json            # Render configuration
    ├── step_07_frame_processing_info.json    # Frame processing details
    ├── step_07_ffmpeg_extract_command.txt    # Frame extraction command
    ├── step_07_frame_coordinate_mapping.json # Frame-to-coordinate mapping
    ├── step_07_crop_operations_log.json      # Per-frame crop operations
    ├── step_07_crop_operations_summary.csv   # Crop operations summary
    ├── step_07_ffmpeg_final_encode_command.txt # Final encoding command
    └── step_07_final_render_summary.json     # Final render summary
```

## Key Debug Files Explained

### Step 1: Gemini AI Analysis

**`step_01_gemini_raw_response.txt`**
- Raw text response from Gemini 2.5 Flash
- Contains the complete AI analysis including JSON and any additional text
- Useful for debugging prompt effectiveness and AI behavior

**`step_02_gemini_bounding_boxes.csv`**
- Extracted bounding boxes in DataFrame format
- Columns: `t_ms`, `x1`, `y1`, `x2`, `y2`
- Shows keyframes detected by AI at ~3 second intervals

### Step 2: Coordinate Normalization

**`step_03_normalized_coordinates.csv`**
- Bounding boxes converted to crop coordinates for the target video resolution
- Columns: `t_ms`, `crop_x`, `crop_y`, `crop_w`, `crop_h`, `zoom_factor`
- Includes padding factor and aspect ratio adjustments

**`step_03_normalization_config.json`**
- Settings used for normalization including target resolution and padding

### Step 3: Kalman Filtering

**`step_04a_interpolated_coordinates.csv`**
- Frame-by-frame coordinates before Kalman smoothing
- Generated using cubic/linear/quadratic interpolation

**`step_04b_kalman_detailed_debug.json`**
- Detailed Kalman filter data for each coordinate dimension
- Includes original measurements, filtered positions, velocities, and gains
- Essential for tuning smoothing parameters

**`step_04b_smoothing_comparison.csv`**
- Side-by-side comparison of interpolated vs smoothed coordinates
- Shows delta values to understand smoothing effect
- Useful for evaluating different smoothing strengths

### Step 4: Quality Analysis

**`step_05_quality_metrics.json`**
- Zoom factors, coverage analysis, and quality scores
- Helps evaluate detection accuracy and crop effectiveness

**`step_05_motion_metrics.json`**
- Motion smoothness, speed consistency, and temporal analysis
- Critical for understanding camera movement quality

### Step 5: FFmpeg Rendering

**`step_07_crop_operations_log.json`**
- Detailed log of every frame crop operation
- Shows success/failure status for each frame
- Includes timing and coordinate information

**`step_07_ffmpeg_extract_command.txt` & `step_07_ffmpeg_final_encode_command.txt`**
- Exact FFmpeg commands used for frame extraction and final encoding
- Enables manual reproduction and debugging of video processing

## Using Debug Outputs

### Enable Debug Outputs

```python
from pipelines.render_video import process_and_render_complete

success = process_and_render_complete(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    enable_debug_outputs=True,  # Enable debug outputs
    debug_outputs_dir=None,     # Auto-create timestamped directory
    rendering_mode='dynamic'
)
```

### Analysis-Only Mode (Faster Testing)

```python
from pipelines.genai_client import process_video_complete_pipeline

results = process_video_complete_pipeline(
    "input.mp4",
    enable_debug_outputs=True,
    debug_outputs_dir="custom_debug_dir"
)
```

### Test Script

Run the included test script:

```bash
python test_debug_pipeline.py
```

Choose from:
1. Full pipeline with rendering
2. Analysis only (faster)
3. Both tests

## Common Analysis Workflows

### Debugging Gemini AI Detection

1. Check `step_01_gemini_raw_response.txt` for AI reasoning
2. Examine `step_02_gemini_bounding_boxes.csv` for detection quality
3. Compare with `step_03_normalized_coordinates.csv` to see normalization effects

### Tuning Kalman Filter

1. Open `step_04b_smoothing_comparison.csv` in spreadsheet software
2. Compare original vs smoothed coordinates
3. Adjust `smoothing_strength` parameter based on delta values
4. Check `step_05_motion_metrics.json` for smoothness scores

### Optimizing Rendering

1. Review `step_07_crop_operations_log.json` for frame processing success rate
2. Check `step_07_final_render_summary.json` for overall statistics
3. Examine FFmpeg command files for optimization opportunities

### Quality Assessment

1. Use `step_05_quality_metrics.json` for zoom and coverage analysis
2. Check `step_05_motion_metrics.json` for camera movement quality
3. Compare different pipeline configurations using metrics

## File Formats

- **CSV files**: Can be opened in Excel, Google Sheets, or pandas
- **JSON files**: Structured data for programmatic analysis
- **TXT files**: Human-readable commands and logs

## Performance Impact

Debug outputs add minimal overhead:
- File I/O operations are non-blocking
- Data is saved incrementally during processing
- Typical debug directory size: 5-50MB depending on video length

## Customization

You can customize debug output behavior:

```python
# Custom debug directory
debug_dir = "/path/to/custom/debug/folder"

# Disable specific debug outputs by modifying the save_step_data calls
# in the pipeline functions
```

## Troubleshooting

If debug outputs aren't being saved:
1. Check file permissions in the outputs directory
2. Ensure sufficient disk space
3. Verify `enable_debug_outputs=True` is set
4. Check console output for debug save messages (🔍 DEBUG:)

## Integration with Existing Tools

Debug outputs are compatible with:
- Data analysis tools (pandas, numpy)
- Visualization libraries (matplotlib, plotly)
- Video analysis software (FFmpeg, OpenCV)
- Spreadsheet applications (Excel, Google Sheets)

This debug system provides complete transparency into every step of the AI Cameraman pipeline, enabling precise tuning and optimization for different video types and use cases. 