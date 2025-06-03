import os, re, json, pandas as pd
from google import genai
from dotenv import load_dotenv
import time
from pathlib import Path
from normalize_coordinates import normalize_bounding_boxes_to_video_resolution, normalize_bounding_boxes_to_1080p, generate_ffmpeg_crop_filter, analyze_crop_quality
from kalman_smoother import interpolate_and_smooth_coordinates, generate_smooth_ffmpeg_filter, analyze_motion_smoothness

load_dotenv()

_PROMPT = (
    "this is footage from a water polo game. i want you to draw a box around the main action of the game (where the majority of the players are, and if a team is scoring or on offense, the box should include the goal they want to score on). the purpose of this box is to determine the optimal zoom in frame for the camerman, as in many frames, the camera should have been more zoomed in. this way, i can see the game better. go at 1 fps and go in intervals of three seconds. you will be returning data about the timestamp and the coordinates of the box (top left x, top left y, bottom right x, bottom right y)."
    "Reply as JSON list of objects {{t_ms:int,x1:int,y1:int,x2:int,y2:int}}."
)

print("Initializing Gemini client...")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  

def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Get video dimensions using ffprobe"""
    import subprocess
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        print(f"📺 Detected video resolution: {width}x{height}")
        return width, height
        
    except Exception as e:
        print(f"⚠️ Could not detect video dimensions: {e}")
        print("🔄 Falling back to default 1920x1080")
        return 1920, 1080

def create_outputs_folder(base_path: str, step_name: str) -> str:
    """Create a timestamped outputs folder for debugging"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"debug_{timestamp}_{step_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def save_step_data(data, step_name: str, output_dir: str, description: str = "", format: str = "csv"):
    """Save data from a pipeline step with descriptive naming"""
    if isinstance(data, pd.DataFrame):
        if format == "csv":
            filepath = Path(output_dir) / f"step_{step_name}.csv"
            data.to_csv(filepath, index=False)
        elif format == "json":
            filepath = Path(output_dir) / f"step_{step_name}.json"
            data.to_json(filepath, orient='records', indent=2)
    elif isinstance(data, dict):
        filepath = Path(output_dir) / f"step_{step_name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif isinstance(data, str):
        filepath = Path(output_dir) / f"step_{step_name}.txt"
        with open(filepath, 'w') as f:
            f.write(data)
    else:
        filepath = Path(output_dir) / f"step_{step_name}.json"
        with open(filepath, 'w') as f:
            json.dump(str(data), f, indent=2)
    
    print(f"🔍 DEBUG: Saved {step_name} → {filepath}")
    if description:
        print(f"   📝 {description}")
    
    return str(filepath)

def upload_and_prompt(mp4_path: str, debug_outputs_dir: str = None) -> pd.DataFrame:
    """Upload video to Gemini and get bounding box coordinates"""
    print("uploading file")
    file_ref = client.files.upload(file=mp4_path)
    print(f"file uploaded: {file_ref.name}, state: {file_ref.state}")

    # Wait for the file to be active
    # Files are ACTIVE once they have been processed.
    # You can use FileUploader.get_file and check the state of the file.
    while file_ref.state.name != "ACTIVE":
        print(f"File {file_ref.name} is not active yet. Current state: {file_ref.state.name}")
        time.sleep(5)  # Wait for 5 seconds before checking again
        file_ref = client.files.get(name=file_ref.name)
        print(f"Re-checked file state: {file_ref.state.name}")

    print(f"File {file_ref.name} is now ACTIVE.")

    # Now that the file is active, you can use it in generate_content
    print("Generating content with the file...")
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20", 
        contents=[file_ref, _PROMPT]
    )
    print("response received")
    
    # Save raw Gemini response if debug is enabled
    if debug_outputs_dir:
        raw_response_path = save_step_data(
            resp.text, 
            "01_gemini_raw_response", 
            debug_outputs_dir, 
            f"Raw response from Gemini 2.5 Flash for {Path(mp4_path).name}",
            format="txt"
        )
    
    # Gemini may wrap JSON in text; find the first [...]
    boxes_json_match = re.search(r"\[.*\]", resp.text, re.S)
    if not boxes_json_match:
        print("Error: Could not find JSON in the response.")
        print("Full response text:", resp.text)
        return pd.DataFrame() # Return empty DataFrame or raise an error
        
    boxes_json = boxes_json_match.group()
    records = json.loads(boxes_json)
    print("records loaded")
    
    df_result = pd.DataFrame(records)
    
    # Save extracted bounding boxes if debug is enabled
    if debug_outputs_dir:
        save_step_data(
            df_result, 
            "02_gemini_bounding_boxes", 
            debug_outputs_dir, 
            f"Extracted {len(df_result)} bounding boxes from Gemini at ~3s intervals"
        )
        
        # Also save the parsed JSON for inspection
        save_step_data(
            records, 
            "02_gemini_bounding_boxes_raw", 
            debug_outputs_dir, 
            "Raw JSON records before DataFrame conversion",
            format="json"
        )
    
    return df_result


def process_video_complete_pipeline(
    mp4_path: str, 
    original_width: int = None,  # Now optional - will auto-detect if not provided
    original_height: int = None, # Now optional - will auto-detect if not provided
    padding_factor: float = 1.1,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    output_crop_file: str = None,
    preserve_aspect_ratio: bool = True,  # New option to preserve original aspect ratio
    enable_debug_outputs: bool = True,   # New option to save debug outputs
    debug_outputs_dir: str = None        # Custom debug directory
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, str]:
    """
    Complete pipeline: Upload → Gemini Analysis → Normalize → Kalman Smooth → FFmpeg Ready
    
    Args:
        mp4_path: Path to input video file
        original_width: Source video width (auto-detected if None)
        original_height: Source video height (auto-detected if None)
        padding_factor: Extra padding around bounding boxes (1.1 = 10%)
        smoothing_strength: 'minimal', 'balanced', 'maximum', or 'cinematic'
        interpolation_method: 'cubic', 'linear', or 'quadratic'
        output_crop_file: Optional path to save crop filter file
        preserve_aspect_ratio: If True, uses original video aspect ratio; if False, uses 16:9
        enable_debug_outputs: If True, saves intermediate data at each step
        debug_outputs_dir: Custom directory for debug outputs (auto-created if None)
        
    Returns:
        Tuple of (original_boxes_df, normalized_crops_df, smoothed_df, quality_metrics, motion_metrics, ffmpeg_filter)
    """
    
    print(f"=== PROCESSING VIDEO: {mp4_path} ===")
    
    # Create debug outputs directory if debugging is enabled
    if enable_debug_outputs:
        if debug_outputs_dir is None:
            # Default to outputs folder in project root
            project_root = Path(__file__).parent.parent
            outputs_base = project_root / "outputs"
            debug_outputs_dir = create_outputs_folder(outputs_base, "pipeline")
        else:
            Path(debug_outputs_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 DEBUG: Saving pipeline steps to {debug_outputs_dir}")
        
        # Save initial video metadata
        video_metadata = {
            "input_video": str(mp4_path),
            "video_name": Path(mp4_path).name,
            "padding_factor": padding_factor,
            "smoothing_strength": smoothing_strength,
            "interpolation_method": interpolation_method,
            "preserve_aspect_ratio": preserve_aspect_ratio
        }
        save_step_data(
            video_metadata, 
            "00_pipeline_config", 
            debug_outputs_dir, 
            "Pipeline configuration and input video metadata"
        )
    
    # Auto-detect video dimensions if not provided
    if original_width is None or original_height is None:
        print("🔍 Auto-detecting video resolution...")
        original_width, original_height = get_video_dimensions(mp4_path)
        
        if enable_debug_outputs:
            video_info = {
                "detected_width": original_width,
                "detected_height": original_height,
                "aspect_ratio": original_width / original_height
            }
            save_step_data(
                video_info, 
                "01_video_dimensions", 
                debug_outputs_dir, 
                f"Auto-detected video dimensions: {original_width}x{original_height}"
            )
    
    # Calculate target aspect ratio
    if preserve_aspect_ratio:
        target_aspect_ratio = original_width / original_height
        print(f"🎯 Using original aspect ratio: {target_aspect_ratio:.2f}:1")
    else:
        target_aspect_ratio = 16/9
        print(f"🎯 Using standard 16:9 aspect ratio")
    
    # Step 1: Get bounding boxes from Gemini
    print("Step 1: Getting bounding boxes from Gemini...")
    original_boxes = upload_and_prompt(mp4_path, debug_outputs_dir if enable_debug_outputs else None)
    
    if original_boxes.empty:
        raise ValueError("No bounding boxes detected by Gemini")
    
    print(f"✓ Detected {len(original_boxes)} bounding boxes")
    
    # Step 2: Normalize coordinates for the actual video resolution
    print(f"Step 2: Normalizing coordinates for {original_width}x{original_height} video...")
    normalized_crops = normalize_bounding_boxes_to_video_resolution(
        original_boxes,
        original_width=original_width,
        original_height=original_height,
        target_aspect_ratio=target_aspect_ratio,
        padding_factor=padding_factor
    )
    
    if enable_debug_outputs:
        save_step_data(
            normalized_crops, 
            "03_normalized_coordinates", 
            debug_outputs_dir, 
            f"Normalized {len(normalized_crops)} crop coordinates for {original_width}x{original_height} video"
        )
        
        # Save normalization settings for reference
        norm_config = {
            "original_width": original_width,
            "original_height": original_height,
            "target_aspect_ratio": target_aspect_ratio,
            "padding_factor": padding_factor,
            "crops_generated": len(normalized_crops)
        }
        save_step_data(
            norm_config, 
            "03_normalization_config", 
            debug_outputs_dir, 
            "Normalization parameters and settings"
        )
    
    print(f"✓ Normalized {len(normalized_crops)} crop coordinates")
    
    # Step 3: Interpolate and smooth with Kalman filtering
    print("Step 3: Interpolating and smoothing with Kalman filtering...")
    smoothed_coords = interpolate_and_smooth_coordinates(
        normalized_crops,
        video_path=mp4_path,
        smoothing_strength=smoothing_strength,
        interpolation_method=interpolation_method,
        enable_debug_outputs=enable_debug_outputs,
        debug_outputs_dir=debug_outputs_dir
    )
    
    if enable_debug_outputs:
        save_step_data(
            smoothed_coords, 
            "04_kalman_smoothed_coords", 
            debug_outputs_dir, 
            f"Kalman filtered coordinates: {len(smoothed_coords)} frames ({smoothing_strength} smoothing)"
        )
    
    print(f"✓ Generated {len(smoothed_coords)} smooth frames")
    
    # Step 4: Generate quality analysis
    print("Step 4: Analyzing quality and motion metrics...")
    quality_metrics = analyze_crop_quality(original_boxes, normalized_crops)
    
    # Detect video FPS for motion analysis
    try:
        import av
        with av.open(mp4_path) as container:
            video_fps = float(container.streams.video[0].average_rate)
    except:
        video_fps = 30.0
    
    motion_metrics = analyze_motion_smoothness(smoothed_coords, video_fps)
    
    if enable_debug_outputs:
        save_step_data(
            quality_metrics, 
            "05_quality_metrics", 
            debug_outputs_dir, 
            f"Quality analysis: {quality_metrics.get('average_zoom', 'N/A')}x avg zoom"
        )
        
        save_step_data(
            motion_metrics, 
            "05_motion_metrics", 
            debug_outputs_dir, 
            f"Motion analysis: {motion_metrics.get('motion_analysis', {}).get('speed_consistency', 'N/A')} smoothness"
        )
    
    # Step 5: Generate FFmpeg crop filter
    print("Step 5: Generating FFmpeg crop filter...")
    ffmpeg_filter = generate_smooth_ffmpeg_filter(smoothed_coords)
    
    if enable_debug_outputs:
        save_step_data(
            ffmpeg_filter, 
            "06_ffmpeg_crop_filter", 
            debug_outputs_dir, 
            f"FFmpeg crop filter ready for {len(smoothed_coords)} frames",
            format="txt"
        )
        
        # Generate sample FFmpeg command for reference
        sample_output = Path(mp4_path).stem + "_cropped.mp4"
        sample_command = f"""# Sample FFmpeg command for dynamic cropping:
ffmpeg -i "{mp4_path}" \\
  -filter_complex "[0:v]sendcmd=f={debug_outputs_dir}/step_06_ffmpeg_crop_filter.txt,crop,scale=1920:1080" \\
  -c:v h264_videotoolbox -b:v 15M -allow_sw 1 \\
  -map 0:a -vsync 2 \\
  "{sample_output}"

# Alternative using frame-by-frame dynamic cropping:
# Use render_cropped_video_dynamic() function for more reliable results
"""
        save_step_data(
            sample_command, 
            "06_sample_ffmpeg_command", 
            debug_outputs_dir, 
            "Sample FFmpeg command for reference",
            format="txt"
        )
    
    # Step 6: Optionally save crop filter to file
    if output_crop_file:
        with open(output_crop_file, 'w') as f:
            f.write(ffmpeg_filter)
        print(f"✓ Crop filter saved to: {output_crop_file}")
    
    # Print comprehensive summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Video resolution: {original_width}x{original_height}")
    print(f"Target aspect ratio: {target_aspect_ratio:.2f}:1")
    print(f"Original keyframes: {len(original_boxes)}")
    print(f"Normalized crops: {len(normalized_crops)}")
    print(f"Smooth frames: {len(smoothed_coords)}")
    print(f"Video duration: {motion_metrics['frame_stats']['duration_seconds']:.1f} seconds")
    print(f"Average zoom: {quality_metrics['average_zoom']:.2f}x")
    print(f"Zoom range: {quality_metrics['recommended_zoom_range']}")
    print(f"Motion smoothness: {motion_metrics['motion_analysis']['speed_consistency']:.2f}")
    if enable_debug_outputs:
        print(f"🔍 DEBUG: All intermediate files saved to {debug_outputs_dir}")
    print(f"Ready for FFmpeg rendering!")
    
    return original_boxes, normalized_crops, smoothed_coords, quality_metrics, motion_metrics, ffmpeg_filter


def save_complete_results(
    original_boxes: pd.DataFrame,
    normalized_crops: pd.DataFrame, 
    smoothed_coords: pd.DataFrame,
    quality_metrics: dict,
    motion_metrics: dict,
    ffmpeg_filter: str,
    base_filename: str
):
    """Save all processing results to files for inspection and debugging"""
    
    # Save original bounding boxes
    original_boxes.to_csv(f"{base_filename}_01_original_boxes.csv", index=False)
    print(f"✓ Saved: {base_filename}_01_original_boxes.csv")
    
    # Save normalized crop coordinates
    normalized_crops.to_csv(f"{base_filename}_02_normalized_crops.csv", index=False)
    print(f"✓ Saved: {base_filename}_02_normalized_crops.csv")
    
    # Save smoothed coordinates
    smoothed_coords.to_csv(f"{base_filename}_03_smoothed_coords.csv", index=False)
    print(f"✓ Saved: {base_filename}_03_smoothed_coords.csv")
    
    # Save quality metrics as JSON
    combined_metrics = {
        "quality_metrics": quality_metrics,
        "motion_metrics": motion_metrics
    }
    with open(f"{base_filename}_04_metrics.json", 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"✓ Saved: {base_filename}_04_metrics.json")
    
    # Save FFmpeg crop filter
    with open(f"{base_filename}_05_crop_filter.txt", 'w') as f:
        f.write(ffmpeg_filter)
    print(f"✓ Saved: {base_filename}_05_crop_filter.txt")
    
    print(f"✓ All results saved with base: {base_filename}")