import os, re, json, pandas as pd
from google import genai
from dotenv import load_dotenv
import time
from normalize_coordinates import normalize_bounding_boxes_to_video_resolution, normalize_bounding_boxes_to_1080p, generate_ffmpeg_crop_filter, analyze_crop_quality
from kalman_smoother import interpolate_and_smooth_coordinates, generate_smooth_ffmpeg_filter, analyze_motion_smoothness
import numpy as np

load_dotenv()

def create_dynamic_prompt(width: int, height: int, aspect_ratio: float) -> str:
    """Generate AI prompt with video-specific dimensions for optimal keyframes"""
    return f"""this is footage from a water polo game. i want you to draw a box around the main action of the game (where the majority of the players are, and if a team is scoring or on offense, the box should include the goal they want to score on). the purpose of this box is to determine the optimal zoom in frame for the camerman, as in many frames, the camera should have been more zoomed in. go at 1 fps and go in intervals of one second. you will be returning data about the timestamp and the coordinates of the box (top left x, top left y, bottom right x, bottom right y).  

The video is {width}x{height} pixels and {aspect_ratio:.2f} aspect ratio, so when determining the box, use these pixel values for x and y coordinates. Further, preserve the aspect ratio when determining the proper coordinates.

For smooth camera panning: ensure the boxes you draw will create natural camera movement when interpolated between keyframes. Consider the motion flow of the game.

do not overcrop, if the width is much wider than the height so that the aspect ratio isn't preserved, increase the height in both the y1 and y2 so that the aspect ratio is preserved, but keep the box centered on the main action. vice versa for if the height is too long, increase the width so that the aspect ratio is preserved  but keep the box centered on the main action. when doing this adjustment, prioritize being sure that the pixels do not go out of bounds of the video (keep within the provided dimensions) over being centered on the main action, meaning that in some cases it may be necessary to only add more width or height to one side rather than the other side.

Reply as JSON list of objects {{t_ms:int,x1:int,y1:int,x2:int,y2:int}}."""

# Legacy prompt for backward compatibility
_PROMPT = (
    "this is footage from a water polo game. i want you to draw a box around the main action of the game (where the majority of the players are, and if a team is scoring or on offense, the box should include the goal they want to score on). the purpose of this box is to determine the optimal zoom in frame for the camerman, as in many frames, the camera should have been more zoomed in. go at 1 fps and go in intervals of three seconds. you will be returning data about the timestamp and the coordinates of the box (top left x, top left y, bottom right x, bottom right y)."
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

def upload_and_prompt(mp4_path: str, custom_prompt: str = None) -> pd.DataFrame:
    """Upload video to Gemini and get bounding box coordinates with enhanced prompt"""
    
    # Get video dimensions for dynamic prompt
    width, height = get_video_dimensions(mp4_path)
    aspect_ratio = width / height
    
    # Use dynamic prompt unless custom one provided
    prompt = custom_prompt or create_dynamic_prompt(width, height, aspect_ratio)
    
    print(f"🎯 Using AI prompt with {width}x{height} dimensions, {aspect_ratio:.2f} aspect ratio")
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
    print("Generating content with enhanced prompt...")
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20", 
        contents=[file_ref, prompt],  # Use the dynamic prompt
    )
    print("response received")
    
    # Gemini may wrap JSON in text; find the first [...]
    boxes_json_match = re.search(r"\[.*\]", resp.text, re.S)
    if not boxes_json_match:
        print("Error: Could not find JSON in the response.")
        print("Full response text:", resp.text)
        return pd.DataFrame() # Return empty DataFrame or raise an error
        
    boxes_json = boxes_json_match.group()
    
    # Clean up common JSON issues from AI responses
    try:
        # First, try direct parsing
        records = json.loads(boxes_json)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parsing failed: {e}")
        print("Attempting to clean JSON response...")
        
        # Clean up common AI JSON issues
        cleaned_json = boxes_json
        
        # Fix common issues:
        # 1. Trailing commas
        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
        
        # 2. Comments (// style)
        cleaned_json = re.sub(r'//.*$', '', cleaned_json, flags=re.MULTILINE)
        
        # 3. Single quotes to double quotes for properties
        cleaned_json = re.sub(r"'(\w+)':", r'"\1":', cleaned_json)
        
        # 4. Missing quotes around string values
        cleaned_json = re.sub(r':(\s*)([a-zA-Z_][a-zA-Z0-9_]*)', r':\1"\2"', cleaned_json)
        
        # 5. Extra text after JSON
        cleaned_json = re.search(r'\[.*\]', cleaned_json, re.S)
        if cleaned_json:
            cleaned_json = cleaned_json.group()
        else:
            print("Error: Could not extract valid JSON structure after cleaning")
            print("Original response:", resp.text)
            return pd.DataFrame()
        
        try:
            records = json.loads(cleaned_json)
            print("✅ Successfully parsed JSON after cleaning")
        except json.JSONDecodeError as e2:
            print(f"JSON parsing still failed after cleaning: {e2}")
            print("Cleaned JSON:", cleaned_json)
            print("Original response:", resp.text)
            return pd.DataFrame()
    
    print("records loaded")
    return pd.DataFrame(records)


def convert_boxes_to_crops(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert AI bounding boxes to crop coordinates (minimal processing needed)
    
    Since the AI now receives video dimensions and handles normalization,
    we only need simple box-to-crop conversion without complex processing.
    
    Args:
        boxes_df: DataFrame with columns ['t_ms', 'x1', 'y1', 'x2', 'y2']
        
    Returns:
        DataFrame with crop coordinates ['t_ms', 'crop_x', 'crop_y', 'crop_w', 'crop_h']
    """
    
    crops = []
    
    for _, row in boxes_df.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        
        # Basic validation
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bounding box at t={row['t_ms']}ms: ({x1},{y1},{x2},{y2})")
            continue
        
        crops.append({
            't_ms': row['t_ms'],
            'crop_x': x1,
            'crop_y': y1,
            'crop_w': x2 - x1,
            'crop_h': y2 - y1
        })
    
    return pd.DataFrame(crops)


def process_video_enhanced_pipeline(
    mp4_path: str,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    output_crop_file: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, str]:
    """
    Enhanced pipeline: Upload → Gemini Analysis → Direct Interpolation → Kalman Smooth → FFmpeg Ready
    
    This is the new improved pipeline that eliminates redundant normalization since
    the AI now handles video dimensions and aspect ratio directly.
    
    Args:
        mp4_path: Path to input video file
        smoothing_strength: 'minimal', 'balanced', 'maximum', or 'cinematic'
        interpolation_method: 'cubic', 'linear', or 'quadratic'
        output_crop_file: Optional path to save crop filter file
        
    Returns:
        Tuple of (original_boxes_df, crop_coords_df, smoothed_df, quality_metrics, motion_metrics, ffmpeg_filter)
    """
    
    print(f"🚀 ENHANCED AI CAMERAMAN PIPELINE")
    print(f"=== PROCESSING VIDEO: {mp4_path} ===")
    
    # Step 1: Get optimized bounding boxes from Gemini (now with proper dimensions)
    print("Step 1: Getting optimized bounding boxes from Gemini...")
    original_boxes = upload_and_prompt(mp4_path)
    
    if original_boxes.empty:
        raise ValueError("No bounding boxes detected by Gemini")
    
    print(f"✓ Detected {len(original_boxes)} bounding boxes")
    
    # Step 2: Convert bounding boxes to crop coordinates (minimal processing)
    print("Step 2: Converting bounding boxes to crop coordinates...")
    crop_coords = convert_boxes_to_crops(original_boxes)
    
    # Add zoom factors for quality analysis
    width, height = get_video_dimensions(mp4_path)
    crop_coords['zoom_factor'] = np.minimum(
        width / crop_coords['crop_w'],
        height / crop_coords['crop_h']
    )
    
    print(f"✓ Converted {len(crop_coords)} crop coordinates")
    
    # Step 3: Apply existing excellent interpolation and smoothing
    print("Step 3: Creating smooth panning with interpolation and Kalman filtering...")
    smoothed_coords = interpolate_and_smooth_coordinates(
        crop_coords,
        video_path=mp4_path,
        smoothing_strength=smoothing_strength,
        interpolation_method=interpolation_method
    )
    
    print(f"✓ Generated {len(smoothed_coords)} smooth frames")
    
    # Step 4: Generate quality analysis (using simplified metrics)
    print("Step 4: Analyzing quality and motion metrics...")
    quality_metrics = analyze_crop_quality(original_boxes, crop_coords)
    
    # Detect video FPS for motion analysis
    try:
        import av
        with av.open(mp4_path) as container:
            video_fps = float(container.streams.video[0].average_rate)
    except:
        video_fps = 30.0
    
    motion_metrics = analyze_motion_smoothness(smoothed_coords, video_fps)
    
    # Step 5: Generate FFmpeg crop filter
    print("Step 5: Generating FFmpeg crop filter...")
    ffmpeg_filter = generate_smooth_ffmpeg_filter(smoothed_coords)
    
    # Step 6: Optionally save crop filter to file
    if output_crop_file:
        with open(output_crop_file, 'w') as f:
            f.write(ffmpeg_filter)
        print(f"✓ Crop filter saved to: {output_crop_file}")
    
    # Print comprehensive summary
    print(f"\n🎯 ENHANCED PROCESSING COMPLETE")
    print(f"Original keyframes: {len(original_boxes)}")
    print(f"Crop coordinates: {len(crop_coords)}")
    print(f"Smooth frames: {len(smoothed_coords)}")
    print(f"Video duration: {motion_metrics['frame_stats']['duration_seconds']:.1f} seconds")
    print(f"Average zoom: {quality_metrics['average_zoom']:.2f}x")
    print(f"Motion smoothness: {motion_metrics['motion_analysis']['speed_consistency']:.2f}")
    print(f"✅ Ready for smooth panning render!")
    
    return original_boxes, crop_coords, smoothed_coords, quality_metrics, motion_metrics, ffmpeg_filter


def process_video_complete_pipeline(
    mp4_path: str, 
    original_width: int = None,  # Now optional - will auto-detect if not provided
    original_height: int = None, # Now optional - will auto-detect if not provided
    padding_factor: float = 1.1,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    output_crop_file: str = None,
    preserve_aspect_ratio: bool = True  # New option to preserve original aspect ratio
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
        
    Returns:
        Tuple of (original_boxes_df, normalized_crops_df, smoothed_df, quality_metrics, motion_metrics, ffmpeg_filter)
    """
    
    print(f"=== PROCESSING VIDEO: {mp4_path} ===")
    
    # Auto-detect video dimensions if not provided
    if original_width is None or original_height is None:
        print("🔍 Auto-detecting video resolution...")
        original_width, original_height = get_video_dimensions(mp4_path)
    
    # Calculate target aspect ratio
    if preserve_aspect_ratio:
        target_aspect_ratio = original_width / original_height
        print(f"🎯 Using original aspect ratio: {target_aspect_ratio:.2f}:1")
    else:
        target_aspect_ratio = 16/9
        print(f"🎯 Using standard 16:9 aspect ratio")
    
    # Step 1: Get bounding boxes from Gemini
    print("Step 1: Getting bounding boxes from Gemini...")
    original_boxes = upload_and_prompt(mp4_path)
    
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
    
    print(f"✓ Normalized {len(normalized_crops)} crop coordinates")
    
    # Step 3: Interpolate and smooth with Kalman filtering
    print("Step 3: Interpolating and smoothing with Kalman filtering...")
    smoothed_coords = interpolate_and_smooth_coordinates(
        normalized_crops,
        video_path=mp4_path,
        smoothing_strength=smoothing_strength,
        interpolation_method=interpolation_method
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
    
    # Step 5: Generate FFmpeg crop filter
    print("Step 5: Generating FFmpeg crop filter...")
    ffmpeg_filter = generate_smooth_ffmpeg_filter(smoothed_coords)
    
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


# Deprecated: old function for backward compatibility
def process_video_to_crop_coordinates(
    mp4_path: str, 
    original_width: int = 1920, 
    original_height: int = 1080,
    padding_factor: float = 1.1,
    output_crop_file: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """
    Legacy function - use process_video_complete_pipeline instead
    """
    print("Warning: Using deprecated function. Please use process_video_complete_pipeline for Kalman smoothing.")
    
    original_boxes = upload_and_prompt(mp4_path)
    if original_boxes.empty:
        raise ValueError("No bounding boxes detected by Gemini")
    
    normalized_crops = normalize_bounding_boxes_to_1080p(
        original_boxes,
        original_width=original_width,
        original_height=original_height, 
        padding_factor=padding_factor
    )
    
    quality_metrics = analyze_crop_quality(original_boxes, normalized_crops)
    ffmpeg_filter = generate_ffmpeg_crop_filter(normalized_crops)
    
    if output_crop_file:
        with open(output_crop_file, 'w') as f:
            f.write(ffmpeg_filter)
    
    return original_boxes, normalized_crops, quality_metrics, ffmpeg_filter


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    video_file_path = os.path.join(project_root, "videos", "waterpolo_trimmed.webm")
    
    print("video file path:", video_file_path)
    
    try:
        # Process video with complete Kalman smoothing pipeline
        (original_boxes, normalized_crops, smoothed_coords, 
         quality_metrics, motion_metrics, ffmpeg_filter) = process_video_complete_pipeline(
            video_file_path,
            padding_factor=1.1,  # 10% padding for safety
            smoothing_strength='balanced',  # Professional smoothing
            interpolation_method='cubic'    # Smooth interpolation
        )
        
        # Save results for inspection
        base_filename = os.path.join(project_root, "outputs", "waterpolo_analysis")
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        
        save_complete_results(
            original_boxes, 
            normalized_crops, 
            smoothed_coords,
            quality_metrics, 
            motion_metrics,
            ffmpeg_filter,
            base_filename
        )
        
        print(f"\n=== READY FOR FFMPEG ===")
        print(f"Use this crop filter file: {base_filename}_05_crop_filter.txt")
        print("\nFFmpeg command:")
        print(f"ffmpeg -i {video_file_path} \\")
        print(f"  -filter_complex \"[0:v]sendcmd=f={base_filename}_05_crop_filter.txt,crop,scale=1920:1080\" \\")
        print(f"  -c:v h264_videotoolbox -b:v 15M -allow_sw 1 \\")
        print(f"  -map 0:a -vsync 2 \\")
        print(f"  {project_root}/outputs/cropped_waterpolo.mp4")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()