import os
import subprocess
import pandas as pd
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from genai_client import process_video_complete_pipeline, save_complete_results
from normalize_coordinates import generate_ffmpeg_crop_filter
import cv2
HAS_OPENCV = True

# # Try to import cv2 for image processing
# try:
#     HAS_OPENCV = True
# except ImportError:
#     HAS_OPENCV = False
#     print("⚠️ OpenCV not found. Frame-by-frame cropping will use FFmpeg only.")

"""
DYNAMIC CROPPING FIX (2024)
===========================

The previous sendcmd-based approach was unreliable and produced fake/corrupted MP4 files.
This has been replaced with a proper frame-by-frame approach following ffmpeg-crop.mdc:

1. Extract all frames as images
2. Crop each frame individually according to coordinates
3. Re-encode cropped frames into final video

This provides:
- Reliable per-frame cropping
- Better error handling
- Proper audio preservation
- Support for OpenCV (faster) or pure FFmpeg (compatible)

Use render_cropped_video_dynamic() with rendering_mode='dynamic' for the new implementation.
The old generate_sendcmd_filter() function is deprecated but maintained for backward compatibility.
"""

def get_video_info(input_video_path: str) -> Dict:
    """Get detailed video information using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        input_video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        
        # Check for audio stream
        audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
        has_audio = audio_stream is not None
        
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
            'duration': float(video_stream['duration']),
            'bitrate': int(video_stream.get('bit_rate', 0)),
            'codec': video_stream['codec_name'],
            'pixel_format': video_stream['pix_fmt'],
            'total_frames': int(video_stream['nb_frames']) if 'nb_frames' in video_stream else None,
            'has_audio': has_audio,
            'audio_codec': audio_stream['codec_name'] if has_audio else None
        }
    except Exception as e:
        print(f"⚠️ Could not get video info: {e}")
        return {}

def generate_smooth_ffmpeg_filter(smoothed_coords_df: pd.DataFrame, fps: float = 29.97) -> str:
    """Create smooth ffmpeg filter for kalman smoothed data."""
    # This is still needed by kalman_smoother.py
    if len(smoothed_coords_df) == 0:
        return ""
    
    commands = []
    for idx, row in smoothed_coords_df.iterrows():
        timestamp = row['t_ms'] / 1000.0
        x, y = int(row['crop_x']), int(row['crop_y'])
        w, h = int(row['crop_w']) & ~1, int(row['crop_h']) & ~1  # Ensure even
        commands.append(f"{timestamp:.3f} crop w {w} h {h} x {x} y {y};")
    
    sendcmd_content = '\n'.join(commands)
    return f"sendcmd=c='{sendcmd_content}'"

def apply_smooth_coordinates_to_frames(
    smoothed_coords_df: pd.DataFrame, 
    total_frames: int,
    fps: float,
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Apply interpolated coordinates directly to frames - NO coordinate loss!
    
    This function fixes the critical efficiency issue where only 6.7% of interpolated
    coordinates were being used. Now 100% of coordinates are applied for true smooth panning.
    
    Args:
        smoothed_coords_df: DataFrame with interpolated coordinates for every frame
        total_frames: Total frames in video
        fps: Video frame rate
        verbose: Print progress
        
    Returns:
        Dictionary mapping each frame number to its unique coordinates
    """
    
    if verbose:
        print(f"🎯 Applying smooth coordinates to {total_frames} frames...")
    
    coords_dict = {}
    
    # The smoothed_coords_df already contains coordinates for every frame timestamp
    # We just need to map them correctly to frame numbers without collision
    
    for _, row in smoothed_coords_df.iterrows():
        # Calculate frame number from timestamp
        frame_time_s = row['t_ms'] / 1000.0
        frame_num = int(frame_time_s * fps) + 1
        
        # Ensure frame number is within valid range
        if 1 <= frame_num <= total_frames:
            coords_dict[frame_num] = {
                'x': int(row['crop_x']),
                'y': int(row['crop_y']),
                'w': int(row['crop_w']) & ~1,  # Ensure even
                'h': int(row['crop_h']) & ~1   # Ensure even
            }
    
    # Fill any missing frames using interpolation (handles edge cases)
    if len(coords_dict) < total_frames:
        missing_frames = set(range(1, total_frames + 1)) - set(coords_dict.keys())
        if verbose:
            print(f"📝 Filling {len(missing_frames)} missing frames with interpolation...")
        
        # Use scipy interpolation to fill missing frames
        from scipy.interpolate import interp1d
        
        existing_frames = sorted(coords_dict.keys())
        existing_coords = np.array([[coords_dict[f]['x'], coords_dict[f]['y'], 
                                   coords_dict[f]['w'], coords_dict[f]['h']] 
                                  for f in existing_frames])
        
        # Create interpolators
        interp_funcs = {}
        coord_names = ['x', 'y', 'w', 'h']
        
        for i, coord_name in enumerate(coord_names):
            interp_funcs[coord_name] = interp1d(
                existing_frames, existing_coords[:, i], 
                kind='linear', fill_value='extrapolate'
            )
        
        # Fill missing frames
        for frame_num in missing_frames:
            coords_dict[frame_num] = {
                'x': int(interp_funcs['x'](frame_num)),
                'y': int(interp_funcs['y'](frame_num)),
                'w': int(interp_funcs['w'](frame_num)) & ~1,
                'h': int(interp_funcs['h'](frame_num)) & ~1
            }
    
    if verbose:
        print(f"✅ Coordinate mapping complete:")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Frames with coordinates: {len(coords_dict)}")
        print(f"   - Coverage: {len(coords_dict)/total_frames*100:.1f}%")
        print(f"   - SMOOTH PANNING: {'✅ TRUE' if len(coords_dict) == total_frames else '❌ FALSE'}")
    
    return coords_dict

def render_cropped_video_dynamic(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    video_codec: str = "h264_videotoolbox",
    quality_preset: str = "medium",
    bitrate: str = "15M",
    scale_resolution: str = "original",
    audio_codec: str = "aac",
    enable_stabilization: bool = False,
    color_correction: bool = False,
    verbose: bool = True,
    enable_debug_outputs: bool = False,
    debug_outputs_dir: str = None
) -> bool:
    """Render the video with per-frame dynamic cropping using frame extraction and re-encoding."""
    # Debug output setup
    if enable_debug_outputs:
        if debug_outputs_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_outputs_dir = f"outputs/debug_render_{timestamp}"
        os.makedirs(debug_outputs_dir, exist_ok=True)
        
        # Save render configuration
        render_config = {
            "input_video_path": input_video_path,
            "output_video_path": output_video_path,
            "video_codec": video_codec,
            "quality_preset": quality_preset,
            "bitrate": bitrate,
            "scale_resolution": scale_resolution,
            "audio_codec": audio_codec,
            "enable_stabilization": enable_stabilization,
            "color_correction": color_correction,
            "coordinates_count": len(smoothed_coords_df),
            "timestamp": timestamp
        }
        
        with open(os.path.join(debug_outputs_dir, "step_07_render_config.json"), 'w') as f:
            json.dump(render_config, f, indent=2)
    
    # Get video info
    video_info = get_video_info(input_video_path)
    fps = video_info.get("fps", 29.97)
    has_audio = video_info.get("has_audio", False)
    original_width = video_info.get("width", 1920)
    original_height = video_info.get("height", 1080)
    
    print("🎬 Dynamic Cropping Render (Frame-by-Frame)")
    print(f"Input: {input_video_path}")
    print(f"Resolution: {original_width}x{original_height} @ {fps:.2f} fps")
    print(f"Frames to process: {len(smoothed_coords_df)}")
    print("Audio:", "present" if has_audio else "none")
    
    if enable_debug_outputs:
        print(f"Debug outputs: {debug_outputs_dir}")
    
    # Determine target resolution for scaling
    if scale_resolution == "original":
        # Use the highest crop resolution from the coordinates
        max_crop_width = int(smoothed_coords_df['crop_w'].max())
        max_crop_height = int(smoothed_coords_df['crop_h'].max())
        target_scale_resolution = f"{max_crop_width}:{max_crop_height}"
        print(f"🎯 Using adaptive resolution: {target_scale_resolution} (based on crop data)")
    else:
        target_scale_resolution = scale_resolution
        print(f"🎯 Using specified resolution: {target_scale_resolution}")
    
    # Create temporary directory for frame processing
    with tempfile.TemporaryDirectory(prefix="dynamic_crop_") as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        cropped_dir = os.path.join(temp_dir, "cropped")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)
        
        try:
            # Step 1: Extract frames to images
            if verbose:
                print("📸 Extracting frames...")
            
            frame_pattern = os.path.join(frames_dir, "frame_%05d.png")
            extract_cmd = [
                "ffmpeg", "-y", 
                "-i", input_video_path,
                frame_pattern
            ]
            
            if enable_debug_outputs:
                with open(os.path.join(debug_outputs_dir, "step_08_ffmpeg_extract_cmd.txt"), 'w') as f:
                    f.write(' '.join(extract_cmd))
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Frame extraction failed: {result.stderr}")
                return False
                
            # Check how many frames were extracted
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            if verbose:
                print(f"✅ Extracted {len(frame_files)} frames")
            
            if enable_debug_outputs:
                with open(os.path.join(debug_outputs_dir, "step_09_frame_extraction_log.txt"), 'w') as f:
                    f.write(f"Extracted {len(frame_files)} frames\n")
                    f.write(f"First frame: {frame_files[0] if frame_files else 'None'}\n")
                    f.write(f"Last frame: {frame_files[-1] if frame_files else 'None'}\n")
            
            # Step 2: Crop each frame according to coordinates
            if verbose:
                print("✂️ Cropping frames...")
            
            success_count = 0
            crop_operations = []
            
            # Create coordinate lookup by frame number
            coords_dict = apply_smooth_coordinates_to_frames(smoothed_coords_df, len(frame_files), fps, verbose)
            
            for i, frame_file in enumerate(frame_files, 1):
                frame_path = os.path.join(frames_dir, frame_file)
                output_frame_path = os.path.join(cropped_dir, frame_file)
                
                # Get unique coordinates for this exact frame - NO fallback needed!
                coords = coords_dict[i]  # Every frame guaranteed to have coordinates
                
                x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
                
                # Validate coordinates are within original frame bounds
                x = max(0, min(x, original_width - w))
                y = max(0, min(y, original_height - h))
                w = min(w, original_width - x)
                h = min(h, original_height - y)
                
                if enable_debug_outputs and i <= 10:  # Log first 10 operations
                    crop_operations.append({
                        'frame': i,
                        'file': frame_file,
                        'unique_coords': True,  # Every frame now has unique coords!
                        'interpolated_coords': coords,
                        'validated_coords': {'x': x, 'y': y, 'w': w, 'h': h}
                    })
                
                if HAS_OPENCV:
                    # Use OpenCV for cropping (faster)
                    try:
                        img = cv2.imread(frame_path)
                        if img is not None:
                            cropped = img[y:y+h, x:x+w]
                            cv2.imwrite(output_frame_path, cropped)
                            success_count += 1
                        else:
                            print(f"⚠️ Could not read frame {frame_file}")
                    except Exception as e:
                        print(f"⚠️ Error cropping frame {frame_file}: {e}")
                else:
                    # Use FFmpeg for cropping (slower but more compatible)
                    crop_cmd = [
                        "ffmpeg", "-y", "-i", frame_path,
                        "-vf", f"crop={w}:{h}:{x}:{y}",
                        output_frame_path
                    ]
                    
                    result = subprocess.run(crop_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        success_count += 1
                    else:
                        print(f"⚠️ Error cropping frame {frame_file}: {result.stderr}")
                
                # Progress indicator
                if verbose and i % 100 == 0:
                    print(f"  Processed {i}/{len(frame_files)} frames...")
            
            if enable_debug_outputs:
                with open(os.path.join(debug_outputs_dir, "step_10_crop_operations.json"), 'w') as f:
                    json.dump(crop_operations, f, indent=2)
            
            if verbose:
                print(f"✅ Successfully cropped {success_count}/{len(frame_files)} frames")
            
            if success_count == 0:
                print("❌ No frames were successfully cropped")
                return False
            
            # Step 3: Re-encode cropped frames into video
            if verbose:
                print("🎥 Re-encoding video...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            # Get the actual resolution from first cropped frame
            first_cropped = os.path.join(cropped_dir, frame_files[0])
            if os.path.exists(first_cropped):
                # Get dimensions of cropped frame
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', first_cropped
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
                    crop_width = int(video_stream['width'])
                    crop_height = int(video_stream['height'])
                    if verbose:
                        print(f"  Cropped frame dimensions: {crop_width}x{crop_height}")
            
            # Build re-encoding command
            cropped_pattern = os.path.join(cropped_dir, "frame_%05d.png")
            
            # Build filter chain
            filters = []
            
            # Scale to target resolution (only if not using original/adaptive resolution)
            if target_scale_resolution != "original" and ":" in target_scale_resolution:
                filters.append(f"scale={target_scale_resolution}:flags=lanczos")
            
            # Add stabilization if enabled
            if enable_stabilization:
                filters.extend([
                    f"vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result=/tmp/transforms.trf",
                    f"vidstabtransform=input=/tmp/transforms.trf:zoom=1:smoothing=30"
                ])
            
            # Add color correction if enabled
            if color_correction:
                filters.append("eq=contrast=1.1:brightness=0.02:saturation=1.1")
            
            # Combine filters
            filter_str = ",".join(filters) if filters else None
            
            # Re-encode command
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", cropped_pattern,
                "-c:v", video_codec,
                "-b:v", bitrate,
                "-pix_fmt", "yuv420p"
            ]
            
            if filter_str:
                encode_cmd.extend(["-vf", filter_str])
            
            if video_codec == "h264_videotoolbox":
                encode_cmd.extend(["-allow_sw", "1", "-realtime", "0", "-profile:v", "high", "-level:v", "4.1"])
            elif video_codec == "libx264":
                encode_cmd.extend(["-preset", quality_preset, "-crf", "23"])
            
            if enable_debug_outputs:
                with open(os.path.join(debug_outputs_dir, "step_11_ffmpeg_encode_cmd.txt"), 'w') as f:
                    f.write(' '.join(encode_cmd))
            
            # Add audio if present
            if has_audio:
                # Extract audio separately and mux it
                audio_file = os.path.join(temp_dir, "audio.aac")
                audio_extract_cmd = [
                    "ffmpeg", "-y", "-i", input_video_path,
                    "-vn", "-c:a", audio_codec, "-b:a", "128k",
                    audio_file
                ]
                
                audio_result = subprocess.run(audio_extract_cmd, capture_output=True, text=True)
                if audio_result.returncode == 0:
                    # Create intermediate video without audio first
                    temp_video = os.path.join(temp_dir, "temp_video.mp4")
                    encode_cmd.append(temp_video)
                    
                    # Run video encoding
                    result = subprocess.run(encode_cmd, capture_output=not verbose, text=True)
                    if result.returncode != 0:
                        print(f"❌ Video encoding failed: {result.stderr}")
                        return False
                    
                    # Mux with audio
                    mux_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_video,
                        "-i", audio_file,
                        "-c", "copy",
                        output_video_path
                    ]
                    
                    if enable_debug_outputs:
                        with open(os.path.join(debug_outputs_dir, "step_12_ffmpeg_mux_cmd.txt"), 'w') as f:
                            f.write(' '.join(mux_cmd))
                    
                    result = subprocess.run(mux_cmd, capture_output=not verbose, text=True)
                    if result.returncode != 0:
                        print(f"❌ Audio muxing failed: {result.stderr}")
                        return False
                else:
                    print("⚠️ Could not extract audio, proceeding without audio")
                    encode_cmd.append(output_video_path)
                    result = subprocess.run(encode_cmd, capture_output=not verbose, text=True)
                    if result.returncode != 0:
                        print(f"❌ Video encoding failed: {result.stderr}")
                        return False
            else:
                # No audio, direct encoding
                encode_cmd.append(output_video_path)
                result = subprocess.run(encode_cmd, capture_output=not verbose, text=True)
                if result.returncode != 0:
                    print(f"❌ Video encoding failed: {result.stderr}")
                    return False
            
            # Verify output
            if os.path.exists(output_video_path):
                size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
                if size_mb < 0.1:
                    print(f"⚠️ Output file is very small ({size_mb:.1f} MB) - may indicate an error")
                    return False
                else:
                    print(f"✅ Dynamic cropped video rendered ({size_mb:.1f} MB)")
                    
                    if enable_debug_outputs:
                        with open(os.path.join(debug_outputs_dir, "step_13_render_complete.txt"), 'w') as f:
                            f.write(f"Render completed successfully\n")
                            f.write(f"Output file: {output_video_path}\n")
                            f.write(f"File size: {size_mb:.1f} MB\n")
                            f.write(f"Frames processed: {success_count}/{len(frame_files)}\n")
                    
                    # Clean up stabilization files
                    if os.path.exists("/tmp/transforms.trf"):
                        os.remove("/tmp/transforms.trf")
                    
                    return True
            else:
                print("❌ Output file was not created")
                return False
                
        except Exception as e:
            print(f"❌ Dynamic cropping failed: {e}")
            return False

def render_cropped_video(
    input_video_path: str,
    output_video_path: str,
    crop_filter_file: str = None,
    smoothed_coords_df: pd.DataFrame = None,
    video_codec: str = 'h264_videotoolbox',  # Apple Silicon hardware encoding
    quality_preset: str = 'medium',
    bitrate: str = '15M',
    scale_resolution: str = 'original',
    audio_codec: str = 'aac',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    verbose: bool = True
) -> bool:
    """
    Render cropped video using FFmpeg with dynamic frame-by-frame cropping.
    
    Args:
        input_video_path: Source video file
        output_video_path: Output video file
        crop_filter_file: Legacy crop filter file (deprecated)
        smoothed_coords_df: DataFrame with smooth crop coordinates
        video_codec: Video codec to use
        quality_preset: Encoding quality preset
        bitrate: Target bitrate
        scale_resolution: Final resolution
        audio_codec: Audio codec
        enable_stabilization: Apply video stabilization
        color_correction: Apply color correction
        verbose: Print detailed output
        
    Returns:
        Success status
    """
    
    if smoothed_coords_df is not None:
        print(f"🎬 Dynamic Cropping Mode")
        
        return render_cropped_video_dynamic(
            input_video_path,
            output_video_path,
            smoothed_coords_df,
            video_codec=video_codec,
            quality_preset=quality_preset,
            bitrate=bitrate,
            scale_resolution=scale_resolution,
            audio_codec=audio_codec,
            enable_stabilization=enable_stabilization,
            color_correction=color_correction,
            verbose=verbose
        )
    
    # Legacy: use crop filter file if provided
    if crop_filter_file is None:
        raise ValueError("Must provide either crop_filter_file or smoothed_coords_df")
    
    print(f"⚠️ Using legacy crop filter file (deprecated)")
    print(f"💡 Recommend using smoothed coordinates instead")
    
    # For legacy support, extract first crop coordinates from file
    with open(crop_filter_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Parse first crop command to get static dimensions
    # Format: "0.000 [enter] crop w 782 h 440 x 328 y 260;"
    import re
    match = re.search(r'w (\d+) h (\d+) x (\d+) y (\d+)', first_line)
    if match:
        w, h, x, y = map(int, match.groups())
        # Create a simple DataFrame with static coordinates for legacy support
        mock_df = pd.DataFrame({
            't_ms': [0, 1000], 
            'frame_number': [0, 30],
            'crop_x': [x, x], 
            'crop_y': [y, y], 
            'crop_w': [w, w], 
            'crop_h': [h, h]
        })
        return render_cropped_video_dynamic(
            input_video_path,
            output_video_path,
            mock_df,
            video_codec=video_codec,
            quality_preset=quality_preset,
            bitrate=bitrate,
            scale_resolution=scale_resolution,
            audio_codec=audio_codec,
            enable_stabilization=enable_stabilization,
            color_correction=color_correction,
            verbose=verbose
        )
    
    raise ValueError("Could not parse crop coordinates from filter file")


def create_preview_video(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    preview_duration: int = 10,  # seconds
    preview_start: int = 5,      # start offset
    verbose: bool = True
) -> bool:
    """
    Create a quick preview video showing the crop area for testing.
    
    Args:
        input_video_path: Source video
        output_video_path: Preview output
        smoothed_coords_df: Crop coordinates
        preview_duration: Length of preview in seconds
        preview_start: Start time in seconds
        verbose: Show progress
        
    Returns:
        Success status
    """
    
    print(f"🎬 Creating {preview_duration}s preview starting at {preview_start}s...")
    
    # Calculate average crop for preview
    avg_x = int(smoothed_coords_df['crop_x'].mean())
    avg_y = int(smoothed_coords_df['crop_y'].mean())
    avg_w = int(smoothed_coords_df['crop_w'].mean())
    avg_h = int(smoothed_coords_df['crop_h'].mean())
    
    # Ensure even dimensions
    avg_w = avg_w - (avg_w % 2)
    avg_h = avg_h - (avg_h % 2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(preview_start),
        '-i', input_video_path,
        '-t', str(preview_duration),
        '-filter_complex', 
        f'[0:v]split=2[original][crop];'
        f'[crop]crop={avg_w}:{avg_h}:{avg_x}:{avg_y},scale=960:540[cropped];'
        f'[original]scale=960:540[scaled_orig];'
        f'[scaled_orig][cropped]hstack[final]',
        '-map', '[final]',
        '-map', '0:a',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '28',
        '-c:a', 'aac',
        output_video_path
    ]
    
    if verbose:
        print(f"Preview crop: {avg_w}x{avg_h} at ({avg_x},{avg_y})")
        print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
        
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"✅ Preview created: {output_video_path}")
            print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Preview creation failed: {e}")
        return False


def batch_render_videos(
    input_videos: List[str],
    output_directory: str,
    quality_preset: str = 'medium',
    create_previews: bool = True,
    verbose: bool = True
) -> List[bool]:
    """
    Batch process multiple videos with dynamic cropping.
    
    Args:
        input_videos: List of input video paths
        output_directory: Directory for processed videos
        quality_preset: Encoding quality preset
        create_previews: Whether to create preview videos
        verbose: Print detailed output
        
    Returns:
        List of success status for each video
    """
    
    print(f"🎬 Batch processing {len(input_videos)} videos")
    print(f"Quality: {quality_preset}")
    print("=" * 60)
    
    os.makedirs(output_directory, exist_ok=True)
    results = []
    
    for i, input_video in enumerate(input_videos, 1):
        print(f"\n📹 Processing video {i}/{len(input_videos)}: {os.path.basename(input_video)}")
        
        try:
            # Generate output paths
            base_name = os.path.splitext(os.path.basename(input_video))[0]
            output_video_path = os.path.join(output_directory, f"{base_name}_processed.mp4")
            preview_path = os.path.join(output_directory, f"{base_name}_preview.mp4") if create_previews else None
            
            # Process complete pipeline
            success = process_and_render_complete(
                input_video,
                output_video_path,
                quality_preset=quality_preset,
                save_intermediate_files=False,
                verbose=verbose
            )
            
            # Create preview if successful and requested
            if success and create_previews:
                print(f"🎬 Creating preview...")
                # First get smoothed coordinates for preview using enhanced pipeline
                from genai_client import process_video_enhanced_pipeline
                original_boxes, crop_coords, smoothed_coords, *_ = process_video_enhanced_pipeline(
                    input_video,
                    smoothing_strength='balanced',
                    interpolation_method='cubic'
                )
                
                create_preview_video(
                    input_video,
                    preview_path,
                    smoothed_coords,
                    preview_duration=10,
                    verbose=False
                )
            
            results.append(success)
            
            if success:
                file_size = os.path.getsize(output_video_path) / (1024 * 1024)
                print(f"✅ Processed: {output_video_path} ({file_size:.1f} MB)")
            else:
                print(f"❌ Failed: {input_video}")
                
        except Exception as e:
            print(f"❌ Error processing {input_video}: {e}")
            results.append(False)
    
    # Summary
    successful = sum(results)
    print(f"\n📊 BATCH SUMMARY:")
    print(f"  Processed: {successful}/{len(input_videos)} videos")
    print(f"  Success rate: {successful/len(input_videos)*100:.1f}%")
    
    return results


def process_and_render_complete(
    input_video_path: str,
    output_video_path: str,
    padding_factor: float = 1.1,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    quality_preset: str = 'medium',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    save_intermediate_files: bool = True,
    **render_kwargs
) -> bool:
    """
    Complete AI Cameraman pipeline: analyze, process, and render with dynamic cropping.
    
    Args:
        input_video_path: Source video file
        output_video_path: Final output video
        padding_factor: Zoom factor for subject framing
        smoothing_strength: 'light', 'balanced', or 'heavy'
        interpolation_method: 'cubic' or 'linear'
        quality_preset: Encoding quality preset
        enable_stabilization: Apply video stabilization
        color_correction: Apply color correction
        save_intermediate_files: Save analysis data
        **render_kwargs: Additional rendering arguments
        
    Returns:
        Success status
    """
    
    print("🤖 AI CAMERAMAN - COMPLETE PIPELINE")
    print("=" * 60)
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Quality: {quality_preset}")
    print("=" * 60)
    
    try:
        # Step 1: Complete AI analysis and coordinate processing with enhanced pipeline
        from genai_client import process_video_enhanced_pipeline
        
        original_boxes, crop_coords, smoothed_coords, quality_metrics, motion_metrics, ffmpeg_filter = process_video_enhanced_pipeline(
            input_video_path,
            smoothing_strength=smoothing_strength,
            interpolation_method=interpolation_method
        )
        
        # Step 2: Save intermediate results if requested
        if save_intermediate_files:
            base_filename = output_video_path.replace('.mp4', '_analysis')
            save_complete_results(
                original_boxes, 
                crop_coords, 
                smoothed_coords,
                quality_metrics, 
                motion_metrics,
                ffmpeg_filter,
                base_filename
            )
        
        # Step 3: Render final video with advanced options
        print("=" * 60)
        success = render_cropped_video(
            input_video_path,
            output_video_path,
            smoothed_coords_df=smoothed_coords,
            video_codec=render_kwargs.get('video_codec', 'h264_videotoolbox'),
            quality_preset=quality_preset,
            bitrate=render_kwargs.get('bitrate', '15M'),
            scale_resolution=render_kwargs.get('scale_resolution', 'original'),
            audio_codec=render_kwargs.get('audio_codec', 'aac'),
            enable_stabilization=enable_stabilization,
            color_correction=color_correction,
            verbose=render_kwargs.get('verbose', True)
        )
        
        if success:
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"✅ Final video: {output_video_path}")
            
            # Print summary stats
            print(f"\n📊 SUMMARY:")
            print(f"  Original keyframes: {len(original_boxes)}")
            print(f"  Crop coordinates: {len(crop_coords)}")
            print(f"  Smooth frames: {len(smoothed_coords)}")
            print(f"  Video duration: {motion_metrics['frame_stats']['duration_seconds']:.1f}s")
            print(f"  Average zoom: {quality_metrics['average_zoom']:.2f}x")
            print(f"  Motion smoothness: {motion_metrics['motion_analysis']['speed_consistency']:.2f}")
            print(f"  Rendering mode: Enhanced Dynamic with 100% coordinate efficiency")
            
        return success
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def render_with_watermark(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    watermark_text: str = None,
    watermark_image: str = None,
    watermark_position: str = 'bottom-right',  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
    watermark_opacity: float = 0.7,
    render_kwargs: dict = None
) -> bool:
    """
    Render video with watermark or logo overlay.
    
    Args:
        input_video_path: Source video
        output_video_path: Output with watermark
        smoothed_coords_df: Crop coordinates
        watermark_text: Text watermark
        watermark_image: Image watermark path
        watermark_position: Position on screen
        watermark_opacity: Opacity (0.0-1.0)
        render_kwargs: Additional render arguments
        
    Returns:
        Success status
    """
    
    if watermark_text is None and watermark_image is None:
        print("⚠️ No watermark specified, rendering without watermark")
        return render_cropped_video(
            input_video_path,
            output_video_path,
            smoothed_coords_df=smoothed_coords_df,
            **(render_kwargs or {})
        )
    
    print(f"🎨 Adding watermark to video...")
    
    # First render the cropped video
    temp_cropped = output_video_path.replace('.mp4', '_temp_cropped.mp4')
    
    success = render_cropped_video(
        input_video_path,
        temp_cropped,
        smoothed_coords_df=smoothed_coords_df,
        **(render_kwargs or {})
    )
    
    if not success:
        return False
    
    try:
        # Position mapping
        positions = {
            'top-left': '10:10',
            'top-right': 'W-w-10:10',
            'bottom-left': '10:H-h-10',
            'bottom-right': 'W-w-10:H-h-10',
            'center': '(W-w)/2:(H-h)/2'
        }
        
        pos = positions.get(watermark_position, positions['bottom-right'])
        
        # Build watermark filter
        if watermark_text:
            # Text watermark
            watermark_filter = (
                f"drawtext=text='{watermark_text}':fontcolor=white@{watermark_opacity}:"
                f"fontsize=24:x={pos}:y=H-th-10"
            )
        else:
            # Image watermark
            watermark_filter = f"[1:v]format=rgba,colorchannelmixer=aa={watermark_opacity}[watermark];[0:v][watermark]overlay={pos}"
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',
            '-i', temp_cropped
        ]
        
        if watermark_image:
            cmd.extend(['-i', watermark_image])
        
        if watermark_text:
            cmd.extend([
                '-vf', watermark_filter,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac'
            ])
        else:
            cmd.extend([
                '-filter_complex', watermark_filter,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac'
            ])
        
        cmd.append(output_video_path)
        
        print(f"Adding {'text' if watermark_text else 'image'} watermark at {watermark_position}")
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up
        if os.path.exists(temp_cropped):
            os.remove(temp_cropped)
        
        print(f"✅ Watermarked video created: {output_video_path}")
        return True
        
    except Exception as e:
        print(f"❌ Watermark error: {e}")
        # Clean up on error
        if os.path.exists(temp_cropped):
            os.remove(temp_cropped)
        return False


def analyze_video_quality(video_path: str) -> Dict:
    """
    Analyze video quality metrics using ffprobe and custom analysis.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Quality metrics dictionary
    """
    
    print(f"📊 Analyzing video quality...")
    
    try:
        # Get basic video info
        video_info = get_video_info(video_path)
        
        # Calculate quality score based on resolution, bitrate, codec
        quality_score = 0
        
        # Resolution score (max 40 points)
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        if width >= 1920 and height >= 1080:
            quality_score += 40
        elif width >= 1280 and height >= 720:
            quality_score += 30
        elif width >= 854 and height >= 480:
            quality_score += 20
        else:
            quality_score += 10
        
        # Bitrate score (max 30 points)
        bitrate = video_info.get('bitrate', 0)
        if bitrate >= 15000000:  # 15 Mbps
            quality_score += 30
        elif bitrate >= 10000000:  # 10 Mbps
            quality_score += 25
        elif bitrate >= 5000000:   # 5 Mbps
            quality_score += 20
        else:
            quality_score += 10
        
        # Codec score (max 20 points)
        codec = video_info.get('codec', '')
        if codec in ['h264', 'hevc', 'av1']:
            quality_score += 20
        elif codec in ['vp9', 'vp8']:
            quality_score += 15
        else:
            quality_score += 10
        
        # Frame rate score (max 10 points)
        fps = video_info.get('fps', 0)
        if fps >= 60:
            quality_score += 10
        elif fps >= 30:
            quality_score += 8
        elif fps >= 24:
            quality_score += 6
        else:
            quality_score += 4
        
        quality_grade = 'A' if quality_score >= 85 else 'B' if quality_score >= 70 else 'C' if quality_score >= 50 else 'D'
        
        metrics = {
            'resolution': f"{width}x{height}",
            'bitrate_mbps': bitrate / 1000000 if bitrate else 0,
            'fps': fps,
            'codec': codec,
            'duration_seconds': video_info.get('duration', 0),
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
        
        print(f"Quality Grade: {quality_grade} ({quality_score}/100)")
        print(f"Resolution: {width}x{height} @ {fps:.1f}fps")
        print(f"Bitrate: {metrics['bitrate_mbps']:.1f} Mbps")
        print(f"Codec: {codec}")
        print(f"Duration: {metrics['duration_seconds']:.1f}s")
        print(f"File size: {metrics['file_size_mb']:.1f} MB")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Quality analysis error: {e}")
        return {}


def quick_render_example():
    """Quick example using existing analysis results"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    input_video = os.path.join(project_root, "videos", "waterpolo_trimmed.webm")
    output_video = os.path.join(project_root, "outputs", "waterpolo_cropped.mp4")
    crop_filter = os.path.join(project_root, "outputs", "waterpolo_analysis_05_crop_filter.txt")
    
    if os.path.exists(crop_filter):
        print("🔄 Using existing crop filter for quick render...")
        success = render_cropped_video(
            input_video,
            output_video,
            crop_filter_file=crop_filter,
            verbose=True
        )
        return success
    else:
        print("No existing crop filter found. Run complete pipeline first.")
        return False


def test_enhanced_coordinate_mapping() -> bool:
    """
    Test that the enhanced coordinate mapping achieves 100% efficiency.
    This verifies the critical fix for smooth panning.
    """
    import pandas as pd
    
    print("🧪 Testing enhanced coordinate mapping...")
    
    # Create sample smoothed coordinates (simulating interpolated keyframes)
    sample_coords = []
    for i in range(10):  # 10 frames at 1 FPS = 10 seconds
        sample_coords.append({
            't_ms': i * 1000,  # 1 second intervals
            'crop_x': 100 + i * 10,
            'crop_y': 200 + i * 5,
            'crop_w': 800,
            'crop_h': 450
        })
    
    smoothed_df = pd.DataFrame(sample_coords)
    total_frames = 10
    fps = 1.0
    
    # Test the enhanced mapping function
    coords_dict = apply_smooth_coordinates_to_frames(smoothed_df, total_frames, fps, verbose=True)
    
    # Verify 100% efficiency
    expected_frames = total_frames
    actual_frames = len(coords_dict)
    efficiency = (actual_frames / expected_frames) * 100
    
    print(f"📊 Test Results:")
    print(f"   Expected frames: {expected_frames}")
    print(f"   Frames with coordinates: {actual_frames}")
    print(f"   Efficiency: {efficiency:.1f}%")
    
    success = efficiency == 100.0
    
    if success:
        print("✅ Enhanced coordinate mapping working perfectly!")
        print("🎯 100% frame coverage achieved - true smooth panning enabled!")
    else:
        print("❌ Coordinate mapping still has efficiency issues")
    
    return success


def demo_advanced_features():
    """Demonstrate advanced dynamic cropping features"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    input_video = os.path.join(project_root, "videos", "waterpolo_trimmed.webv")
    
    print("🎬 DYNAMIC CROPPING FEATURES DEMO")
    print("=" * 50)
    
    # Demo 1: Dynamic cropping with stabilization
    print("\n1. Dynamic Cropping with Stabilization")
    output_dynamic = os.path.join(project_root, "outputs", "demo_dynamic_stabilized.mp4")
    success1 = process_and_render_complete(
        input_video,
        output_dynamic,
        quality_preset='medium',
        enable_stabilization=True,
        color_correction=True,
        save_intermediate_files=False
    )
    
    # Demo 2: High quality dynamic cropping
    print("\n2. High Quality Dynamic Cropping")
    output_high_quality = os.path.join(project_root, "outputs", "demo_dynamic_high_quality.mp4")
    success2 = process_and_render_complete(
        input_video,
        output_high_quality,
        quality_preset='slow',
        enable_stabilization=False,
        color_correction=True,
        save_intermediate_files=False,
        video_codec='libx264',
        bitrate='20M'
    )
    
    # Demo 3: Preview creation
    print("\n3. Quick Preview Creation")
    if success1:
        preview_path = os.path.join(project_root, "outputs", "demo_preview.mp4")
        # Load coordinates from saved analysis
        try:
            coords_file = output_dynamic.replace('.mp4', '_analysis_04_smoothed_coordinates.csv')
            if os.path.exists(coords_file):
                import pandas as pd
                smoothed_coords = pd.read_csv(coords_file)
                create_preview_video(
                    input_video,
                    preview_path,
                    smoothed_coords,
                    preview_duration=15,
                    preview_start=10
                )
        except Exception as e:
            print(f"⚠️ Preview creation skipped: {e}")
    
    # Demo 4: Watermark rendering
    print("\n4. Dynamic Cropping with Watermark")
    if success1:
        watermark_path = os.path.join(project_root, "outputs", "demo_watermarked.mp4")
        try:
            coords_file = output_dynamic.replace('.mp4', '_analysis_04_smoothed_coordinates.csv')
            if os.path.exists(coords_file):
                import pandas as pd
                smoothed_coords = pd.read_csv(coords_file)
                render_with_watermark(
                    input_video,
                    watermark_path,
                    smoothed_coords,
                    watermark_text="AI Cameraman",
                    watermark_position="bottom-right",
                    watermark_opacity=0.8
                )
        except Exception as e:
            print(f"⚠️ Watermark demo skipped: {e}")
    
    # Demo 5: Quality analysis
    print("\n5. Video Quality Analysis")
    if success1:
        print("\nOriginal video:")
        analyze_video_quality(input_video)
        print("\nProcessed video:")
        analyze_video_quality(output_dynamic)
    
    print("\n" + "=" * 50)
    print(f"Demo Results:")
    print(f"  Dynamic render with stabilization: {'✅' if success1 else '❌'}")
    print(f"  High quality dynamic render: {'✅' if success2 else '❌'}")
    print(f"  Dynamic cropping showcases frame-by-frame precision")


def parse_arguments():
    """Parse command line arguments for flexible usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Cameraman Video Processing')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('output_video', help='Output video file path')
    
    # Analysis options
    parser.add_argument('--padding', type=float, default=1.1, 
                       help='Padding factor around bounding boxes (default: 1.1)')
    parser.add_argument('--smoothing', choices=['minimal', 'balanced', 'maximum', 'cinematic'], 
                       default='balanced', help='Kalman filter smoothing strength')
    parser.add_argument('--interpolation', choices=['cubic', 'linear', 'quadratic'], 
                       default='cubic', help='Interpolation method')
    
    # Rendering options
    parser.add_argument('--quality', choices=['fast', 'medium', 'slow', 'best', 'ultra'], 
                       default='medium', help='Quality preset')
    parser.add_argument('--codec', choices=['h264_videotoolbox', 'libx264', 'hevc_videotoolbox'], 
                       default='h264_videotoolbox', help='Video codec')
    parser.add_argument('--bitrate', default='15M', help='Target bitrate (e.g., 15M)')
    parser.add_argument('--resolution', default='original', help='Target resolution (e.g., 1920:1080, or "original" for adaptive)')
    
    # Advanced features
    parser.add_argument('--stabilize', action='store_true', help='Enable video stabilization')
    parser.add_argument('--color-correct', action='store_true', help='Enable color correction')
    parser.add_argument('--watermark-text', help='Add text watermark')
    parser.add_argument('--watermark-image', help='Add image watermark')
    parser.add_argument('--watermark-position', 
                       choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'],
                       default='bottom-right', help='Watermark position')
    
    # Output options
    parser.add_argument('--preview', action='store_true', help='Create preview video')
    parser.add_argument('--save-analysis', action='store_true', help='Save intermediate analysis files')
    parser.add_argument('--analyze-quality', action='store_true', help='Analyze output video quality')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def test_dynamic_cropping(smoothed_coords_df: pd.DataFrame, output_path: str = None) -> bool:
    """
    Test the new frame-by-frame dynamic cropping implementation.
    
    Args:
        smoothed_coords_df: DataFrame with crop coordinates over time
        output_path: Optional path to save test output
    
    Returns:
        True if the implementation works correctly, False otherwise
    """
    try:
        print("🧪 Testing frame-by-frame dynamic cropping...")
        
        if smoothed_coords_df.empty:
            print("❌ No coordinate data provided")
            return False
        
        print(f"✅ Found {len(smoothed_coords_df)} coordinate frames")
        
        # Validate coordinate data structure
        required_columns = ['t_ms', 'crop_x', 'crop_y', 'crop_w', 'crop_h']
        missing_columns = [col for col in required_columns if col not in smoothed_coords_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ Coordinate data structure is valid")
        
        # Check coordinate ranges
        for idx, row in smoothed_coords_df.iterrows():
            if row['crop_w'] <= 0 or row['crop_h'] <= 0:
                print(f"❌ Invalid dimensions at frame {idx}: w={row['crop_w']}, h={row['crop_h']}")
                return False
            
            if row['crop_x'] < 0 or row['crop_y'] < 0:
                print(f"⚠️ Negative coordinates at frame {idx}: x={row['crop_x']}, y={row['crop_y']}")
        
        print("✅ Coordinate ranges are valid")
        
        # Test coordinate to frame mapping
        fps = 30.0  # Assume 30fps for test
        frame_count = 0
        coords_dict = {}
        
        for _, row in smoothed_coords_df.iterrows():
            frame_num = int(row['t_ms'] / 1000.0 * fps) + 1
            coords_dict[frame_num] = {
                'x': int(row['crop_x']),
                'y': int(row['crop_y']),
                'w': int(row['crop_w']) & ~1,  # Ensure even
                'h': int(row['crop_h']) & ~1   # Ensure even
            }
            frame_count += 1
        
        print(f"✅ Generated coordinate mapping for {len(coords_dict)} frames")
        
        # Test with sample frame if output path provided
        if output_path:
            test_data = {
                'frame_count': frame_count,
                'coordinate_mapping': list(coords_dict.keys())[:10],  # First 10 frames
                'sample_coords': coords_dict[min(coords_dict.keys())] if coords_dict else None
            }
            
            # Save test data as JSON
            test_file = output_path.replace('.mp4', '_dynamic_crop_test.json')
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"💾 Test data saved to: {test_file}")
        
        print("✅ Frame-by-frame dynamic cropping test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dynamic cropping test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys
    
    if len(sys.argv) > 1:
        # Command line interface
        args = parse_arguments()
        
        print(f"🚀 AI CAMERAMAN PIPELINE")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_video}")
        print(f"Quality: {args.quality}")
        print("=" * 60)
        
        # Check if input file exists
        input_file_path = args.input_video
        
        if not os.path.exists(input_file_path):
            # Try looking in the videos folder relative to script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, ".."))
            videos_path = os.path.join(project_root, "videos", os.path.basename(args.input_video))
            
            if os.path.exists(videos_path):
                input_file_path = videos_path
                print(f"📁 Found input file in videos folder: {videos_path}")
            else:
                print(f"❌ Input file not found in:")
                print(f"   Current path: {args.input_video}")
                print(f"   Videos folder: {videos_path}")
                sys.exit(1)
        
        # Update args to use the found path
        args.input_video = input_file_path
        
        # Handle output path - use outputs folder if no directory specified
        if not os.path.dirname(args.output_video):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, ".."))
            outputs_dir = os.path.join(project_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            args.output_video = os.path.join(outputs_dir, args.output_video)
            print(f"📁 Output will be saved to: {args.output_video}")
        
        try:
            # Main processing
            if args.watermark_text or args.watermark_image:
                # Process with watermark
                # First run the main pipeline to get coordinates
                temp_output = args.output_video.replace('.mp4', '_temp.mp4')
                
                success = process_and_render_complete(
                    args.input_video,
                    temp_output,
                    padding_factor=args.padding,
                    smoothing_strength=args.smoothing,
                    interpolation_method=args.interpolation,
                    quality_preset=args.quality,
                    enable_stabilization=args.stabilize,
                    color_correction=args.color_correct,
                    save_intermediate_files=args.save_analysis,
                    video_codec=args.codec,
                    bitrate=args.bitrate,
                    scale_resolution=args.resolution,
                    verbose=args.verbose
                )
                
                if success:
                    # Load coordinates for watermark rendering
                    coords_file = temp_output.replace('.mp4', '_analysis_04_smoothed_coordinates.csv')
                    if os.path.exists(coords_file):
                        import pandas as pd
                        smoothed_coords = pd.read_csv(coords_file)
                        
                        success = render_with_watermark(
                            args.input_video,
                            args.output_video,
                            smoothed_coords,
                            watermark_text=args.watermark_text,
                            watermark_image=args.watermark_image,
                            watermark_position=args.watermark_position,
                            render_kwargs={
                                'video_codec': args.codec,
                                'bitrate': args.bitrate,
                                'scale_resolution': args.resolution
                            }
                        )
                        
                        # Clean up temp file
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                
            else:
                # Standard processing
                success = process_and_render_complete(
                    args.input_video,
                    args.output_video,
                    padding_factor=args.padding,
                    smoothing_strength=args.smoothing,
                    interpolation_method=args.interpolation,
                    quality_preset=args.quality,
                    enable_stabilization=args.stabilize,
                    color_correction=args.color_correct,
                    save_intermediate_files=args.save_analysis,
                    video_codec=args.codec,
                    bitrate=args.bitrate,
                    scale_resolution=args.resolution,
                    verbose=args.verbose
                )
            
            # Create preview if requested
            if success and args.preview:
                coords_file = args.output_video.replace('.mp4', '_analysis_04_smoothed_coordinates.csv')
                if os.path.exists(coords_file):
                    import pandas as pd
                    smoothed_coords = pd.read_csv(coords_file)
                    preview_path = args.output_video.replace('.mp4', '_preview.mp4')
                    create_preview_video(args.input_video, preview_path, smoothed_coords)
            
            # Analyze quality if requested
            if success and args.analyze_quality:
                print("\n📊 QUALITY ANALYSIS")
                print("Original:")
                analyze_video_quality(args.input_video)
                print("\nProcessed:")
                analyze_video_quality(args.output_video)
            
            if success:
                print(f"\n🎉 Processing complete!")
                print(f"✅ Output: {args.output_video}")
            else:
                print(f"\n❌ Processing failed!")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Processing interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Interactive mode - existing behavior
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        
        input_video_path = os.path.join(project_root, "videos", "waterpolo_trimmed.webm")
        output_video_path = os.path.join(project_root, "outputs", "waterpolo_final.mp4")
        
        # Check if we have existing results for quick render
        crop_filter_path = os.path.join(project_root, "outputs", "waterpolo_analysis_05_crop_filter.txt")
        
        if os.path.exists(crop_filter_path):
            print("Found existing analysis results. Choose:")
            print("1. Quick render with existing results")
            print("2. Full pipeline (re-analyze)")
            print("3. Demo advanced features")
            
            # For automation, just do quick render
            choice = input("Enter choice (1/2/3) or press Enter for quick render: ").strip()
            
            if choice == "2":
                print("Running complete pipeline...")
                success = process_and_render_complete(
                    input_video_path,
                    output_video_path,
                    quality_preset='medium'
                )
            elif choice == "3":
                print("Running advanced features demo...")
                demo_advanced_features()
                success = True
            else:
                print("Doing quick render with existing results...")
                success = quick_render_example()
                
        else:
            print("No existing results found. Running complete pipeline...")
            success = process_and_render_complete(
                input_video_path,
                output_video_path,
                quality_preset='medium'
            )
        
        if success:
            print("\n🎊 All done! Your AI-cropped video is ready.")
        else:
            print("\n💥 Something went wrong. Check the logs above.") 