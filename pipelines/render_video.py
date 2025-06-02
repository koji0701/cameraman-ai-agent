import os
import subprocess
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from genai_client import process_video_complete_pipeline, save_complete_results
from normalize_coordinates import generate_ffmpeg_crop_filter
from kalman_smoother import generate_smooth_ffmpeg_filter


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
        
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
            'duration': float(video_stream['duration']),
            'bitrate': int(video_stream.get('bit_rate', 0)),
            'codec': video_stream['codec_name'],
            'pixel_format': video_stream['pix_fmt'],
            'total_frames': int(video_stream['nb_frames']) if 'nb_frames' in video_stream else None
        }
    except Exception as e:
        print(f"⚠️ Could not get video info: {e}")
        return {}


def generate_sendcmd_filter(smoothed_coords_df: pd.DataFrame, fps: float = 29.97) -> str:
    """
    Generate FFmpeg sendcmd filter for dynamic frame-by-frame cropping.
    
    Args:
        smoothed_coords_df: DataFrame with crop coordinates per frame
        fps: Video frame rate
        
    Returns:
        FFmpeg sendcmd filter string
    """
    commands = []
    
    for idx, row in smoothed_coords_df.iterrows():
        timestamp = row['frame_number'] / fps
        
        # Ensure even dimensions for codec compatibility
        w = int(row['crop_w']) - (int(row['crop_w']) % 2)
        h = int(row['crop_h']) - (int(row['crop_h']) % 2)
        x = int(row['crop_x'])
        y = int(row['crop_y'])
        
        # FFmpeg sendcmd format: timestamp [enter] filtername param1 value1 param2 value2;
        cmd = f"{timestamp:.3f} [enter] crop w {w} h {h} x {x} y {y};"
        commands.append(cmd)
    
    # Create sendcmd filter
    sendcmd_content = "\\n".join(commands)
    sendcmd_filter = f"sendcmd=c='{sendcmd_content}'"
    
    return sendcmd_filter


def create_dynamic_crop_filter(smoothed_coords_df: pd.DataFrame, fps: float = 29.97) -> str:
    """
    Create a complete FFmpeg filter graph for dynamic cropping with smooth transitions.
    
    Returns a complex filter that includes:
    - Dynamic cropping with sendcmd
    - Smooth transitions between crop changes
    - Scale to target resolution
    """
    
    # Generate sendcmd for dynamic cropping
    sendcmd = generate_sendcmd_filter(smoothed_coords_df, fps)
    
    # Complete filter graph with smooth transitions
    filter_graph = (
        f"[0:v]{sendcmd}[crop_cmd];"
        "[crop_cmd]crop=w=1920:h=1080:x=0:y=0:keep_aspect=1[cropped];"
        "[cropped]scale=1920:1080:flags=lanczos[scaled]"
    )
    
    return filter_graph


def render_cropped_video_dynamic(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    video_codec: str = 'h264_videotoolbox',
    quality_preset: str = 'medium',
    bitrate: str = '15M',
    scale_resolution: str = '1920:1080',
    audio_codec: str = 'aac',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    verbose: bool = True
) -> bool:
    """
    Render video with dynamic frame-by-frame cropping using FFmpeg sendcmd.
    
    Args:
        input_video_path: Source video file
        output_video_path: Output video file
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
    
    # Get video information
    video_info = get_video_info(input_video_path)
    fps = video_info.get('fps', 29.97)
    
    print(f"🎬 Dynamic Cropping Render")
    print(f"Input: {input_video_path}")
    print(f"Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')} @ {fps:.2f}fps")
    print(f"Frames to process: {len(smoothed_coords_df)}")
    
    # Create sendcmd file for dynamic cropping
    sendcmd_file = output_video_path.replace('.mp4', '_sendcmd.txt')
    
    commands = []
    for idx, row in smoothed_coords_df.iterrows():
        timestamp = row['frame_number'] / fps
        
        # Ensure even dimensions
        w = int(row['crop_w']) - (int(row['crop_w']) % 2)
        h = int(row['crop_h']) - (int(row['crop_h']) % 2)
        x = int(row['crop_x'])
        y = int(row['crop_y'])
        
        cmd = f"{timestamp:.3f} [enter] crop w {w} h {h} x {x} y {y};"
        commands.append(cmd)
    
    # Write sendcmd file
    with open(sendcmd_file, 'w') as f:
        f.write('\n'.join(commands))
    
    # Build complex filter graph
    filters = []
    
    # Dynamic cropping with sendcmd
    filters.append(f"[0:v]sendcmd=f='{sendcmd_file}',crop=w=1920:h=1080[cropped]")
    
    # Optional stabilization
    if enable_stabilization:
        filters.append("[cropped]vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result=/tmp/transforms.trf[stab_detect]")
        filters.append("[stab_detect]vidstabtransform=input=/tmp/transforms.trf:zoom=1:smoothing=30[stabilized]")
        current_label = "stabilized"
    else:
        current_label = "cropped"
    
    # Optional color correction
    if color_correction:
        filters.append(f"[{current_label}]eq=contrast=1.1:brightness=0.02:saturation=1.1[color_corrected]")
        current_label = "color_corrected"
    
    # Final scaling
    filters.append(f"[{current_label}]scale={scale_resolution}:flags=lanczos[final]")
    
    filter_complex = ';'.join(filters)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', input_video_path,
        '-filter_complex', filter_complex,
        '-map', '[final]',
        '-map', '0:a',  # Copy audio
        '-c:v', video_codec,
        '-b:v', bitrate,
        '-c:a', audio_codec,
    ]
    
    # Add codec-specific options
    if video_codec == 'h264_videotoolbox':
        cmd.extend([
            '-allow_sw', '1',
            '-realtime', '0',
            '-profile:v', 'high',
            '-level:v', '4.1'
        ])
        
        # Quality presets for VideoToolbox
        quality_map = {
            'fast': ['-q:v', '30'],
            'medium': ['-q:v', '25'], 
            'slow': ['-q:v', '20'],
            'best': ['-q:v', '15']
        }
        if quality_preset in quality_map:
            cmd.extend(quality_map[quality_preset])
    
    elif video_codec == 'libx264':
        cmd.extend([
            '-preset', quality_preset,
            '-tune', 'film',
            '-movflags', '+faststart'
        ])
    
    cmd.append(output_video_path)
    
    print(f"🎨 Applying {len(smoothed_coords_df)} dynamic crop adjustments...")
    if enable_stabilization:
        print("📹 Video stabilization enabled")
    if color_correction:
        print("🎨 Color correction enabled")
    
    if verbose:
        print(f"FFmpeg command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        print(f"✅ Dynamic crop video rendered successfully!")
        
        # Clean up temporary files
        if os.path.exists(sendcmd_file):
            os.remove(sendcmd_file)
        if os.path.exists('/tmp/transforms.trf'):
            os.remove('/tmp/transforms.trf')
        
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"Output: {output_video_path}")
            print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg first.")
        print("Install with: brew install ffmpeg")
        return False


def render_multipass_video(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    target_quality: str = 'high',  # 'medium', 'high', 'ultra'
    verbose: bool = True
) -> bool:
    """
    Render video using multi-pass encoding for optimal quality/size ratio.
    
    Args:
        input_video_path: Source video
        output_video_path: Final output
        smoothed_coords_df: Smooth coordinates
        target_quality: Quality level
        verbose: Show progress
        
    Returns:
        Success status
    """
    
    print(f"🎯 Multi-pass encoding for {target_quality} quality")
    
    # Quality settings
    quality_settings = {
        'medium': {'crf': 23, 'preset': 'medium', 'bitrate': '10M'},
        'high': {'crf': 20, 'preset': 'slow', 'bitrate': '15M'},
        'ultra': {'crf': 18, 'preset': 'veryslow', 'bitrate': '20M'}
    }
    
    settings = quality_settings.get(target_quality, quality_settings['high'])
    
    # Temporary files
    temp_cropped = output_video_path.replace('.mp4', '_temp_cropped.mp4')
    pass1_log = output_video_path.replace('.mp4', '_pass1.log')
    
    try:
        # Pass 1: Create cropped video (fast preset)
        print("📹 Pass 1: Creating cropped video...")
        success = render_cropped_video_simple(
            input_video_path,
            temp_cropped,
            smoothed_coords_df,
            video_codec='libx264',
            bitrate=settings['bitrate'],
            verbose=verbose
        )
        
        if not success:
            return False
        
        # Pass 2: Optimal encoding
        print("🎨 Pass 2: Optimizing encoding...")
        
        cmd = [
            'ffmpeg',
            '-y',
            '-i', temp_cropped,
            '-c:v', 'libx264',
            '-preset', settings['preset'],
            '-crf', str(settings['crf']),
            '-tune', 'film',
            '-movflags', '+faststart',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_video_path
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        print(f"✅ Multi-pass encoding complete!")
        
        # Clean up
        if os.path.exists(temp_cropped):
            os.remove(temp_cropped)
        
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"Final output: {output_video_path}")
            print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-pass encoding error: {e}")
        # Clean up on error
        for temp_file in [temp_cropped, pass1_log]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return False


def render_cropped_video_simple(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    video_codec: str = 'h264_videotoolbox',
    bitrate: str = '15M',
    scale_resolution: str = '1920:1080',
    audio_codec: str = 'aac',
    verbose: bool = True
) -> bool:
    """
    Render cropped video using a simpler approach with average crop coordinates.
    """
    
    # Calculate average crop coordinates for a static crop
    avg_x = int(smoothed_coords_df['crop_x'].mean())
    avg_y = int(smoothed_coords_df['crop_y'].mean())
    avg_w = int(smoothed_coords_df['crop_w'].mean())
    avg_h = int(smoothed_coords_df['crop_h'].mean())
    
    # Ensure even dimensions
    avg_w = avg_w - (avg_w % 2)
    avg_h = avg_h - (avg_h % 2)
    
    print(f"Using average crop: {avg_w}x{avg_h} at ({avg_x},{avg_y})")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Build simple FFmpeg command with static crop
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-i', input_video_path,
        '-filter_complex', f'[0:v]crop={avg_w}:{avg_h}:{avg_x}:{avg_y}[cropped];[cropped]scale={scale_resolution}',
        '-c:v', video_codec,
        '-b:v', bitrate,
        '-c:a', audio_codec,
        '-map', '0:a',
        '-vsync', '2',
    ]
    
    # Add hardware acceleration flags for Apple Silicon
    if video_codec == 'h264_videotoolbox':
        cmd.extend(['-allow_sw', '1'])
    
    cmd.append(output_video_path)
    
    print(f"🎬 Rendering video with static crop...")
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Codec: {video_codec} @ {bitrate}")
    
    if verbose:
        print(f"FFmpeg command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        print(f"✅ Video rendered successfully!")
        print(f"Output file: {output_video_path}")
        
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg first.")
        print("Install with: brew install ffmpeg")
        return False


def render_cropped_video(
    input_video_path: str,
    output_video_path: str,
    crop_filter_file: str = None,
    smoothed_coords_df: pd.DataFrame = None,
    video_codec: str = 'h264_videotoolbox',  # Apple Silicon hardware encoding
    rendering_mode: str = 'simple',  # 'simple', 'dynamic', 'multipass'
    quality_preset: str = 'medium',
    bitrate: str = '15M',
    scale_resolution: str = '1920:1080',
    audio_codec: str = 'aac',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    verbose: bool = True
) -> bool:
    """
    Render cropped video using FFmpeg with multiple rendering modes.
    
    Args:
        input_video_path: Source video file
        output_video_path: Output video file
        crop_filter_file: Legacy crop filter file (deprecated)
        smoothed_coords_df: DataFrame with smooth crop coordinates
        video_codec: Video codec to use
        rendering_mode: 'simple' (static crop), 'dynamic' (frame-by-frame), 'multipass' (quality optimized)
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
        print(f"🎬 Rendering Mode: {rendering_mode.upper()}")
        
        if rendering_mode == 'dynamic':
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
        
        elif rendering_mode == 'multipass':
            return render_multipass_video(
                input_video_path,
                output_video_path,
                smoothed_coords_df,
                target_quality=quality_preset,
                verbose=verbose
            )
        
        else:  # 'simple' mode
            return render_cropped_video_simple(
                input_video_path,
                output_video_path,
                smoothed_coords_df,
                video_codec,
                bitrate,
                scale_resolution,
                audio_codec,
                verbose
            )
    
    # Legacy: use crop filter file if provided
    if crop_filter_file is None:
        raise ValueError("Must provide either crop_filter_file or smoothed_coords_df")
    
    print(f"⚠️ Using legacy crop filter file (deprecated)")
    print(f"💡 Recommend using smoothed coordinates with rendering_mode instead")
    
    # For now, extract first crop coordinates from file
    with open(crop_filter_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Parse first crop command to get static dimensions
    # Format: "0.000 [enter] crop w 782 h 440 x 328 y 260;"
    import re
    match = re.search(r'w (\d+) h (\d+) x (\d+) y (\d+)', first_line)
    if match:
        w, h, x, y = map(int, match.groups())
        mock_df = pd.DataFrame({'crop_x': [x], 'crop_y': [y], 'crop_w': [w], 'crop_h': [h]})
        return render_cropped_video_simple(
            input_video_path,
            output_video_path,
            mock_df,
            video_codec,
            bitrate,
            scale_resolution,
            audio_codec,
            verbose
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
    rendering_mode: str = 'simple',
    quality_preset: str = 'medium',
    create_previews: bool = True,
    verbose: bool = True
) -> List[bool]:
    """
    Batch process multiple videos with the same settings.
    
    Args:
        input_videos: List of input video paths
        output_directory: Directory for outputs
        rendering_mode: Rendering mode for all videos
        quality_preset: Quality preset for all videos
        create_previews: Create preview videos
        verbose: Show progress
        
    Returns:
        List of success statuses for each video
    """
    
    print(f"🎬 Batch processing {len(input_videos)} videos")
    print(f"Mode: {rendering_mode}, Quality: {quality_preset}")
    print("=" * 60)
    
    os.makedirs(output_directory, exist_ok=True)
    results = []
    
    for i, input_video in enumerate(input_videos, 1):
        print(f"\n📹 Processing {i}/{len(input_videos)}: {os.path.basename(input_video)}")
        
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_video))[0]
            output_video = os.path.join(output_directory, f"{base_name}_cropped.mp4")
            
            # Run complete pipeline
            success = process_and_render_complete(
                input_video,
                output_video,
                padding_factor=1.1,
                smoothing_strength='balanced',
                interpolation_method='cubic',
                rendering_mode=rendering_mode,
                quality_preset=quality_preset,
                enable_stabilization=False,
                color_correction=False,
                save_intermediate_files=verbose
            )
            
            results.append(success)
            
            # Create preview if requested and successful
            if success and create_previews:
                preview_path = os.path.join(output_directory, f"{base_name}_preview.mp4")
                print(f"🎬 Creating preview for {base_name}...")
                
                # Need to load the smoothed coordinates for preview
                # This is a simplified approach - in practice, you'd save/load the coordinates
                print("⚠️ Preview creation requires saved coordinate data")
            
            if success:
                print(f"✅ {base_name} completed successfully")
            else:
                print(f"❌ {base_name} failed")
                
        except Exception as e:
            print(f"❌ Error processing {input_video}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    successful = sum(results)
    print(f"🎉 Batch complete: {successful}/{len(input_videos)} videos processed successfully")
    
    return results


def process_and_render_complete(
    input_video_path: str,
    output_video_path: str,
    padding_factor: float = 1.1,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    rendering_mode: str = 'simple',  # 'simple', 'dynamic', 'multipass'
    quality_preset: str = 'medium',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    save_intermediate_files: bool = True,
    **render_kwargs
) -> bool:
    """
    Complete end-to-end pipeline: Analyze → Normalize → Smooth → Render
    
    Args:
        input_video_path: Source video file
        output_video_path: Final cropped video file
        padding_factor: Padding around bounding boxes (1.1 = 10%)
        smoothing_strength: Kalman filter strength ('minimal', 'balanced', 'maximum', 'cinematic')
        interpolation_method: Interpolation method ('cubic', 'linear', 'quadratic')
        rendering_mode: Rendering mode ('simple', 'dynamic', 'multipass')
        quality_preset: Quality preset ('fast', 'medium', 'slow', 'best')
        enable_stabilization: Apply video stabilization
        color_correction: Apply color correction
        save_intermediate_files: Save analysis files for debugging
        **render_kwargs: Additional arguments for render_cropped_video
        
    Returns:
        Success status
    """
    
    print(f"🚀 STARTING COMPLETE AI CAMERAMAN PIPELINE")
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Mode: {rendering_mode.upper()}, Quality: {quality_preset}")
    if enable_stabilization:
        print("📹 Video stabilization: ENABLED")
    if color_correction:
        print("🎨 Color correction: ENABLED")
    print("=" * 60)
    
    try:
        # Step 1: Complete analysis and smoothing pipeline
        (original_boxes, normalized_crops, smoothed_coords, 
         quality_metrics, motion_metrics, ffmpeg_filter) = process_video_complete_pipeline(
            input_video_path,
            padding_factor=padding_factor,
            smoothing_strength=smoothing_strength,
            interpolation_method=interpolation_method
        )
        
        # Step 2: Save intermediate results if requested
        if save_intermediate_files:
            base_filename = output_video_path.replace('.mp4', '_analysis')
            save_complete_results(
                original_boxes, 
                normalized_crops, 
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
            rendering_mode=rendering_mode,
            quality_preset=quality_preset,
            enable_stabilization=enable_stabilization,
            color_correction=color_correction,
            **render_kwargs
        )
        
        if success:
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"✅ Final video: {output_video_path}")
            
            # Print summary stats
            print(f"\n📊 SUMMARY:")
            print(f"  Original keyframes: {len(original_boxes)}")
            print(f"  Smooth frames: {len(smoothed_coords)}")
            print(f"  Video duration: {motion_metrics['frame_stats']['duration_seconds']:.1f}s")
            print(f"  Average zoom: {quality_metrics['average_zoom']:.2f}x")
            print(f"  Motion smoothness: {motion_metrics['motion_analysis']['speed_consistency']:.2f}")
            print(f"  Rendering mode: {rendering_mode}")
            
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


def demo_advanced_features():
    """Demonstrate advanced FFmpeg features"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    input_video = os.path.join(project_root, "videos", "waterpolo_trimmed.webv")
    
    print("🎬 ADVANCED FFMPEG FEATURES DEMO")
    print("=" * 50)
    
    # Demo 1: Dynamic cropping with stabilization
    print("\n1. Dynamic Cropping with Stabilization")
    output_dynamic = os.path.join(project_root, "outputs", "demo_dynamic_stabilized.mp4")
    success1 = process_and_render_complete(
        input_video,
        output_dynamic,
        rendering_mode='dynamic',
        quality_preset='medium',
        enable_stabilization=True,
        color_correction=True,
        save_intermediate_files=False
    )
    
    # Demo 2: Multi-pass encoding for high quality
    print("\n2. Multi-pass High Quality Encoding")
    output_multipass = os.path.join(project_root, "outputs", "demo_multipass_high.mp4")
    success2 = process_and_render_complete(
        input_video,
        output_multipass,
        rendering_mode='multipass',
        quality_preset='high',
        save_intermediate_files=False
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
    
    # Demo 4: Quality analysis
    print("\n4. Video Quality Analysis")
    if success1:
        print("\nOriginal video:")
        analyze_video_quality(input_video)
        print("\nProcessed video:")
        analyze_video_quality(output_dynamic)
    
    print("\n" + "=" * 50)
    print(f"Demo Results:")
    print(f"  Dynamic render: {'✅' if success1 else '❌'}")
    print(f"  Multi-pass render: {'✅' if success2 else '❌'}")
    

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
    parser.add_argument('--mode', choices=['simple', 'dynamic', 'multipass'], 
                       default='simple', help='Rendering mode')
    parser.add_argument('--quality', choices=['fast', 'medium', 'slow', 'best', 'ultra'], 
                       default='medium', help='Quality preset')
    parser.add_argument('--codec', choices=['h264_videotoolbox', 'libx264', 'hevc_videotoolbox'], 
                       default='h264_videotoolbox', help='Video codec')
    parser.add_argument('--bitrate', default='15M', help='Target bitrate (e.g., 15M)')
    parser.add_argument('--resolution', default='1920:1080', help='Target resolution (e.g., 1920:1080)')
    
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


if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys
    
    if len(sys.argv) > 1:
        # Command line interface
        args = parse_arguments()
        
        print(f"🚀 AI CAMERAMAN PIPELINE")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_video}")
        print(f"Mode: {args.mode}, Quality: {args.quality}")
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
                    rendering_mode=args.mode,
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
                    rendering_mode=args.mode,
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
                    rendering_mode='simple',
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
                rendering_mode='simple',
                quality_preset='medium'
            )
        
        if success:
            print("\n🎊 All done! Your AI-cropped video is ready.")
        else:
            print("\n💥 Something went wrong. Check the logs above.") 