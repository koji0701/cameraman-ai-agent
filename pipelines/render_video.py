import os
import subprocess
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Dict, List
from genai_client import process_video_complete_pipeline, save_complete_results
import cv2
HAS_OPENCV = True

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
        
        video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
        has_audio = audio_stream is not None
        
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
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
    debug_outputs_dir: str = None,
) -> bool:
    """Render video with per-frame dynamic cropping using frame extraction and re-encoding."""
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
    
    if enable_debug_outputs and debug_outputs_dir:
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
            "detected_video_info": video_info,
            "frames_to_process": len(smoothed_coords_df),
            "rendering_mode": "dynamic_frame_by_frame"
        }
        
        config_path = Path(debug_outputs_dir) / "step_07_render_config.json"
        with open(config_path, 'w') as f:
            json.dump(render_config, f, indent=2)
        print(f"🔍 RENDER DEBUG: Saved render config → {config_path}")
    
    if scale_resolution == "original":
        max_crop_width = int(smoothed_coords_df['crop_w'].max())
        max_crop_height = int(smoothed_coords_df['crop_h'].max())
        target_scale_resolution = f"{max_crop_width}:{max_crop_height}"
        print(f"🎯 Using adaptive resolution: {target_scale_resolution} (based on crop data)")
    else:
        target_scale_resolution = scale_resolution
        print(f"🎯 Using specified resolution: {target_scale_resolution}")
    
    with tempfile.TemporaryDirectory(prefix="dynamic_crop_") as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        cropped_dir = os.path.join(temp_dir, "cropped")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)
        
        if enable_debug_outputs and debug_outputs_dir:
            frame_processing_info = {
                "temp_directory": temp_dir,
                "frames_extraction_dir": frames_dir,
                "cropped_frames_dir": cropped_dir,
                "target_scale_resolution": target_scale_resolution,
                "max_crop_dimensions": {
                    "width": int(smoothed_coords_df['crop_w'].max()),
                    "height": int(smoothed_coords_df['crop_h'].max())
                }
            }
            
            info_path = Path(debug_outputs_dir) / "step_07_frame_processing_info.json"
            with open(info_path, 'w') as f:
                json.dump(frame_processing_info, f, indent=2)
            print(f"🔍 RENDER DEBUG: Saved frame processing info → {info_path}")
        
        try:
            if verbose:
                print("📸 Extracting frames...")
            
            frame_pattern = os.path.join(frames_dir, "frame_%05d.png")
            extract_cmd = [
                "ffmpeg", "-y", 
                "-i", input_video_path,
                frame_pattern
            ]
            
            if enable_debug_outputs and debug_outputs_dir:
                extract_cmd_str = " ".join(extract_cmd)
                cmd_path = Path(debug_outputs_dir) / "step_07_ffmpeg_extract_command.txt"
                with open(cmd_path, 'w') as f:
                    f.write(f"# FFmpeg frame extraction command:\n{extract_cmd_str}\n")
                print(f"🔍 RENDER DEBUG: Saved extraction command → {cmd_path}")
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Frame extraction failed: {result.stderr}")
                if enable_debug_outputs and debug_outputs_dir:
                    error_path = Path(debug_outputs_dir) / "step_07_ffmpeg_extract_error.txt"
                    with open(error_path, 'w') as f:
                        f.write(f"STDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}\n")
                    print(f"🔍 RENDER DEBUG: Saved extraction error → {error_path}")
                return False
                
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            if verbose:
                print(f"✅ Extracted {len(frame_files)} frames")
            
            if verbose:
                print("✂️ Cropping frames...")
            
            success_count = 0
            coords_dict = {}
            for _, row in smoothed_coords_df.iterrows():
                frame_num = int(row['t_ms'] / 1000.0 * fps) + 1
                coords_dict[frame_num] = {
                    'x': int(row['crop_x']),
                    'y': int(row['crop_y']),
                    'w': int(row['crop_w']) & ~1,
                    'h': int(row['crop_h']) & ~1
                }
            
            if enable_debug_outputs and debug_outputs_dir:
                coords_mapping_path = Path(debug_outputs_dir) / "step_07_frame_coordinate_mapping.json"
                coords_mapping_data = {
                    "total_frames_extracted": len(frame_files),
                    "coordinate_mappings": coords_dict,
                    "fps_used_for_mapping": fps,
                    "mapping_method": "time_to_frame_conversion"
                }
                with open(coords_mapping_path, 'w') as f:
                    json.dump(coords_mapping_data, f, indent=2)
                print(f"🔍 RENDER DEBUG: Saved coordinate mapping → {coords_mapping_path}")
            
            opencv_crop_operations = []
            ffmpeg_crop_operations = []
            
            for i, frame_file in enumerate(frame_files, 1):
                frame_path = os.path.join(frames_dir, frame_file)
                output_frame_path = os.path.join(cropped_dir, frame_file)
                
                if i in coords_dict:
                    coords = coords_dict[i]
                else:
                    closest_frame = min(coords_dict.keys(), key=lambda x: abs(x - i))
                    coords = coords_dict[closest_frame]
                
                x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
                
                crop_operation = {
                    "frame_number": i,
                    "frame_file": frame_file,
                    "crop_coordinates": {"x": x, "y": y, "w": w, "h": h},
                    "method": "opencv" if HAS_OPENCV else "ffmpeg"
                }
                
                try:
                    if HAS_OPENCV:
                        img = cv2.imread(frame_path)
                        if img is not None:
                            cropped = img[y:y+h, x:x+w]
                            cv2.imwrite(output_frame_path, cropped)
                            success_count += 1
                            crop_operation["status"] = "success"
                            opencv_crop_operations.append(crop_operation)
                        else:
                            crop_operation["status"] = "failed_to_read_image"
                            opencv_crop_operations.append(crop_operation)
                    else:
                        crop_cmd = [
                            "ffmpeg", "-y", "-i", frame_path,
                            "-vf", f"crop={w}:{h}:{x}:{y}",
                            output_frame_path
                        ]
                        result = subprocess.run(crop_cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            success_count += 1
                            crop_operation["status"] = "success"
                        else:
                            crop_operation["status"] = f"ffmpeg_error: {result.stderr[:100]}"
                        ffmpeg_crop_operations.append(crop_operation)
                        
                except Exception as e:
                    crop_operation["status"] = f"exception: {str(e)[:100]}"
                    if HAS_OPENCV:
                        opencv_crop_operations.append(crop_operation)
                    else:
                        ffmpeg_crop_operations.append(crop_operation)
            
            if enable_debug_outputs and debug_outputs_dir:
                all_operations = opencv_crop_operations + ffmpeg_crop_operations
                operations_log = {
                    "total_frames_processed": len(frame_files),
                    "successful_crops": success_count,
                    "failed_crops": len(frame_files) - success_count,
                    "opencv_available": HAS_OPENCV,
                    "operations": all_operations
                }
                
                ops_path = Path(debug_outputs_dir) / "step_07_crop_operations_log.json"
                with open(ops_path, 'w') as f:
                    json.dump(operations_log, f, indent=2)
                print(f"🔍 RENDER DEBUG: Saved crop operations log → {ops_path}")
                
                if all_operations:
                    ops_df = pd.DataFrame(all_operations)
                    ops_summary_path = Path(debug_outputs_dir) / "step_07_crop_operations_summary.csv"
                    ops_df.to_csv(ops_summary_path, index=False)
                    print(f"🔍 RENDER DEBUG: Saved crop operations summary → {ops_summary_path}")
            
            if success_count == 0:
                print("❌ No frames were successfully cropped")
                return False
                
            if verbose:
                print(f"✅ Successfully cropped {success_count}/{len(frame_files)} frames")
            
            if verbose:
                print("🎞️ Re-encoding video...")
            
            cropped_pattern = os.path.join(cropped_dir, "frame_%05d.png")
            
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", cropped_pattern,
            ]
            
            if has_audio:
                encode_cmd.extend(["-i", input_video_path])
            
            if video_codec == "h264_videotoolbox":
                encode_cmd.extend([
                    "-c:v", "h264_videotoolbox",
                    "-b:v", bitrate,
                    "-allow_sw", "1"
                ])
            else:
                encode_cmd.extend([
                    "-c:v", video_codec,
                    "-b:v", bitrate
                ])
            
            if quality_preset in ["fast", "medium", "slow", "veryslow"]:
                encode_cmd.extend(["-preset", quality_preset])
            
            if target_scale_resolution != "original":
                encode_cmd.extend(["-vf", f"scale={target_scale_resolution}"])
            
            if has_audio:
                encode_cmd.extend(["-c:a", audio_codec, "-map", "1:a"])
            
            encode_cmd.extend(["-vsync", "2", output_video_path])
            
            if enable_debug_outputs and debug_outputs_dir:
                encode_cmd_str = " ".join(encode_cmd)
                final_cmd_path = Path(debug_outputs_dir) / "step_07_ffmpeg_final_encode_command.txt"
                with open(final_cmd_path, 'w') as f:
                    f.write(f"# Final FFmpeg encoding command:\n{encode_cmd_str}\n")
                print(f"🔍 RENDER DEBUG: Saved final encode command → {final_cmd_path}")
            
            result = subprocess.run(encode_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Video encoding failed: {result.stderr}")
                if enable_debug_outputs and debug_outputs_dir:
                    encode_error_path = Path(debug_outputs_dir) / "step_07_ffmpeg_encode_error.txt"
                    with open(encode_error_path, 'w') as f:
                        f.write(f"STDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}\n")
                    print(f"🔍 RENDER DEBUG: Saved encode error → {encode_error_path}")
                return False
            
            if verbose:
                print(f"✅ Video successfully rendered: {output_video_path}")
                
            if enable_debug_outputs and debug_outputs_dir:
                render_summary = {
                    "rendering_successful": True,
                    "output_video_path": output_video_path,
                    "input_resolution": f"{original_width}x{original_height}",
                    "output_resolution": target_scale_resolution,
                    "total_frames_processed": len(frame_files),
                    "successful_crops": success_count,
                    "fps": fps,
                    "has_audio": has_audio,
                    "video_codec": video_codec,
                    "audio_codec": audio_codec if has_audio else None,
                    "bitrate": bitrate,
                    "quality_preset": quality_preset
                }
                
                summary_path = Path(debug_outputs_dir) / "step_07_final_render_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(render_summary, f, indent=2)
                print(f"🔍 RENDER DEBUG: Saved final render summary → {summary_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during frame processing: {e}")
            if enable_debug_outputs and debug_outputs_dir:
                exception_path = Path(debug_outputs_dir) / "step_07_processing_exception.txt"
                with open(exception_path, 'w') as f:
                    import traceback
                    f.write(f"Exception during frame processing:\n{traceback.format_exc()}\n")
                print(f"🔍 RENDER DEBUG: Saved exception details → {exception_path}")
            return False

def render_cropped_video_simple(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    video_codec: str = 'h264_videotoolbox',
    bitrate: str = '15M',
    scale_resolution: str = 'original',
    audio_codec: str = 'aac',
    verbose: bool = True
) -> bool:
    """Render video with static crop using average coordinates."""
    video_info = get_video_info(input_video_path)
    has_audio = video_info.get('has_audio', False)
    original_width = video_info.get("width", 1920)
    original_height = video_info.get("height", 1080)

    avg_x = int(smoothed_coords_df['crop_x'].mean())
    avg_y = int(smoothed_coords_df['crop_y'].mean())
    avg_w = int(smoothed_coords_df['crop_w'].mean()) & ~1
    avg_h = int(smoothed_coords_df['crop_h'].mean()) & ~1

    print(f"Using average crop: {avg_w}x{avg_h} at ({avg_x},{avg_y})")
    print(f"Original video: {original_width}x{original_height}")
    print("Audio stream:" + (" present" if has_audio else " none"))

    if scale_resolution == "original":
        target_scale_resolution = f"{avg_w}:{avg_h}"
        print(f"🎯 Using adaptive resolution: {target_scale_resolution} (based on crop)")
    else:
        target_scale_resolution = scale_resolution
        print(f"🎯 Using specified resolution: {target_scale_resolution}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    if target_scale_resolution == "original" or ":" not in target_scale_resolution:
        filter_graph = f"[0:v]crop={avg_w}:{avg_h}:{avg_x}:{avg_y}[outv]"
    else:
        filter_graph = (
            f"[0:v]crop={avg_w}:{avg_h}:{avg_x}:{avg_y},"
            f"scale={target_scale_resolution}[outv]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-filter_complex", filter_graph,
        "-map", "[outv]",
        "-c:v", video_codec,
        "-b:v", bitrate,
        "-fps_mode", "cfr",
    ]

    if has_audio:
        cmd.extend(["-map", "0:a", "-c:a", audio_codec])

    if video_codec == "h264_videotoolbox":
        cmd.extend(["-allow_sw", "1"])

    cmd.append(output_video_path)

    if verbose:
        print("FFmpeg command:", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"✅ Video rendered – {size_mb:.1f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ FFmpeg binary not found. Install FFmpeg first.")
        return False

def render_cropped_video(
    input_video_path: str,
    output_video_path: str,
    smoothed_coords_df: pd.DataFrame,
    video_codec: str = 'h264_videotoolbox',
    rendering_mode: str = 'simple',
    quality_preset: str = 'medium',
    bitrate: str = '15M',
    scale_resolution: str = 'original',
    audio_codec: str = 'aac',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    verbose: bool = True
) -> bool:
    """Render cropped video using FFmpeg with multiple rendering modes."""
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
    else:
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

def process_and_render_complete(
    input_video_path: str,
    output_video_path: str,
    padding_factor: float = 1.1,
    smoothing_strength: str = 'balanced',
    interpolation_method: str = 'cubic',
    rendering_mode: str = 'simple',
    quality_preset: str = 'medium',
    enable_stabilization: bool = False,
    color_correction: bool = False,
    save_intermediate_files: bool = True,
    enable_debug_outputs: bool = True,
    debug_outputs_dir: str = None,
    **render_kwargs
) -> bool:
    """Complete end-to-end pipeline: Analyze → Normalize → Smooth → Render"""
    print(f"🚀 STARTING COMPLETE AI CAMERAMAN PIPELINE")
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Mode: {rendering_mode.upper()}, Quality: {quality_preset}")
    if enable_stabilization:
        print("📹 Video stabilization: ENABLED")
    if color_correction:
        print("🎨 Color correction: ENABLED")
    if enable_debug_outputs:
        print("🔍 Debug outputs: ENABLED")
    print("=" * 60)
    
    try:
        (original_boxes, normalized_crops, smoothed_coords, 
         quality_metrics, motion_metrics, ffmpeg_filter) = process_video_complete_pipeline(
            input_video_path,
            padding_factor=padding_factor,
            smoothing_strength=smoothing_strength,
            interpolation_method=interpolation_method,
            enable_debug_outputs=enable_debug_outputs,
            debug_outputs_dir=debug_outputs_dir
        )
        
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
        
        print("=" * 60)
        
        if enable_debug_outputs and debug_outputs_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_root = Path(__file__).parent.parent
            outputs_base = project_root / "outputs"
            debug_outputs_dir = str(outputs_base / f"debug_{timestamp}_pipeline")
        
        if rendering_mode == 'dynamic':
            success = render_cropped_video_dynamic(
                input_video_path,
                output_video_path,
                smoothed_coords,
                quality_preset=quality_preset,
                enable_stabilization=enable_stabilization,
                color_correction=color_correction,
                enable_debug_outputs=enable_debug_outputs,
                debug_outputs_dir=debug_outputs_dir,
                **render_kwargs
            )
        else:
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
            
            if enable_debug_outputs and debug_outputs_dir:
                render_info = {
                    "rendering_mode": rendering_mode,
                    "function_used": "render_cropped_video",
                    "input_video_path": input_video_path,
                    "output_video_path": output_video_path,
                    "quality_preset": quality_preset,
                    "enable_stabilization": enable_stabilization,
                    "color_correction": color_correction,
                    "additional_kwargs": render_kwargs
                }
                render_info_path = Path(debug_outputs_dir) / "step_07_render_mode_info.json"
                with open(render_info_path, 'w') as f:
                    json.dump(render_info, f, indent=2)
                print(f"🔍 RENDER DEBUG: Saved render mode info → {render_info_path}")
        
        if success:
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"✅ Final video: {output_video_path}")
            
            print(f"\n📊 SUMMARY:")
            print(f"  Original keyframes: {len(original_boxes)}")
            print(f"  Smooth frames: {len(smoothed_coords)}")
            print(f"  Video duration: {motion_metrics['frame_stats']['duration_seconds']:.1f}s")
            print(f"  Average zoom: {quality_metrics['average_zoom']:.2f}x")
            print(f"  Motion smoothness: {motion_metrics['motion_analysis']['speed_consistency']:.2f}")
            print(f"  Rendering mode: {rendering_mode}")
            
            if enable_debug_outputs:
                print(f"\n🔍 DEBUG DATA:")
                print(f"  All intermediate files saved to: {debug_outputs_dir}")
                print(f"  Step-by-step outputs available for analysis")
            
        return success
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        
        if enable_debug_outputs and debug_outputs_dir:
            exception_path = Path(debug_outputs_dir) / "pipeline_exception.txt"
            with open(exception_path, 'w') as f:
                f.write(f"Pipeline Exception:\n{traceback.format_exc()}\n")
            print(f"🔍 DEBUG: Exception details saved to {exception_path}")
        
        return False 