"""
File management utilities for AI Cameraman processing
Handles file validation, directory management, and video statistics
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import shutil


class FileManager:
    """
    Manages file operations for video processing
    """
    
    # Supported video formats
    SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.flv', '.wmv', '.webm'}
    
    def __init__(self):
        self.temp_dirs = []  # Track temporary directories for cleanup
    
    def validate_video_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is a supported video format
        
        Args:
            file_path: Path to the video file
            
        Returns:
            bool: True if valid video file
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False
            
            # Check if it's a file (not directory)
            if not path.is_file():
                return False
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_VIDEO_EXTENSIONS:
                return False
            
            # Try to get video info to verify it's actually a video
            video_info = self.get_video_info(file_path)
            return video_info is not None and 'width' in video_info
            
        except Exception:
            return False
    
    def ensure_output_directory(self, output_path: str) -> bool:
        """
        Ensure the output directory exists
        
        Args:
            output_path: Path to the output file
            
        Returns:
            bool: True if directory was created or already exists
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video file information using ffprobe
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            audio_stream = None
            
            for stream in info['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                return None
            
            # Calculate duration more accurately
            duration = float(video_stream.get('duration', 0))
            if duration == 0 and 'format' in info:
                duration = float(info['format'].get('duration', 0))
            
            # Calculate FPS
            fps = 30.0  # default
            if 'r_frame_rate' in video_stream:
                try:
                    fps_str = video_stream['r_frame_rate']
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den)
                    else:
                        fps = float(fps_str)
                except:
                    pass
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'duration': duration,
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'codec': video_stream['codec_name'],
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'total_frames': int(video_stream.get('nb_frames', duration * fps)) if duration > 0 else None,
                'has_audio': audio_stream is not None,
                'audio_codec': audio_stream['codec_name'] if audio_stream else None,
                'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
            
        except Exception as e:
            print(f"⚠️ Could not get video info for {video_path}: {e}")
            return None
    
    def get_file_stats(self, input_path: str, output_path: str) -> Dict:
        """
        Get file size statistics and compression ratio
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            
        Returns:
            Dictionary with file statistics
        """
        stats = {
            'input_size_mb': 0.0,
            'output_size_mb': 0.0,
            'compression_ratio': 0.0,
            'space_saved_mb': 0.0
        }
        
        try:
            if os.path.exists(input_path):
                input_size = os.path.getsize(input_path)
                stats['input_size_mb'] = round(input_size / (1024 * 1024), 2)
            
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                stats['output_size_mb'] = round(output_size / (1024 * 1024), 2)
                
                if stats['input_size_mb'] > 0:
                    stats['compression_ratio'] = round(
                        (1 - stats['output_size_mb'] / stats['input_size_mb']) * 100, 1
                    )
                    stats['space_saved_mb'] = round(
                        stats['input_size_mb'] - stats['output_size_mb'], 2
                    )
        
        except Exception as e:
            print(f"⚠️ Could not calculate file stats: {e}")
        
        return stats
    
    def create_temp_directory(self, prefix: str = "ai_cameraman_") -> str:
        """
        Create a temporary directory for processing
        
        Args:
            prefix: Prefix for the temporary directory name
            
        Returns:
            Path to the temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_temp_directories(self):
        """Clean up all temporary directories created during processing"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Could not clean up temp directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def get_safe_output_path(self, input_path: str, output_dir: str = None, suffix: str = "_ai_cropped") -> str:
        """
        Generate a safe output path that doesn't overwrite existing files
        
        Args:
            input_path: Path to the input file
            output_dir: Directory for output (default: same as input)
            suffix: Suffix to add to filename
            
        Returns:
            Safe output path
        """
        input_path = Path(input_path)
        
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
        
        # Create base output name
        base_name = input_path.stem + suffix
        extension = input_path.suffix
        
        # Find a unique filename
        counter = 1
        while True:
            if counter == 1:
                output_path = output_dir / f"{base_name}{extension}"
            else:
                output_path = output_dir / f"{base_name}_{counter}{extension}"
            
            if not output_path.exists():
                return str(output_path)
            
            counter += 1
            if counter > 1000:  # Safety check
                raise RuntimeError("Could not find unique output filename")
    
    def list_video_files(self, directory: str) -> List[str]:
        """
        List all video files in a directory
        
        Args:
            directory: Directory to search
            
        Returns:
            List of video file paths
        """
        video_files = []
        
        try:
            directory = Path(directory)
            if directory.exists() and directory.is_dir():
                for file_path in directory.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix.lower() in self.SUPPORTED_VIDEO_EXTENSIONS):
                        video_files.append(str(file_path))
        
        except Exception as e:
            print(f"⚠️ Error listing video files in {directory}: {e}")
        
        return sorted(video_files)
    
    def estimate_processing_time(self, video_info: Dict) -> Dict[str, float]:
        """
        Estimate processing time based on video characteristics
        
        Args:
            video_info: Video information dictionary
            
        Returns:
            Dictionary with time estimates in seconds
        """
        duration = video_info.get('duration', 0)
        resolution = video_info.get('width', 1920) * video_info.get('height', 1080)
        fps = video_info.get('fps', 30)
        
        # Base processing factor (seconds of processing per second of video)
        base_factor = 0.5  # Conservative estimate
        
        # Adjust for resolution (higher resolution = more processing time)
        resolution_factor = resolution / (1920 * 1080)  # Normalize to 1080p
        
        # Adjust for frame rate
        fps_factor = fps / 30.0  # Normalize to 30 fps
        
        # Combined factor
        processing_factor = base_factor * resolution_factor * fps_factor
        
        estimates = {
            'gemini_analysis': duration * 0.3,  # Gemini analysis is relatively fast
            'coordinate_smoothing': duration * 0.1,  # Smoothing is quick
            'video_processing': duration * processing_factor,  # Main processing
            'total_estimated': duration * (0.4 + processing_factor)
        }
        
        return estimates
    
    def __del__(self):
        """Cleanup temporary directories on destruction"""
        self.cleanup_temp_directories() 