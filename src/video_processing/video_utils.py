"""
Shared utilities for video processing
"""

import cv2
import json
import subprocess
from typing import Dict, Optional
import numpy as np


class VideoUtils:
    """Shared utilities for video processing operations"""
    
    @staticmethod
    def get_video_info_opencv(video_path: str) -> Dict:
        """Get video information using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def get_video_info_ffprobe(video_path: str) -> Dict:
        """Get detailed video information using ffprobe (fallback method)"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
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
            print(f"⚠️ Could not get video info with ffprobe: {e}")
            return {}
    
    @staticmethod
    def validate_crop_bounds(x: int, y: int, w: int, h: int, 
                           frame_width: int, frame_height: int) -> tuple:
        """Validate and adjust crop coordinates to stay within frame bounds"""
        # Ensure coordinates are non-negative
        x = max(0, x)
        y = max(0, y)
        
        # Ensure crop doesn't exceed frame boundaries
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        
        # Ensure minimum crop size
        w = max(1, w)
        h = max(1, h)
        
        # Ensure even dimensions for video encoding
        w = w & ~1
        h = h & ~1
        
        return x, y, w, h
    
    @staticmethod
    def calculate_optimal_output_size(crop_coordinates: list) -> tuple:
        """Calculate optimal output dimensions from list of crop coordinates"""
        if not crop_coordinates:
            return 640, 480  # Default size
        
        max_width = max(coord.get('w', 0) for coord in crop_coordinates)
        max_height = max(coord.get('h', 0) for coord in crop_coordinates)
        
        # Ensure even dimensions
        max_width = max_width & ~1
        max_height = max_height & ~1
        
        # Ensure minimum size
        max_width = max(64, max_width)
        max_height = max(64, max_height)
        
        return max_width, max_height
    
    @staticmethod
    def timestamp_to_frame_number(timestamp: float, fps: float) -> int:
        """Convert timestamp to frame number"""
        return int(timestamp * fps)
    
    @staticmethod
    def frame_number_to_timestamp(frame_number: int, fps: float) -> float:
        """Convert frame number to timestamp"""
        return frame_number / fps 