"""
OpenCV-based video processor for AI Cameraman

This module provides efficient video processing using OpenCV for dynamic cropping
with significantly reduced storage requirements compared to FFmpeg.
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import interp1d

from .video_utils import VideoUtils


@dataclass
class ProcessingStats:
    """Statistics for video processing performance"""
    total_frames: int = 0
    frames_processed: int = 0
    processing_time: float = 0.0
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    compression_ratio: float = 0.0
    average_fps: float = 0.0
    peak_memory_mb: float = 0.0


class OpenCVCameraman:
    """OpenCV-based video processor for dynamic cropping"""
    
    def __init__(self, input_video_path: str, output_video_path: str, verbose: bool = True):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.verbose = verbose
        
        # Video properties
        self.cap = None
        self.writer = None
        self.video_info = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        
        # Processing state
        self.stats = ProcessingStats()
        self.frame_coords = {}
        
        # Initialize video capture
        self._initialize_video_capture()
    
    def _initialize_video_capture(self):
        """Initialize video capture and get video properties"""
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.input_video_path}")
        
        # Get video properties
        self.video_info = VideoUtils.get_video_info_opencv(self.input_video_path)
        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']
        self.frame_width = self.video_info['width']
        self.frame_height = self.video_info['height']
        
        if self.verbose:
            print(f"📹 Video initialized:")
            print(f"   Resolution: {self.frame_width}x{self.frame_height}")
            print(f"   FPS: {self.fps:.2f}")
            print(f"   Total frames: {self.total_frames}")
            print(f"   Duration: {self.video_info['duration']:.2f}s")
    
    def process_with_gemini_coords(self, gemini_coordinates: List[Dict], **kwargs) -> bool:
        """
        Main processing method - converts Gemini coordinates and processes video
        
        Args:
            gemini_coordinates: List of coordinate dictionaries from Gemini API
            **kwargs: Additional processing options
            
        Returns:
            bool: Success status
        """
        try:
            # Convert Gemini coordinates to frame-indexed format
            self.frame_coords = self.convert_gemini_to_frame_coords(gemini_coordinates)
            
            # Calculate optimal output dimensions
            output_size = self.calculate_optimal_dimensions(gemini_coordinates)
            
            # Setup video writer with optimized settings
            self._setup_optimized_writer(output_size, **kwargs)
            
            # Process video frame by frame
            success = self._process_video_frames()
            
            # Cleanup
            self._cleanup()
            
            if self.verbose:
                self._print_processing_summary()
            
            return success
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            self._cleanup()
            return False
    
    def convert_gemini_to_frame_coords(self, gemini_data: List[Dict]) -> Dict[int, Dict]:
        """
        Convert Gemini timestamp-based coordinates to frame-indexed parameters
        
        Args:
            gemini_data: List of coordinate dictionaries from Gemini API
            
        Returns:
            Dictionary mapping frame numbers to crop coordinates
        """
        if self.verbose:
            print(f"🎯 Converting {len(gemini_data)} Gemini coordinates to frame mapping...")
        
        frame_coords = {}
        
        # Convert timestamps to frame numbers
        for coord in gemini_data:
            timestamp = coord.get("timestamp", 0)
            frame_num = VideoUtils.timestamp_to_frame_number(timestamp, self.fps)
            
            # Ensure frame number is within valid range
            if 0 <= frame_num < self.total_frames:
                x, y, w, h = VideoUtils.validate_crop_bounds(
                    coord.get('x', 0),
                    coord.get('y', 0), 
                    coord.get('width', 640),
                    coord.get('height', 480),
                    self.frame_width,
                    self.frame_height
                )
                
                frame_coords[frame_num] = {'x': x, 'y': y, 'w': w, 'h': h}
        
        # Interpolate between keyframes for smooth transitions
        interpolated_coords = self._interpolate_coordinates(frame_coords)
        
        if self.verbose:
            print(f"✅ Coordinate conversion complete:")
            print(f"   Keyframes: {len(frame_coords)}")
            print(f"   Total frames with coordinates: {len(interpolated_coords)}")
            print(f"   Coverage: {len(interpolated_coords)/self.total_frames*100:.1f}%")
        
        return interpolated_coords
    
    def _interpolate_coordinates(self, frame_coords: Dict[int, Dict]) -> Dict[int, Dict]:
        """Interpolate coordinates between keyframes for smooth transitions"""
        if len(frame_coords) < 2:
            # If we have only one coordinate, use it for all frames
            if frame_coords:
                coord = list(frame_coords.values())[0]
                return {i: coord for i in range(self.total_frames)}
            else:
                # Default coordinates if none provided
                default_coord = {'x': 0, 'y': 0, 'w': self.frame_width, 'h': self.frame_height}
                return {i: default_coord for i in range(self.total_frames)}
        
        # Sort frames by frame number
        sorted_frames = sorted(frame_coords.keys())
        
        # Extract coordinate arrays
        frames_array = np.array(sorted_frames)
        coords_array = np.array([[frame_coords[f]['x'], frame_coords[f]['y'], 
                                 frame_coords[f]['w'], frame_coords[f]['h']] 
                                for f in sorted_frames])
        
        # Create interpolation functions
        interp_funcs = []
        for i in range(4):  # x, y, w, h
            interp_funcs.append(interp1d(
                frames_array, coords_array[:, i], 
                kind='cubic', fill_value='extrapolate'
            ))
        
        # Generate coordinates for all frames
        interpolated = {}
        for frame_num in range(self.total_frames):
            x = int(interp_funcs[0](frame_num))
            y = int(interp_funcs[1](frame_num))
            w = int(interp_funcs[2](frame_num))
            h = int(interp_funcs[3](frame_num))
            
            # Validate bounds
            x, y, w, h = VideoUtils.validate_crop_bounds(
                x, y, w, h, self.frame_width, self.frame_height
            )
            
            interpolated[frame_num] = {'x': x, 'y': y, 'w': w, 'h': h}
        
        return interpolated
    
    def calculate_optimal_dimensions(self, gemini_coordinates: List[Dict]) -> Tuple[int, int]:
        """Calculate optimal output dimensions from Gemini coordinates"""
        if not gemini_coordinates:
            return 640, 480  # Default size
        
        # Extract all coordinate values
        crop_coords = []
        for coord in gemini_coordinates:
            crop_coords.append({
                'w': coord.get('width', 640),
                'h': coord.get('height', 480)
            })
        
        return VideoUtils.calculate_optimal_output_size(crop_coords)
    
    def _setup_optimized_writer(self, output_size: Tuple[int, int], **kwargs):
        """Setup VideoWriter with optimal compression settings"""
        width, height = output_size
        
        # Store output dimensions for later reference
        self.output_width = width
        self.output_height = height
        
        # Try different codecs in order of preference
        codec_preference = ['HEVC', 'H264', 'XVID', 'mp4v']
        
        for codec in codec_preference:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(
                    self.output_video_path, 
                    fourcc, 
                    self.fps, 
                    (width, height)
                )
                
                if self.writer.isOpened():
                    if self.verbose:
                        print(f"🎬 Video writer initialized with {codec} codec")
                        print(f"   Output size: {width}x{height}")
                        print(f"   FPS: {self.fps}")
                    break
                else:
                    self.writer.release()
                    self.writer = None
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not initialize {codec} codec: {e}")
                continue
        
        if self.writer is None:
            raise RuntimeError("Could not initialize any video codec")
    
    def _process_video_frames(self) -> bool:
        """Process video frames with dynamic cropping"""
        if self.verbose:
            print(f"🎞️ Processing {self.total_frames} frames...")
        
        start_time = time.time()
        frames_processed = 0
        
        # Get input file size
        self.stats.input_size_mb = os.path.getsize(self.input_video_path) / (1024 * 1024)
        
        for frame_num in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                if self.verbose:
                    print(f"⚠️ Could not read frame {frame_num}")
                break
            
            # Get crop parameters for this frame
            if frame_num in self.frame_coords:
                crop_params = self.frame_coords[frame_num]
            else:
                # Fallback to full frame if no coordinates
                crop_params = {'x': 0, 'y': 0, 'w': self.frame_width, 'h': self.frame_height}
            
            # Apply crop
            cropped_frame = self._apply_dynamic_crop(frame, crop_params)
            
            # Resize to output dimensions if needed
            if cropped_frame.shape[:2] != (self.output_height, self.output_width):
                cropped_frame = cv2.resize(cropped_frame, (self.output_width, self.output_height))
            
            # Write frame
            self.writer.write(cropped_frame)
            frames_processed += 1
            
            # Progress update
            if self.verbose and frames_processed % 100 == 0:
                progress = frames_processed / self.total_frames * 100
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                print(f"📈 Progress: {progress:.1f}% ({frames_processed}/{self.total_frames}) @ {fps:.1f} FPS")
        
        # Update statistics
        self.stats.total_frames = self.total_frames
        self.stats.frames_processed = frames_processed
        self.stats.processing_time = time.time() - start_time
        self.stats.average_fps = frames_processed / self.stats.processing_time if self.stats.processing_time > 0 else 0
        
        return frames_processed == self.total_frames
    
    def _apply_dynamic_crop(self, frame: np.ndarray, crop_params: Dict) -> np.ndarray:
        """Apply crop with smart bounds checking"""
        x, y, w, h = crop_params['x'], crop_params['y'], crop_params['w'], crop_params['h']
        
        # Additional bounds validation
        x, y, w, h = VideoUtils.validate_crop_bounds(
            x, y, w, h, frame.shape[1], frame.shape[0]
        )
        
        # Apply crop using NumPy slicing
        cropped = frame[y:y+h, x:x+w]
        
        return cropped
    
    def _cleanup(self):
        """Clean up video capture and writer resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        
        # Calculate final statistics
        if os.path.exists(self.output_video_path):
            self.stats.output_size_mb = os.path.getsize(self.output_video_path) / (1024 * 1024)
            self.stats.compression_ratio = (
                self.stats.input_size_mb / max(self.stats.output_size_mb, 0.001)
            )
    
    def _print_processing_summary(self):
        """Print processing performance summary"""
        print(f"\n{'='*60}")
        print(f"🎬 VIDEO PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"⏱️  Processing time: {self.stats.processing_time:.2f}s")
        print(f"🎞️  Frames processed: {self.stats.frames_processed}/{self.stats.total_frames}")
        print(f"📈 Average FPS: {self.stats.average_fps:.1f}")
        print(f"💾 Input size: {self.stats.input_size_mb:.1f}MB")
        print(f"📤 Output size: {self.stats.output_size_mb:.1f}MB")
        print(f"🗜️  Compression ratio: {self.stats.compression_ratio:.2f}x")
        print(f"💡 Storage efficiency: {(1 - self.stats.output_size_mb/max(self.stats.input_size_mb, 0.001))*100:.1f}%")
        print(f"✅ Success rate: {self.stats.frames_processed/self.stats.total_frames*100:.1f}%")
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get processing statistics"""
        return self.stats 