"""
Integration bridge between OpenCV processor and existing Gemini API pipeline

This module provides compatibility layer to integrate the new OpenCV-based
video processor with the existing AI cameraman pipeline.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

# Add pipelines directory to path for existing modules
sys.path.append(str(Path(__file__).parent.parent.parent / "pipelines"))

try:
    from genai_client import process_video_complete_pipeline, save_complete_results
    from kalman_smoother import apply_kalman_smoothing
    from normalize_coordinates import normalize_action_coordinates
except ImportError as e:
    print(f"⚠️ Could not import existing pipeline modules: {e}")
    process_video_complete_pipeline = None
    apply_kalman_smoothing = None
    normalize_action_coordinates = None

from .opencv_processor import OpenCVCameraman
from .ffmpeg_processor import FFmpegProcessor
from .video_utils import VideoUtils


class AICameramanPipeline:
    """
    Integrated AI Cameraman pipeline that can use either OpenCV or FFmpeg
    for video processing while maintaining compatibility with existing systems.
    """
    
    def __init__(self, 
                 processor_type: str = "opencv",  # "opencv" or "ffmpeg"
                 verbose: bool = True):
        self.processor_type = processor_type.lower()
        self.verbose = verbose
        
        if self.processor_type not in ["opencv", "ffmpeg"]:
            raise ValueError("processor_type must be 'opencv' or 'ffmpeg'")
    
    def process_video_complete(self,
                             input_video_path: str,
                             output_video_path: str,
                             padding_factor: float = 1.1,
                             smoothing_strength: str = 'balanced',
                             interpolation_method: str = 'cubic',
                             **kwargs) -> bool:
        """
        Complete video processing pipeline using Gemini API and chosen processor
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path for output video
            padding_factor: Padding around detected action area
            smoothing_strength: Kalman smoothing strength
            interpolation_method: Coordinate interpolation method
            **kwargs: Additional processing parameters
            
        Returns:
            bool: Success status
        """
        
        try:
            if self.verbose:
                print(f"🎬 Starting AI Cameraman pipeline with {self.processor_type.upper()} processor")
                print(f"   Input: {input_video_path}")
                print(f"   Output: {output_video_path}")
            
            # Step 1: Get Gemini API coordinates
            if self.verbose:
                print("1️⃣ Extracting action coordinates using Gemini API...")
            
            gemini_results = self._extract_gemini_coordinates(
                input_video_path, padding_factor
            )
            
            if not gemini_results:
                print("❌ Failed to extract coordinates from Gemini API")
                return False
            
            # Step 2: Apply Kalman smoothing
            if self.verbose:
                print("2️⃣ Applying Kalman smoothing to coordinates...")
            
            smoothed_coords = self._apply_coordinate_smoothing(
                gemini_results, smoothing_strength, interpolation_method
            )
            
            # Step 3: Process video with chosen processor
            if self.verbose:
                print(f"3️⃣ Processing video with {self.processor_type.upper()} processor...")
            
            success = self._process_video_with_processor(
                input_video_path, output_video_path, smoothed_coords, **kwargs
            )
            
            if success and self.verbose:
                print("✅ AI Cameraman pipeline completed successfully!")
            
            return success
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return False
    
    def _extract_gemini_coordinates(self, video_path: str, padding_factor: float) -> Optional[List[Dict]]:
        """Extract action coordinates using existing Gemini API integration"""
        
        if process_video_complete_pipeline is None:
            print("⚠️ Gemini API pipeline not available - using mock coordinates")
            return self._create_mock_coordinates(video_path)
        
        try:
            # Use existing Gemini API processing
            results = process_video_complete_pipeline(
                video_path,
                padding_factor=padding_factor,
                verbose=self.verbose
            )
            
            # Convert results to standard format
            coordinates = []
            if results and 'coordinates' in results:
                for coord in results['coordinates']:
                    coordinates.append({
                        'timestamp': coord.get('timestamp', 0),
                        'x': coord.get('x', 0),
                        'y': coord.get('y', 0),
                        'width': coord.get('width', 640),
                        'height': coord.get('height', 480),
                        'confidence': coord.get('confidence', 1.0)
                    })
            
            return coordinates
            
        except Exception as e:
            print(f"⚠️ Gemini API extraction failed: {e}")
            return self._create_mock_coordinates(video_path)
    
    def _create_mock_coordinates(self, video_path: str) -> List[Dict]:
        """Create mock coordinates for testing when Gemini API is unavailable"""
        
        # Get video info to create realistic coordinates
        try:
            video_info = VideoUtils.get_video_info_opencv(video_path)
            duration = video_info['duration']
            width = video_info['width']
            height = video_info['height']
        except:
            # Fallback values
            duration = 10.0
            width = 1920
            height = 1080
        
        # Create coordinates that simulate water polo action movement
        import numpy as np
        
        coordinates = []
        num_coords = max(5, int(duration))  # At least 5 coordinates
        
        for i in range(num_coords):
            timestamp = (i / (num_coords - 1)) * duration
            
            # Simulate circular motion (water polo action)
            center_x = width // 2
            center_y = height // 2
            radius = min(width, height) // 4
            
            angle = (timestamp / duration) * 2 * np.pi
            action_x = int(center_x + radius * np.cos(angle))
            action_y = int(center_y + radius * np.sin(angle))
            
            # Action area size
            action_width = width // 3
            action_height = height // 3
            
            coordinates.append({
                'timestamp': timestamp,
                'x': max(0, action_x - action_width // 2),
                'y': max(0, action_y - action_height // 2),
                'width': action_width,
                'height': action_height,
                'confidence': 0.9
            })
        
        if self.verbose:
            print(f"   🎯 Created {len(coordinates)} mock coordinates for testing")
        
        return coordinates
    
    def _apply_coordinate_smoothing(self, 
                                  coordinates: List[Dict], 
                                  smoothing_strength: str,
                                  interpolation_method: str) -> List[Dict]:
        """Apply Kalman smoothing to coordinates using existing pipeline"""
        
        if apply_kalman_smoothing is None:
            print("⚠️ Kalman smoothing not available - using original coordinates")
            return coordinates
        
        try:
            # Convert to DataFrame format expected by existing code
            coords_df = pd.DataFrame(coordinates)
            
            # Apply existing smoothing logic
            smoothed_df = apply_kalman_smoothing(
                coords_df, 
                smoothing_strength=smoothing_strength,
                interpolation_method=interpolation_method
            )
            
            # Convert back to list format
            smoothed_coords = []
            for _, row in smoothed_df.iterrows():
                smoothed_coords.append({
                    'timestamp': row.get('timestamp', row.get('t_ms', 0) / 1000),
                    'x': int(row.get('x', row.get('crop_x', 0))),
                    'y': int(row.get('y', row.get('crop_y', 0))),
                    'width': int(row.get('width', row.get('crop_w', 640))),
                    'height': int(row.get('height', row.get('crop_h', 480)))
                })
            
            if self.verbose:
                print(f"   ✅ Applied {smoothing_strength} smoothing to {len(smoothed_coords)} coordinates")
            
            return smoothed_coords
            
        except Exception as e:
            print(f"⚠️ Smoothing failed: {e} - using original coordinates")
            return coordinates
    
    def _process_video_with_processor(self,
                                    input_path: str,
                                    output_path: str,
                                    coordinates: List[Dict],
                                    **kwargs) -> bool:
        """Process video using the selected processor"""
        
        try:
            if self.processor_type == "opencv":
                processor = OpenCVCameraman(input_path, output_path, verbose=self.verbose)
                return processor.process_with_gemini_coords(coordinates, **kwargs)
            
            elif self.processor_type == "ffmpeg":
                processor = FFmpegProcessor(input_path, output_path)
                return processor.process_with_gemini_coords(coordinates, **kwargs)
            
            else:
                raise ValueError(f"Unknown processor type: {self.processor_type}")
                
        except Exception as e:
            print(f"❌ Processor failed: {e}")
            return False
    
    def benchmark_processors(self, 
                           video_path: str, 
                           output_dir: str = "benchmark_comparison") -> Dict:
        """
        Benchmark both OpenCV and FFmpeg processors on the same video
        
        Args:
            video_path: Path to test video
            output_dir: Directory for benchmark outputs
            
        Returns:
            Dictionary with performance comparison results
        """
        
        from .benchmark import VideoProcessorBenchmark, create_test_coordinates
        
        # Create benchmark suite
        benchmark = VideoProcessorBenchmark(output_dir)
        
        # Get coordinates (use mock if Gemini API unavailable)
        coordinates = self._extract_gemini_coordinates(video_path, 1.1)
        if not coordinates:
            coordinates = create_test_coordinates()
        
        # Run comparison
        results = benchmark.run_comparison_benchmark(video_path, coordinates)
        benchmark.save_results()
        benchmark.print_summary()
        
        return results
    
    def get_storage_savings_estimate(self, video_path: str) -> Dict:
        """
        Estimate storage savings from using dynamic cropping
        
        Returns:
            Dictionary with storage analysis
        """
        
        try:
            video_info = VideoUtils.get_video_info_opencv(video_path)
            input_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Get sample coordinates to estimate crop area
            coordinates = self._extract_gemini_coordinates(video_path, 1.1)
            
            if coordinates:
                # Calculate average crop area vs full frame
                total_frame_area = video_info['width'] * video_info['height']
                total_crop_area = sum(coord['width'] * coord['height'] for coord in coordinates)
                avg_crop_area = total_crop_area / len(coordinates)
                
                crop_ratio = avg_crop_area / total_frame_area
                estimated_size_reduction = 1 - crop_ratio
                estimated_output_size = input_size_mb * crop_ratio
                
                return {
                    'input_size_mb': input_size_mb,
                    'estimated_output_size_mb': estimated_output_size,
                    'estimated_size_reduction_percent': estimated_size_reduction * 100,
                    'estimated_savings_mb': input_size_mb - estimated_output_size,
                    'average_crop_ratio': crop_ratio,
                    'coordinates_analyzed': len(coordinates)
                }
            else:
                return {
                    'input_size_mb': input_size_mb,
                    'estimated_output_size_mb': input_size_mb * 0.3,  # Default assumption
                    'estimated_size_reduction_percent': 70.0,
                    'estimated_savings_mb': input_size_mb * 0.7,
                    'average_crop_ratio': 0.3,
                    'coordinates_analyzed': 0
                }
                
        except Exception as e:
            print(f"⚠️ Could not analyze storage savings: {e}")
            return {}


# Convenience function for backward compatibility
def process_video_with_ai_cameraman(input_video_path: str,
                                   output_video_path: str,
                                   processor_type: str = "opencv",
                                   **kwargs) -> bool:
    """
    Convenience function for processing video with AI cameraman
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video  
        processor_type: "opencv" or "ffmpeg"
        **kwargs: Additional processing parameters
        
    Returns:
        bool: Success status
    """
    
    pipeline = AICameramanPipeline(processor_type=processor_type)
    return pipeline.process_video_complete(input_video_path, output_video_path, **kwargs) 