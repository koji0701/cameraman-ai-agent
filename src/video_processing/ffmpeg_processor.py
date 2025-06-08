"""
FFmpeg-based video processor (Legacy implementation)

This wraps the existing FFmpeg functionality to maintain compatibility
during the migration to OpenCV.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Add pipelines directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent / "pipelines"))

try:
    from render_video import (
        render_cropped_video_dynamic,
        get_video_info,
        apply_smooth_coordinates_to_frames
    )
except ImportError as e:
    print(f"⚠️ Could not import FFmpeg modules: {e}")
    render_cropped_video_dynamic = None
    get_video_info = None
    apply_smooth_coordinates_to_frames = None


class FFmpegProcessor:
    """Legacy FFmpeg video processor for comparison and fallback"""
    
    def __init__(self, input_video_path: str, output_video_path: str):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.video_info = None
        
    def get_video_info(self) -> Dict:
        """Get video information using FFmpeg"""
        if self.video_info is None:
            if get_video_info:
                self.video_info = get_video_info(self.input_video_path)
            else:
                raise RuntimeError("FFmpeg modules not available")
        return self.video_info
    
    def process_with_gemini_coords(self, gemini_coordinates: List[Dict], **kwargs) -> bool:
        """Process video using existing FFmpeg pipeline"""
        if render_cropped_video_dynamic is None:
            raise RuntimeError("FFmpeg render function not available")
        
        # Convert Gemini coordinates to DataFrame format expected by existing code
        smoothed_coords_df = self._convert_gemini_to_dataframe(gemini_coordinates)
        
        # Use existing FFmpeg implementation
        return render_cropped_video_dynamic(
            self.input_video_path,
            self.output_video_path,
            smoothed_coords_df,
            **kwargs
        )
    
    def _convert_gemini_to_dataframe(self, gemini_coordinates: List[Dict]) -> pd.DataFrame:
        """Convert Gemini API coordinates to DataFrame format"""
        data = []
        for coord in gemini_coordinates:
            data.append({
                't_ms': coord.get('timestamp', 0) * 1000,  # Convert to milliseconds
                'crop_x': coord.get('x', 0),
                'crop_y': coord.get('y', 0),
                'crop_w': coord.get('width', 640),
                'crop_h': coord.get('height', 480)
            })
        
        return pd.DataFrame(data)
    
    def benchmark_performance(self) -> Dict:
        """Benchmark FFmpeg performance for comparison"""
        import time
        import os
        
        # Get file sizes
        input_size = os.path.getsize(self.input_video_path) if os.path.exists(self.input_video_path) else 0
        output_size = os.path.getsize(self.output_video_path) if os.path.exists(self.output_video_path) else 0
        
        video_info = self.get_video_info()
        
        return {
            'processor': 'FFmpeg',
            'input_file_size_mb': input_size / (1024 * 1024),
            'output_file_size_mb': output_size / (1024 * 1024),
            'compression_ratio': input_size / max(output_size, 1),
            'video_info': video_info
        } 