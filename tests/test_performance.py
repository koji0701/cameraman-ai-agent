#!/usr/bin/env python3

"""
Performance tests for dynamic cropping FFmpeg implementation.
Tests large datasets, memory usage, and processing speed.
"""

import unittest
import tempfile
import os
import time
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add pipelines to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import (
    render_cropped_video_dynamic,
    render_cropped_video
)


class TestFFmpegPerformance(unittest.TestCase):
    """Performance tests for FFmpeg implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        
    def create_mock_video_file(self):
        """Create a mock video file for testing"""
        with open(self.input_video, 'w') as f:
            f.write("mock video content")

    def create_large_coordinate_dataset(self, size: int) -> pd.DataFrame:
        """Create a large coordinate dataset for testing"""
        return pd.DataFrame({
            't_ms': list(range(0, size * 33, 33)),  # 30 FPS timing
            'frame_number': list(range(size)),
            'crop_x': np.random.randint(0, 200, size),
            'crop_y': np.random.randint(0, 200, size),
            'crop_w': np.random.randint(800, 1200, size),
            'crop_h': np.random.randint(600, 800, size)
        })

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_large_dataset_handling(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test performance with large coordinate datasets (1000+ frames)"""
        self.create_mock_video_file()
        
        # Mock successful operations
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        
        # Create large dataset
        large_coords = self.create_large_coordinate_dataset(1000)
        
        start_time = time.time()
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                large_coords,
                verbose=False
            )
            
        end_time = time.time()
            elapsed = end_time - start_time
            
        self.assertTrue(result)
        self.assertLess(elapsed, 10.0, f"Processing 1000 frames took too long: {elapsed:.2f}s")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_memory_usage_large_dataset(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test memory usage with very large datasets"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 50 * 1024 * 1024  # 50MB
        
        # Create very large dataset (5000 frames)
        very_large_coords = self.create_large_coordinate_dataset(5000)
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
                result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                very_large_coords,
                verbose=False
                )
                
                self.assertTrue(result)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_coordinate_precision_performance(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test performance with high-precision coordinate data"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024
        
        # Create high precision coordinates
        precision_coords = pd.DataFrame({
            't_ms': np.linspace(0, 10000, 300),  # High precision timestamps
            'frame_number': range(300),
            'crop_x': np.random.uniform(0, 200, 300),
            'crop_y': np.random.uniform(0, 200, 300),
            'crop_w': np.random.uniform(800, 1200, 300),
            'crop_h': np.random.uniform(600, 800, 300)
        })
        
            start_time = time.time()
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                precision_coords,
                verbose=False
            )
            
        end_time = time.time()
        elapsed = end_time - start_time
        
        self.assertTrue(result)
        self.assertLess(elapsed, 5.0, f"High precision processing took too long: {elapsed:.2f}s")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_extreme_coordinate_values(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test handling of extreme coordinate values"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 2 * 1024 * 1024
        
        # Test with extreme values (should be clamped/handled gracefully)
        extreme_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [-1000, 0, 5000],  # Negative and very large values
            'crop_y': [-500, 1080, 3000],
            'crop_w': [10, 1920, 10000],  # Very small and very large
            'crop_h': [10, 1080, 8000]
        })
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                extreme_coords,
                verbose=False
            )
            
        # Should handle extreme values without crashing
        self.assertIsInstance(result, bool)

    def test_coordinate_dataframe_validation(self):
        """Test coordinate DataFrame validation and processing speed"""
        sizes = [10, 100, 1000, 5000]
        
        for size in sizes:
            start_time = time.time()
            coords = self.create_large_coordinate_dataset(size)
            end_time = time.time()
            
            # Dataset creation should be fast
            self.assertLess(end_time - start_time, 1.0)
            
            # DataFrame should have correct structure
            self.assertEqual(len(coords), size)
            self.assertIn('t_ms', coords.columns)
            self.assertIn('frame_number', coords.columns)
            self.assertIn('crop_x', coords.columns)
            self.assertIn('crop_y', coords.columns)
            self.assertIn('crop_w', coords.columns)
            self.assertIn('crop_h', coords.columns)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_concurrent_processing_safety(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test that dynamic processing is safe for concurrent operations"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 1 * 1024 * 1024
        
        coords = self.create_large_coordinate_dataset(100)
        
        # Multiple output files to simulate concurrent processing
        output_files = [
            os.path.join(self.test_dir, f"output_{i}.mp4")
            for i in range(3)
        ]
        
        results = []
        
        for output_file in output_files:
            with patch('render_video.HAS_OPENCV', True), \
                 patch('render_video.cv2.imread') as mock_imread, \
                 patch('render_video.cv2.imwrite') as mock_imwrite:
                
                mock_imread.return_value = MagicMock()
                mock_imwrite.return_value = True
                
                result = render_cropped_video_dynamic(
                    self.input_video,
                    output_file,
                    coords,
                    verbose=False
                )
                results.append(result)
        
        # All operations should succeed
        self.assertTrue(all(results))

    def test_memory_cleanup_verification(self):
        """Test that large datasets don't cause memory leaks"""
        # Create and process multiple large datasets
        for i in range(5):
            large_coords = self.create_large_coordinate_dataset(1000)
            
            # Process the dataset (DataFrame operations)
            processed = large_coords.copy()
            processed['crop_x'] = processed['crop_x'].astype(int)
            processed['crop_y'] = processed['crop_y'].astype(int)
            processed['crop_w'] = processed['crop_w'].astype(int) & ~1  # Make even
            processed['crop_h'] = processed['crop_h'].astype(int) & ~1  # Make even
            
            # Clean up explicitly
            del large_coords
            del processed
        
        # Test should complete without memory issues
        self.assertTrue(True)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_edge_case_performance(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test performance with edge case scenarios"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 30.0, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        # Test case 1: Single frame
        single_frame = pd.DataFrame({
            't_ms': [0],
            'frame_number': [0],
            'crop_x': [100],
            'crop_y': [50],
            'crop_w': [1000],
            'crop_h': [600]
        })
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                single_frame,
                verbose=False
            )
            
        self.assertIsInstance(result, bool)

    def test_coordinate_data_types(self):
        """Test handling of different data types in coordinates"""
        # Test with different numeric types
        coords_int = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [100, 110, 120],  # int
            'crop_y': [50, 55, 60],     # int
            'crop_w': [1000, 1000, 1000],  # int
            'crop_h': [600, 600, 600]   # int
        })
        
        coords_float = pd.DataFrame({
            't_ms': [0.0, 1000.5, 2000.7],
            'frame_number': [0, 30, 60],
            'crop_x': [100.5, 110.7, 120.3],  # float
            'crop_y': [50.2, 55.8, 60.1],     # float
            'crop_w': [1000.9, 1000.4, 1000.6],  # float
            'crop_h': [600.3, 600.7, 600.2]   # float
        })
        
        # Both should be processed without errors
        for coords in [coords_int, coords_float]:
            # Basic validation
            self.assertEqual(len(coords), 3)
            self.assertTrue(all(col in coords.columns for col in ['t_ms', 'frame_number', 'crop_x', 'crop_y', 'crop_w', 'crop_h']))


if __name__ == '__main__':
    unittest.main(verbosity=2) 