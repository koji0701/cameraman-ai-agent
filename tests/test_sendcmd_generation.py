import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add pipelines to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import generate_smooth_ffmpeg_filter


class TestSendcmdGeneration(unittest.TestCase):
    """Specialized tests for smooth ffmpeg filter generation (sendcmd-based)"""
    
    def setUp(self):
        """Set up test data"""
        self.basic_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [100, 110, 120],
            'crop_y': [50, 55, 60],
            'crop_w': [1000, 1020, 980],
            'crop_h': [600, 610, 590]
        })
    
    def test_sendcmd_basic_structure(self):
        """Test basic sendcmd filter structure"""
        result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=30.0)
        
        # Should wrap in sendcmd filter
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertTrue(result.endswith("'"))
        
        # Should contain crop commands
        self.assertIn('crop w', result)
        self.assertIn('h', result)
        self.assertIn('x', result)
        self.assertIn('y', result)
    
    def test_sendcmd_timeline_accuracy(self):
        """Test that timeline is accurately converted from t_ms"""
        fps = 29.97
        result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=fps)
        
        # Extract timestamps from the result
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        timestamps = []
        
        for line in lines:
            if line.strip():
                parts = line.split()
                timestamps.append(float(parts[0]))
        
        # Check that timestamps match expected values from t_ms
        expected_timestamps = [0.0, 1.0, 2.0]  # From t_ms / 1000
        
        for i, expected in enumerate(expected_timestamps):
            self.assertAlmostEqual(timestamps[i], expected, places=3)
    
    def test_sendcmd_coordinate_format(self):
        """Test coordinate formatting in sendcmd"""
        result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=30.0)
        
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                # Remove semicolon at the end before parsing
                parts = line.rstrip(';').split()
                
                # Should have correct format: timestamp crop w WIDTH h HEIGHT x X y Y
                self.assertEqual(parts[1], 'crop')
                self.assertEqual(parts[2], 'w')
                self.assertEqual(parts[4], 'h')
                self.assertEqual(parts[6], 'x')
                self.assertEqual(parts[8], 'y')
                
                # Values should be integers
                width = int(parts[3])
                height = int(parts[5])
                x = int(parts[7])
                y = int(parts[9])
                
                # Width and height should be even (for video encoding)
                self.assertEqual(width % 2, 0)
                self.assertEqual(height % 2, 0)
    
    def test_sendcmd_even_dimension_enforcement(self):
        """Test that odd dimensions are made even"""
        odd_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [100, 110],
            'crop_y': [50, 55],
            'crop_w': [1001, 983],  # Odd widths
            'crop_h': [601, 589]    # Odd heights
        })
        
        result = generate_smooth_ffmpeg_filter(odd_coords, fps=30.0)
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                parts = line.split()
                width = int(parts[3])
                height = int(parts[5])
                
                # Should be even
                self.assertEqual(width % 2, 0)
                self.assertEqual(height % 2, 0)
                
                # Should be <= original odd values
                self.assertLessEqual(width, 1001)
                self.assertLessEqual(height, 601)
    
    def test_sendcmd_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # Should return empty string for empty dataframe
        result = generate_smooth_ffmpeg_filter(empty_df)
        self.assertEqual(result, "")
    
    def test_sendcmd_single_frame(self):
        """Test sendcmd generation with single frame"""
        single_frame = pd.DataFrame({
            't_ms': [0],
            'frame_number': [0],
            'crop_x': [100],
            'crop_y': [50],
            'crop_w': [1000],
            'crop_h': [600]
        })
        
        result = generate_smooth_ffmpeg_filter(single_frame, fps=30.0)
        
        # Should still be valid
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertIn('0.000 crop w 1000 h 600 x 100 y 50', result)
    
    def test_sendcmd_high_fps(self):
        """Test sendcmd generation with high frame rate"""
        high_fps = 120.0
        result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=high_fps)
        
        # Timeline should use t_ms regardless of FPS
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        timestamps = []
        
        for line in lines:
            if line.strip():
                parts = line.split()
                timestamps.append(float(parts[0]))
        
        # With any FPS, timestamps should match the original t_ms values
        expected = [0.0, 1.0, 2.0]
        for i, exp in enumerate(expected):
            self.assertAlmostEqual(timestamps[i], exp, places=3)
    
    def test_sendcmd_precision(self):
        """Test timestamp precision in sendcmd"""
        # Create coordinates with sub-second timing
        precise_coords = pd.DataFrame({
            't_ms': [0, 333, 666, 999],
            'frame_number': [0, 10, 20, 30],
            'crop_x': [100, 110, 120, 115],
            'crop_y': [50, 55, 60, 58],
            'crop_w': [1000, 1020, 980, 990],
            'crop_h': [600, 610, 590, 595]
        })
        
        result = generate_smooth_ffmpeg_filter(precise_coords, fps=30.0)
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        timestamps = []
        for line in lines:
            if line.strip():
                parts = line.split()
                timestamps.append(float(parts[0]))
        
        # Should preserve millisecond precision
        expected = [0.0, 0.333, 0.666, 0.999]
        for i, exp in enumerate(expected):
            self.assertAlmostEqual(timestamps[i], exp, places=3)
    
    def test_sendcmd_large_coordinates(self):
        """Test sendcmd with large coordinate values"""
        large_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [0, 1920],
            'crop_y': [0, 1080],
            'crop_w': [1920, 3840],  # 4K width
            'crop_h': [1080, 2160],  # 4K height
        })
        
        result = generate_smooth_ffmpeg_filter(large_coords, fps=30.0)
        
        # Should handle large values without issues
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertIn('crop w 1920', result)
        self.assertIn('crop w 3840', result)
    
    def test_sendcmd_negative_coordinates(self):
        """Test sendcmd with negative coordinates (should be accepted)"""
        negative_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [-100, 50],
            'crop_y': [-50, 25],
            'crop_w': [1000, 900],
            'crop_h': [600, 500]
        })
        
        result = generate_smooth_ffmpeg_filter(negative_coords, fps=30.0)
        
        # Should include negative coordinates in output
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertIn('x -100', result)
        self.assertIn('y -50', result)

    def test_dynamic_crop_filter_complete(self):
        """Test complete dynamic crop filter generation process"""
        # This test verifies the integration with the smooth filter function
        result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=30.0)
        
        # Should be a complete sendcmd filter string
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertTrue(result.endswith("'"))
        
        # Should have all coordinate data - 3 lines separated by \n within the sendcmd
        inner_content = result.replace("sendcmd=c='", "").replace("'", "")
        lines = [line for line in inner_content.split('\n') if line.strip()]
        self.assertEqual(len(lines), 3)  # 3 data lines for our test coordinates
    
    def test_dynamic_crop_filter_empty_input(self):
        """Test dynamic crop filter with empty input"""
        empty_df = pd.DataFrame()
        result = generate_smooth_ffmpeg_filter(empty_df)
        self.assertEqual(result, "")
    
    def test_sendcmd_different_fps_values(self):
        """Test sendcmd generation with various FPS values"""
        fps_values = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]
        
        for fps in fps_values:
            result = generate_smooth_ffmpeg_filter(self.basic_coords, fps=fps)
            
            # Should generate valid sendcmd regardless of FPS
            self.assertTrue(result.startswith("sendcmd=c='"))
            self.assertIn('crop w', result)
            
            # Timeline should be based on t_ms, not FPS
            lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
            timestamps = []
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    timestamps.append(float(parts[0]))
            
            # Should always be [0.0, 1.0, 2.0] regardless of FPS
            expected = [0.0, 1.0, 2.0]
            for i, exp in enumerate(expected):
                self.assertAlmostEqual(timestamps[i], exp, places=3)
    
    def test_sendcmd_coordinate_bounds(self):
        """Test sendcmd with extreme coordinate bounds"""
        extreme_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [0, 5000],
            'crop_y': [0, 3000],
            'crop_w': [10, 8000],
            'crop_h': [10, 5000]
        })
        
        result = generate_smooth_ffmpeg_filter(extreme_coords, fps=30.0)
        
        # Should handle extreme values
        self.assertTrue(result.startswith("sendcmd=c='"))
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                parts = line.split()
                width = int(parts[3])
                height = int(parts[5])
                
                # Dimensions should still be even
                self.assertEqual(width % 2, 0)
                self.assertEqual(height % 2, 0)
    
    def test_sendcmd_fractional_coordinates(self):
        """Test sendcmd with fractional coordinates (should be converted to integers)"""
        fractional_coords = pd.DataFrame({
            't_ms': [0.0, 1000.5],
            'frame_number': [0, 30],
            'crop_x': [100.7, 110.3],
            'crop_y': [50.2, 55.8],
            'crop_w': [1000.9, 1020.1],
            'crop_h': [600.6, 610.4]
        })
        
        result = generate_smooth_ffmpeg_filter(fractional_coords, fps=30.0)
        
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                # Remove semicolon at the end before parsing
                parts = line.rstrip(';').split()
                
                # All coordinate values should be integers
                width = int(parts[3])
                height = int(parts[5])
                x = int(parts[7])
                y = int(parts[9])
                
                # Should be valid integers
                self.assertIsInstance(width, int)
                self.assertIsInstance(height, int)
                self.assertIsInstance(x, int)
                self.assertIsInstance(y, int)


if __name__ == '__main__':
    unittest.main(verbosity=2) 