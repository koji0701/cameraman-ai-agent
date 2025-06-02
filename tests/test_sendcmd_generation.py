import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add pipelines to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import generate_sendcmd_filter, create_dynamic_crop_filter


class TestSendcmdGeneration(unittest.TestCase):
    """Specialized tests for sendcmd filter generation"""
    
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
        result = generate_sendcmd_filter(self.basic_coords, fps=30.0)
        
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
        result = generate_sendcmd_filter(self.basic_coords, fps=fps)
        
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
        result = generate_sendcmd_filter(self.basic_coords, fps=30.0)
        
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                parts = line.split()
                
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
        
        result = generate_sendcmd_filter(odd_coords, fps=30.0)
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
        
        with self.assertRaises(ValueError) as context:
            generate_sendcmd_filter(empty_df)
        
        self.assertIn("No crop data supplied", str(context.exception))
    
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
        
        result = generate_sendcmd_filter(single_frame, fps=30.0)
        
        # Should still be valid
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertIn('0.000 crop w 1000 h 600 x 100 y 50', result)
    
    def test_sendcmd_high_fps(self):
        """Test sendcmd generation with high frame rate"""
        high_fps = 120.0
        result = generate_sendcmd_filter(self.basic_coords, fps=high_fps)
        
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
        
        result = generate_sendcmd_filter(precise_coords, fps=30.0)
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
        
        result = generate_sendcmd_filter(large_coords, fps=30.0)
        
        # Should handle large values without issues
        self.assertIsInstance(result, str)
        self.assertIn('3840', result)  # Should contain large width
        self.assertIn('2160', result)  # Should contain large height
    
    def test_sendcmd_negative_coordinates(self):
        """Test sendcmd with negative coordinates"""
        negative_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [-50, 100],  # Negative X
            'crop_y': [50, -25],   # Negative Y
            'crop_w': [1000, 980],
            'crop_h': [600, 590]
        })
        
        result = generate_sendcmd_filter(negative_coords, fps=30.0)
        
        # Should still generate valid sendcmd (negatives will be preserved)
        self.assertIsInstance(result, str)
        self.assertIn('-50', result)
        self.assertIn('-25', result)
    
    def test_dynamic_crop_filter_complete(self):
        """Test complete dynamic crop filter creation"""
        filter_graph = create_dynamic_crop_filter(self.basic_coords, fps=30.0)
        
        # Should be a complete filter graph
        self.assertIsInstance(filter_graph, str)
        
        # Should contain all required components
        self.assertIn('sendcmd', filter_graph)
        self.assertIn('[crop_cmd]', filter_graph)
        self.assertIn('crop=w=1920:h=1080', filter_graph)
        self.assertIn('[cropped]', filter_graph)
        self.assertIn('scale=1920:1080', filter_graph)
        self.assertIn('[scaled]', filter_graph)
        
        # Should use semicolons to separate filter stages
        self.assertIn(';', filter_graph)
    
    def test_dynamic_crop_filter_empty_input(self):
        """Test dynamic crop filter with empty input"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            create_dynamic_crop_filter(empty_df)
    
    def test_sendcmd_different_fps_values(self):
        """Test sendcmd generation with various FPS values"""
        fps_values = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]
        
        for fps in fps_values:
            result = generate_sendcmd_filter(self.basic_coords, fps=fps)
            
            # Should work for all common FPS values
            self.assertTrue(result.startswith("sendcmd=c='"))
            self.assertIn('crop w', result)
            
            # Timeline should be consistent regardless of FPS (uses t_ms)
            lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
            timestamps = []
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    timestamps.append(float(parts[0]))
            
            # Should always have 3 timestamps for our test data
            self.assertEqual(len(timestamps), 3)
            
            # Should be in ascending order
            self.assertEqual(timestamps, sorted(timestamps))
    
    def test_sendcmd_coordinate_bounds(self):
        """Test sendcmd generation with extreme coordinate values"""
        extreme_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [0, 32000, 100],      # Very large X
            'crop_y': [0, 18000, 50],       # Very large Y
            'crop_w': [100, 32000, 1000],   # From tiny to huge
            'crop_h': [100, 18000, 600],    # From tiny to huge
        })
        
        result = generate_sendcmd_filter(extreme_coords, fps=30.0)
        
        # Should handle extreme values
        self.assertIsInstance(result, str)
        self.assertIn('32000', result)
        self.assertIn('18000', result)
    
    def test_sendcmd_fractional_coordinates(self):
        """Test sendcmd generation with fractional coordinates"""
        fractional_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [100.7, 110.3],
            'crop_y': [50.2, 55.8],
            'crop_w': [1000.9, 1020.1],
            'crop_h': [600.4, 610.6]
        })
        
        result = generate_sendcmd_filter(fractional_coords, fps=30.0)
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        
        for line in lines:
            if line.strip():
                parts = line.split()
                # All coordinate values should be integers
                for i in [3, 5, 7, 9]:  # w, h, x, y positions
                    self.assertTrue(parts[i].lstrip('-').isdigit())


if __name__ == '__main__':
    unittest.main(verbosity=2) 