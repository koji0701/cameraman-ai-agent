import unittest
import tempfile
import pandas as pd
import numpy as np
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add pipelines to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import (
    generate_sendcmd_filter,
    create_dynamic_crop_filter,
    render_cropped_video_simple,
    render_cropped_video_dynamic,
    render_multipass_video
)


class TestFFmpegPerformance(unittest.TestCase):
    """Performance tests for FFmpeg implementations"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        Path(self.input_video).touch()
        
        # Mock video info for performance tests
        self.mock_video_info = {
            'width': 1920, 'height': 1080, 'fps': 30.0,
            'duration': 10.0, 'has_audio': True, 'audio_codec': 'aac'
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sendcmd_generation_large_dataset(self):
        """Test sendcmd generation performance with large datasets"""
        # Test different dataset sizes
        sizes = [100, 1000, 5000, 10000]
        
        for size in sizes:
            coords = pd.DataFrame({
                't_ms': range(0, size * 33, 33),  # 30 FPS spacing
                'frame_number': range(size),
                'crop_x': np.random.randint(0, 500, size),
                'crop_y': np.random.randint(0, 300, size),
                'crop_w': np.random.randint(800, 1400, size),
                'crop_h': np.random.randint(600, 1000, size)
            })
            
            start_time = time.time()
            result = generate_sendcmd_filter(coords, fps=30.0)
            end_time = time.time()
            
            # Should complete within reasonable time
            elapsed = end_time - start_time
            self.assertLess(elapsed, 5.0, f"Sendcmd generation took too long for {size} frames: {elapsed:.2f}s")
            
            # Result should be properly formatted
            self.assertIsInstance(result, str)
            self.assertTrue(result.startswith("sendcmd=c='"))
            
            # Should contain expected number of crop commands
            crop_commands = result.count('crop w')
            self.assertEqual(crop_commands, size)

    def test_memory_usage_large_coordinates(self):
        """Test memory usage with large coordinate datasets"""
        # Create very large dataset (simulating 1 hour at 60 FPS)
        large_size = 60 * 60 * 60  # 216,000 frames
        
        # Generate coordinates in chunks to avoid memory issues during generation
        coords = pd.DataFrame({
            't_ms': range(0, large_size * 16, 16),  # 60 FPS spacing
            'frame_number': range(large_size),
            'crop_x': np.random.randint(0, 200, large_size),
            'crop_y': np.random.randint(0, 100, large_size),
            'crop_w': np.random.randint(1000, 1200, large_size),
            'crop_h': np.random.randint(600, 800, large_size)
        })
        
        # Test that sendcmd generation doesn't crash with memory errors
        try:
            start_time = time.time()
            result = generate_sendcmd_filter(coords, fps=60.0)
            end_time = time.time()
            
            # Should complete without memory errors
            self.assertIsInstance(result, str)
            elapsed = end_time - start_time
            print(f"Large dataset ({large_size} frames) processed in {elapsed:.2f}s")
            
        except MemoryError:
            self.fail("Memory error with large coordinate dataset")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    def test_simple_render_performance_scaling(self, mock_get_info, mock_subprocess):
        """Test simple render performance with different dataset sizes"""
        mock_get_info.return_value = self.mock_video_info
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        sizes = [10, 100, 1000, 5000]
        
        for size in sizes:
            coords = pd.DataFrame({
                't_ms': list(range(0, size * 33, 33)),
                'frame_number': list(range(size)),
                'crop_x': [100] * size,
                'crop_y': [50] * size,
                'crop_w': [1000] * size,
                'crop_h': [600] * size
            })
            
            start_time = time.time()
            result = render_cropped_video_simple(
                self.input_video, self.output_video, coords, verbose=False
            )
            end_time = time.time()
            
            self.assertTrue(result)
            elapsed = end_time - start_time
            
            # Simple render should be fast regardless of dataset size (uses averages)
            self.assertLess(elapsed, 1.0, f"Simple render too slow for {size} frames: {elapsed:.2f}s")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('builtins.open', create=True)
    def test_dynamic_render_performance_scaling(self, mock_open, mock_get_info, mock_subprocess):
        """Test dynamic render performance with different dataset sizes"""
        mock_get_info.return_value = self.mock_video_info
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        sizes = [10, 100, 1000]  # Dynamic mode is more expensive
        
        for size in sizes:
            coords = pd.DataFrame({
                't_ms': list(range(0, size * 33, 33)),
                'frame_number': list(range(size)),
                'crop_x': np.random.randint(50, 150, size),
                'crop_y': np.random.randint(25, 75, size),
                'crop_w': np.random.randint(900, 1100, size),
                'crop_h': np.random.randint(550, 650, size)
            })
            
            with patch('render_video.os.makedirs'), \
                 patch('render_video.os.remove'), \
                 patch('render_video.os.path.exists', return_value=False), \
                 patch('render_video.os.path.getsize', return_value=1024*1024):
                
                start_time = time.time()
                result = render_cropped_video_dynamic(
                    self.input_video, self.output_video, coords, verbose=False
                )
                end_time = time.time()
                
                self.assertTrue(result)
                elapsed = end_time - start_time
                
                # Dynamic render should scale reasonably with dataset size
                # Allow more time as dataset grows
                max_time = 0.1 + (size / 1000) * 2  # Base time + scaling factor
                self.assertLess(elapsed, max_time, f"Dynamic render too slow for {size} frames: {elapsed:.2f}s")

    def test_sendcmd_string_size_limits(self):
        """Test sendcmd string generation with size limits"""
        # Test with dataset that would create very large sendcmd string
        large_coords = pd.DataFrame({
            't_ms': range(0, 100000, 10),  # 10,000 frames
            'frame_number': range(10000),
            'crop_x': [100] * 10000,
            'crop_y': [50] * 10000,
            'crop_w': [1000] * 10000,
            'crop_h': [600] * 10000
        })
        
        result = generate_sendcmd_filter(large_coords, fps=30.0)
        
        # Should generate a very large string
        self.assertGreater(len(result), 100000)  # Should be substantial
        
        # But should still be properly formatted
        self.assertTrue(result.startswith("sendcmd=c='"))
        self.assertTrue(result.endswith("'"))
        
        # Check that all commands are present
        crop_count = result.count('crop w')
        self.assertEqual(crop_count, 10000)

    def test_fps_conversion_performance(self):
        """Test performance of FPS-related calculations"""
        # Test with various FPS values and large datasets
        fps_values = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0]
        
        size = 900  # Fixed size for all arrays
        coords = pd.DataFrame({
            't_ms': list(range(0, size * 33, 33)),  # ~900 frames
            'frame_number': list(range(size)),
            'crop_x': [100] * size,
            'crop_y': [50] * size,
            'crop_w': [1000] * size,
            'crop_h': [600] * size
        })
        
        for fps in fps_values:
            start_time = time.time()
            result = generate_sendcmd_filter(coords, fps=fps)
            end_time = time.time()
            
            elapsed = end_time - start_time
            self.assertLess(elapsed, 1.0, f"FPS {fps} processing too slow: {elapsed:.2f}s")
            
            # Should generate valid result regardless of FPS
            self.assertIsInstance(result, str)
            self.assertTrue(result.startswith("sendcmd=c='"))

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    def test_multipass_performance_vs_simple(self, mock_subprocess, mock_simple):
        """Test that multipass doesn't add excessive overhead"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        size = 90  # Fixed size for all arrays
        coords = pd.DataFrame({
            't_ms': list(range(0, size * 33, 33)),
            'frame_number': list(range(size)),
            'crop_x': [100] * size,
            'crop_y': [50] * size,
            'crop_w': [1000] * size,
            'crop_h': [600] * size
        })
        
        # Mock file operations to avoid actual I/O
        with patch('render_video.os.path.exists', return_value=True), \
             patch('render_video.os.remove'), \
             patch('render_video.os.path.getsize', return_value=1024*1024):
            
            start_time = time.time()
            result = render_multipass_video(
                self.input_video, self.output_video, coords,
                target_quality='medium', verbose=False
            )
            end_time = time.time()
            
            self.assertTrue(result)
            elapsed = end_time - start_time
            
            # Multipass should complete quickly (most time is in actual FFmpeg execution)
            self.assertLess(elapsed, 1.0, f"Multipass overhead too high: {elapsed:.2f}s")

    def test_coordinate_precision_performance(self):
        """Test performance with high-precision coordinates"""
        # Create dataset with many decimal places
        size = 1000
        coords = pd.DataFrame({
            't_ms': [i * 33.333333 for i in range(size)],
            'frame_number': list(range(size)),
            'crop_x': [100.123456789 + i * 0.001 for i in range(size)],
            'crop_y': [50.987654321 + i * 0.002 for i in range(size)],
            'crop_w': [1000.555666777 + i * 0.003 for i in range(size)],
            'crop_h': [600.444333222 + i * 0.004 for i in range(size)]
        })
        
        start_time = time.time()
        result = generate_sendcmd_filter(coords, fps=29.97)
        end_time = time.time()
        
        elapsed = end_time - start_time
        self.assertLess(elapsed, 2.0, f"High precision processing too slow: {elapsed:.2f}s")
        
        # Coordinates should be converted to integers properly
        lines = result.replace("sendcmd=c='", "").replace("'", "").split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.strip():
                parts = line.split()
                # Should have format: timestamp crop w WIDTH h HEIGHT x X y Y
                self.assertEqual(parts[1], 'crop')
                self.assertTrue(parts[3].isdigit())  # width
                self.assertTrue(parts[5].isdigit())  # height
                self.assertTrue(parts[7].lstrip('-').isdigit())  # x (can be negative)
                self.assertTrue(parts[9].lstrip('-').isdigit())  # y (can be negative)

    def test_extreme_coordinate_values(self):
        """Test performance with extreme coordinate values"""
        # Create dataset with extreme values
        coords = pd.DataFrame({
            't_ms': range(0, 1000, 1),
            'frame_number': range(1000),
            'crop_x': [0, -1000, 50000, 0] * 250,  # Mix of extreme values
            'crop_y': [0, -500, 25000, 0] * 250,
            'crop_w': [1, 100, 32000, 1920] * 250,
            'crop_h': [1, 100, 18000, 1080] * 250
        })
        
        start_time = time.time()
        result = generate_sendcmd_filter(coords, fps=1000.0)  # High FPS
        end_time = time.time()
        
        elapsed = end_time - start_time
        self.assertLess(elapsed, 2.0, f"Extreme values processing too slow: {elapsed:.2f}s")
        
        # Should handle extreme values without errors
        self.assertIsInstance(result, str)
        self.assertIn('32000', result)  # Should contain large values
        self.assertIn('-1000', result)  # Should contain negative values

    def test_concurrent_operation_safety(self):
        """Test that operations are safe for concurrent use"""
        import threading
        
        coords = pd.DataFrame({
            't_ms': range(0, 1000, 10),
            'frame_number': range(100),
            'crop_x': [100] * 100,
            'crop_y': [50] * 100,
            'crop_w': [1000] * 100,
            'crop_h': [600] * 100
        })
        
        results = []
        errors = []
        
        def worker():
            try:
                result = generate_sendcmd_filter(coords, fps=30.0)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All should succeed
        self.assertEqual(len(errors), 0, f"Concurrent operations failed: {errors}")
        self.assertEqual(len(results), 5)
        
        # All results should be identical
        for result in results[1:]:
            self.assertEqual(result, results[0])

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up"""
        import gc
        
        # Generate large dataset multiple times
        for iteration in range(5):
            coords = pd.DataFrame({
                't_ms': range(0, 50000, 50),  # 1000 frames
                'frame_number': range(1000),
                'crop_x': np.random.randint(0, 100, 1000),
                'crop_y': np.random.randint(0, 100, 1000),
                'crop_w': np.random.randint(900, 1100, 1000),
                'crop_h': np.random.randint(500, 700, 1000)
            })
            
            result = generate_sendcmd_filter(coords, fps=20.0)
            
            # Explicitly delete large objects
            del coords
            del result
            
            # Force garbage collection
            gc.collect()
        
        # If we reach here without memory errors, test passes
        self.assertTrue(True)

    def test_edge_case_performance(self):
        """Test performance with edge cases"""
        edge_cases = [
            # Single frame
            pd.DataFrame({
                't_ms': [0], 'frame_number': [0], 'crop_x': [100],
                'crop_y': [50], 'crop_w': [1000], 'crop_h': [600]
            }),
            # Very sparse timeline (large time gaps)
            pd.DataFrame({
                't_ms': [0, 10000, 20000, 30000],
                'frame_number': [0, 300, 600, 900],
                'crop_x': [100, 110, 120, 130],
                'crop_y': [50, 55, 60, 65],
                'crop_w': [1000, 1010, 1020, 1030],
                'crop_h': [600, 610, 620, 630]
            }),
            # High frequency changes
            pd.DataFrame({
                't_ms': range(0, 1000, 1),  # Every millisecond
                'frame_number': range(1000),
                'crop_x': [100 + i % 10 for i in range(1000)],
                'crop_y': [50 + i % 5 for i in range(1000)],
                'crop_w': [1000 + i % 20 for i in range(1000)],
                'crop_h': [600 + i % 15 for i in range(1000)]
            })
        ]
        
        for i, coords in enumerate(edge_cases):
            start_time = time.time()
            result = generate_sendcmd_filter(coords, fps=30.0)
            end_time = time.time()
            
            elapsed = end_time - start_time
            self.assertLess(elapsed, 1.0, f"Edge case {i} too slow: {elapsed:.2f}s")
            
            # Should generate valid result
            self.assertIsInstance(result, str)
            self.assertTrue(result.startswith("sendcmd=c='"))


if __name__ == '__main__':
    # Run performance tests
    unittest.main(verbosity=2) 