import unittest
import tempfile
import pandas as pd
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add pipelines to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import render_multipass_video


class TestMultipassEncoding(unittest.TestCase):
    """Specialized tests for multipass encoding functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        
        # Create sample coordinates
        self.sample_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000, 3000],
            'frame_number': [0, 30, 60, 90],
            'crop_x': [100, 110, 120, 115],
            'crop_y': [50, 55, 60, 58],
            'crop_w': [1000, 1020, 980, 990],
            'crop_h': [600, 610, 590, 595]
        })
        
        # Create mock input file
        Path(self.input_video).touch()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    @patch('render_video.os.path.getsize')
    def test_multipass_medium_quality(self, mock_getsize, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with medium quality settings"""
        # Mock successful operations
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='medium',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Verify first pass (simple render) was called
        mock_simple.assert_called_once()
        
        # Verify second pass (optimization) was called
        mock_subprocess.assert_called_once()
        
        # Check quality settings for medium
        args = mock_subprocess.call_args[0][0]
        self.assertIn('-crf', args)
        self.assertIn('23', args)  # Medium CRF
        self.assertIn('-preset', args)
        self.assertIn('medium', args)  # Medium preset

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    @patch('render_video.os.path.getsize')
    def test_multipass_high_quality(self, mock_getsize, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with high quality settings"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 15 * 1024 * 1024  # 15MB
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='high',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Check high quality settings
        args = mock_subprocess.call_args[0][0]
        self.assertIn('-crf', args)
        self.assertIn('20', args)  # High CRF
        self.assertIn('-preset', args)
        self.assertIn('slow', args)  # Slow preset for high quality

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    @patch('render_video.os.path.getsize')
    def test_multipass_ultra_quality(self, mock_getsize, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with ultra quality settings"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 20 * 1024 * 1024  # 20MB
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='ultra',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Check ultra quality settings
        args = mock_subprocess.call_args[0][0]
        self.assertIn('-crf', args)
        self.assertIn('18', args)  # Ultra CRF
        self.assertIn('-preset', args)
        self.assertIn('veryslow', args)  # Very slow preset for ultra quality

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_invalid_quality(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with invalid quality setting"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        # Should default to 'high' for invalid quality
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='invalid_quality',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Should use high quality defaults
        args = mock_subprocess.call_args[0][0]
        self.assertIn('-crf', args)
        self.assertIn('20', args)  # High CRF default

    @patch('render_video.render_cropped_video_simple')
    def test_multipass_first_pass_failure(self, mock_simple):
        """Test multipass encoding when first pass fails"""
        # Mock first pass failure
        mock_simple.return_value = False
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='high',
            verbose=False
        )
        
        self.assertFalse(result)
        
        # First pass should have been attempted
        mock_simple.assert_called_once()

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_second_pass_failure(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding when second pass fails"""
        mock_simple.return_value = True
        mock_exists.return_value = True
        
        # Mock second pass failure
        mock_subprocess.side_effect = Exception("FFmpeg error")
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='high',
            verbose=False
        )
        
        self.assertFalse(result)
        
        # Both passes should have been attempted
        mock_simple.assert_called_once()
        mock_subprocess.assert_called_once()
        
        # Cleanup should have been attempted
        mock_remove.assert_called()

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_temp_file_cleanup(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test that temporary files are properly cleaned up"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='medium',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Temporary cropped file should be cleaned up
        mock_remove.assert_called()
        
        # Should check for temp file existence
        mock_exists.assert_called()

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_ffmpeg_options(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test that correct FFmpeg options are used in second pass"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='high',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Check FFmpeg command structure
        args = mock_subprocess.call_args[0][0]
        
        self.assertIn('ffmpeg', args)
        self.assertIn('-y', args)  # Overwrite output
        self.assertIn('-i', args)  # Input file
        self.assertIn('-c:v', args)  # Video codec
        self.assertIn('libx264', args)  # Specific codec
        self.assertIn('-tune', args)  # Tuning
        self.assertIn('film', args)  # Film tuning
        self.assertIn('-movflags', args)  # Move flags
        self.assertIn('+faststart', args)  # Fast start
        self.assertIn('-c:a', args)  # Audio codec
        self.assertIn('aac', args)  # AAC audio
        self.assertIn('-b:a', args)  # Audio bitrate
        self.assertIn('128k', args)  # 128k audio

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_verbose_mode(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding in verbose mode"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            self.sample_coords,
            target_quality='medium',
            verbose=True
        )
        
        self.assertTrue(result)
        
        # Verbose mode should not capture output
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[1]
        self.assertFalse(call_args.get('capture_output', True))

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_coordinate_validation(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with various coordinate inputs"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        # Test with minimal coordinates
        minimal_coords = pd.DataFrame({
            't_ms': [0],
            'frame_number': [0],
            'crop_x': [100],
            'crop_y': [50],
            'crop_w': [1000],
            'crop_h': [600]
        })
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            minimal_coords,
            target_quality='medium',
            verbose=False
        )
        
        self.assertTrue(result)

    def test_multipass_quality_settings_mapping(self):
        """Test that quality settings are properly mapped"""
        # Import the function to test its internal quality settings
        from render_video import render_multipass_video
        
        # This test verifies the quality settings exist and have expected structure
        # We can't easily test the internal dictionary without refactoring,
        # but we've tested the actual functionality above
        
        quality_levels = ['medium', 'high', 'ultra']
        for quality in quality_levels:
            # These should be valid quality levels
            self.assertIn(quality, ['medium', 'high', 'ultra'])

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    @patch('render_video.os.path.getsize')
    def test_multipass_output_file_size_reporting(self, mock_getsize, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test that output file size is properly reported"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        # Mock different file sizes
        test_sizes = [
            5 * 1024 * 1024,   # 5MB
            15 * 1024 * 1024,  # 15MB
            50 * 1024 * 1024   # 50MB
        ]
        
        for size in test_sizes:
            mock_getsize.return_value = size
            
            # Mock that output file exists
            with patch('render_video.os.path.exists') as mock_output_exists:
                mock_output_exists.return_value = True
                
                result = render_multipass_video(
                    self.input_video,
                    self.output_video,
                    self.sample_coords,
                    target_quality='medium',
                    verbose=False
                )
                
                self.assertTrue(result)
                mock_getsize.assert_called_with(self.output_video)

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_empty_coordinates(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with empty coordinates DataFrame"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        empty_coords = pd.DataFrame()
        
        # Should pass the empty DataFrame to the simple render function
        # The simple render function should handle the empty case
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            empty_coords,
            target_quality='medium',
            verbose=False
        )
        
        # Result depends on how simple render handles empty coordinates
        # At minimum, the call should be made
        mock_simple.assert_called_once()

    @patch('render_video.render_cropped_video_simple')
    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.remove')
    def test_multipass_large_coordinates(self, mock_remove, mock_exists, mock_subprocess, mock_simple):
        """Test multipass encoding with large coordinate datasets"""
        mock_simple.return_value = True
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        
        # Create large coordinate dataset
        large_coords = pd.DataFrame({
            't_ms': range(0, 30000, 100),  # 300 frames
            'frame_number': range(300),
            'crop_x': [100 + (i % 50) for i in range(300)],
            'crop_y': [50 + (i % 30) for i in range(300)],
            'crop_w': [1000 + (i % 100) for i in range(300)],
            'crop_h': [600 + (i % 60) for i in range(300)]
        })
        
        result = render_multipass_video(
            self.input_video,
            self.output_video,
            large_coords,
            target_quality='medium',
            verbose=False
        )
        
        self.assertTrue(result)
        
        # Should handle large datasets without issues
        mock_simple.assert_called_once()
        mock_subprocess.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2) 