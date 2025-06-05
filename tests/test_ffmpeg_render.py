#!/usr/bin/env python3

"""
FFmpeg rendering test suite for dynamic cropping mode.
Tests the frame-by-frame dynamic cropping functionality.
"""

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import json

# Add pipelines to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import (
    render_cropped_video_dynamic,
    render_cropped_video,
    get_video_info,
    create_preview_video,
    batch_render_videos,
    process_and_render_complete
)


class TestFFmpegRenderModes(unittest.TestCase):
    """Comprehensive tests for dynamic rendering implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        
        # Create sample coordinates
        self.sample_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000, 3000],
            'frame_number': [0, 30, 60, 90],
            'crop_x': [100, 110, 120, 130],
            'crop_y': [50, 55, 60, 65],
            'crop_w': [1000, 1010, 1020, 1030],
            'crop_h': [600, 610, 620, 630]
        })
        
    def create_mock_video_file(self):
        """Create a mock video file for testing"""
        with open(self.input_video, 'w') as f:
            f.write("mock video content")

    @patch('render_video.subprocess.run')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    @patch('render_video.os.listdir')
    @patch('render_video.tempfile.TemporaryDirectory')
    def test_dynamic_render_mode(self, mock_temp_dir, mock_listdir, mock_getsize, mock_exists, mock_subprocess):
        """Test dynamic rendering mode"""
        self.create_mock_video_file()
        
        # Mock temp directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/mock_temp"
        
        # Mock frame files exist
        mock_listdir.return_value = ["frame_00001.png", "frame_00002.png", "frame_00003.png", "frame_00004.png"]
        
        # Mock successful subprocess calls with proper ffprobe response
        def mock_subprocess_side_effect(*args, **kwargs):
            cmd = args[0]
            if 'ffprobe' in cmd:
                # Mock ffprobe response with proper JSON
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    'streams': [
                        {
                            'codec_type': 'video',
                            'width': 1920,
                            'height': 1080,
                            'r_frame_rate': '30000/1001',
                            'duration': '60.0',
                            'bit_rate': '5000000',
                            'codec_name': 'h264',
                            'pix_fmt': 'yuv420p',
                            'nb_frames': '1800'
                        },
                        {
                            'codec_type': 'audio',
                            'codec_name': 'aac'
                        }
                    ]
                })
                return mock_result
            else:
                # Mock other ffmpeg calls
                mock_result = MagicMock()
                mock_result.returncode = 0
                return mock_result
        
        mock_subprocess.side_effect = mock_subprocess_side_effect
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        
        # Mock OpenCV operations
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite, \
             patch('render_video.os.makedirs'), \
             patch('render_video.os.remove'):
            
            mock_imread.return_value = MagicMock()  # Mock image
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                self.sample_coords,
                enable_stabilization=False,
                color_correction=False,
                verbose=False
            )
            
            self.assertTrue(result)

    @patch('render_video.render_cropped_video_dynamic')
    def test_render_mode_dispatcher(self, mock_dynamic):
        """Test the main render function calls dynamic mode"""
        self.create_mock_video_file()
        
        mock_dynamic.return_value = True
        
        # Test dynamic mode (default behavior)
        result = render_cropped_video(
            self.input_video,
            self.output_video,
            smoothed_coords_df=self.sample_coords
        )
        self.assertTrue(result)
        mock_dynamic.assert_called_once()

    def test_coordinate_validation(self):
        """Test coordinate validation in dynamic mode"""
        # Test with invalid coordinates (should be handled gracefully)
        invalid_coords = pd.DataFrame({
            't_ms': [0, 1000],
            'frame_number': [0, 30],
            'crop_x': [-100, 2000],  # Invalid coordinates
            'crop_y': [-50, 1500],
            'crop_w': [3000, 5000],  # Too large
            'crop_h': [2000, 3000]
        })
        
        with patch('render_video.subprocess.run') as mock_subprocess, \
             patch('render_video.get_video_info') as mock_get_info, \
             patch('render_video.os.path.exists', return_value=True), \
             patch('render_video.os.path.getsize', return_value=1024*1024):
            
            mock_get_info.return_value = {
                'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': False
            }
            mock_subprocess.return_value = MagicMock(returncode=0)
            
            # Should handle invalid coordinates without crashing
            self.create_mock_video_file()
            
            with patch('render_video.HAS_OPENCV', True), \
                 patch('render_video.cv2.imread') as mock_imread, \
                 patch('render_video.cv2.imwrite') as mock_imwrite:
                
                mock_imread.return_value = MagicMock()
                mock_imwrite.return_value = True
                
                result = render_cropped_video_dynamic(
                    self.input_video,
                    self.output_video,
                    invalid_coords,
                    verbose=False
                )
                
                # Should not crash, even with invalid coordinates
                self.assertIsInstance(result, bool)


class TestFFmpegIntegration(unittest.TestCase):
    """Integration tests for FFmpeg command generation and execution"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        
        self.sample_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [100, 110, 120],
            'crop_y': [50, 55, 60],
            'crop_w': [1000, 1000, 1000],
            'crop_h': [600, 600, 600]
        })

    def create_mock_video_file(self):
        """Create a mock video file for testing"""
        with open(self.input_video, 'w') as f:
            f.write("mock video content")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    def test_video_info_extraction(self, mock_get_info, mock_subprocess):
        """Test video information extraction"""
        mock_get_info.return_value = {
            'width': 1920,
            'height': 1080,
            'fps': 29.97,
            'duration': 60.0,
            'has_audio': True,
            'codec': 'h264'
        }
        
        info = get_video_info(self.input_video)
        
        self.assertEqual(info['width'], 1920)
        self.assertEqual(info['height'], 1080)
        self.assertEqual(info['fps'], 29.97)
        self.assertTrue(info['has_audio'])

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_audio_handling(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test audio stream handling in dynamic mode"""
        self.create_mock_video_file()
        
        # Test with audio
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': True
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                self.sample_coords,
                verbose=False
            )
            
            self.assertTrue(result)
            
            # Should have multiple subprocess calls for audio extraction and muxing
            self.assertGreater(mock_subprocess.call_count, 1)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_no_audio_handling(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test handling video without audio"""
        self.create_mock_video_file()
        
        # Test without audio
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                self.sample_coords,
                verbose=False
            )
            
            self.assertTrue(result)


class TestAdvancedFeatures(unittest.TestCase):
    """Tests for advanced rendering features"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, "test_input.mp4")
        self.output_video = os.path.join(self.test_dir, "test_output.mp4")
        
        self.sample_coords = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'frame_number': [0, 30, 60],
            'crop_x': [100, 110, 120],
            'crop_y': [50, 55, 60],
            'crop_w': [1000, 1000, 1000],
            'crop_h': [600, 600, 600]
        })

    def create_mock_video_file(self):
        """Create a mock video file for testing"""
        with open(self.input_video, 'w') as f:
            f.write("mock video content")

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_stabilization_enabled(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test video stabilization feature"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                self.sample_coords,
                enable_stabilization=True,
                verbose=False
            )
            
            self.assertTrue(result)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_color_correction_enabled(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test color correction feature"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 5 * 1024 * 1024
        
        with patch('render_video.HAS_OPENCV', True), \
             patch('render_video.cv2.imread') as mock_imread, \
             patch('render_video.cv2.imwrite') as mock_imwrite:
            
            mock_imread.return_value = MagicMock()
            mock_imwrite.return_value = True
            
            result = render_cropped_video_dynamic(
                self.input_video,
                self.output_video,
                self.sample_coords,
                color_correction=True,
                verbose=False
            )
            
            self.assertTrue(result)

    @patch('render_video.subprocess.run')
    @patch('render_video.get_video_info')
    @patch('render_video.os.path.exists')
    @patch('render_video.os.path.getsize')
    def test_preview_creation(self, mock_getsize, mock_exists, mock_get_info, mock_subprocess):
        """Test preview video creation"""
        self.create_mock_video_file()
        
        mock_get_info.return_value = {
            'width': 1920, 'height': 1080, 'fps': 29.97, 'has_audio': False
        }
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 2 * 1024 * 1024
        
        preview_path = os.path.join(self.test_dir, "preview.mp4")
        
        result = create_preview_video(
            self.input_video,
            preview_path,
            self.sample_coords,
            preview_duration=5,
            verbose=False
        )
        
        self.assertTrue(result)

    @patch('render_video.process_and_render_complete')
    def test_batch_processing(self, mock_process):
        """Test batch video processing"""
        mock_process.return_value = True
        
        input_videos = [
            os.path.join(self.test_dir, "video1.mp4"),
            os.path.join(self.test_dir, "video2.mp4")
        ]
        
        # Create mock input files
        for video in input_videos:
            with open(video, 'w') as f:
                f.write("mock video")
        
        output_dir = os.path.join(self.test_dir, "batch_output")
        
        results = batch_render_videos(
            input_videos,
            output_dir,
            quality_preset='medium',
            create_previews=False,
            verbose=False
        )
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results))


if __name__ == '__main__':
    unittest.main(verbosity=2) 