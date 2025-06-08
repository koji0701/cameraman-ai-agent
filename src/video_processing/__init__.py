"""
Video Processing Module for AI Cameraman

This module provides video processing capabilities for dynamic cropping
and action-following video generation.
"""

from .opencv_processor import OpenCVCameraman
from .video_utils import VideoUtils

__all__ = ['OpenCVCameraman', 'VideoUtils'] 