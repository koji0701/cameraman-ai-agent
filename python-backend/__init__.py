"""
Python Backend for AI Cameraman Desktop GUI
============================================

This module provides the Python-shell bridge and utilities for integrating
the AI Cameraman video processing pipeline with the Electron desktop GUI.

Key Components:
- video_processor.py: Main python-shell bridge for Electron communication
- progress_tracker.py: Progress tracking and reporting utilities
- file_manager.py: File operations and video validation

Usage:
    This module is designed to be used via python-shell from the Electron
    main process, providing JSON-based communication for video processing.
"""

__version__ = "1.0.0"
__author__ = "AI Cameraman Team"

from .video_processor import VideoCameramanBridge
from .progress_tracker import ProgressTracker, MultiStageProgressTracker
from .file_manager import FileManager

__all__ = [
    'VideoCameramanBridge',
    'ProgressTracker', 
    'MultiStageProgressTracker',
    'FileManager'
] 