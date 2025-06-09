"""
Progress tracking utilities for AI Cameraman processing
Provides standardized progress reporting and timestamp management
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading


class ProgressTracker:
    """
    Manages progress tracking and reporting for video processing operations
    """
    
    def __init__(self):
        self.start_time = None
        self.current_stage = ""
        self.current_percentage = 0.0
        self.stage_times = {}
        self.lock = threading.Lock()
    
    def start_job(self, job_name: str = "video_processing"):
        """Start tracking a new job"""
        with self.lock:
            self.start_time = time.time()
            self.current_stage = "initializing"
            self.current_percentage = 0.0
            self.stage_times = {job_name: self.start_time}
    
    def update_progress(self, percentage: float, stage: str, details: str = ""):
        """Update current progress"""
        with self.lock:
            self.current_percentage = max(0.0, min(100.0, percentage))
            if stage != self.current_stage:
                current_time = time.time()
                self.stage_times[stage] = current_time
                self.current_stage = stage
    
    def get_timestamp(self) -> str:
        """Get current ISO timestamp"""
        return datetime.now().isoformat()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since job start in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time based on current progress"""
        if self.current_percentage <= 0 or self.start_time is None:
            return None
        
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return None
        
        # Simple linear estimation
        total_estimated = elapsed * (100.0 / self.current_percentage)
        remaining = total_estimated - elapsed
        return max(0.0, remaining)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        elapsed = self.get_elapsed_time()
        remaining = self.get_estimated_remaining_time()
        
        return {
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self.format_duration(elapsed),
            'estimated_remaining_seconds': remaining,
            'estimated_remaining_formatted': self.format_duration(remaining) if remaining else None,
            'current_stage': self.current_stage,
            'current_percentage': self.current_percentage,
            'stages_completed': list(self.stage_times.keys())
        }
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds is None or seconds < 0:
            return "Unknown"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def create_stage_callback(self, stage_name: str, start_percentage: float, end_percentage: float):
        """
        Create a callback function for a specific processing stage
        
        Args:
            stage_name: Name of the processing stage
            start_percentage: Starting percentage for this stage
            end_percentage: Ending percentage for this stage
            
        Returns:
            Callback function that maps sub-progress to overall progress
        """
        def stage_callback(sub_percentage: float, details: str = ""):
            # Map sub-percentage to overall percentage range
            overall_percentage = start_percentage + (sub_percentage / 100.0) * (end_percentage - start_percentage)
            self.update_progress(overall_percentage, stage_name, details)
        
        return stage_callback
    
    def get_stage_definitions(self) -> Dict[str, Dict[str, float]]:
        """Get standard stage definitions for AI Cameraman processing"""
        return {
            'initializing': {'start': 0, 'end': 5},
            'extracting_frames': {'start': 5, 'end': 15},
            'analyzing_with_gemini': {'start': 15, 'end': 40},
            'applying_smoothing': {'start': 40, 'end': 50},
            'processing_video': {'start': 50, 'end': 90},
            'finalizing': {'start': 90, 'end': 100}
        }
    
    def create_standard_callbacks(self) -> Dict[str, callable]:
        """Create standard callback functions for all processing stages"""
        stages = self.get_stage_definitions()
        callbacks = {}
        
        for stage_name, stage_range in stages.items():
            callbacks[stage_name] = self.create_stage_callback(
                stage_name, stage_range['start'], stage_range['end']
            )
        
        return callbacks


class MultiStageProgressTracker(ProgressTracker):
    """
    Enhanced progress tracker for complex multi-stage operations
    """
    
    def __init__(self):
        super().__init__()
        self.substage_progress = {}
        self.stage_weights = {}
    
    def set_stage_weights(self, weights: Dict[str, float]):
        """Set relative weights for different stages"""
        total_weight = sum(weights.values())
        self.stage_weights = {k: v/total_weight for k, v in weights.items()}
    
    def update_substage_progress(self, stage: str, substage: str, percentage: float, details: str = ""):
        """Update progress for a substage within a stage"""
        with self.lock:
            if stage not in self.substage_progress:
                self.substage_progress[stage] = {}
            
            self.substage_progress[stage][substage] = percentage
            
            # Calculate overall stage progress
            if self.substage_progress[stage]:
                stage_progress = sum(self.substage_progress[stage].values()) / len(self.substage_progress[stage])
            else:
                stage_progress = 0.0
            
            # Calculate overall progress using stage weights
            if self.stage_weights:
                overall_progress = sum(
                    self.stage_weights.get(s, 0) * (sum(subs.values()) / len(subs) if subs else 0)
                    for s, subs in self.substage_progress.items()
                ) * 100
            else:
                overall_progress = stage_progress
            
            self.update_progress(overall_progress, f"{stage}:{substage}", details)
    
    def get_detailed_progress(self) -> Dict[str, Any]:
        """Get detailed progress information including substages"""
        base_stats = self.get_processing_stats()
        base_stats['substage_progress'] = dict(self.substage_progress)
        base_stats['stage_weights'] = dict(self.stage_weights)
        return base_stats 