#!/usr/bin/env python3
"""
Python-shell bridge for AI Cameraman processing
Handles JSON commands from Electron and provides progress updates
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.video_processing.pipeline_integration import AICameramanPipeline
from progress_tracker import ProgressTracker
from file_manager import FileManager

class VideoCameramanBridge:
    def __init__(self):
        self.pipeline = AICameramanPipeline(processor_type="opencv", verbose=False)
        self.progress_tracker = ProgressTracker()
        self.file_manager = FileManager()
        self.current_job = None
        self.processing_thread = None
        self.cancel_requested = False
    
    def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send structured JSON message to Electron"""
        message = {
            'type': message_type,
            'timestamp': self.progress_tracker.get_timestamp(),
            'data': data
        }
        print(json.dumps(message))
        sys.stdout.flush()
    
    def process_video(self, input_path: str, output_path: str, options: Dict[str, Any]):
        """Process video with progress reporting"""
        try:
            self.cancel_requested = False
            self.send_message('status', {'status': 'starting', 'message': 'Initializing AI Cameraman...'})
            
            # Validate input file
            if not self.file_manager.validate_video_file(input_path):
                self.send_message('error', {'message': f'Invalid video file: {input_path}'})
                return
            
            # Ensure output directory exists
            self.file_manager.ensure_output_directory(output_path)
            
            # Create progress callback
            def progress_callback(percentage: float, stage: str, details: str = ""):
                if self.cancel_requested:
                    raise InterruptedError("Processing cancelled by user")
                
                self.send_message('progress', {
                    'percentage': percentage,
                    'stage': stage,
                    'details': details
                })
            
            # Process video using existing pipeline
            success = self.pipeline.process_video_complete(
                input_video_path=input_path,
                output_video_path=output_path,
                padding_factor=options.get('padding_factor', 1.1),
                smoothing_strength=options.get('smoothing_strength', 'balanced'),
                interpolation_method=options.get('interpolation_method', 'cubic'),
                progress_callback=progress_callback,
                verbose=True
            )
            
            if success and not self.cancel_requested:
                # Get file statistics
                stats = self.file_manager.get_file_stats(input_path, output_path)
                
                self.send_message('completed', {
                    'output_path': output_path,
                    'input_size_mb': stats['input_size_mb'],
                    'output_size_mb': stats['output_size_mb'],
                    'compression_ratio': stats['compression_ratio'],
                    'processing_time': stats.get('processing_time', 0)
                })
            elif self.cancel_requested:
                self.send_message('cancelled', {'message': 'Processing cancelled by user'})
            else:
                self.send_message('error', {'message': 'Video processing failed'})
                
        except InterruptedError:
            self.send_message('cancelled', {'message': 'Processing cancelled by user'})
        except Exception as e:
            self.send_message('error', {'message': f'Processing error: {str(e)}'})
    
    def handle_command(self, command: Dict[str, Any]):
        """Handle incoming JSON command from Electron"""
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'process_video':
                # Start processing in separate thread
                self.processing_thread = threading.Thread(
                    target=self.process_video,
                    args=(
                        command['input_path'],
                        command['output_path'],
                        command.get('options', {})
                    )
                )
                self.processing_thread.start()
                
            elif cmd_type == 'cancel':
                self.cancel_requested = True
                self.send_message('status', {'status': 'cancelling', 'message': 'Cancelling processing...'})
                
            elif cmd_type == 'ping':
                self.send_message('pong', {'message': 'Python bridge is ready'})
                
            elif cmd_type == 'get_video_info':
                video_info = self.file_manager.get_video_info(command['video_path'])
                self.send_message('video_info', video_info)
                
            else:
                self.send_message('error', {'message': f'Unknown command type: {cmd_type}'})
                
        except KeyError as e:
            self.send_message('error', {'message': f'Missing required parameter: {e}'})
        except Exception as e:
            self.send_message('error', {'message': f'Command error: {str(e)}'})
    
    def run(self):
        """Main event loop - listen for commands from Electron"""
        # Send ready signal
        self.send_message('ready', {'message': 'Python bridge initialized and ready'})
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    command = json.loads(line)
                    self.handle_command(command)
                except json.JSONDecodeError:
                    self.send_message('error', {'message': f'Invalid JSON command: {line}'})
                except Exception as e:
                    self.send_message('error', {'message': f'Error processing command: {str(e)}'})
                    
        except KeyboardInterrupt:
            self.send_message('status', {'status': 'shutdown', 'message': 'Python bridge shutting down'})
        except Exception as e:
            self.send_message('error', {'message': f'Bridge error: {str(e)}'})

if __name__ == "__main__":
    bridge = VideoCameramanBridge()
    bridge.run() 