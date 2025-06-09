#!/usr/bin/env python3
"""
Test script for the Python backend bridge
Simulates Electron communication and tests the video processing pipeline
"""

import json
import sys
import subprocess
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "python-backend"))

from video_processor import VideoCameramanBridge
from file_manager import FileManager


def test_bridge_communication():
    """Test the basic communication with the bridge"""
    print("🧪 Testing bridge communication...")
    
    bridge = VideoCameramanBridge()
    
    # Test basic functionality
    file_manager = FileManager()
    
    # Create a mock video info
    test_commands = [
        {'type': 'ping'},
        {'type': 'get_video_info', 'video_path': '/fake/path/test.mp4'},
    ]
    
    for command in test_commands:
        print(f"   Testing command: {command['type']}")
        try:
            bridge.handle_command(command)
            print(f"   ✅ Command {command['type']} handled successfully")
        except Exception as e:
            print(f"   ❌ Command {command['type']} failed: {e}")


def test_file_manager():
    """Test file manager functionality"""
    print("\n🧪 Testing file manager...")
    
    file_manager = FileManager()
    
    # Test video format validation
    test_files = [
        'test.mp4',
        'test.mov', 
        'test.avi',
        'test.txt',
        'test.jpg'
    ]
    
    for filename in test_files:
        # Create temporary test file
        temp_path = Path(f"/tmp/{filename}")
        try:
            temp_path.touch()
            is_valid = file_manager.validate_video_file(str(temp_path))
            expected = filename.endswith(('.mp4', '.mov', '.avi'))
            
            if filename.endswith('.txt') or filename.endswith('.jpg'):
                expected = False  # These should fail validation
            
            print(f"   {filename}: {'✅' if is_valid == expected else '❌'} (valid: {is_valid})")
            
            # Clean up
            temp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"   {filename}: ❌ Error: {e}")


def test_progress_tracker():
    """Test progress tracking functionality"""
    print("\n🧪 Testing progress tracker...")
    
    from progress_tracker import ProgressTracker
    
    tracker = ProgressTracker()
    tracker.start_job("test_job")
    
    # Simulate progress updates
    stages = [
        (10, "initializing", "Setting up..."),
        (30, "processing", "Processing frames..."),
        (60, "encoding", "Encoding video..."),
        (90, "finalizing", "Finishing up..."),
        (100, "completed", "Done!")
    ]
    
    for percentage, stage, details in stages:
        tracker.update_progress(percentage, stage, details)
        stats = tracker.get_processing_stats()
        print(f"   {percentage}% - {stage}: {details}")
        print(f"     Elapsed: {stats['elapsed_formatted']}")
        time.sleep(0.1)  # Simulate work
    
    print("   ✅ Progress tracking test completed")


def test_mock_processing():
    """Test mock video processing without actual video file"""
    print("\n🧪 Testing mock video processing...")
    
    bridge = VideoCameramanBridge()
    
    # Test with mock command
    mock_command = {
        'type': 'process_video',
        'input_path': '/fake/input.mp4',
        'output_path': '/fake/output.mp4',
        'options': {
            'padding_factor': 1.1,
            'smoothing_strength': 'balanced'
        }
    }
    
    print("   Sending mock processing command...")
    
    # Capture output
    messages = []
    original_stdout = sys.stdout
    
    class MessageCapture:
        def write(self, text):
            if text.strip():
                try:
                    msg = json.loads(text.strip())
                    messages.append(msg)
                    print(f"   📨 {msg['type']}: {msg.get('data', {}).get('message', 'No message')}")
                except:
                    print(f"   📝 {text.strip()}")
        
        def flush(self):
            pass
    
    # Run the command in a separate thread
    def run_command():
        sys.stdout = MessageCapture()
        try:
            bridge.handle_command(mock_command)
        finally:
            sys.stdout = original_stdout
    
    thread = threading.Thread(target=run_command)
    thread.start()
    thread.join(timeout=5)  # Wait max 5 seconds
    
    if messages:
        print(f"   ✅ Received {len(messages)} messages from bridge")
    else:
        print("   ⚠️ No messages received (this is expected for mock processing)")


def main():
    """Run all tests"""
    print("🚀 AI Cameraman Python Backend Test Suite")
    print("=" * 50)
    
    try:
        test_bridge_communication()
        test_file_manager()
        test_progress_tracker()
        test_mock_processing()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\n💡 Next steps:")
        print("   1. Create the Electron desktop GUI")
        print("   2. Set up python-shell integration")
        print("   3. Test with real video files")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 