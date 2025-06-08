#!/usr/bin/env python3
"""
Test script for OpenCV video processor

This script tests the OpenCV-based video processing implementation
to ensure it works correctly before full migration.
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from video_processing.opencv_processor import OpenCVCameraman
    from video_processing.benchmark import VideoProcessorBenchmark, create_test_coordinates
    from video_processing.video_utils import VideoUtils
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)


def create_test_video(output_path: str, duration: int = 5) -> bool:
    """Create a simple test video for testing purposes"""
    try:
        import cv2
        import numpy as np
        
        # Video properties
        width, height = 640, 480
        fps = 30
        total_frames = duration * fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"⚠️ Could not create test video at {output_path}")
            return False
        
        print(f"🎬 Creating test video: {width}x{height} @ {fps}fps for {duration}s")
        
        # Generate frames with moving colored rectangle
        for frame_num in range(total_frames):
            # Create black background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Moving rectangle (simulates water polo action)
            rect_size = 100
            center_x = int(width/2 + 200 * np.sin(frame_num * 0.1))
            center_y = int(height/2 + 100 * np.cos(frame_num * 0.1))
            
            # Ensure rectangle stays within bounds
            x1 = max(0, center_x - rect_size//2)
            y1 = max(0, center_y - rect_size//2)
            x2 = min(width, center_x + rect_size//2)
            y2 = min(height, center_y + rect_size//2)
            
            # Draw colored rectangle (simulates action)
            color = (
                int(128 + 127 * np.sin(frame_num * 0.05)),  # Blue
                int(128 + 127 * np.cos(frame_num * 0.05)),  # Green  
                255                                          # Red
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            
            # Add frame number text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()
        print(f"✅ Test video created successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        return False


def test_opencv_processor_basic():
    """Test basic OpenCV processor functionality"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING OPENCV PROCESSOR - BASIC FUNCTIONALITY")
    print(f"{'='*60}")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_input = temp_dir / "test_input.mp4"
        test_output = temp_dir / "test_output.mp4"
        
        # Create test video
        print("1️⃣ Creating test video...")
        if not create_test_video(str(test_input)):
            print("❌ Failed to create test video")
            return False
        
        # Test video info extraction
        print("2️⃣ Testing video info extraction...")
        try:
            video_info = VideoUtils.get_video_info_opencv(str(test_input))
            print(f"   📹 Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f}fps")
            print(f"   ⏱️  Duration: {video_info['duration']:.2f}s ({video_info['total_frames']} frames)")
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return False
        
        # Test coordinate conversion
        print("3️⃣ Testing coordinate conversion...")
        test_coords = create_test_coordinates()
        try:
            processor = OpenCVCameraman(str(test_input), str(test_output), verbose=False)
            frame_coords = processor.convert_gemini_to_frame_coords(test_coords)
            print(f"   🎯 Converted {len(test_coords)} coordinates to {len(frame_coords)} frame mappings")
        except Exception as e:
            print(f"❌ Error in coordinate conversion: {e}")
            return False
        
        # Test video processing
        print("4️⃣ Testing video processing...")
        try:
            start_time = time.time()
            processor = OpenCVCameraman(str(test_input), str(test_output), verbose=True)
            success = processor.process_with_gemini_coords(test_coords)
            processing_time = time.time() - start_time
            
            if success and os.path.exists(test_output):
                input_size = os.path.getsize(test_input) / (1024 * 1024)
                output_size = os.path.getsize(test_output) / (1024 * 1024)
                print(f"   ✅ Processing successful in {processing_time:.2f}s")
                print(f"   📊 Input: {input_size:.1f}MB → Output: {output_size:.1f}MB")
                print(f"   🗜️  Compression: {input_size/output_size:.2f}x")
                return True
            else:
                print(f"❌ Processing failed or output file not created")
                return False
                
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return False


def test_opencv_vs_ffmpeg_benchmark():
    """Test OpenCV vs FFmpeg performance comparison"""
    print(f"\n{'='*60}")
    print(f"⚖️ TESTING OPENCV vs FFMPEG PERFORMANCE")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_input = temp_dir / "benchmark_input.mp4"
        
        # Create larger test video for meaningful benchmark
        print("📹 Creating test video for benchmarking...")
        if not create_test_video(str(test_input), duration=10):  # 10 second video
            print("❌ Failed to create benchmark video")
            return False
        
        # Run benchmark comparison
        try:
            benchmark = VideoProcessorBenchmark(str(temp_dir / "benchmark_results"))
            test_coords = create_test_coordinates()
            
            results = benchmark.run_comparison_benchmark(str(test_input), test_coords)
            benchmark.save_results()
            benchmark.print_summary()
            
            # Analyze results
            if results['opencv'] and results['ffmpeg']:
                opencv_stats = results['opencv']
                ffmpeg_stats = results['ffmpeg']
                
                print(f"\n🏆 PERFORMANCE COMPARISON:")
                print(f"   ⏱️  Speed: OpenCV vs FFmpeg = {opencv_stats.processing_time_seconds:.2f}s vs {ffmpeg_stats.processing_time_seconds:.2f}s")
                print(f"   💾 Storage: OpenCV vs FFmpeg = {opencv_stats.storage_efficiency:.1f}% vs {ffmpeg_stats.storage_efficiency:.1f}%")
                print(f"   🧠 Memory: OpenCV vs FFmpeg = {opencv_stats.peak_memory_usage_mb:.1f}MB vs {ffmpeg_stats.peak_memory_usage_mb:.1f}MB")
                
                return True
            else:
                print("⚠️ Benchmark completed but some processors failed")
                return False
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False


def main():
    """Main test function"""
    print(f"🚀 OpenCV Video Processor Test Suite")
    print(f"{'='*60}")
    
    # Test basic functionality
    basic_test_passed = test_opencv_processor_basic()
    
    # Test performance comparison (only if basic tests pass)
    if basic_test_passed:
        benchmark_test_passed = test_opencv_vs_ffmpeg_benchmark()
    else:
        print("⚠️ Skipping benchmark tests due to basic test failures")
        benchmark_test_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Basic functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"Performance benchmark: {'✅ PASS' if benchmark_test_passed else '❌ FAIL'}")
    
    if basic_test_passed and benchmark_test_passed:
        print(f"\n🎉 All tests passed! OpenCV processor is ready for production.")
        return True
    else:
        print(f"\n⚠️ Some tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 