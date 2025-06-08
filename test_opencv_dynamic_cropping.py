#!/usr/bin/env python3
"""
Test script to demonstrate OpenCV dynamic cropping for AI Cameraman
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from video_processing.opencv_processor import OpenCVCameraman
from video_processing.pipeline_integration import AICameramanPipeline

def test_opencv_dynamic_cropping():
    """Test OpenCV dynamic cropping with waterpolo_trimmed_1080.mp4"""
    
    # Setup paths
    video_path = "videos/waterpolo_trimmed_1080.mp4"
    output_path = "outputs/waterpolo_opencv_test.mp4"
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    print("🎬 AI CAMERAMAN - OPENCV DYNAMIC CROPPING TEST")
    print("=" * 60)
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Test 1: Basic OpenCV processor initialization
        print("\n📹 Test 1: Initializing OpenCV processor...")
        processor = OpenCVCameraman(video_path, output_path, verbose=True)
        
        # Test 2: Create mock coordinates for demo (simulates Gemini API results)
        print("\n🎯 Test 2: Creating mock coordinates for water polo action...")
        
        # Sample coordinates that simulate following water polo action
        # These represent where the "action" is happening in the video
        mock_gemini_coords = [
            {"timestamp": 0.0, "x": 400, "y": 300, "width": 800, "height": 600},
            {"timestamp": 1.0, "x": 500, "y": 250, "width": 700, "height": 550},
            {"timestamp": 2.0, "x": 600, "y": 200, "width": 750, "height": 600},
            {"timestamp": 3.0, "x": 550, "y": 350, "width": 800, "height": 650},
            {"timestamp": 4.0, "x": 450, "y": 400, "width": 700, "height": 600},
        ]
        
        print(f"   Created {len(mock_gemini_coords)} coordinate keyframes")
        
        # Test 3: Convert Gemini coordinates to frame-based coordinates
        print("\n🔄 Test 3: Converting coordinates to frame mapping...")
        frame_coords = processor.convert_gemini_to_frame_coords(mock_gemini_coords)
        
        # Test 4: Calculate optimal output dimensions
        print("\n📐 Test 4: Calculating optimal output dimensions...")
        optimal_width, optimal_height = processor.calculate_optimal_dimensions(mock_gemini_coords)
        print(f"   Optimal dimensions: {optimal_width}x{optimal_height}")
        
        # Test 5: Setup video writer
        print("\n🎬 Test 5: Setting up optimized video writer...")
        processor._setup_optimized_writer((optimal_width, optimal_height))
        
        # Test 6: Process video frames with dynamic cropping
        print("\n✂️ Test 6: Processing video with dynamic cropping...")
        print("   This will crop each frame based on the coordinates...")
        success = processor._process_video_frames()
        
        # Test 7: Cleanup and show results
        print("\n🧹 Test 7: Cleanup and final results...")
        processor._cleanup()
        
        if success:
            print(f"\n🎉 SUCCESS!")
            print(f"✅ Dynamically cropped video created: {output_path}")
            print(f"\n📊 Processing Statistics:")
            print(f"   Total frames: {processor.stats.total_frames}")
            print(f"   Frames processed: {processor.stats.frames_processed}")
            print(f"   Processing time: {processor.stats.processing_time:.2f}s")
            print(f"   Average FPS: {processor.stats.average_fps:.2f}")
            print(f"   Input size: {processor.stats.input_size_mb:.1f} MB")
            print(f"   Output size: {processor.stats.output_size_mb:.1f} MB")
            print(f"   Compression ratio: {processor.stats.compression_ratio:.2f}x")
            
            print(f"\n✨ The video shows dynamic cropping following the mock 'action' coordinates")
            print(f"   In a real scenario, these coordinates would come from Gemini API")
            print(f"   analyzing the water polo footage to identify the main action areas.")
            
            return True
        else:
            print("❌ Processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_pipeline_integration():
    """Demonstrate the integrated pipeline with mock Gemini coordinates"""
    
    print("\n" + "=" * 60)
    print("🤖 DEMONSTRATION: INTEGRATED AI CAMERAMAN PIPELINE")
    print("=" * 60)
    
    video_path = "videos/waterpolo_trimmed_1080.mp4"
    output_path = "outputs/waterpolo_pipeline_test.mp4"
    
    try:
        # Initialize the pipeline
        pipeline = AICameramanPipeline(
            input_video_path=video_path,
            output_video_path=output_path,
            use_mock_coordinates=True,  # Use mock instead of Gemini API
            verbose=True
        )
        
        # Run the complete pipeline
        success = pipeline.process_video()
        
        if success:
            print(f"\n🎊 PIPELINE COMPLETE!")
            print(f"✅ AI-processed video: {output_path}")
            
            # Show storage savings
            savings = pipeline.get_storage_savings_estimate(video_path)
            print(f"\n💾 Storage Analysis:")
            print(f"   Input size: {savings['input_size_mb']:.1f} MB")
            print(f"   Estimated output: {savings['estimated_output_size_mb']:.1f} MB")
            print(f"   Size reduction: {savings['estimated_size_reduction_percent']:.1f}%")
            print(f"   Savings: {savings['estimated_savings_mb']:.1f} MB")
            
            return True
        else:
            print("❌ Pipeline failed")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting OpenCV Dynamic Cropping Tests...")
    
    # Test 1: Basic OpenCV dynamic cropping
    test1_success = test_opencv_dynamic_cropping()
    
    # Test 2: Integrated pipeline demonstration
    test2_success = demo_pipeline_integration()
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"OpenCV Dynamic Cropping: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Pipeline Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Your cameraman can fully use OpenCV for dynamic cropping.")
        print("\n📋 What you can do next:")
        print("1. Replace mock coordinates with real Gemini API analysis")
        print("2. Adjust cropping parameters for your specific use case")
        print("3. Run the full pipeline: python pipelines/render_video.py")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.") 