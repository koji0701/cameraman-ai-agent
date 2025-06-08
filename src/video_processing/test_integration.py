#!/usr/bin/env python3
"""
Test script for the integrated AI Cameraman pipeline

Tests the integration between OpenCV processor and the existing Gemini API pipeline.
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from video_processing.pipeline_integration import (
        AICameramanPipeline, 
        process_video_with_ai_cameraman
    )
    from video_processing.test_opencv_processor import create_test_video
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_pipeline_integration():
    """Test the integrated pipeline functionality"""
    print(f"\n{'='*60}")
    print(f"🔗 TESTING PIPELINE INTEGRATION")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test video
        test_input = temp_dir / "test_input.mp4"
        opencv_output = temp_dir / "opencv_output.mp4"
        ffmpeg_output = temp_dir / "ffmpeg_output.mp4"
        
        print("1️⃣ Creating test video...")
        if not create_test_video(str(test_input), duration=8):
            print("❌ Failed to create test video")
            return False
        
        # Test OpenCV pipeline
        print("2️⃣ Testing OpenCV pipeline integration...")
        try:
            opencv_pipeline = AICameramanPipeline(processor_type="opencv", verbose=True)
            opencv_success = opencv_pipeline.process_video_complete(
                str(test_input),
                str(opencv_output),
                padding_factor=1.1
            )
            
            if opencv_success and os.path.exists(opencv_output):
                opencv_size = os.path.getsize(opencv_output) / (1024 * 1024)
                print(f"   ✅ OpenCV pipeline successful, output: {opencv_size:.1f}MB")
            else:
                print(f"   ❌ OpenCV pipeline failed")
                return False
                
        except Exception as e:
            print(f"   ❌ OpenCV pipeline error: {e}")
            return False
        
        # Test FFmpeg pipeline (if available)
        print("3️⃣ Testing FFmpeg pipeline integration...")
        try:
            ffmpeg_pipeline = AICameramanPipeline(processor_type="ffmpeg", verbose=True)
            ffmpeg_success = ffmpeg_pipeline.process_video_complete(
                str(test_input),
                str(ffmpeg_output),
                padding_factor=1.1
            )
            
            if ffmpeg_success and os.path.exists(ffmpeg_output):
                ffmpeg_size = os.path.getsize(ffmpeg_output) / (1024 * 1024)
                print(f"   ✅ FFmpeg pipeline successful, output: {ffmpeg_size:.1f}MB")
            else:
                print(f"   ⚠️ FFmpeg pipeline failed (expected if FFmpeg modules unavailable)")
                
        except Exception as e:
            print(f"   ⚠️ FFmpeg pipeline error: {e} (expected if FFmpeg modules unavailable)")
        
        # Test convenience function
        print("4️⃣ Testing convenience function...")
        try:
            convenience_output = temp_dir / "convenience_output.mp4"
            convenience_success = process_video_with_ai_cameraman(
                str(test_input),
                str(convenience_output),
                processor_type="opencv"
            )
            
            if convenience_success and os.path.exists(convenience_output):
                convenience_size = os.path.getsize(convenience_output) / (1024 * 1024)
                print(f"   ✅ Convenience function successful, output: {convenience_size:.1f}MB")
            else:
                print(f"   ❌ Convenience function failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Convenience function error: {e}")
            return False
        
        return True


def test_storage_analysis():
    """Test storage savings estimation"""
    print(f"\n{'='*60}")
    print(f"💾 TESTING STORAGE ANALYSIS")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_input = temp_dir / "analysis_input.mp4"
        
        # Create test video
        print("1️⃣ Creating test video for analysis...")
        if not create_test_video(str(test_input), duration=10):
            print("❌ Failed to create test video")
            return False
        
        try:
            pipeline = AICameramanPipeline(processor_type="opencv", verbose=False)
            
            # Test storage savings estimation
            print("2️⃣ Analyzing storage savings potential...")
            savings_analysis = pipeline.get_storage_savings_estimate(str(test_input))
            
            if savings_analysis:
                print(f"📊 Storage Analysis Results:")
                print(f"   Input size: {savings_analysis.get('input_size_mb', 0):.1f}MB")
                print(f"   Estimated output: {savings_analysis.get('estimated_output_size_mb', 0):.1f}MB")
                print(f"   Estimated savings: {savings_analysis.get('estimated_size_reduction_percent', 0):.1f}%")
                print(f"   Average crop ratio: {savings_analysis.get('average_crop_ratio', 0):.2f}")
                print(f"   Coordinates analyzed: {savings_analysis.get('coordinates_analyzed', 0)}")
                return True
            else:
                print("❌ Storage analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Storage analysis error: {e}")
            return False


def test_benchmark_comparison():
    """Test benchmark comparison functionality"""
    print(f"\n{'='*60}")
    print(f"⚖️ TESTING BENCHMARK COMPARISON")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_input = temp_dir / "benchmark_input.mp4"
        
        # Create test video
        print("1️⃣ Creating test video for benchmarking...")
        if not create_test_video(str(test_input), duration=6):
            print("❌ Failed to create test video")
            return False
        
        try:
            pipeline = AICameramanPipeline(processor_type="opencv", verbose=False)
            
            print("2️⃣ Running benchmark comparison...")
            results = pipeline.benchmark_processors(
                str(test_input),
                str(temp_dir / "benchmark_results")
            )
            
            if results:
                print("✅ Benchmark comparison completed")
                return True
            else:
                print("⚠️ Benchmark comparison completed with some failures")
                return False
                
        except Exception as e:
            print(f"❌ Benchmark comparison error: {e}")
            return False


def test_mock_coordinate_generation():
    """Test mock coordinate generation for testing"""
    print(f"\n{'='*60}")
    print(f"🎯 TESTING MOCK COORDINATE GENERATION")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_input = temp_dir / "coord_test_input.mp4"
        
        # Create test video
        print("1️⃣ Creating test video...")
        if not create_test_video(str(test_input), duration=5):
            print("❌ Failed to create test video")
            return False
        
        try:
            pipeline = AICameramanPipeline(processor_type="opencv", verbose=True)
            
            print("2️⃣ Testing mock coordinate generation...")
            mock_coords = pipeline._create_mock_coordinates(str(test_input))
            
            if mock_coords:
                print(f"   📊 Generated {len(mock_coords)} mock coordinates")
                print(f"   📝 Sample coordinate: {mock_coords[0]}")
                
                # Validate coordinate structure
                required_keys = ['timestamp', 'x', 'y', 'width', 'height']
                for coord in mock_coords[:3]:  # Check first few
                    for key in required_keys:
                        if key not in coord:
                            print(f"❌ Missing key '{key}' in coordinate")
                            return False
                
                print("   ✅ All coordinates have required structure")
                return True
            else:
                print("❌ No mock coordinates generated")
                return False
                
        except Exception as e:
            print(f"❌ Mock coordinate generation error: {e}")
            return False


def main():
    """Main test function"""
    print(f"🚀 AI Cameraman Pipeline Integration Test Suite")
    print(f"{'='*60}")
    
    # Run all tests
    tests = [
        ("Pipeline Integration", test_pipeline_integration),
        ("Storage Analysis", test_storage_analysis),
        ("Benchmark Comparison", test_benchmark_comparison),
        ("Mock Coordinate Generation", test_mock_coordinate_generation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 All integration tests passed! The pipeline is ready for production.")
        return True
    else:
        print(f"\n⚠️ Some integration tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 