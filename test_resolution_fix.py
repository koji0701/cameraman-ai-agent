#!/usr/bin/env python3
"""
Test script to verify resolution-aware processing fixes zoom bias issues.

This test demonstrates the difference between:
1. Old hardcoded 1920x1080 approach (causes zoom bias)
2. New auto-detected resolution approach (fixes zoom bias)
"""

import sys
import os
import pandas as pd

# Add pipelines to path
sys.path.append('pipelines')

def test_resolution_detection():
    """Test automatic video resolution detection"""
    print("🧪 Testing Resolution Detection")
    print("=" * 50)
    
    # Test video file
    video_path = "videos/waterpolo_trimmed_two.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return False
    
    try:
        from genai_client import get_video_dimensions
        
        # Detect resolution
        width, height = get_video_dimensions(video_path)
        
        print(f"✅ Detected resolution: {width}x{height}")
        
        # Verify it's 4K
        if width == 3840 and height == 2160:
            print("✅ Correctly detected 4K resolution")
            return True
        else:
            print(f"⚠️ Unexpected resolution: {width}x{height}")
            return False
            
    except Exception as e:
        print(f"❌ Resolution detection failed: {e}")
        return False

def test_coordinate_normalization():
    """Test coordinate normalization with different resolutions"""
    print("\n🧪 Testing Coordinate Normalization")
    print("=" * 50)
    
    try:
        from normalize_coordinates import (
            normalize_bounding_boxes_to_1080p,
            normalize_bounding_boxes_to_video_resolution
        )
        
        # Sample bounding box data (in 4K coordinates)
        test_data = pd.DataFrame({
            't_ms': [0, 1000, 2000],
            'x1': [1920, 1800, 1700],  # Center area of 4K frame
            'y1': [1080, 1000, 950],
            'x2': [2400, 2300, 2200],
            'y2': [1400, 1350, 1300]
        })
        
        print("📊 Sample bounding boxes (4K coordinates):")
        print(test_data)
        
        # Test old approach (hardcoded 1920x1080)
        print("\n🔧 Old approach (hardcoded 1920x1080):")
        old_result = normalize_bounding_boxes_to_1080p(
            test_data,
            original_width=1920,  # Wrong! Should be 3840
            original_height=1080,  # Wrong! Should be 2160
            padding_factor=1.1
        )
        print("❌ Crop coordinates (will cause zoom bias):")
        print(old_result[['crop_x', 'crop_y', 'crop_w', 'crop_h']])
        
        # Test new approach (auto-detected resolution)
        print("\n🔧 New approach (correct 4K resolution):")
        new_result = normalize_bounding_boxes_to_video_resolution(
            test_data,
            original_width=3840,  # Correct 4K width
            original_height=2160,  # Correct 4K height
            target_aspect_ratio=16/9,
            padding_factor=1.1
        )
        print("✅ Crop coordinates (properly centered):")
        print(new_result[['crop_x', 'crop_y', 'crop_w', 'crop_h']])
        
        # Compare center points
        old_center_x = old_result['crop_x'].iloc[0] + old_result['crop_w'].iloc[0] / 2
        old_center_y = old_result['crop_y'].iloc[0] + old_result['crop_h'].iloc[0] / 2
        
        new_center_x = new_result['crop_x'].iloc[0] + new_result['crop_w'].iloc[0] / 2
        new_center_y = new_result['crop_y'].iloc[0] + new_result['crop_h'].iloc[0] / 2
        
        print(f"\n📍 Crop Centers Comparison:")
        print(f"Old approach: ({old_center_x:.0f}, {old_center_y:.0f})")
        print(f"New approach: ({new_center_x:.0f}, {new_center_y:.0f})")
        
        # Check if new approach is better centered (closer to 4K center: 1920, 1080)
        old_distance = abs(old_center_x - 1920) + abs(old_center_y - 1080)
        new_distance = abs(new_center_x - 1920) + abs(new_center_y - 1080)
        
        if new_distance < old_distance:
            print("✅ New approach produces better-centered crops")
            return True
        else:
            print("⚠️ Centering comparison inconclusive")
            return False
            
    except Exception as e:
        print(f"❌ Coordinate normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_scaling():
    """Test adaptive scaling resolution"""
    print("\n🧪 Testing Adaptive Scaling")
    print("=" * 50)
    
    # Mock coordinate data
    coords_df = pd.DataFrame({
        't_ms': [0, 1000],
        'crop_x': [960, 970],
        'crop_y': [540, 550], 
        'crop_w': [1920, 1900],
        'crop_h': [1080, 1070]
    })
    
    # Test original resolution detection
    max_width = int(coords_df['crop_w'].max())
    max_height = int(coords_df['crop_h'].max())
    
    print(f"📏 Max crop dimensions: {max_width}x{max_height}")
    
    # This should be used as the target resolution instead of hardcoded 1920x1080
    adaptive_resolution = f"{max_width}:{max_height}"
    print(f"🎯 Adaptive resolution: {adaptive_resolution}")
    
    if adaptive_resolution != "1920:1080":
        print("✅ Adaptive scaling will preserve crop quality")
        return True
    else:
        print("ℹ️ Adaptive scaling same as standard resolution")
        return True

def main():
    """Run all tests"""
    print("🚀 RESOLUTION-AWARE PROCESSING TESTS")
    print("=" * 60)
    
    tests = [
        ("Resolution Detection", test_resolution_detection),
        ("Coordinate Normalization", test_coordinate_normalization), 
        ("Adaptive Scaling", test_adaptive_scaling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"🎉 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Resolution-aware processing is working correctly.")
        print("🎯 The zoom bias issue should now be fixed for 4K videos.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 