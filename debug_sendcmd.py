#!/usr/bin/env python3
"""
Debug script for sendcmd generation and dynamic cropping issues.
Run this to test your sendcmd generation before full video rendering.
"""

import os
import sys
import pandas as pd
import subprocess
import tempfile

# Add pipelines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipelines'))

from render_video import generate_sendcmd_filter, test_sendcmd_generation, get_video_info


def create_test_coordinates(num_frames=100):
    """Create test coordinates for debugging"""
    import numpy as np
    
    # Create smooth test data
    t_values = np.linspace(0, 3000, num_frames)  # 3 seconds at various frame rates
    
    # Simulate a simple pan and zoom
    base_x = 100 + 50 * np.sin(t_values / 1000)  # Slow pan
    base_y = 50 + 30 * np.cos(t_values / 1000)   # Slight vertical movement
    base_w = 1000 + 100 * np.sin(t_values / 500) # Zoom in/out
    base_h = 600 + 60 * np.sin(t_values / 500)   # Maintain aspect ratio
    
    coords_df = pd.DataFrame({
        't_ms': t_values,
        'frame_number': range(num_frames),
        'crop_x': base_x.astype(int),
        'crop_y': base_y.astype(int), 
        'crop_w': base_w.astype(int),
        'crop_h': base_h.astype(int)
    })
    
    return coords_df


def test_ffmpeg_sendcmd():
    """Test FFmpeg sendcmd filter with a simple video"""
    
    print("🎬 FFmpeg sendcmd filter test")
    print("=" * 50)
    
    # Create test coordinates
    coords = create_test_coordinates(50)  # Small test
    print(f"📊 Created {len(coords)} test coordinates")
    
    # Test sendcmd generation
    success = test_sendcmd_generation(coords, "test_output.mp4")
    
    if not success:
        print("❌ Sendcmd generation failed")
        return False
    
    # Generate sendcmd content
    sendcmd_content = generate_sendcmd_filter(coords)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sendcmd.txt', delete=False) as f:
        f.write(sendcmd_content)
        sendcmd_file = f.name
    
    print(f"📄 Sendcmd file: {sendcmd_file}")
    
    # Test if FFmpeg can parse the sendcmd file
    test_cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'testsrc=duration=3:size=1920x1080:rate=30',
        '-vf', f'sendcmd=f={sendcmd_file},crop=w=iw:h=ih:x=0:y=0',
        '-t', '1',  # Only 1 second for test
        '-f', 'null',
        '-'
    ]
    
    print("🧪 Testing FFmpeg sendcmd parsing...")
    print(f"Command: {' '.join(test_cmd)}")
    
    try:
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ FFmpeg successfully parsed sendcmd file")
            # Clean up
            os.unlink(sendcmd_file)
            return True
        else:
            print("❌ FFmpeg failed to parse sendcmd file")
            print("STDERR:", result.stderr)
            print(f"🔧 Sendcmd file kept for inspection: {sendcmd_file}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg test timed out")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found - please install FFmpeg")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def analyze_existing_coordinates(csv_path):
    """Analyze existing coordinate data"""
    
    if not os.path.exists(csv_path):
        print(f"❌ Coordinate file not found: {csv_path}")
        return False
    
    print(f"📊 Analyzing coordinates: {csv_path}")
    
    try:
        coords_df = pd.read_csv(csv_path)
        
        print(f"📈 Data shape: {coords_df.shape}")
        print(f"📈 Columns: {list(coords_df.columns)}")
        
        # Check required columns
        required_cols = ['t_ms', 'crop_x', 'crop_y', 'crop_w', 'crop_h']
        missing_cols = [col for col in required_cols if col not in coords_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Show data ranges
        print(f"📈 Time range: {coords_df['t_ms'].min():.1f} - {coords_df['t_ms'].max():.1f} ms")
        print(f"📈 X range: {coords_df['crop_x'].min()} - {coords_df['crop_x'].max()}")
        print(f"📈 Y range: {coords_df['crop_y'].min()} - {coords_df['crop_y'].max()}")
        print(f"📈 Width range: {coords_df['crop_w'].min()} - {coords_df['crop_w'].max()}")
        print(f"📈 Height range: {coords_df['crop_h'].min()} - {coords_df['crop_h'].max()}")
        
        # Test sendcmd generation
        success = test_sendcmd_generation(coords_df, csv_path.replace('.csv', '_test.mp4'))
        
        return success
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def main():
    """Main debug routine"""
    
    print("🔧 SENDCMD DEBUG UTILITY")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Analyze existing coordinate file
        csv_file = sys.argv[1]
        success = analyze_existing_coordinates(csv_file)
    else:
        # Run basic test
        success = test_ffmpeg_sendcmd()
    
    if success:
        print("\n✅ All tests passed!")
        print("💡 Your sendcmd generation should work for dynamic cropping")
    else:
        print("\n❌ Tests failed!")
        print("💡 Check the error messages above to fix sendcmd issues")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 