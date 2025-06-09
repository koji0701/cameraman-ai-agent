#!/usr/bin/env python3
"""
Desktop Launcher for AI Cameraman
=================================

This script serves as the entry point for the AI Cameraman desktop application.
It can either launch the Python backend bridge for Electron or provide a CLI interface.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "python-backend"))

from video_processor import VideoCameramanBridge
from file_manager import FileManager


def launch_python_bridge():
    """Launch the Python-shell bridge for Electron communication"""
    print("🚀 Starting AI Cameraman Python Bridge")
    print("   Waiting for commands from Electron...")
    print("   Send JSON commands via stdin, receive responses via stdout")
    
    bridge = VideoCameramanBridge()
    bridge.run()


def launch_electron_gui():
    """Launch the Electron desktop GUI (when implemented)"""
    desktop_gui_path = project_root / "desktop-gui"
    
    if not desktop_gui_path.exists():
        print("❌ Desktop GUI not yet implemented")
        print("   The Electron GUI will be created in Phase 2")
        print("   For now, you can use the CLI interface or Python bridge")
        return False
    
    try:
        # Try to start the Electron app
        subprocess.run(["npm", "start"], cwd=desktop_gui_path, check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Electron GUI")
        print("   Make sure to run 'npm install' in the desktop-gui directory")
        return False
    except FileNotFoundError:
        print("❌ Node.js/npm not found")
        print("   Please install Node.js to run the desktop GUI")
        return False


def run_cli_processing(input_video: str, output_video: str, **options):
    """Run video processing via CLI interface"""
    print("🎬 AI Cameraman CLI Processing")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_video}")
    
    # Validate input file
    file_manager = FileManager()
    if not file_manager.validate_video_file(input_video):
        print(f"❌ Invalid video file: {input_video}")
        return False
    
    # Import and use the existing pipeline
    from src.video_processing.pipeline_integration import AICameramanPipeline
    
    pipeline = AICameramanPipeline(processor_type="opencv", verbose=True)
    
    # Create progress callback for CLI
    def progress_callback(percentage: float, stage: str, details: str = ""):
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        print(f"   [{progress_bar}] {percentage:5.1f}% - {stage}: {details}")
    
    try:
        success = pipeline.process_video_complete(
            input_video_path=input_video,
            output_video_path=output_video,
            progress_callback=progress_callback,
            **options
        )
        
        if success:
            # Show file statistics
            stats = file_manager.get_file_stats(input_video, output_video)
            print("\n✅ Processing completed successfully!")
            print(f"   Input size: {stats['input_size_mb']} MB")
            print(f"   Output size: {stats['output_size_mb']} MB")
            print(f"   Compression: {stats['compression_ratio']}%")
            print(f"   Space saved: {stats['space_saved_mb']} MB")
        else:
            print("\n❌ Processing failed!")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Processing error: {e}")
        return False


def show_status():
    """Show the current status of the AI Cameraman system"""
    print("🔍 AI Cameraman System Status")
    print("=" * 40)
    
    # Check Python dependencies
    print("\n📦 Python Dependencies:")
    dependencies = [
        ('opencv-cv2', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
    ]
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {dep_name}")
        except ImportError:
            print(f"   ❌ {dep_name} (missing)")
    
    # Check pipeline components
    print("\n🧩 Pipeline Components:")
    pipeline_components = [
        ('Gemini API Client', 'pipelines.genai_client'),
        ('Kalman Smoother', 'pipelines.kalman_smoother'),
        ('OpenCV Processor', 'src.video_processing.opencv_processor'),
        ('Pipeline Integration', 'src.video_processing.pipeline_integration'),
    ]
    
    for comp_name, import_path in pipeline_components:
        try:
            __import__(import_path)
            print(f"   ✅ {comp_name}")
        except ImportError:
            print(f"   ❌ {comp_name} (missing)")
    
    # Check external tools
    print("\n🛠️  External Tools:")
    external_tools = ['ffmpeg', 'ffprobe']
    
    for tool in external_tools:
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {tool}")
            else:
                print(f"   ❌ {tool} (not working)")
        except FileNotFoundError:
            print(f"   ❌ {tool} (not installed)")
    
    # Check desktop GUI status
    print("\n🖥️  Desktop GUI:")
    desktop_gui_path = project_root / "desktop-gui"
    if desktop_gui_path.exists():
        print(f"   ✅ Desktop GUI directory exists")
        package_json = desktop_gui_path / "package.json"
        if package_json.exists():
            print(f"   ✅ package.json found")
        else:
            print(f"   ⚠️ package.json missing")
    else:
        print(f"   ⚠️ Desktop GUI not yet implemented (Phase 2)")
    
    print("\n" + "=" * 40)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Cameraman Desktop Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gui                     # Launch desktop GUI
  %(prog)s bridge                  # Start Python bridge for Electron
  %(prog)s cli input.mp4 output.mp4  # Process video via CLI
  %(prog)s status                  # Show system status
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['gui', 'bridge', 'cli', 'status'],
        help='Launch mode'
    )
    
    parser.add_argument(
        'input_video',
        nargs='?',
        help='Input video file (for cli mode)'
    )
    
    parser.add_argument(
        'output_video', 
        nargs='?',
        help='Output video file (for cli mode)'
    )
    
    parser.add_argument(
        '--padding',
        type=float,
        default=1.1,
        help='Padding factor around action area (default: 1.1)'
    )
    
    parser.add_argument(
        '--smoothing',
        choices=['light', 'balanced', 'heavy'],
        default='balanced',
        help='Coordinate smoothing strength (default: balanced)'
    )
    
    parser.add_argument(
        '--processor',
        choices=['opencv', 'ffmpeg'],
        default='opencv',
        help='Video processor type (default: opencv)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        success = launch_electron_gui()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'bridge':
        launch_python_bridge()
    
    elif args.mode == 'status':
        show_status()
    
    elif args.mode == 'cli':
        if not args.input_video or not args.output_video:
            print("❌ CLI mode requires input and output video paths")
            parser.print_help()
            sys.exit(1)
        
        success = run_cli_processing(
            args.input_video,
            args.output_video,
            padding_factor=args.padding,
            smoothing_strength=args.smoothing,
            processor_type=args.processor
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 