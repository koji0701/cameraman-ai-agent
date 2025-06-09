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
    """Launch the Electron desktop GUI"""
    desktop_gui_path = project_root / "desktop-gui"
    
    if not desktop_gui_path.exists():
        print("❌ Desktop GUI directory not found")
        print("   Expected directory: desktop-gui")
        return False
    
    # Check if package.json exists
    package_json = desktop_gui_path / "package.json"
    if not package_json.exists():
        print("❌ Desktop GUI not properly set up (missing package.json)")
        print("   Run 'npm install' in the desktop-gui directory")
        return False
    
    try:
        print("🚀 Starting AI Cameraman Desktop GUI...")
        print("   This will open the Electron application")
        
        # Try to start the Electron app
        subprocess.run(["npm", "start"], cwd=desktop_gui_path, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed to start Electron GUI")
        print(f"   Error: {e}")
        print("   Make sure to run 'npm install' in the desktop-gui directory")
        return False
    except FileNotFoundError:
        print("❌ Node.js/npm not found")
        print("   Please install Node.js to run the desktop GUI")
        print("   Visit: https://nodejs.org/")
        return False


def run_cli_processing(input_video: str, output_video: str, **options):
    """Run video processing via CLI interface"""
    print("🎬 AI Cameraman Enhanced CLI Processing")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_video}")
    
    # Display processing options
    print(f"   Quality: {options.get('quality_preset', 'medium')}")
    print(f"   Codec: {options.get('video_codec', 'h264_videotoolbox')}")
    print(f"   Streaming: {'Disabled' if not options.get('use_streaming', True) else 'Enabled'}")
    print(f"   Bitrate: {options.get('bitrate', '15M')}")
    print(f"   Stabilization: {'Enabled' if options.get('enable_stabilization', False) else 'Disabled'}")
    
    # Validate input file
    file_manager = FileManager()
    if not file_manager.validate_video_file(input_video):
        print(f"❌ Invalid video file: {input_video}")
        return False
    
    # Check if we should use the direct pipeline (for full compatibility)
    processor_type = options.get('processor_type', 'opencv')
    use_direct_pipeline = True  # Use direct pipeline for full feature support
    
    if use_direct_pipeline:
        print("   Using direct pipeline for maximum compatibility...")
        return run_direct_pipeline_processing(input_video, output_video, **options)
    else:
        # Use integrated pipeline (for future desktop GUI compatibility)
        from src.video_processing.pipeline_integration import AICameramanPipeline
        
        pipeline = AICameramanPipeline(processor_type=processor_type, verbose=True)
        
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
                file_manager = FileManager()
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


def run_direct_pipeline_processing(input_video: str, output_video: str, **options):
    """Run video processing using the direct pipeline (like your current workflow)"""
    
    # Import the direct pipeline functions
    sys.path.append(str(project_root / "pipelines"))
    
    try:
        from render_video import process_and_render_complete
        
        # Create progress callback for CLI
        def progress_callback(percentage: float, stage: str, details: str = ""):
            progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
            print(f"   [{progress_bar}] {percentage:5.1f}% - {stage}: {details}")
        
        # Map CLI options to render_video parameters
        render_kwargs = {
            'quality_preset': options.get('quality_preset', 'medium'),
            'video_codec': options.get('video_codec', 'h264_videotoolbox'),
            'bitrate': options.get('bitrate', '15M'),
            'enable_stabilization': options.get('enable_stabilization', False),
            'color_correction': options.get('color_correction', False),
            'use_streaming': options.get('use_streaming', True),
            'save_intermediate_files': True,
            'verbose': True
        }
        
        print(f"   🚀 Starting direct pipeline processing...")
        print(f"   📋 Processing options: {render_kwargs}")
        
        # Call the direct pipeline function
        success = process_and_render_complete(
            input_video_path=input_video,
            output_video_path=output_video,
            padding_factor=options.get('padding_factor', 1.1),
            smoothing_strength=options.get('smoothing_strength', 'balanced'),
            interpolation_method=options.get('interpolation_method', 'cubic'),
            **render_kwargs
        )
        
        if success:
            # Show file statistics
            file_manager = FileManager()
            stats = file_manager.get_file_stats(input_video, output_video)
            print("\n✅ Direct pipeline processing completed successfully!")
            print(f"   Input size: {stats['input_size_mb']} MB")
            print(f"   Output size: {stats['output_size_mb']} MB") 
            print(f"   Compression: {stats['compression_ratio']}%")
            print(f"   Space saved: {stats['space_saved_mb']} MB")
        else:
            print("\n❌ Direct pipeline processing failed!")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Direct pipeline processing error: {e}")
        print("   This may indicate missing dependencies or pipeline issues")
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
            node_modules = desktop_gui_path / "node_modules"
            if node_modules.exists():
                print(f"   ✅ Dependencies installed")
            else:
                print(f"   ⚠️ Dependencies not installed (run 'npm install')")
            print(f"   ✅ AI Cameraman GUI ready to launch")
        else:
            print(f"   ❌ package.json missing")
    else:
        print(f"   ❌ Desktop GUI directory not found")
    
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
  %(prog)s cli input.mp4 output.mp4  # Basic video processing
  %(prog)s cli input.mp4 output.mp4 --quality medium --codec h264_videotoolbox --disable-streaming
  %(prog)s status                  # Show system status

Enhanced CLI Options:
  --quality {low,medium,high,ultra}    Video quality preset
  --codec CODEC                        Video codec (h264_videotoolbox, libx264, etc.)
  --disable-streaming                  Use file-based processing instead of streaming
  --bitrate RATE                       Video bitrate (e.g., 15M, 20M)
  --enable-stabilization               Enable video stabilization
  --color-correction                   Enable color correction
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
    
    # Video encoding options (matching existing pipeline)
    parser.add_argument(
        '--quality',
        choices=['low', 'medium', 'high', 'ultra'],
        default='medium',
        help='Video quality preset (default: medium)'
    )
    
    parser.add_argument(
        '--codec',
        default='h264_videotoolbox',
        help='Video codec (default: h264_videotoolbox)'
    )
    
    parser.add_argument(
        '--disable-streaming',
        action='store_true',
        help='Disable streaming mode (use file-based processing)'
    )
    
    parser.add_argument(
        '--bitrate',
        default='15M',
        help='Video bitrate (default: 15M)'
    )
    
    parser.add_argument(
        '--enable-stabilization',
        action='store_true',
        help='Enable video stabilization'
    )
    
    parser.add_argument(
        '--color-correction',
        action='store_true',
        help='Enable color correction'
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
            processor_type=args.processor,
            # Video encoding options
            quality_preset=args.quality,
            video_codec=args.codec,
            use_streaming=not args.disable_streaming,
            bitrate=args.bitrate,
            enable_stabilization=args.enable_stabilization,
            color_correction=args.color_correction
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 