"""
Benchmarking tool for FFmpeg vs OpenCV performance comparison
"""

import time
import os
import json
import psutil
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import tempfile
import shutil

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    processor_type: str
    video_path: str
    processing_time_seconds: float
    input_file_size_mb: float
    output_file_size_mb: float
    compression_ratio: float
    peak_memory_usage_mb: float
    storage_efficiency: float
    temp_files_created: int
    temp_storage_used_mb: float
    success: bool
    error_message: str = ""


class VideoProcessorBenchmark:
    """Benchmark suite for video processors"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="videobench_")
        self.results = []
        
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def monitor_system_resources(self) -> Dict:
        """Monitor system resource usage"""
        process = psutil.Process()
        return {
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent()
        }
    
    def count_temp_files(self, directory: str) -> tuple:
        """Count temporary files and their total size"""
        temp_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(ext in file.lower() for ext in ['.tmp', '.temp', '.cache']):
                    temp_files += 1
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass
        
        return temp_files, total_size / (1024 * 1024)  # Size in MB
    
    def benchmark_processor(self, processor, gemini_coordinates: List[Dict], 
                          test_name: str) -> BenchmarkResult:
        """Benchmark a video processor"""
        
        print(f"🔍 Benchmarking {test_name}...")
        
        # Get initial system state
        initial_resources = self.monitor_system_resources()
        initial_temp_files, initial_temp_size = self.count_temp_files(self.temp_dir)
        
        # Get input file size
        input_size = os.path.getsize(processor.input_video_path) / (1024 * 1024)
        
        # Start timing
        start_time = time.time()
        peak_memory = initial_resources['memory_mb']
        success = False
        error_message = ""
        
        try:
            # Monitor resources during processing
            resource_monitor = []
            
            # Process video
            success = processor.process_with_gemini_coords(gemini_coordinates)
            
            # Monitor peak memory usage (simplified - in real scenario would need threading)
            current_resources = self.monitor_system_resources()
            peak_memory = max(peak_memory, current_resources['memory_mb'])
            
        except Exception as e:
            error_message = str(e)
            success = False
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get final measurements
        output_size = 0
        if os.path.exists(processor.output_video_path):
            output_size = os.path.getsize(processor.output_video_path) / (1024 * 1024)
        
        final_temp_files, final_temp_size = self.count_temp_files(self.temp_dir)
        
        # Calculate metrics
        compression_ratio = input_size / max(output_size, 0.001)
        storage_efficiency = (input_size - output_size) / input_size * 100 if input_size > 0 else 0
        temp_files_created = final_temp_files - initial_temp_files
        temp_storage_used = final_temp_size - initial_temp_size
        
        result = BenchmarkResult(
            processor_type=test_name,
            video_path=processor.input_video_path,
            processing_time_seconds=processing_time,
            input_file_size_mb=input_size,
            output_file_size_mb=output_size,
            compression_ratio=compression_ratio,
            peak_memory_usage_mb=peak_memory - initial_resources['memory_mb'],
            storage_efficiency=storage_efficiency,
            temp_files_created=temp_files_created,
            temp_storage_used_mb=temp_storage_used,
            success=success,
            error_message=error_message
        )
        
        self.results.append(result)
        return result
    
    def run_comparison_benchmark(self, video_path: str, gemini_coordinates: List[Dict]):
        """Run benchmark comparing FFmpeg and OpenCV processors"""
        
        print("🚀 Starting video processor benchmark comparison...")
        
        # Test FFmpeg processor
        try:
            from .ffmpeg_processor import FFmpegProcessor
            
            ffmpeg_output = Path(self.temp_dir) / "ffmpeg_output.mp4"
            ffmpeg_processor = FFmpegProcessor(video_path, str(ffmpeg_output))
            
            ffmpeg_result = self.benchmark_processor(
                ffmpeg_processor, gemini_coordinates, "FFmpeg"
            )
            
            print(f"✅ FFmpeg benchmark completed:")
            print(f"   Time: {ffmpeg_result.processing_time_seconds:.2f}s")
            print(f"   Storage: {ffmpeg_result.storage_efficiency:.1f}% reduction")
            print(f"   Memory: {ffmpeg_result.peak_memory_usage_mb:.1f}MB peak")
            
        except Exception as e:
            print(f"⚠️ FFmpeg benchmark failed: {e}")
            ffmpeg_result = None
        
        # Test OpenCV processor
        try:
            from .opencv_processor import OpenCVCameraman
            
            opencv_output = Path(self.temp_dir) / "opencv_output.mp4"
            opencv_processor = OpenCVCameraman(video_path, str(opencv_output))
            
            opencv_result = self.benchmark_processor(
                opencv_processor, gemini_coordinates, "OpenCV"
            )
            
            print(f"✅ OpenCV benchmark completed:")
            print(f"   Time: {opencv_result.processing_time_seconds:.2f}s")
            print(f"   Storage: {opencv_result.storage_efficiency:.1f}% reduction")
            print(f"   Memory: {opencv_result.peak_memory_usage_mb:.1f}MB peak")
            
        except Exception as e:
            print(f"⚠️ OpenCV benchmark failed: {e}")
            opencv_result = None
        
        return {
            'ffmpeg': ffmpeg_result,
            'opencv': opencv_result
        }
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        results_file = self.output_dir / filename
        
        results_data = {
            'timestamp': time.time(),
            'results': [asdict(result) for result in self.results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📊 Benchmark results saved to: {results_file}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"\n{result.processor_type}:")
            print(f"  ⏱️  Processing time: {result.processing_time_seconds:.2f}s")
            print(f"  💾 Input size: {result.input_file_size_mb:.1f}MB")
            print(f"  📤 Output size: {result.output_file_size_mb:.1f}MB")
            print(f"  🗜️  Compression: {result.compression_ratio:.2f}x")
            print(f"  💡 Storage efficiency: {result.storage_efficiency:.1f}%")
            print(f"  🧠 Peak memory: {result.peak_memory_usage_mb:.1f}MB")
            print(f"  📁 Temp files: {result.temp_files_created}")
            print(f"  💿 Temp storage: {result.temp_storage_used_mb:.1f}MB")
            print(f"  ✅ Success: {result.success}")
            if result.error_message:
                print(f"  ❌ Error: {result.error_message}")


def create_test_coordinates() -> List[Dict]:
    """Create test coordinates for benchmarking"""
    # Simulate water polo action coordinates
    return [
        {"timestamp": 0.0, "x": 100, "y": 200, "width": 800, "height": 600},
        {"timestamp": 1.0, "x": 150, "y": 180, "width": 750, "height": 580},
        {"timestamp": 2.0, "x": 200, "y": 160, "width": 700, "height": 560},
        {"timestamp": 3.0, "x": 250, "y": 140, "width": 680, "height": 540},
        {"timestamp": 4.0, "x": 300, "y": 120, "width": 660, "height": 520},
    ]


if __name__ == "__main__":
    # Example usage
    benchmark = VideoProcessorBenchmark()
    
    # You would run this with an actual video file
    # test_video = "path/to/test/video.mp4"
    # test_coords = create_test_coordinates()
    # results = benchmark.run_comparison_benchmark(test_video, test_coords)
    # benchmark.save_results()
    # benchmark.print_summary()
    
    print("📋 Benchmark tool ready. Run with actual video file to compare processors.") 