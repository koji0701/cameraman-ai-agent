#!/usr/bin/env python3
"""
Comprehensive test runner for FFmpeg dynamic cropping implementation.

This script runs all test suites for the dynamic rendering mode:
- Dynamic (frame-by-frame cropping)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --performance-only
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path

# Add pipelines to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

# Import all test modules
try:
    from test_ffmpeg_render import TestFFmpegRenderModes, TestFFmpegIntegration, TestAdvancedFeatures
    from test_sendcmd_generation import TestSendcmdGeneration
    from test_performance import TestFFmpegPerformance
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class TestRunner:
    """Comprehensive test runner with detailed reporting"""
    
    def __init__(self, verbosity: int = 2, performance_only: bool = False):
        self.verbosity = verbosity
        self.performance_only = performance_only
        self.results = {}
        
    def run_test_suite(self, test_class, suite_name: str) -> bool:
        """Run a single test suite and collect results"""
        print(f"\n{'='*60}")
        print(f"Running: {suite_name}")
        print(f"{'='*60}")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests with custom runner
        stream = open(os.devnull, 'w') if self.verbosity < 2 else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=self.verbosity,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Close stream if we opened it
        if stream != sys.stdout:
            stream.close()
        
        # Collect results
        self.results[suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100,
            'duration': end_time - start_time,
            'successful': result.wasSuccessful()
        }
        
        # Print summary
        print(f"\nResults: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        print(f"Success rate: {self.results[suite_name]['success_rate']:.1f}%")
        
        return result.wasSuccessful()
    
    def run_all_tests(self):
        """Run all test suites"""
        total_start = time.time()
        all_successful = True
        
        print(f"FFmpeg Dynamic Cropping Test Suite")
        print(f"Testing frame-by-frame dynamic cropping implementation")
        if self.performance_only:
            print("Running performance tests only")
        print(f"Verbosity level: {self.verbosity}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define test suites to run
        test_suites = []
        
        if not self.performance_only:
            test_suites.extend([
                (TestFFmpegRenderModes, "Dynamic Rendering Tests"),
                (TestFFmpegIntegration, "FFmpeg Integration Tests"),
                (TestAdvancedFeatures, "Advanced Features"),
                (TestSendcmdGeneration, "Sendcmd Filter Generation")
            ])
        
        test_suites.append((TestFFmpegPerformance, "Performance Tests"))
        
        # Run each test suite
        for test_class, suite_name in test_suites:
            success = self.run_test_suite(test_class, suite_name)
            if not success:
                all_successful = False
        
        total_end = time.time()
        
        # Print final summary
        self.print_final_summary(total_end - total_start, all_successful)
        
        return all_successful
    
    def print_final_summary(self, total_duration: float, all_successful: bool):
        """Print comprehensive test summary"""
        print(f"\n{'='*80}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        
        overall_success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1) * 100
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful: {total_tests - total_failures - total_errors}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f} seconds")
        
        print(f"\nPer-Suite Results:")
        print(f"{'Suite':<30} {'Tests':<8} {'Success Rate':<12} {'Duration':<10}")
        print(f"{'-'*65}")
        
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result['successful'] else "❌ FAIL"
            print(f"{suite_name:<30} {result['tests_run']:<8} {result['success_rate']:>8.1f}% {result['duration']:>8.2f}s {status}")
        
        print(f"\n{'='*80}")
        if all_successful:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print(f"{'='*80}")
    
    def print_mode_coverage(self):
        """Print coverage information for dynamic rendering mode"""
        print(f"\n{'='*80}")
        print("DYNAMIC CROPPING COVERAGE ANALYSIS")
        print(f"{'='*80}")
        
        mode_tests = {
            'Dynamic Mode': 0,
            'Sendcmd Generation': 0,
            'Performance': 0,
            'Integration': 0
        }
        
        # Count tests by category
        for suite_name, result in self.results.items():
            if 'Sendcmd' in suite_name:
                mode_tests['Sendcmd Generation'] += result['tests_run']
            elif 'Performance' in suite_name:
                mode_tests['Performance'] += result['tests_run']
            elif 'Dynamic' in suite_name or 'Rendering' in suite_name:
                mode_tests['Dynamic Mode'] += result['tests_run']
            else:
                mode_tests['Integration'] += result['tests_run']
        
        for mode, count in mode_tests.items():
            print(f"{mode:<25}: {count} tests")
        
        print(f"\nDynamic cropping implementation comprehensively tested:")
        print(f"✓ Frame-by-frame cropping with coordinate precision")
        print(f"✓ Sendcmd filter generation and validation")
        print(f"✓ Performance testing with large datasets")
        print(f"✓ Integration testing with FFmpeg pipeline")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for FFmpeg dynamic cropping implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_all_tests.py                    # Run all tests
    python tests/run_all_tests.py --verbose          # High verbosity
    python tests/run_all_tests.py --performance-only # Performance tests only
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests with high verbosity'
    )
    
    parser.add_argument(
        '--performance-only', '-p',
        action='store_true',
        help='Run only performance tests'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Run tests with minimal output'
    )
    
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 1 if args.quiet else (3 if args.verbose else 2)
    
    # Create and run test runner
    runner = TestRunner(
        verbosity=verbosity,
        performance_only=args.performance_only
    )
    
    try:
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUnexpected error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 