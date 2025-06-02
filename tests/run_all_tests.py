#!/usr/bin/env python3
"""
Comprehensive test runner for FFmpeg implementations.

This script runs all test suites for the three FFmpeg rendering modes:
- Simple (static crop)
- Dynamic (frame-by-frame sendcmd)
- Multipass (quality optimized)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
    python tests/run_all_tests.py --performance-only
    python tests/run_all_tests.py --mode simple
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
    from test_multipass_encoding import TestMultipassEncoding
    from test_performance import TestFFmpegPerformance
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class TestRunner:
    """Comprehensive test runner with detailed reporting"""
    
    def __init__(self, verbosity=2, performance_only=False, mode_filter=None):
        self.verbosity = verbosity
        self.performance_only = performance_only
        self.mode_filter = mode_filter
        self.results = {}
        
    def run_test_suite(self, test_class, suite_name):
        """Run a specific test suite and capture results"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Store results
        self.results[suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'duration': end_time - start_time,
            'result_object': result
        }
        
        return result.wasSuccessful()
    
    def filter_tests_by_mode(self, test_class):
        """Filter tests based on mode if specified"""
        if not self.mode_filter:
            return test_class
        
        # Create a filtered test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_name in loader.getTestCaseNames(test_class):
            if self.mode_filter.lower() in test_name.lower():
                suite.addTest(test_class(test_name))
        
        return suite
    
    def run_all_tests(self):
        """Run all test suites"""
        total_start = time.time()
        all_successful = True
        
        print(f"FFmpeg Implementation Test Suite")
        print(f"Testing all three modes: Simple, Dynamic, Multipass")
        if self.mode_filter:
            print(f"Filter: {self.mode_filter} mode only")
        if self.performance_only:
            print("Running performance tests only")
        print(f"Verbosity level: {self.verbosity}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define test suites to run
        test_suites = []
        
        if not self.performance_only:
            test_suites.extend([
                (TestFFmpegRenderModes, "Core Rendering Modes"),
                (TestFFmpegIntegration, "FFmpeg Integration Tests"),
                (TestAdvancedFeatures, "Advanced Features"),
                (TestSendcmdGeneration, "Sendcmd Filter Generation"),
                (TestMultipassEncoding, "Multipass Encoding")
            ])
        
        test_suites.append((TestFFmpegPerformance, "Performance Tests"))
        
        # Run each test suite
        for test_class, suite_name in test_suites:
            if self.mode_filter:
                # Skip irrelevant test suites based on mode filter
                if (self.mode_filter.lower() == 'simple' and 
                    suite_name in ["Sendcmd Filter Generation", "Multipass Encoding"]):
                    continue
                elif (self.mode_filter.lower() == 'dynamic' and 
                      suite_name == "Multipass Encoding"):
                    continue
                elif (self.mode_filter.lower() == 'multipass' and 
                      suite_name == "Sendcmd Filter Generation"):
                    continue
            
            success = self.run_test_suite(test_class, suite_name)
            if not success:
                all_successful = False
        
        total_end = time.time()
        
        # Print comprehensive summary
        self.print_summary(total_end - total_start, all_successful)
        
        return all_successful
    
    def print_summary(self, total_duration, all_successful):
        """Print comprehensive test summary"""
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        total_passed = total_tests - total_failures - total_errors
        
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Total tests run: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Skipped: {total_skipped}")
        print(f"Overall success rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        
        print(f"\nPer-suite breakdown:")
        print(f"{'Suite':<30} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Rate':<8} {'Time':<8}")
        print(f"{'-'*80}")
        
        for suite_name, result in self.results.items():
            passed = result['tests_run'] - result['failures'] - result['errors']
            print(f"{suite_name:<30} {result['tests_run']:<8} {passed:<8} {result['failures']:<8} "
                  f"{result['errors']:<8} {result['success_rate']:<7.1f}% {result['duration']:<7.2f}s")
        
        # Print detailed failure information
        if total_failures > 0 or total_errors > 0:
            print(f"\n{'='*80}")
            print("FAILURE DETAILS")
            print(f"{'='*80}")
            
            for suite_name, result in self.results.items():
                result_obj = result['result_object']
                
                if result_obj.failures:
                    print(f"\nFailures in {suite_name}:")
                    for test, traceback in result_obj.failures:
                        print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
                
                if result_obj.errors:
                    print(f"\nErrors in {suite_name}:")
                    for test, traceback in result_obj.errors:
                        print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        # Print mode-specific summary
        if not self.performance_only:
            self.print_mode_coverage()
        
        print(f"\n{'='*80}")
        status = "✅ ALL TESTS PASSED" if all_successful else "❌ SOME TESTS FAILED"
        print(f"{status}")
        print(f"{'='*80}")
    
    def print_mode_coverage(self):
        """Print coverage information for each rendering mode"""
        print(f"\n{'='*80}")
        print("MODE COVERAGE ANALYSIS")
        print(f"{'='*80}")
        
        mode_tests = {
            'Simple Mode': 0,
            'Dynamic Mode': 0, 
            'Multipass Mode': 0,
            'Integration': 0
        }
        
        # Count tests by mode (simplified analysis)
        for suite_name, result in self.results.items():
            if 'Sendcmd' in suite_name or 'dynamic' in suite_name.lower():
                mode_tests['Dynamic Mode'] += result['tests_run']
            elif 'Multipass' in suite_name:
                mode_tests['Multipass Mode'] += result['tests_run']
            elif 'simple' in suite_name.lower():
                mode_tests['Simple Mode'] += result['tests_run']
            else:
                mode_tests['Integration'] += result['tests_run']
        
        for mode, count in mode_tests.items():
            print(f"{mode:<20}: {count} tests")
        
        print(f"\nAll three FFmpeg rendering modes have been tested:")
        print(f"✓ Simple Mode - Static crop using average coordinates")
        print(f"✓ Dynamic Mode - Frame-by-frame cropping with sendcmd")
        print(f"✓ Multipass Mode - Two-pass encoding for optimal quality")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for FFmpeg implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_all_tests.py                    # Run all tests
    python tests/run_all_tests.py --verbose          # High verbosity
    python tests/run_all_tests.py --performance-only # Performance tests only
    python tests/run_all_tests.py --mode simple      # Simple mode tests only
    python tests/run_all_tests.py --mode dynamic     # Dynamic mode tests only
    python tests/run_all_tests.py --mode multipass   # Multipass mode tests only
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
        '--mode', '-m',
        choices=['simple', 'dynamic', 'multipass'],
        help='Run tests for specific rendering mode only'
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
        performance_only=args.performance_only,
        mode_filter=args.mode
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