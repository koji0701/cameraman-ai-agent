# FFmpeg Implementation Test Suite

Comprehensive test suite for all FFmpeg rendering implementations across three modes: **Simple**, **Dynamic**, and **Multipass**.

## Overview

This test suite validates the functionality, performance, and reliability of your FFmpeg video rendering pipeline. It covers:

- 🎥 **Simple Mode**: Static crop using average coordinates
- 🎬 **Dynamic Mode**: Frame-by-frame cropping with sendcmd filters
- 🏆 **Multipass Mode**: Two-pass encoding for optimal quality
- ⚡ **Performance Testing**: Large datasets and edge cases
- 🔧 **Integration Testing**: FFmpeg command generation and execution

## Test Files

### Core Test Suites

| File | Description | Coverage |
|------|-------------|----------|
| `test_ffmpeg_render.py` | Main rendering functionality tests | All three modes, error handling, codec options |
| `test_sendcmd_generation.py` | Sendcmd filter generation tests | Dynamic mode, coordinate precision, edge cases |
| `test_multipass_encoding.py` | Multipass encoding tests | Quality settings, two-pass workflow, cleanup |
| `test_performance.py` | Performance and stress tests | Large datasets, memory usage, concurrency |
| `run_all_tests.py` | Comprehensive test runner | Orchestrates all test suites with reporting |

### Test Categories

#### 1. Functional Tests (`test_ffmpeg_render.py`)
- ✅ **Simple render mode validation**
- ✅ **Dynamic render mode validation** 
- ✅ **Multipass render mode validation**
- ✅ **Mode dispatcher functionality**
- ✅ **Audio stream handling**
- ✅ **Codec-specific options**
- ✅ **Error handling and recovery**
- ✅ **Video information extraction**
- ✅ **Command validation**

#### 2. Sendcmd Generation Tests (`test_sendcmd_generation.py`)
- ✅ **Sendcmd filter structure**
- ✅ **Timeline accuracy**
- ✅ **Coordinate formatting**
- ✅ **Even dimension enforcement**
- ✅ **Precision handling**
- ✅ **Extreme coordinate values**
- ✅ **Different FPS values**
- ✅ **Empty data handling**

#### 3. Multipass Encoding Tests (`test_multipass_encoding.py`)
- ✅ **Quality preset validation** (medium, high, ultra)
- ✅ **Two-pass workflow**
- ✅ **Temporary file cleanup**
- ✅ **FFmpeg option verification**
- ✅ **Failure handling**
- ✅ **File size reporting**
- ✅ **Verbose mode testing**

#### 4. Performance Tests (`test_performance.py`)
- ✅ **Large dataset handling** (10k+ frames)
- ✅ **Memory usage validation**
- ✅ **Processing speed benchmarks**
- ✅ **Extreme coordinate values**
- ✅ **Concurrent operation safety**
- ✅ **Memory cleanup verification**
- ✅ **Edge case performance**

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_all_tests.py

# Run with high verbosity
python tests/run_all_tests.py --verbose

# Run only performance tests
python tests/run_all_tests.py --performance-only
```

### Mode-Specific Testing

```bash
# Test only Simple mode
python tests/run_all_tests.py --mode simple

# Test only Dynamic mode
python tests/run_all_tests.py --mode dynamic

# Test only Multipass mode
python tests/run_all_tests.py --mode multipass
```

### Individual Test Files

```bash
# Run specific test suite
python -m pytest tests/test_sendcmd_generation.py -v
python -m pytest tests/test_multipass_encoding.py -v
python -m pytest tests/test_performance.py -v

# Run with unittest directly
python tests/test_ffmpeg_render.py
python tests/test_sendcmd_generation.py
```

### Advanced Options

```bash
# Minimal output
python tests/run_all_tests.py --quiet

# Performance tests only with verbose output
python tests/run_all_tests.py --performance-only --verbose

# Test specific mode with minimal output
python tests/run_all_tests.py --mode dynamic --quiet
```

## Test Results

The test runner provides comprehensive reporting:

### Summary Report
- Total tests run
- Pass/fail/error counts
- Success rates per suite
- Execution timing
- Mode coverage analysis

### Detailed Output
- Per-test results
- Failure details with context
- Performance metrics
- Memory usage validation

### Example Output
```
==================================================================================
TEST SUMMARY
==================================================================================
Total duration: 45.23 seconds
Total tests run: 87
Passed: 85
Failed: 2
Errors: 0
Skipped: 0
Overall success rate: 97.7%

Per-suite breakdown:
Suite                          Tests    Pass     Fail     Error    Rate     Time    
--------------------------------------------------------------------------------
Core Rendering Modes           25       25       0        0        100.0%   12.34s
Sendcmd Filter Generation      18       18       0        0        100.0%   3.45s
Multipass Encoding            15       15       0        0        100.0%   8.67s
Performance Tests             29       27       2        0        93.1%    20.77s

==================================================================================
MODE COVERAGE ANALYSIS
==================================================================================
Simple Mode         : 23 tests
Dynamic Mode        : 31 tests
Multipass Mode      : 19 tests
Integration         : 14 tests

All three FFmpeg rendering modes have been tested:
✓ Simple Mode - Static crop using average coordinates
✓ Dynamic Mode - Frame-by-frame cropping with sendcmd
✓ Multipass Mode - Two-pass encoding for optimal quality
```

## Test Dependencies

### Required Packages
```
pandas
numpy
unittest (built-in)
tempfile (built-in)
subprocess (built-in)
pathlib (built-in)
```

### Mock Dependencies
Tests use extensive mocking to avoid requiring:
- Actual FFmpeg installation
- Real video files
- File system operations

## Writing New Tests

### Test Structure
```python
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add pipelines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))

from render_video import your_function

class TestYourFeature(unittest.TestCase):
    def setUp(self):
        # Test setup
        pass
    
    def tearDown(self):
        # Cleanup
        pass
    
    @patch('render_video.subprocess.run')
    def test_your_feature(self, mock_subprocess):
        # Your test implementation
        pass
```

### Testing Guidelines

1. **Use descriptive test names**: `test_sendcmd_generation_with_large_dataset`
2. **Mock external dependencies**: FFmpeg, file operations, subprocess calls
3. **Test edge cases**: Empty data, extreme values, error conditions
4. **Validate both success and failure paths**
5. **Include performance considerations**: Memory usage, execution time
6. **Document test purpose**: Clear docstrings explaining what's being tested

### Adding Tests to Runner

Add new test classes to `run_all_tests.py`:

```python
from your_new_test_file import YourTestClass

# In TestRunner.run_all_tests():
test_suites.extend([
    (YourTestClass, "Your Test Description")
])
```

## Continuous Integration

The test suite is designed for CI/CD environments:

```yaml
# Example GitHub Actions
- name: Run FFmpeg Tests
  run: python tests/run_all_tests.py --quiet
  
- name: Run Performance Tests
  run: python tests/run_all_tests.py --performance-only
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **Missing Modules**: Check pipelines directory path
3. **Test Failures**: Review mock configurations
4. **Performance Issues**: Adjust timeout values in performance tests

### Debug Mode
```bash
# Run with maximum verbosity and Python debugging
python -v tests/run_all_tests.py --verbose
```

### Individual Test Debugging
```bash
# Run single test with full output
python -m unittest tests.test_ffmpeg_render.TestFFmpegRenderModes.test_simple_render_mode -v
```

## Coverage Goals

- **Functional Coverage**: 100% of public API methods
- **Edge Case Coverage**: All known edge cases and error conditions
- **Performance Coverage**: Large datasets and stress testing
- **Integration Coverage**: End-to-end workflow validation

## Contributing

When adding new features to the FFmpeg implementation:

1. Add corresponding tests to appropriate test file
2. Update test documentation
3. Ensure all three modes are covered
4. Add performance tests for new functionality
5. Update this README with new test descriptions

---

**Happy Testing! 🎬✨**

The comprehensive test suite ensures your FFmpeg implementations are robust, performant, and reliable across all three rendering modes. 