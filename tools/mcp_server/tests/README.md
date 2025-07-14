# MCP Server Test Suite

This directory contains a comprehensive test suite for the Redis-py MCP Server. The tests are designed to ensure code quality, reliability, and maintainability.

## Test Structure

The test suite is organized into several modules:

### Core Test Files

- **`test_config.py`** - Tests for configuration management (`config.py`)
- **`test_ast_parsing.py`** - Tests for AST parsing and analysis functions
- **`test_file_operations.py`** - Tests for file system operations and utilities
- **`test_analysis.py`** - Tests for test analysis and suggestion functions
- **`test_run_server.py`** - Tests for the server launcher (`run_server.py`)
- **`test_server_tools.py`** - Tests for MCP server tools and handlers
- **`test_integration.py`** - Integration tests for complete workflows

### Support Files

- **`conftest.py`** - Pytest fixtures and configuration
- **`__init__.py`** - Test package initialization
- **`README.md`** - This documentation file

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific test type
python run_tests.py unit
python run_tests.py integration

# Run with verbose output
python run_tests.py --verbose
```

### Test Types

- **`all`** - Run all tests (default)
- **`unit`** - Run unit tests only
- **`integration`** - Run integration tests only
- **`server`** - Run server tool tests only
- **`config`** - Run configuration tests only
- **`ast`** - Run AST parsing tests only
- **`file`** - Run file operations tests only
- **`analysis`** - Run analysis tests only
- **`run_server`** - Run server launcher tests only

### Advanced Usage

```bash
# Run tests in parallel (faster)
python run_tests.py --parallel

# Exclude slow tests
python run_tests.py -m "not slow"

# Run only unit tests
python run_tests.py -m "unit"

# Check test dependencies
python run_tests.py --check-deps

# Install test dependencies
python run_tests.py --install-deps
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=config --cov=main --cov=run_server --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run specific test function
pytest tests/test_config.py::TestMCPServerConfig::test_init_default_values -v
```

## Test Markers

Tests are categorized with markers for flexible execution:

- **`unit`** - Unit tests
- **`integration`** - Integration tests
- **`ast`** - AST parsing related tests
- **`filesystem`** - File system related tests
- **`server`** - Server functionality tests
- **`slow`** - Slow tests (can be excluded for faster runs)

## Test Coverage

The test suite aims for high test coverage across all modules:

- **Configuration management** - 100% coverage of config.py
- **AST parsing** - Comprehensive testing of all AST functions
- **File operations** - Full testing of file system utilities
- **Test analysis** - Complete coverage of analysis functions
- **Server tools** - Testing of all MCP server handlers
- **Integration** - End-to-end workflow testing

### Viewing Coverage Reports

After running tests with `--coverage`, view the HTML report:

```bash
open test_coverage_html/index.html  # macOS
xdg-open test_coverage_html/index.html  # Linux
start test_coverage_html/index.html  # Windows
```

## Test Features

### Fixtures

The test suite includes comprehensive fixtures in `conftest.py`:

- **`temp_project_dir`** - Temporary project structure for testing
- **`sample_python_file`** - Sample Python code for AST testing
- **`sample_test_file`** - Sample test file for analysis testing
- **`mock_config`** - Mock configuration for isolated testing
- **`temp_python_file`** - Temporary Python file with sample content
- **`invalid_python_file`** - Python file with syntax errors for error testing

### Test Categories

1. **Unit Tests** - Test individual functions and classes in isolation
2. **Integration Tests** - Test interactions between components
3. **Error Handling Tests** - Test error conditions and edge cases
4. **Performance Tests** - Test with large datasets and concurrent operations
5. **Compatibility Tests** - Test backwards compatibility and API stability

### Mocking Strategy

Tests use strategic mocking to:
- Isolate units under test
- Control external dependencies
- Simulate error conditions
- Test edge cases safely

## Adding New Tests

### Test File Naming

- Test files should be named `test_<module>.py`
- Test classes should be named `Test<ClassName>`
- Test functions should be named `test_<description>`

### Example Test Structure

```python
#!/usr/bin/env python3
"""
Tests for new_module.py
"""

import pytest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from new_module import new_function


class TestNewFunction:
    """Test the new_function"""
    
    def test_normal_case(self):
        """Test normal operation"""
        result = new_function("input")
        assert result == "expected"
    
    def test_error_case(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            new_function(None)
    
    @pytest.mark.slow
    def test_performance_case(self):
        """Test with large input"""
        large_input = "x" * 10000
        result = new_function(large_input)
        assert isinstance(result, str)
```

### Test Guidelines

1. **One assertion per test** - Keep tests focused
2. **Descriptive names** - Test names should explain what they test
3. **Arrange-Act-Assert** - Structure tests clearly
4. **Use fixtures** - Leverage existing fixtures for common setup
5. **Mock external dependencies** - Keep tests isolated
6. **Test edge cases** - Include error conditions and boundary cases
7. **Add markers** - Use appropriate markers for categorization

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    cd tools/mcp_server
    python run_tests.py --coverage
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./tools/mcp_server/coverage.xml
```

## Dependencies

### Required

- `pytest>=7.0.0` - Test framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.0.0` - Coverage reporting
- `pytest-mock>=3.10.0` - Mocking utilities

### Optional

- `pytest-xdist` - Parallel test execution
- `pytest-html` - HTML test reports

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **Import errors** - Ensure you're running from the correct directory
2. **Missing dependencies** - Run `python run_tests.py --check-deps`
3. **Slow tests** - Use `--parallel` flag or exclude slow tests
4. **Coverage not working** - Ensure pytest-cov is installed

### Getting Help

- Check test output for detailed error messages
- Use `--verbose` flag for more detailed output
- Review individual test files for specific test documentation
- Check the main project README for overall setup instructions

## Best Practices

1. **Run tests before committing** - Ensure all tests pass
2. **Write tests for new features** - Maintain test coverage
3. **Update tests when changing code** - Keep tests current
4. **Use appropriate markers** - Help others run relevant tests
5. **Mock external dependencies** - Keep tests fast and reliable
6. **Test error conditions** - Ensure robust error handling
7. **Keep tests simple** - One concept per test 