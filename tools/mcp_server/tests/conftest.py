#!/usr/bin/env python3
"""
Pytest fixtures and configuration for MCP server tests
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MCPServerConfig


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create basic project structure
        (temp_path / "src").mkdir()
        (temp_path / "tests").mkdir()
        (temp_path / "docs").mkdir()
        (temp_path / ".git").mkdir()
        (temp_path / "__pycache__").mkdir()
        
        # Create some test files
        (temp_path / "src" / "module.py").write_text("""
def hello_world():
    '''A simple hello world function'''
    return "Hello, World!"

class TestClass:
    '''A test class'''
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
""")
        
        (temp_path / "src" / "utils.py").write_text("""
import os
from typing import Optional

def get_file_size(filepath: str) -> Optional[int]:
    '''Get file size in bytes'''
    try:
        return os.path.getsize(filepath)
    except OSError:
        return None
""")
        
        (temp_path / "tests" / "test_module.py").write_text("""
import pytest
from src.module import hello_world, TestClass

def test_hello_world():
    assert hello_world() == "Hello, World!"

class TestTestClass:
    def test_init(self):
        obj = TestClass("Alice")
        assert obj.name == "Alice"
    
    def test_greet(self):
        obj = TestClass("Bob")
        assert obj.greet() == "Hello, Bob!"
""")
        
        (temp_path / "README.md").write_text("# Test Project")
        (temp_path / "pyproject.toml").write_text("""
[tool.pytest.ini_options]
testpaths = ["tests"]
""")
        
        # Create some ignored files
        (temp_path / ".DS_Store").write_text("binary data")
        (temp_path / "__pycache__" / "cache.pyc").write_text("compiled python")
        
        yield temp_path


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for AST parsing tests"""
    return """
'''Sample module for testing AST parsing'''

import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

class SampleClass:
    '''A sample class for testing'''
    
    class_var: int = 42
    
    def __init__(self, name: str, age: int = 0):
        '''Initialize the instance'''
        self.name = name
        self.age = age
    
    @property
    def display_name(self) -> str:
        '''Get the display name'''
        return f"{self.name} ({self.age})"
    
    def greet(self, greeting: str = "Hello") -> str:
        '''Greet someone'''
        return f"{greeting}, {self.name}!"
    
    async def async_method(self) -> None:
        '''An async method'''
        pass
    
    def _private_method(self) -> None:
        '''A private method'''
        pass

def simple_function(x: int, y: int) -> int:
    '''Add two numbers'''
    return x + y

async def async_function(data: List[str]) -> Dict[str, int]:
    '''Process data asynchronously'''
    return {item: len(item) for item in data}

def function_with_defaults(name: str, age: int = 25, *args, **kwargs) -> str:
    '''Function with default parameters'''
    return f"{name} is {age} years old"

def function_with_complex_types(
    items: List[Dict[str, Optional[int]]], 
    callback: callable = None
) -> Optional[Dict[str, List[int]]]:
    '''Function with complex type annotations'''
    if callback:
        return callback(items)
    return None

# Module-level variable
MODULE_CONSTANT = "test_value"
"""


@pytest.fixture
def sample_test_file():
    """Create a sample test file for analysis"""
    return """
import pytest
from unittest.mock import Mock, patch
from src.module import SampleClass, simple_function

class TestSampleClass:
    '''Test class for SampleClass'''
    
    @pytest.fixture
    def sample_instance(self):
        '''Create a sample instance'''
        return SampleClass("Test", 30)
    
    def test_init(self, sample_instance):
        '''Test initialization'''
        assert sample_instance.name == "Test"
        assert sample_instance.age == 30
    
    @pytest.mark.parametrize("name,age,expected", [
        ("Alice", 25, "Alice (25)"),
        ("Bob", 30, "Bob (30)"),
    ])
    def test_display_name(self, name, age, expected):
        '''Test display name property'''
        obj = SampleClass(name, age)
        assert obj.display_name == expected
    
    @patch('src.module.some_external_function')
    def test_with_mock(self, mock_func):
        '''Test with mocking'''
        mock_func.return_value = "mocked"
        obj = SampleClass("Test")
        result = obj.greet()
        assert result == "Hello, Test!"

def test_simple_function():
    '''Test simple function'''
    assert simple_function(2, 3) == 5

@pytest.mark.asyncio
async def test_async_function():
    '''Test async function'''
    from src.module import async_function
    result = await async_function(["a", "bb", "ccc"])
    assert result == {"a": 1, "bb": 2, "ccc": 3}

@pytest.mark.slow
def test_slow_operation():
    '''A slow test'''
    import time
    time.sleep(0.1)  # Simulate slow operation
    assert True
"""


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    config = MCPServerConfig()
    with patch('config.config', config):
        yield config


@pytest.fixture
def temp_python_file(temp_project_dir, sample_python_file):
    """Create a temporary Python file with sample content"""
    file_path = temp_project_dir / "test_module.py"
    file_path.write_text(sample_python_file)
    return file_path


@pytest.fixture
def temp_test_file(temp_project_dir, sample_test_file):
    """Create a temporary test file with sample content"""
    file_path = temp_project_dir / "test_sample.py"
    file_path.write_text(sample_test_file)
    return file_path


@pytest.fixture
def invalid_python_file(temp_project_dir):
    """Create a Python file with syntax errors"""
    file_path = temp_project_dir / "invalid.py"
    file_path.write_text("""
def broken_function(
    # Missing closing parenthesis and colon
    print("This is invalid Python syntax"
    return "error"
""")
    return file_path


@pytest.fixture
def mock_file_system():
    """Create a mock file system for testing"""
    return {
        "src/module.py": "# Sample module\nclass TestClass:\n    pass",
        "src/utils.py": "# Utilities\ndef helper():\n    pass",
        "tests/test_module.py": "# Test module\nimport pytest\n\ndef test_function():\n    pass",
        "README.md": "# Project README",
        "pyproject.toml": "[tool.pytest.ini_options]\ntestpaths = ['tests']",
        ".gitignore": "*.pyc\n__pycache__/",
    }


@pytest.fixture(autouse=True)
def clean_sys_modules():
    """Clean up sys.modules after each test to avoid import conflicts"""
    import sys
    modules_before = set(sys.modules.keys())
    yield
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith('test_') or module.startswith('conftest'):
            sys.modules.pop(module, None) 