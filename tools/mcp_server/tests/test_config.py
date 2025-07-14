#!/usr/bin/env python3
"""
Comprehensive unit tests for config.py module
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MCPServerConfig, config


class TestMCPServerConfig:
    """Test the MCPServerConfig class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a fresh config instance for testing"""
        return MCPServerConfig()
    
    def test_init_default_values(self, mock_config):
        """Test default configuration values"""
        assert mock_config.server_name == "redis-py-test-infra"
        assert mock_config.server_version == "1.0.0"
        assert mock_config.max_file_size == 1024 * 1024
        assert mock_config.max_directory_depth == 3
        assert mock_config.python_extensions == {'.py', '.pyi'}
        assert '.txt' in mock_config.text_extensions
        assert '.md' in mock_config.text_extensions
        assert '.json' in mock_config.text_extensions
        assert '.git' in mock_config.ignore_dirs
        assert '__pycache__' in mock_config.ignore_dirs
        assert '.DS_Store' in mock_config.ignore_files
        assert '.gitignore' in mock_config.ignore_files
    
    def test_project_root_calculation(self, mock_config):
        """Test that project root is calculated correctly"""
        # Should be three levels up from the config file
        expected_root = Path(__file__).parent.parent.parent.parent
        assert mock_config.project_root == expected_root
    
    @patch.dict(os.environ, {'MCP_DEBUG': 'true'})
    def test_debug_environment_variable_true(self):
        """Test debug flag from environment variable - true"""
        test_config = MCPServerConfig()
        assert test_config.debug is True
    
    @patch.dict(os.environ, {'MCP_DEBUG': 'false'})
    def test_debug_environment_variable_false(self):
        """Test debug flag from environment variable - false"""
        test_config = MCPServerConfig()
        assert test_config.debug is False
    
    @patch.dict(os.environ, {'MCP_DEBUG': 'True'})
    def test_debug_environment_variable_case_insensitive(self):
        """Test debug flag is case insensitive"""
        test_config = MCPServerConfig()
        assert test_config.debug is True
    
    @patch.dict(os.environ, {'MCP_DEBUG': 'invalid'})
    def test_debug_environment_variable_invalid(self):
        """Test debug flag with invalid value defaults to false"""
        test_config = MCPServerConfig()
        assert test_config.debug is False
    
    @patch.dict(os.environ, {'MCP_LOG_LEVEL': 'DEBUG'})
    def test_log_level_environment_variable(self):
        """Test log level from environment variable"""
        test_config = MCPServerConfig()
        assert test_config.log_level == 'DEBUG'
    
    def test_log_level_default(self, mock_config):
        """Test default log level"""
        assert mock_config.log_level == 'INFO'
    
    def test_is_ignored_path_git_directory(self, mock_config):
        """Test that .git directory is ignored"""
        path = Path('.git')
        assert mock_config.is_ignored_path(path) is True
    
    def test_is_ignored_path_pycache_directory(self, mock_config):
        """Test that __pycache__ directory is ignored"""
        path = Path('__pycache__')
        assert mock_config.is_ignored_path(path) is True
    
    def test_is_ignored_path_nested_ignored_directory(self, mock_config):
        """Test that nested ignored directories are ignored"""
        path = Path('src/__pycache__/module.py')
        assert mock_config.is_ignored_path(path) is True
    
    def test_is_ignored_path_dotfile_in_path(self, mock_config):
        """Test that paths with dot files are ignored"""
        path = Path('src/.hidden/file.py')
        assert mock_config.is_ignored_path(path) is True
    
    def test_is_ignored_path_ignored_file(self, mock_config):
        """Test that ignored files are ignored"""
        path = Path('.DS_Store')
        assert mock_config.is_ignored_path(path) is True
    
    def test_is_ignored_path_normal_file(self, mock_config):
        """Test that normal files are not ignored"""
        path = Path('src/module.py')
        assert mock_config.is_ignored_path(path) is False
    
    def test_is_ignored_path_normal_directory(self, mock_config):
        """Test that normal directories are not ignored"""
        path = Path('src/utils')
        assert mock_config.is_ignored_path(path) is False
    
    def test_is_python_file_py_extension(self, mock_config):
        """Test that .py files are identified as Python files"""
        path = Path('module.py')
        assert mock_config.is_python_file(path) is True
    
    def test_is_python_file_pyi_extension(self, mock_config):
        """Test that .pyi files are identified as Python files"""
        path = Path('module.pyi')
        assert mock_config.is_python_file(path) is True
    
    def test_is_python_file_non_python_extension(self, mock_config):
        """Test that non-Python files are not identified as Python files"""
        path = Path('README.md')
        assert mock_config.is_python_file(path) is False
    
    def test_is_text_file_python_file(self, mock_config):
        """Test that Python files are identified as text files"""
        path = Path('module.py')
        assert mock_config.is_text_file(path) is True
    
    def test_is_text_file_markdown_file(self, mock_config):
        """Test that Markdown files are identified as text files"""
        path = Path('README.md')
        assert mock_config.is_text_file(path) is True
    
    def test_is_text_file_json_file(self, mock_config):
        """Test that JSON files are identified as text files"""
        path = Path('config.json')
        assert mock_config.is_text_file(path) is True
    
    def test_is_text_file_binary_file(self, mock_config):
        """Test that binary files are not identified as text files"""
        path = Path('image.png')
        assert mock_config.is_text_file(path) is False
    
    def test_get_project_info_structure(self, mock_config):
        """Test that get_project_info returns expected structure"""
        info = mock_config.get_project_info()
        
        required_keys = ['name', 'version', 'project_root', 'debug', 'max_file_size', 'supported_extensions']
        for key in required_keys:
            assert key in info
        
        assert info['name'] == mock_config.server_name
        assert info['version'] == mock_config.server_version
        assert info['project_root'] == str(mock_config.project_root)
        assert info['debug'] == mock_config.debug
        assert info['max_file_size'] == mock_config.max_file_size
        
        # Check supported extensions structure
        assert 'supported_extensions' in info
        assert 'python' in info['supported_extensions']
        assert 'text' in info['supported_extensions']
        assert isinstance(info['supported_extensions']['python'], list)
        assert isinstance(info['supported_extensions']['text'], list)
    
    def test_get_project_info_extensions_content(self, mock_config):
        """Test that get_project_info returns correct extensions"""
        info = mock_config.get_project_info()
        
        python_extensions = info['supported_extensions']['python']
        text_extensions = info['supported_extensions']['text']
        
        assert '.py' in python_extensions
        assert '.pyi' in python_extensions
        assert '.txt' in text_extensions
        assert '.md' in text_extensions
        assert '.json' in text_extensions


class TestGlobalConfig:
    """Test the global config instance"""
    
    def test_global_config_exists(self):
        """Test that global config instance exists"""
        assert config is not None
        assert isinstance(config, MCPServerConfig)
    
    def test_global_config_properties(self):
        """Test that global config has expected properties"""
        assert hasattr(config, 'server_name')
        assert hasattr(config, 'server_version')
        assert hasattr(config, 'project_root')
        assert hasattr(config, 'max_file_size')
        assert hasattr(config, 'python_extensions')
        assert hasattr(config, 'text_extensions')
        assert hasattr(config, 'ignore_dirs')
        assert hasattr(config, 'ignore_files')
    
    def test_global_config_methods(self):
        """Test that global config has expected methods"""
        assert hasattr(config, 'is_ignored_path')
        assert hasattr(config, 'is_python_file')
        assert hasattr(config, 'is_text_file')
        assert hasattr(config, 'get_project_info')
        assert callable(config.is_ignored_path)
        assert callable(config.is_python_file)
        assert callable(config.is_text_file)
        assert callable(config.get_project_info)


class TestConfigEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_path(self):
        """Test behavior with empty path"""
        empty_path = Path('')
        assert config.is_ignored_path(empty_path) is False
    
    def test_root_path(self):
        """Test behavior with root path"""
        root_path = Path('/')
        assert config.is_ignored_path(root_path) is False
    
    def test_relative_path_with_dots(self):
        """Test behavior with relative paths containing dots"""
        path = Path('../some/path')
        # Should not be ignored just because it has dots in the path
        assert config.is_ignored_path(path) is False
    
    def test_path_with_hidden_file_in_middle(self):
        """Test path with hidden file in the middle"""
        path = Path('src/.hidden/file.py')
        assert config.is_ignored_path(path) is True
    
    def test_path_case_sensitivity(self):
        """Test case sensitivity of path checking"""
        # Test with different case
        path_lower = Path('__pycache__')
        path_upper = Path('__PYCACHE__')
        
        assert config.is_ignored_path(path_lower) is True
        # Different case should not be ignored (case sensitive)
        assert config.is_ignored_path(path_upper) is False
    
    def test_file_extension_case_sensitivity(self):
        """Test case sensitivity of file extension checking"""
        path_lower = Path('module.py')
        path_upper = Path('module.PY')
        
        assert config.is_python_file(path_lower) is True
        # Different case should not be recognized as Python file
        assert config.is_python_file(path_upper) is False
    
    def test_no_extension_file(self):
        """Test behavior with files without extensions"""
        path = Path('Makefile')
        assert config.is_python_file(path) is False
        assert config.is_text_file(path) is False
    
    def test_multiple_dots_in_filename(self):
        """Test behavior with multiple dots in filename"""
        path = Path('test.backup.py')
        assert config.is_python_file(path) is True
        assert config.is_text_file(path) is True
    
    def test_hidden_python_file(self):
        """Test behavior with hidden Python files"""
        path = Path('.hidden.py')
        assert config.is_python_file(path) is True
        assert config.is_text_file(path) is True
        # Should be ignored because it starts with a dot
        assert config.is_ignored_path(path) is True 