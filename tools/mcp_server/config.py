"""
Configuration settings for the Redis-py MCP Server
"""

import os
from pathlib import Path
from typing import Set, Dict, Any


class MCPServerConfig:
    """Configuration class for the MCP server."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.server_name = "redis-py-test-infra"
        self.server_version = "1.0.0"
        
        # File handling settings
        self.max_file_size = 1024 * 1024  # 1MB default
        self.python_extensions = {'.py', '.pyi'}
        self.text_extensions = {'.txt', '.md', '.rst', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini'}
        
        # Directories to ignore
        self.ignore_dirs = {
            '.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 
            'node_modules', '.tox', 'build', 'dist', '.eggs', 'env',
            '.coverage', '.mypy_cache', '.ruff_cache'
        }
        
        # Files to ignore
        self.ignore_files = {
            '.DS_Store', '.gitignore', '.gitmodules', 'Thumbs.db',
            '.coverage', '.coveragerc', '.python-version'
        }
        
        # Directory structure settings
        self.max_directory_depth = 3
        
        # Debug settings
        self.debug = os.getenv('MCP_DEBUG', 'false').lower() == 'true'
        
        # Logging settings
        self.log_level = os.getenv('MCP_LOG_LEVEL', 'INFO')
        
    def is_ignored_path(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        parts = path.parts
        for part in parts:
            if part in self.ignore_dirs:
                return True
            # Only ignore hidden files/directories, not path navigation elements like '..'
            if part.startswith('.') and part not in ('..', '.'):
                return True
        return path.name in self.ignore_files
    
    def is_python_file(self, path: Path) -> bool:
        """Check if a file is a Python file."""
        return path.suffix in self.python_extensions
    
    def is_text_file(self, path: Path) -> bool:
        """Check if a file is a text file."""
        return path.suffix in self.text_extensions or self.is_python_file(path)
    
    def is_test_file(self, path: Path) -> bool:
        """Check if a file is a test file."""
        if not self.is_python_file(path):
            return False
        
        # Check if file is in a test directory
        path_parts = path.parts
        for part in path_parts:
            if part.startswith('test') or part == 'tests':
                return True
        
        # Check if filename indicates it's a test file
        filename = path.name
        return (filename.startswith('test_') or 
                filename.endswith('_test.py') or
                filename == 'test.py')
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get basic project information."""
        return {
            'name': self.server_name,
            'version': self.server_version,
            'project_root': str(self.project_root),
            'debug': self.debug,
            'max_file_size': self.max_file_size,
            'supported_extensions': {
                'python': list(self.python_extensions),
                'text': list(self.text_extensions)
            }
        }


# Global configuration instance
config = MCPServerConfig() 