#!/usr/bin/env python3
"""
Comprehensive unit tests for file operations functions in main.py
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    find_test_files,
    get_relative_path,
    is_ignored_path,
    find_python_files,
    read_file_content,
    get_directory_structure,
    get_project_info
)


class TestFindTestFiles:
    """Test the find_test_files function"""
    
    def test_find_test_files_default_directory(self, temp_project_dir):
        """Test finding test files in default directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_test_files()
            
            assert len(result) > 0
            test_file = next((f for f in result if f['path'].endswith('test_module.py')), None)
            assert test_file is not None
            assert test_file['is_test'] is True
            assert test_file['size'] > 0
    
    def test_find_test_files_specific_directory(self, temp_project_dir):
        """Test finding test files in specific directory"""
        test_dir = temp_project_dir / "tests"
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_test_files(test_dir)
            
            assert len(result) > 0
            # All files should be from the tests directory
            for file_info in result:
                assert "tests" in file_info['path']
    
    def test_find_test_files_empty_directory(self, temp_project_dir):
        """Test finding test files in empty directory"""
        empty_dir = temp_project_dir / "empty"
        empty_dir.mkdir()
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_test_files(empty_dir)
            
            assert len(result) == 0
    
    def test_find_test_files_nonexistent_directory(self, temp_project_dir):
        """Test finding test files in non-existent directory"""
        nonexistent_dir = temp_project_dir / "nonexistent"
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_test_files(nonexistent_dir)
            
            assert len(result) == 0


class TestGetRelativePath:
    """Test the get_relative_path function"""
    
    def test_relative_path_normal(self, temp_project_dir):
        """Test getting relative path for normal file"""
        test_file = temp_project_dir / "src" / "module.py"
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_relative_path(test_file)
            
            assert result == "src/module.py"
    
    def test_relative_path_root_file(self, temp_project_dir):
        """Test getting relative path for file in root"""
        test_file = temp_project_dir / "README.md"
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_relative_path(test_file)
            
            assert result == "README.md"
    
    def test_relative_path_outside_project(self, temp_project_dir):
        """Test getting relative path for file outside project"""
        outside_file = Path("/tmp/outside.py")
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_relative_path(outside_file)
            
            # Should return the absolute path as string
            assert result == str(outside_file)


class TestIsIgnoredPath:
    """Test the is_ignored_path function"""
    
    def test_ignored_git_directory(self):
        """Test that .git directory is ignored"""
        path = Path(".git")
        assert is_ignored_path(path) is True
    
    def test_ignored_pycache_directory(self):
        """Test that __pycache__ directory is ignored"""
        path = Path("__pycache__")
        assert is_ignored_path(path) is True
    
    def test_ignored_nested_directory(self):
        """Test that nested ignored directories are ignored"""
        path = Path("src/__pycache__/module.pyc")
        assert is_ignored_path(path) is True
    
    def test_normal_file_not_ignored(self):
        """Test that normal files are not ignored"""
        path = Path("src/module.py")
        assert is_ignored_path(path) is False
    
    def test_ignored_file(self):
        """Test that ignored files are ignored"""
        path = Path(".DS_Store")
        assert is_ignored_path(path) is True


class TestFindPythonFiles:
    """Test the find_python_files function"""
    
    def test_find_python_files_default_directory(self, temp_project_dir):
        """Test finding Python files in default directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_python_files()
            
            assert len(result) > 0
            
            # Check that we found expected files
            file_paths = [f['path'] for f in result]
            assert any('module.py' in path for path in file_paths)
            assert any('utils.py' in path for path in file_paths)
    
    def test_find_python_files_specific_directory(self, temp_project_dir):
        """Test finding Python files in specific directory"""
        src_dir = temp_project_dir / "src"
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_python_files(src_dir)
            
            assert len(result) > 0
            
            # All files should be from the src directory
            for file_info in result:
                assert "src" in file_info['path']
                assert file_info['path'].endswith('.py')
    
    def test_find_python_files_structure(self, temp_project_dir):
        """Test the structure of Python file information"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_python_files()
            
            for file_info in result:
                assert 'path' in file_info
                assert 'size' in file_info
                assert 'modified' in file_info
                assert 'is_test' in file_info
                assert isinstance(file_info['size'], int)
                assert file_info['size'] > 0
    
    def test_find_python_files_ignores_pycache(self, temp_project_dir):
        """Test that __pycache__ files are ignored"""
        pycache_dir = temp_project_dir / "__pycache__"
        pycache_file = pycache_dir / "module.pyc"
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_python_files()
            
            file_paths = [f['path'] for f in result]
            assert not any('__pycache__' in path for path in file_paths)
    
    def test_find_python_files_empty_directory(self, temp_project_dir):
        """Test finding Python files in empty directory"""
        empty_dir = temp_project_dir / "empty"
        empty_dir.mkdir()
        
        with patch('config.config.project_root', temp_project_dir):
            result = find_python_files(empty_dir)
            
            assert len(result) == 0


class TestReadFileContent:
    """Test the read_file_content function"""
    
    def test_read_python_file(self, temp_project_dir):
        """Test reading a Python file"""
        python_file = temp_project_dir / "src" / "module.py"
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("src/module.py")
            
            assert 'error' not in result
            assert 'content' in result
            assert 'size' in result
            assert 'lines' in result
            assert 'is_python' in result
            assert 'is_text' in result
            assert result['is_python'] is True
            assert result['is_text'] is True
            assert result['size'] > 0
            assert result['lines'] > 0
            assert 'def hello_world' in result['content']
    
    def test_read_text_file(self, temp_project_dir):
        """Test reading a text file"""
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("README.md")
            
            assert 'error' not in result
            assert result['is_python'] is False
            assert result['is_text'] is True
            assert result['content'] == "# Test Project"
    
    def test_read_nonexistent_file(self, temp_project_dir):
        """Test reading a non-existent file"""
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("nonexistent.py")
            
            assert 'error' in result
            assert 'not found' in result['error'].lower()
    
    def test_read_file_with_max_size(self, temp_project_dir):
        """Test reading file with max size limit"""
        # Create a file with known content
        large_file = temp_project_dir / "large.py"
        content = "# This is a test file\n" * 1000
        large_file.write_text(content)
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("large.py", max_size=100)
            
            assert 'error' not in result
            assert len(result['content']) <= 100
            assert result['truncated'] is True
    
    def test_read_file_ignored_file(self, temp_project_dir):
        """Test reading an ignored file"""
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content(".DS_Store")
            
            assert 'error' in result
            assert 'ignored' in result['error'].lower() or 'not found' in result['error'].lower()
    
    def test_read_file_permission_error(self, temp_project_dir):
        """Test handling of permission errors"""
        with patch('config.config.project_root', temp_project_dir):
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                result = read_file_content("src/module.py")
                
                assert 'error' in result
                assert 'permission' in result['error'].lower()


class TestGetDirectoryStructure:
    """Test the get_directory_structure function"""
    
    def test_get_directory_structure_default(self, temp_project_dir):
        """Test getting directory structure for default directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure()
            
            assert result is not None
            assert result['type'] == 'directory'
            assert result['name'] == temp_project_dir.name
            assert 'children' in result
            assert len(result['children']) > 0
            
            # Check that we have expected directories
            child_names = [child['name'] for child in result['children']]
            assert 'src' in child_names
            assert 'tests' in child_names
            assert 'docs' in child_names
    
    def test_get_directory_structure_specific_directory(self, temp_project_dir):
        """Test getting directory structure for specific directory"""
        src_dir = temp_project_dir / "src"
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure(src_dir)
            
            assert result is not None
            assert result['type'] == 'directory'
            assert result['name'] == 'src'
            assert 'children' in result
            
            # Check that we have expected files
            child_names = [child['name'] for child in result['children']]
            assert 'module.py' in child_names
            assert 'utils.py' in child_names
    
    def test_get_directory_structure_with_max_depth(self, temp_project_dir):
        """Test getting directory structure with max depth"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure(max_depth=1)
            
            assert result is not None
            assert 'children' in result
            
            # Check that subdirectories don't have children (depth limit)
            for child in result['children']:
                if child['type'] == 'directory':
                    if 'children' in child:
                        # At depth 1, children should be empty or not deeply nested
                        assert len(child['children']) == 0 or all(
                            'children' not in grandchild or len(grandchild['children']) == 0
                            for grandchild in child['children']
                        )
    
    def test_get_directory_structure_ignores_hidden(self, temp_project_dir):
        """Test that directory structure ignores hidden directories"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure()
            
            # Check that ignored directories are not included
            child_names = [child['name'] for child in result['children']]
            assert '.git' not in child_names
            assert '__pycache__' not in child_names
    
    def test_get_directory_structure_nonexistent_directory(self, temp_project_dir):
        """Test getting directory structure for non-existent directory"""
        nonexistent_dir = temp_project_dir / "nonexistent"
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure(nonexistent_dir)
            
            assert result is None
    
    def test_get_directory_structure_file_info(self, temp_project_dir):
        """Test that file information is correctly included"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_directory_structure()
            
            # Find a file in the structure
            for child in result['children']:
                if child['type'] == 'file':
                    assert 'size' in child
                    assert 'modified' in child
                    assert isinstance(child['size'], int)
                    break
            else:
                # If no files at root level, check subdirectories
                for child in result['children']:
                    if child['type'] == 'directory' and 'children' in child:
                        for grandchild in child['children']:
                            if grandchild['type'] == 'file':
                                assert 'size' in grandchild
                                assert 'modified' in grandchild
                                assert isinstance(grandchild['size'], int)
                                break


class TestGetProjectInfo:
    """Test the get_project_info function"""
    
    def test_get_project_info_structure(self, temp_project_dir):
        """Test that get_project_info returns expected structure"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            required_keys = [
                'name', 'version', 'project_root', 'file_counts', 
                'main_directories', 'python_files', 'test_files',
                'total_lines', 'configuration'
            ]
            
            for key in required_keys:
                assert key in result, f"Missing key: {key}"
    
    def test_get_project_info_file_counts(self, temp_project_dir):
        """Test that file counts are correctly calculated"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            file_counts = result['file_counts']
            assert 'python' in file_counts
            assert 'test' in file_counts
            assert 'total' in file_counts
            assert isinstance(file_counts['python'], int)
            assert isinstance(file_counts['test'], int)
            assert isinstance(file_counts['total'], int)
            assert file_counts['python'] > 0
            assert file_counts['test'] > 0
            assert file_counts['total'] >= file_counts['python'] + file_counts['test']
    
    def test_get_project_info_main_directories(self, temp_project_dir):
        """Test that main directories are correctly identified"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            main_dirs = result['main_directories']
            assert isinstance(main_dirs, list)
            assert len(main_dirs) > 0
            
            # Check that we have expected directories
            dir_names = [d['name'] for d in main_dirs]
            assert 'src' in dir_names
            assert 'tests' in dir_names
            assert 'docs' in dir_names
    
    def test_get_project_info_python_files(self, temp_project_dir):
        """Test that Python files are correctly counted"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            python_files = result['python_files']
            assert isinstance(python_files, list)
            assert len(python_files) > 0
            
            # Check structure of Python file info
            for file_info in python_files:
                assert 'path' in file_info
                assert 'size' in file_info
                assert 'is_test' in file_info
    
    def test_get_project_info_test_files(self, temp_project_dir):
        """Test that test files are correctly identified"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            test_files = result['test_files']
            assert isinstance(test_files, list)
            assert len(test_files) > 0
            
            # All test files should have is_test = True
            for file_info in test_files:
                assert file_info['is_test'] is True
    
    def test_get_project_info_configuration(self, temp_project_dir):
        """Test that configuration information is included"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            config_info = result['configuration']
            assert isinstance(config_info, dict)
            assert 'debug' in config_info
            assert 'max_file_size' in config_info
            assert 'max_directory_depth' in config_info
    
    def test_get_project_info_total_lines(self, temp_project_dir):
        """Test that total lines are calculated"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_project_info()
            
            total_lines = result['total_lines']
            assert isinstance(total_lines, int)
            assert total_lines > 0


class TestFileOperationsEdgeCases:
    """Test edge cases and error conditions for file operations"""
    
    def test_file_operations_with_unicode(self, temp_project_dir):
        """Test file operations with Unicode content"""
        unicode_file = temp_project_dir / "unicode.py"
        unicode_content = "# -*- coding: utf-8 -*-\n# Unicode: café, naïve, résumé 🐍\nprint('Hello, 世界')"
        unicode_file.write_text(unicode_content, encoding='utf-8')
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("unicode.py")
            
            assert 'error' not in result
            assert 'café' in result['content']
            assert '世界' in result['content']
            assert '🐍' in result['content']
    
    def test_file_operations_with_large_files(self, temp_project_dir):
        """Test file operations with large files"""
        large_file = temp_project_dir / "large.py"
        # Create a file larger than default max size
        large_content = "# " + "x" * (1024 * 1024 + 1000)  # Slightly over 1MB
        large_file.write_text(large_content)
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("large.py")
            
            assert 'error' not in result
            assert result['truncated'] is True
            assert len(result['content']) <= 1024 * 1024  # Should be truncated
    
    def test_file_operations_with_binary_files(self, temp_project_dir):
        """Test file operations with binary files"""
        binary_file = temp_project_dir / "binary.bin"
        binary_content = b'\x00\x01\x02\x03\xFF\xFE\xFD'
        binary_file.write_bytes(binary_content)
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("binary.bin")
            
            # Should handle binary files gracefully
            assert 'error' in result or result['is_text'] is False
    
    def test_file_operations_with_symlinks(self, temp_project_dir):
        """Test file operations with symbolic links"""
        if os.name == 'nt':  # Skip on Windows
            pytest.skip("Symbolic links not fully supported on Windows")
        
        target_file = temp_project_dir / "target.py"
        target_file.write_text("# Target file")
        
        symlink_file = temp_project_dir / "symlink.py"
        symlink_file.symlink_to(target_file)
        
        with patch('config.config.project_root', temp_project_dir):
            result = read_file_content("symlink.py")
            
            assert 'error' not in result
            assert result['content'] == "# Target file" 