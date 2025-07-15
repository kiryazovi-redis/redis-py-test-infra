#!/usr/bin/env python3
"""
Comprehensive integration tests for the complete MCP server functionality
"""

import os
import json
import pytest
import asyncio
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MCPServerConfig
from main import (
    parse_module_ast,
    find_python_files,
    read_file_content,
    get_project_info,
    analyze_test_files,
    suggest_test_cases
)
from run_server import check_dependencies, run_server


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_project_analysis_workflow(self, temp_project_dir):
        """Test complete project analysis workflow"""
        with patch('config.config.project_root', temp_project_dir):
            # Step 1: Get project info
            project_info = get_project_info()
            assert 'name' in project_info
            assert 'file_counts' in project_info
            assert project_info['file_counts']['python'] > 0
            
            # Step 2: Find Python files
            python_files = find_python_files()
            assert len(python_files) > 0
            
            # Step 3: Analyze a Python file
            test_file = next((f for f in python_files if 'module.py' in f['path']), None)
            assert test_file is not None
            
            file_analysis = parse_module_ast(test_file['path'])
            assert 'classes' in file_analysis
            assert 'functions' in file_analysis
            assert len(file_analysis['functions']) > 0
            
            # Step 4: Read file content
            file_content = read_file_content(test_file['path'])
            assert 'content' in file_content
            assert 'def hello_world' in file_content['content']
    
    def test_test_analysis_workflow(self, temp_project_dir):
        """Test complete test analysis workflow"""
        with patch('config.config.project_root', temp_project_dir):
            # Step 1: Analyze existing test files
            test_analysis = analyze_test_files("tests")
            assert 'total_test_files' in test_analysis
            assert test_analysis['total_test_files'] > 0
            
            # Step 2: Find a source file to suggest tests for
            python_files = find_python_files()
            source_file = next((f for f in python_files if 'module.py' in f['path']), None)
            assert source_file is not None
            
            # Step 3: Suggest test cases
            test_suggestions = suggest_test_cases(source_file['path'])
            assert 'test_suggestions' in test_suggestions
            assert 'recommended_framework' in test_suggestions
            assert len(test_suggestions['test_suggestions']) > 0
    
    def test_configuration_integration(self, temp_project_dir):
        """Test that configuration integrates properly with all components"""
        config = MCPServerConfig()
        
        with patch('main.config', config):
            with patch('main.config.project_root', temp_project_dir):
                # Test that config affects file operations
                python_files = find_python_files()
                assert len(python_files) > 0
                
                # Test that ignored paths are respected
                for file_info in python_files:
                    path = Path(file_info['path'])
                    assert not config.is_ignored_path(path)
                
                # Test file size limits
                large_file = temp_project_dir / "large.py"
                large_content = "# " + "x" * (config.max_file_size + 1000)
                large_file.write_text(large_content)
                
                file_content = read_file_content("large.py")
                assert file_content['truncated'] is True
                assert len(file_content['content']) <= config.max_file_size


class TestCrossModuleIntegration:
    """Test integration between different modules"""
    
    def test_ast_and_file_operations_integration(self, temp_python_file):
        """Test integration between AST parsing and file operations"""
        with patch('config.config.project_root', temp_python_file.parent):
            # First, read the file content
            file_content = read_file_content(temp_python_file.name)
            assert 'content' in file_content
            assert file_content['is_python'] is True
            
            # Then, parse its AST
            ast_result = parse_module_ast(temp_python_file.name)
            assert 'classes' in ast_result
            assert 'functions' in ast_result
            
            # The content should match what AST parsing found
            assert len(ast_result['functions']) > 0
            assert len(ast_result['classes']) > 0
            
            # Function names from AST should be in the file content
            for func in ast_result['functions']:
                assert func['name'] in file_content['content']
    
    def test_config_and_analysis_integration(self, temp_project_dir):
        """Test integration between config and analysis functions"""
        config = MCPServerConfig()
        
        with patch('config.config', config):
            with patch('config.config.project_root', temp_project_dir):
                # Test that project info respects config
                project_info = get_project_info()
                assert project_info['configuration']['debug'] == config.debug
                assert project_info['configuration']['max_file_size'] == config.max_file_size
                
                # Test that analysis functions respect config settings
                python_files = find_python_files()
                
                # All found files should be valid according to config
                for file_info in python_files:
                    path = Path(file_info['path'])
                    assert config.is_python_file(path)
                    assert not config.is_ignored_path(path)
    
    def test_server_and_functions_integration(self, temp_python_file):
        """Test integration between server tools and underlying functions"""
        # This test would require MCP server to be running, so we'll mock it
        with patch('config.config.project_root', temp_python_file.parent):
            # Test that functions work independently (as they would be called by server)
            
            # Test parse_module
            result = parse_module_ast(temp_python_file.name)
            assert 'error' not in result
            assert 'functions' in result
            
            # Test that result can be JSON serialized (as server would do)
            json_result = json.dumps(result, indent=2)
            assert isinstance(json_result, str)
            parsed_back = json.loads(json_result)
            assert parsed_back == result


class TestErrorHandlingIntegration:
    """Test error handling across module boundaries"""
    
    def test_cascading_error_handling(self, temp_project_dir):
        """Test that errors are properly handled across function calls"""
        with patch('config.config.project_root', temp_project_dir):
            # Test with non-existent file
            result = parse_module_ast("nonexistent.py")
            assert 'error' in result
            assert 'not found' in result['error'].lower()
            
            # Test that this doesn't break subsequent operations
            python_files = find_python_files()
            assert isinstance(python_files, list)
            
            # Test with invalid Python file
            invalid_file = temp_project_dir / "invalid.py"
            invalid_file.write_text("def broken(:\n    pass")
            
            result = parse_module_ast("invalid.py")
            assert 'error' in result
            assert 'syntax error' in result['error'].lower()
    
    def test_permission_error_handling(self, temp_project_dir):
        """Test handling of permission errors across modules"""
        with patch('main.config.project_root', temp_project_dir):
            # Create a file first
            test_file = temp_project_dir / "test.py"
            test_file.write_text("print('hello')")
            
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                # Test file reading with permission error
                result = read_file_content("test.py")
                assert 'error' in result
                assert 'permission' in result['error'].lower() or 'access denied' in result['error'].lower()
                
                # Test AST parsing with permission error
                result = parse_module_ast("test.py")
                assert 'error' in result
                assert 'permission' in result['error'].lower() or 'access denied' in result['error'].lower()
    
    def test_configuration_error_recovery(self, temp_project_dir):
        """Test recovery from configuration-related errors"""
        # Test with invalid configuration
        invalid_config = MCPServerConfig()
        invalid_config.max_file_size = -1  # Invalid size
        
        with patch('config.config', invalid_config):
            with patch('config.config.project_root', temp_project_dir):
                # Functions should handle invalid config gracefully
                python_files = find_python_files()
                assert isinstance(python_files, list)
                
                # File reading should work despite invalid max_file_size
                test_file = temp_project_dir / "test.py"
                test_file.write_text("print('hello')")
                
                result = read_file_content("test.py")
                # Should either work or fail gracefully
                assert 'content' in result or 'error' in result


class TestPerformanceIntegration:
    """Test performance aspects of integrated functionality"""
    
    def test_large_project_analysis(self, temp_project_dir):
        """Test analysis performance with larger projects"""
        # Create a larger project structure
        for i in range(20):
            dir_path = temp_project_dir / f"subdir_{i}"
            dir_path.mkdir()
            
            for j in range(5):
                file_path = dir_path / f"module_{j}.py"
                content = f"""
def function_{j}():
    '''Function {j} in subdir {i}'''
    return {j}

class Class_{j}:
    '''Class {j} in subdir {i}'''
    def method_{j}(self):
        return {j}
"""
                file_path.write_text(content)
        
        with patch('config.config.project_root', temp_project_dir):
            # Test that operations complete in reasonable time
            python_files = find_python_files()
            assert len(python_files) >= 100  # 20 dirs * 5 files each
            
            # Test project info with many files
            project_info = get_project_info()
            assert project_info['file_counts']['python'] >= 100
    
    def test_concurrent_operations(self, temp_project_dir):
        """Test that operations can be performed concurrently"""
        with patch('config.config.project_root', temp_project_dir):
            # Simulate concurrent operations (as might happen in server)
            import threading
            results = []
            errors = []
            
            def analyze_files():
                try:
                    result = find_python_files()
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            def get_info():
                try:
                    result = get_project_info()
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Start concurrent operations
            threads = [
                threading.Thread(target=analyze_files),
                threading.Thread(target=get_info),
                threading.Thread(target=analyze_files),
            ]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Check that all operations completed successfully
            assert len(errors) == 0
            assert len(results) == 3


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""
    
    def test_typical_code_exploration_session(self, temp_project_dir):
        """Test a typical code exploration session"""
        with patch('config.config.project_root', temp_project_dir):
            # 1. User starts by getting project overview
            project_info = get_project_info()
            assert 'main_directories' in project_info
            assert len(project_info['main_directories']) > 0
            
            # 2. User explores main source directory
            src_files = find_python_files(temp_project_dir / "src")
            assert len(src_files) > 0
            
            # 3. User examines specific file
            target_file = src_files[0]
            file_content = read_file_content(target_file['path'])
            assert 'content' in file_content
            
            # 4. User analyzes AST of the file
            ast_info = parse_module_ast(target_file['path'])
            assert 'functions' in ast_info or 'classes' in ast_info
            
            # 5. User looks for testing opportunities
            test_suggestions = suggest_test_cases(target_file['path'])
            assert 'test_suggestions' in test_suggestions
    
    def test_test_generation_workflow(self, temp_project_dir):
        """Test a typical test generation workflow"""
        with patch('config.config.project_root', temp_project_dir):
            # 1. Analyze existing test coverage
            test_analysis = analyze_test_files()
            initial_test_count = test_analysis['total_test_files']
            
            # 2. Find source files that need testing
            python_files = find_python_files()
            source_files = [f for f in python_files if not f['is_test']]
            assert len(source_files) > 0
            
            # 3. Generate test suggestions for each source file
            all_suggestions = []
            for source_file in source_files:
                suggestions = suggest_test_cases(source_file['path'])
                if 'test_suggestions' in suggestions:
                    all_suggestions.extend(suggestions['test_suggestions'])
            
            assert len(all_suggestions) > 0
            
            # 4. Verify suggestions have proper structure
            for suggestion in all_suggestions:
                assert 'test_name' in suggestion
                assert 'test_type' in suggestion
                assert 'priority' in suggestion


class TestBackwardsCompatibility:
    """Test backwards compatibility and version stability"""
    
    def test_function_signatures_stable(self):
        """Test that main function signatures are stable"""
        # Test that key functions have expected signatures
        import inspect
        
        # Test parse_module_ast signature
        sig = inspect.signature(parse_module_ast)
        assert 'file_path' in sig.parameters
        
        # Test find_python_files signature
        sig = inspect.signature(find_python_files)
        assert 'directory' in sig.parameters or len(sig.parameters) == 0
        
        # Test read_file_content signature
        sig = inspect.signature(read_file_content)
        assert 'file_path' in sig.parameters
        assert 'max_size' in sig.parameters
    
    def test_output_format_stability(self, temp_python_file):
        """Test that output formats are stable"""
        with patch('config.config.project_root', temp_python_file.parent):
            # Test parse_module_ast output format
            result = parse_module_ast(temp_python_file.name)
            required_keys = ['file_path', 'classes', 'functions', 'imports']
            for key in required_keys:
                assert key in result
            
            # Test find_python_files output format
            files = find_python_files()
            if files:
                file_info = files[0]
                required_keys = ['path', 'size', 'modified', 'is_test']
                for key in required_keys:
                    assert key in file_info
    
    def test_configuration_backwards_compatibility(self):
        """Test that configuration is backwards compatible"""
        config = MCPServerConfig()
        
        # Test that required attributes exist
        required_attrs = [
            'project_root', 'server_name', 'server_version',
            'max_file_size', 'python_extensions', 'text_extensions',
            'ignore_dirs', 'ignore_files'
        ]
        
        for attr in required_attrs:
            assert hasattr(config, attr)
        
        # Test that required methods exist
        required_methods = ['is_ignored_path', 'is_python_file', 'is_text_file', 'get_project_info']
        
        for method in required_methods:
            assert hasattr(config, method)
            assert callable(getattr(config, method))


class TestModuleDependencies:
    """Test module dependencies and imports"""
    
    def test_required_dependencies_available(self):
        """Test that required dependencies are available"""
        # Test that core Python modules are available
        import ast
        import os
        import sys
        import json
        import pathlib
        
        # Test that all required functions can be imported
        from main import (
            parse_module_ast,
            find_python_files,
            read_file_content,
            get_project_info
        )
        
        from config import MCPServerConfig, config
        from run_server import check_dependencies, run_server
    
    def test_optional_dependencies_handling(self):
        """Test handling of optional dependencies"""
        # Test MCP dependency
        try:
            import mcp
            mcp_available = True
        except ImportError:
            mcp_available = False
        
        # Test coverage dependency
        try:
            import coverage
            coverage_available = True
        except ImportError:
            coverage_available = False
        
        # Functions should work regardless of optional dependencies
        config = MCPServerConfig()
        assert config is not None
    
    def test_import_error_resilience(self):
        """Test resilience to import errors"""
        # Test that basic functionality works even with missing optional imports
        config = MCPServerConfig()
        
        # Core functionality should work
        assert config.is_python_file(Path("test.py"))
        assert config.is_ignored_path(Path(".git"))
        assert isinstance(config.get_project_info(), dict) 