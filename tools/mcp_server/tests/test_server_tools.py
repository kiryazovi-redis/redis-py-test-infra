#!/usr/bin/env python3
"""
Comprehensive unit tests for MCP server tools and handlers in main.py
"""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP types for testing
try:
    import mcp.types as types
    from mcp.server import Server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    
    # Mock MCP types for testing
    class MockTypes:
        class Tool:
            pass
        class TextContent:
            def __init__(self, type, text):
                self.type = type
                self.text = text

    types = MockTypes()

from main import (
    server,
    handle_list_tools,
    handle_call_tool
)


@pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
class TestServerTools:
    """Test the MCP server tools functionality"""
    
    @pytest.mark.asyncio
    async def test_handle_list_tools(self):
        """Test the handle_list_tools function"""
        result = await handle_list_tools()
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that all expected tools are present
        tool_names = [tool.name for tool in result]
        expected_tools = [
            'parse_module',
            'get_function_details',
            'get_class_details',
            'find_imports',
            'get_type_hints',
            'find_python_files',
            'read_file',
            'get_directory_structure',
            'get_project_info',
            'analyze_test_files',
            'get_test_patterns',
            'find_untested_code',
            'suggest_test_cases',
            'get_test_coverage_info'
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    @pytest.mark.asyncio
    async def test_handle_list_tools_structure(self):
        """Test that handle_list_tools returns properly structured tools"""
        result = await handle_list_tools()
        
        for tool in result:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            assert isinstance(tool.inputSchema, dict)
            assert 'type' in tool.inputSchema
            assert tool.inputSchema['type'] == 'object'
    
    def test_server_instance(self):
        """Test that server instance is properly created"""
        assert server is not None
        assert server.name == "redis-py-test-infra"


@pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
class TestHandleCallTool:
    """Test the handle_call_tool function"""
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_parse_module(self, temp_python_file):
        """Test calling parse_module tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("parse_module", {"file_path": temp_python_file.name})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            assert result[0].type == "text"
            
            # Parse the JSON response
            response_data = json.loads(result[0].text)
            assert 'file_path' in response_data
            assert 'classes' in response_data
            assert 'functions' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_function_details(self, temp_python_file):
        """Test calling get_function_details tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("get_function_details", {
                "file_path": temp_python_file.name,
                "function_name": "simple_function"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'name' in response_data
            assert response_data['name'] == "simple_function"
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_class_details(self, temp_python_file):
        """Test calling get_class_details tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("get_class_details", {
                "file_path": temp_python_file.name,
                "class_name": "SampleClass"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'name' in response_data
            assert response_data['name'] == "SampleClass"
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_find_imports(self, temp_python_file):
        """Test calling find_imports tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("find_imports", {
                "file_path": temp_python_file.name
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'imports' in response_data
            assert 'total_imports' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_type_hints(self, temp_python_file):
        """Test calling get_type_hints tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("get_type_hints", {
                "file_path": temp_python_file.name
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'functions' in response_data
            assert 'classes' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_find_python_files(self, temp_project_dir):
        """Test calling find_python_files tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("find_python_files", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert isinstance(response_data, list)
            assert len(response_data) > 0
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_find_python_files_with_directory(self, temp_project_dir):
        """Test calling find_python_files tool with directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("find_python_files", {
                "directory": "src"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert isinstance(response_data, list)
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_read_file(self, temp_python_file):
        """Test calling read_file tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("read_file", {
                "file_path": temp_python_file.name
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'content' in response_data
            assert 'size' in response_data
            assert 'lines' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_read_file_with_max_size(self, temp_python_file):
        """Test calling read_file tool with max_size"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("read_file", {
                "file_path": temp_python_file.name,
                "max_size": 100
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'content' in response_data
            assert len(response_data['content']) <= 100
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_directory_structure(self, temp_project_dir):
        """Test calling get_directory_structure tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_directory_structure", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'type' in response_data
            assert response_data['type'] == 'directory'
            assert 'children' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_directory_structure_with_options(self, temp_project_dir):
        """Test calling get_directory_structure tool with options"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_directory_structure", {
                "directory": "src",
                "max_depth": 2
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'type' in response_data
            assert response_data['type'] == 'directory'
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_project_info(self, temp_project_dir):
        """Test calling get_project_info tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_project_info", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'name' in response_data
            assert 'version' in response_data
            assert 'project_root' in response_data
            assert 'file_counts' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_analyze_test_files(self, temp_project_dir):
        """Test calling analyze_test_files tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("analyze_test_files", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'total_test_files' in response_data
            assert 'test_classes' in response_data
            assert 'test_functions' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_analyze_test_files_with_directory(self, temp_project_dir):
        """Test calling analyze_test_files tool with directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("analyze_test_files", {
                "directory": "tests"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'total_test_files' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_test_patterns(self, temp_project_dir):
        """Test calling get_test_patterns tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_test_patterns", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'testing_frameworks' in response_data
            assert 'common_fixtures' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_find_untested_code(self, temp_project_dir):
        """Test calling find_untested_code tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("find_untested_code", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'analysis_summary' in response_data
            assert 'untested_functions' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_find_untested_code_with_directories(self, temp_project_dir):
        """Test calling find_untested_code tool with directories"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("find_untested_code", {
                "source_dir": "src",
                "test_dir": "tests"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'analysis_summary' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_suggest_test_cases(self, temp_python_file):
        """Test calling suggest_test_cases tool"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("suggest_test_cases", {
                "file_path": temp_python_file.name
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'test_suggestions' in response_data
            assert 'recommended_framework' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_suggest_test_cases_with_options(self, temp_python_file):
        """Test calling suggest_test_cases tool with options"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = await handle_call_tool("suggest_test_cases", {
                "file_path": temp_python_file.name,
                "function_name": "simple_function",
                "framework": "pytest"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            response_data = json.loads(result[0].text)
            assert 'test_suggestions' in response_data
            assert response_data['recommended_framework'] == "pytest"
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_test_coverage_info(self, temp_project_dir):
        """Test calling get_test_coverage_info tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_test_coverage_info", {})
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            # This should return an error since no coverage file exists
            response_data = json.loads(result[0].text)
            assert 'error' in response_data
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_get_test_coverage_info_with_file(self, temp_project_dir):
        """Test calling get_test_coverage_info tool with custom file"""
        coverage_file = temp_project_dir / "custom_coverage"
        coverage_file.write_text("mock coverage")
        
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("get_test_coverage_info", {
                "coverage_file": str(coverage_file)
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_unknown_tool(self):
        """Test calling unknown tool"""
        result = await handle_call_tool("unknown_tool", {})
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Unknown tool" in result[0].text
    
    @pytest.mark.asyncio
    async def test_handle_call_tool_error_handling(self, temp_project_dir):
        """Test error handling in handle_call_tool"""
        with patch('config.config.project_root', temp_project_dir):
            result = await handle_call_tool("parse_module", {
                "file_path": "nonexistent.py"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            # Should contain error information
            response_data = json.loads(result[0].text)
            assert 'error' in response_data


class TestServerToolsWithoutMCP:
    """Test server tools functionality without MCP dependency"""
    
    @pytest.mark.skipif(HAS_MCP, reason="Testing without MCP")
    def test_server_tools_without_mcp(self):
        """Test that server tools can be imported without MCP"""
        # This test ensures the module can be imported even without MCP
        # The actual server functionality would need MCP to work
        assert True  # If we get here, import worked
    
    def test_mock_types_structure(self):
        """Test that mock types have expected structure"""
        if not HAS_MCP:
            content = types.TextContent("text", "test content")
            assert content.type == "text"
            assert content.text == "test content"


class TestServerToolsEdgeCases:
    """Test edge cases and error conditions for server tools"""
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_with_missing_arguments(self):
        """Test calling tools with missing required arguments"""
        result = await handle_call_tool("get_function_details", {
            "file_path": "test.py"
            # Missing function_name
        })
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        
        response_data = json.loads(result[0].text)
        assert 'error' in response_data
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_with_invalid_arguments(self):
        """Test calling tools with invalid arguments"""
        result = await handle_call_tool("read_file", {
            "file_path": "test.py",
            "max_size": "invalid"  # Should be integer
        })
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        
        # Should handle the error gracefully
        response_text = result[0].text
        assert isinstance(response_text, str)
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_with_empty_arguments(self):
        """Test calling tools with empty arguments"""
        result = await handle_call_tool("find_python_files", {})
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        
        # Should work with default values
        response_data = json.loads(result[0].text)
        assert isinstance(response_data, list)
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_json_serialization_error(self):
        """Test handling of JSON serialization errors"""
        with patch('main.parse_module_ast', return_value={'circular': None}):
            # Create a circular reference that can't be serialized
            data = {'test': 'value'}
            data['circular'] = data
            
            with patch('main.parse_module_ast', return_value=data):
                result = await handle_call_tool("parse_module", {
                    "file_path": "test.py"
                })
                
                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], types.TextContent)
                
                # Should handle serialization error gracefully
                response_text = result[0].text
                assert isinstance(response_text, str)
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_exception_handling(self):
        """Test handling of exceptions in tool calls"""
        with patch('main.parse_module_ast', side_effect=Exception("Test exception")):
            result = await handle_call_tool("parse_module", {
                "file_path": "test.py"
            })
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], types.TextContent)
            
            # Should handle exception gracefully
            response_text = result[0].text
            assert isinstance(response_text, str)
    
    @pytest.mark.skipif(not HAS_MCP, reason="MCP not available for testing")
    @pytest.mark.asyncio
    async def test_handle_call_tool_with_none_arguments(self):
        """Test calling tools with None arguments"""
        result = await handle_call_tool("get_directory_structure", {
            "directory": None,
            "max_depth": None
        })
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        
        # Should handle None values gracefully
        response_data = json.loads(result[0].text)
        assert 'type' in response_data or 'error' in response_data 