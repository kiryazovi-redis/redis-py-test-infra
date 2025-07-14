#!/usr/bin/env python3
"""
Comprehensive unit tests for AST parsing functions in main.py
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    get_ast_from_file,
    extract_docstring,
    get_type_annotation,
    extract_function_info,
    extract_class_info,
    parse_module_ast,
    get_function_details,
    get_class_details,
    find_imports_in_file,
    get_type_hints_from_file
)


class TestGetASTFromFile:
    """Test the get_ast_from_file function"""
    
    def test_valid_python_file(self, temp_python_file):
        """Test parsing a valid Python file"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_ast_from_file(temp_python_file.name)
            assert isinstance(result, ast.Module)
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file"""
        with patch('config.config.project_root', Path("/tmp")):
            result = get_ast_from_file("nonexistent.py")
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'File not found' in result['error']
    
    def test_non_python_file(self, temp_project_dir):
        """Test handling of non-Python file"""
        text_file = temp_project_dir / "test.txt"
        text_file.write_text("This is not Python code")
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_ast_from_file("test.txt")
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'Not a Python file' in result['error']
    
    def test_syntax_error_file(self, invalid_python_file):
        """Test handling of file with syntax errors"""
        with patch('config.config.project_root', invalid_python_file.parent):
            result = get_ast_from_file(invalid_python_file.name)
            assert isinstance(result, dict)
            assert 'error' in result
            assert 'Syntax error' in result['error']
    
    def test_file_read_error(self, temp_project_dir):
        """Test handling of file read errors"""
        python_file = temp_project_dir / "test.py"
        python_file.write_text("print('hello')")
        
        with patch('config.config.project_root', temp_project_dir):
            with patch('builtins.open', side_effect=IOError("Permission denied")):
                result = get_ast_from_file("test.py")
                assert isinstance(result, dict)
                assert 'error' in result
                assert 'Error parsing' in result['error']


class TestExtractDocstring:
    """Test the extract_docstring function"""
    
    def test_function_docstring(self, sample_python_file):
        """Test extracting docstring from function"""
        tree = ast.parse(sample_python_file)
        
        # Find the simple_function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "simple_function":
                docstring = extract_docstring(node)
                assert docstring == "Add two numbers"
                break
        else:
            pytest.fail("simple_function not found")
    
    def test_class_docstring(self, sample_python_file):
        """Test extracting docstring from class"""
        tree = ast.parse(sample_python_file)
        
        # Find the SampleClass
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SampleClass":
                docstring = extract_docstring(node)
                assert docstring == "A sample class for testing"
                break
        else:
            pytest.fail("SampleClass not found")
    
    def test_module_docstring(self, sample_python_file):
        """Test extracting docstring from module"""
        tree = ast.parse(sample_python_file)
        docstring = extract_docstring(tree)
        assert docstring == "Sample module for testing AST parsing"
    
    def test_no_docstring(self):
        """Test handling of nodes without docstring"""
        code = "def no_docstring(): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        docstring = extract_docstring(func_node)
        assert docstring is None
    
    def test_empty_body(self):
        """Test handling of nodes with empty body"""
        code = "if True: pass"
        tree = ast.parse(code)
        if_node = tree.body[0]
        docstring = extract_docstring(if_node)
        assert docstring is None


class TestGetTypeAnnotation:
    """Test the get_type_annotation function"""
    
    def test_simple_type(self):
        """Test simple type annotation"""
        code = "def func(x: int): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        arg_annotation = func_node.args.args[0].annotation
        result = get_type_annotation(arg_annotation)
        assert result == "int"
    
    def test_complex_type(self):
        """Test complex type annotation"""
        code = "def func(x: List[Dict[str, Optional[int]]]): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        arg_annotation = func_node.args.args[0].annotation
        result = get_type_annotation(arg_annotation)
        assert result == "List[Dict[str, Optional[int]]]"
    
    def test_none_annotation(self):
        """Test None annotation"""
        result = get_type_annotation(None)
        assert result is None
    
    def test_unparseable_annotation(self):
        """Test handling of unparseable annotations"""
        # Create a mock annotation that will fail to unparse
        mock_annotation = MagicMock()
        
        with patch('ast.unparse', side_effect=Exception("Unparseable")):
            result = get_type_annotation(mock_annotation)
            assert result is None


class TestExtractFunctionInfo:
    """Test the extract_function_info function"""
    
    def test_simple_function(self, sample_python_file):
        """Test extracting info from simple function"""
        tree = ast.parse(sample_python_file)
        
        # Find the simple_function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "simple_function":
                info = extract_function_info(node)
                assert info['name'] == "simple_function"
                assert info['type'] == "function"
                assert info['docstring'] == "Add two numbers"
                assert info['return_type'] == "int"
                assert len(info['parameters']) == 2
                assert info['parameters'][0]['name'] == "x"
                assert info['parameters'][0]['type'] == "int"
                assert info['parameters'][1]['name'] == "y"
                assert info['parameters'][1]['type'] == "int"
                break
        else:
            pytest.fail("simple_function not found")
    
    def test_async_function(self, sample_python_file):
        """Test extracting info from async function"""
        tree = ast.parse(sample_python_file)
        
        # Find the async_function
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "async_function":
                info = extract_function_info(node)
                assert info['name'] == "async_function"
                assert info['type'] == "async_function"
                assert info['return_type'] == "Dict[str, int]"
                assert len(info['parameters']) == 1
                assert info['parameters'][0]['name'] == "data"
                assert info['parameters'][0]['type'] == "List[str]"
                break
        else:
            pytest.fail("async_function not found")
    
    def test_function_with_defaults(self, sample_python_file):
        """Test extracting info from function with default parameters"""
        tree = ast.parse(sample_python_file)
        
        # Find the function_with_defaults
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "function_with_defaults":
                info = extract_function_info(node)
                assert info['name'] == "function_with_defaults"
                assert len(info['parameters']) == 4  # name, age, *args, **kwargs
                
                # Check default value
                age_param = next(p for p in info['parameters'] if p['name'] == 'age')
                assert age_param['default'] == "25"
                
                # Check *args
                args_param = next(p for p in info['parameters'] if p['name'] == 'args')
                assert args_param['kind'] == 'vararg'
                
                # Check **kwargs
                kwargs_param = next(p for p in info['parameters'] if p['name'] == 'kwargs')
                assert kwargs_param['kind'] == 'kwarg'
                break
        else:
            pytest.fail("function_with_defaults not found")
    
    def test_function_with_decorators(self, sample_python_file):
        """Test extracting info from function with decorators"""
        tree = ast.parse(sample_python_file)
        
        # Find a method with @property decorator
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "display_name":
                info = extract_function_info(node)
                assert info['name'] == "display_name"
                assert len(info['decorators']) == 1
                assert info['decorators'][0] == "property"
                break
        else:
            pytest.fail("display_name method not found")


class TestExtractClassInfo:
    """Test the extract_class_info function"""
    
    def test_class_basic_info(self, sample_python_file):
        """Test extracting basic class information"""
        tree = ast.parse(sample_python_file)
        
        # Find the SampleClass
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SampleClass":
                info = extract_class_info(node)
                assert info['name'] == "SampleClass"
                assert info['type'] == "class"
                assert info['docstring'] == "A sample class for testing"
                assert info['base_classes'] == []
                break
        else:
            pytest.fail("SampleClass not found")
    
    def test_class_methods_and_properties(self, sample_python_file):
        """Test extracting methods and properties from class"""
        tree = ast.parse(sample_python_file)
        
        # Find the SampleClass
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SampleClass":
                info = extract_class_info(node)
                
                # Check methods
                method_names = [m['name'] for m in info['methods']]
                assert '__init__' in method_names
                assert 'greet' in method_names
                assert 'async_method' in method_names
                assert '_private_method' in method_names
                
                # Check properties
                property_names = [p['name'] for p in info['properties']]
                assert 'display_name' in property_names
                
                # Check visibility
                private_method = next(m for m in info['methods'] if m['name'] == '_private_method')
                assert private_method['visibility'] == 'private'
                
                public_method = next(m for m in info['methods'] if m['name'] == 'greet')
                assert public_method['visibility'] == 'public'
                break
        else:
            pytest.fail("SampleClass not found")
    
    def test_class_variables(self, sample_python_file):
        """Test extracting class variables"""
        tree = ast.parse(sample_python_file)
        
        # Find the SampleClass
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SampleClass":
                info = extract_class_info(node)
                
                # Check class variables
                var_names = [v['name'] for v in info['class_variables']]
                assert 'class_var' in var_names
                
                class_var = next(v for v in info['class_variables'] if v['name'] == 'class_var')
                assert class_var['type'] == 'int'
                break
        else:
            pytest.fail("SampleClass not found")


class TestParseModuleAST:
    """Test the parse_module_ast function"""
    
    def test_valid_module(self, temp_python_file):
        """Test parsing a valid module"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = parse_module_ast(temp_python_file.name)
            
            assert 'error' not in result
            assert result['file_path'] == temp_python_file.name
            assert result['docstring'] == "Sample module for testing AST parsing"
            assert len(result['classes']) > 0
            assert len(result['functions']) > 0
            assert len(result['imports']) > 0
    
    def test_invalid_module(self, invalid_python_file):
        """Test parsing an invalid module"""
        with patch('config.config.project_root', invalid_python_file.parent):
            result = parse_module_ast(invalid_python_file.name)
            
            assert 'error' in result
            assert 'Syntax error' in result['error']
    
    def test_module_imports(self, temp_python_file):
        """Test that imports are correctly identified"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = parse_module_ast(temp_python_file.name)
            
            imports = result['imports']
            import_modules = [imp['module'] for imp in imports if imp['type'] == 'import']
            from_imports = [imp['module'] for imp in imports if imp['type'] == 'from_import']
            
            assert 'os' in import_modules
            assert 'sys' in import_modules
            assert 'typing' in from_imports
            assert 'pathlib' in from_imports


class TestGetFunctionDetails:
    """Test the get_function_details function"""
    
    def test_existing_function(self, temp_python_file):
        """Test getting details for existing function"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_function_details(temp_python_file.name, "simple_function")
            
            assert 'error' not in result
            assert result['name'] == "simple_function"
            assert result['type'] == "function"
            assert result['docstring'] == "Add two numbers"
    
    def test_nonexistent_function(self, temp_python_file):
        """Test getting details for non-existent function"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_function_details(temp_python_file.name, "nonexistent_function")
            
            assert 'error' in result
            assert 'not found' in result['error']
    
    def test_invalid_file(self):
        """Test getting function details from invalid file"""
        with patch('config.config.project_root', Path("/tmp")):
            result = get_function_details("nonexistent.py", "some_function")
            
            assert 'error' in result


class TestGetClassDetails:
    """Test the get_class_details function"""
    
    def test_existing_class(self, temp_python_file):
        """Test getting details for existing class"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_class_details(temp_python_file.name, "SampleClass")
            
            assert 'error' not in result
            assert result['name'] == "SampleClass"
            assert result['type'] == "class"
            assert result['docstring'] == "A sample class for testing"
    
    def test_nonexistent_class(self, temp_python_file):
        """Test getting details for non-existent class"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_class_details(temp_python_file.name, "NonexistentClass")
            
            assert 'error' in result
            assert 'not found' in result['error']
    
    def test_invalid_file(self):
        """Test getting class details from invalid file"""
        with patch('config.config.project_root', Path("/tmp")):
            result = get_class_details("nonexistent.py", "SomeClass")
            
            assert 'error' in result


class TestFindImportsInFile:
    """Test the find_imports_in_file function"""
    
    def test_find_imports(self, temp_python_file):
        """Test finding imports in a file"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = find_imports_in_file(temp_python_file.name)
            
            assert 'error' not in result
            assert result['file_path'] == temp_python_file.name
            assert result['total_imports'] > 0
            assert len(result['imports']) > 0
            
            # Check that we have both import and from_import types
            import_types = [imp['type'] for imp in result['imports']]
            assert 'import' in import_types
            assert 'from_import' in import_types
    
    def test_imports_structure(self, temp_python_file):
        """Test the structure of import information"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = find_imports_in_file(temp_python_file.name)
            
            imports = result['imports']
            for imp in imports:
                assert 'type' in imp
                assert 'module' in imp
                assert 'line_number' in imp
                assert imp['type'] in ['import', 'from_import']
                
                if imp['type'] == 'from_import':
                    assert 'name' in imp
    
    def test_invalid_file_imports(self):
        """Test finding imports in invalid file"""
        with patch('config.config.project_root', Path("/tmp")):
            result = find_imports_in_file("nonexistent.py")
            
            assert 'error' in result


class TestGetTypeHintsFromFile:
    """Test the get_type_hints_from_file function"""
    
    def test_find_type_hints(self, temp_python_file):
        """Test finding type hints in a file"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_type_hints_from_file(temp_python_file.name)
            
            assert 'error' not in result
            assert result['file_path'] == temp_python_file.name
            assert len(result['functions']) > 0
            assert len(result['classes']) > 0
            assert len(result['variables']) > 0
    
    def test_function_type_hints(self, temp_python_file):
        """Test function type hints extraction"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_type_hints_from_file(temp_python_file.name)
            
            functions = result['functions']
            simple_func = next((f for f in functions if f['name'] == 'simple_function'), None)
            assert simple_func is not None
            assert simple_func['return_type'] == 'int'
            
            # Check parameter types
            params = simple_func['parameters']
            assert len(params) == 2
            assert params[0]['type'] == 'int'
            assert params[1]['type'] == 'int'
    
    def test_class_type_hints(self, temp_python_file):
        """Test class type hints extraction"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = get_type_hints_from_file(temp_python_file.name)
            
            classes = result['classes']
            sample_class = next((c for c in classes if c['name'] == 'SampleClass'), None)
            assert sample_class is not None
            
            # Check class variables with type hints
            class_vars = sample_class['class_variables']
            class_var = next((v for v in class_vars if v['name'] == 'class_var'), None)
            assert class_var is not None
            assert class_var['type'] == 'int'
    
    def test_invalid_file_type_hints(self):
        """Test getting type hints from invalid file"""
        with patch('config.config.project_root', Path("/tmp")):
            result = get_type_hints_from_file("nonexistent.py")
            
            assert 'error' in result 