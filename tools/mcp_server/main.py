#!/usr/bin/env python3
"""
MCP Server for Redis-py Test Infrastructure

This MCP server provides tools to help with test generation and code exploration
for the redis-py client library project.
"""

import os
import sys
import json
import asyncio
import argparse
import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import re # Added for regex in suggest_test_cases

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import configuration
from config import config

# Server instance
server = Server(config.server_name)


# AST Analysis Functions

def get_ast_from_file(file_path: str) -> Union[ast.AST, Dict[str, str]]:
    """Parse a Python file and return its AST or error information."""
    try:
        full_path = config.project_root / file_path
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        if not config.is_python_file(full_path):
            return {'error': f'Not a Python file: {file_path}'}
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return ast.parse(content)
    except SyntaxError as e:
        return {'error': f'Syntax error in {file_path}: {str(e)}'}
    except Exception as e:
        return {'error': f'Error parsing {file_path}: {str(e)}'}


def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and
        node.body and isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def get_type_annotation(annotation: Optional[ast.expr]) -> Optional[str]:
    """Convert type annotation AST node to string."""
    if annotation is None:
        return None
    try:
        return ast.unparse(annotation)
    except:
        return None


def extract_function_info(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
    """Extract detailed information from a function/method AST node."""
    params = []
    for arg in node.args.args:
        param_info = {
            'name': arg.arg,
            'type': get_type_annotation(arg.annotation),
            'kind': 'positional'
        }
        params.append(param_info)
    
    # Handle *args
    if node.args.vararg:
        params.append({
            'name': node.args.vararg.arg,
            'type': get_type_annotation(node.args.vararg.annotation),
            'kind': 'vararg'
        })
    
    # Handle **kwargs
    if node.args.kwarg:
        params.append({
            'name': node.args.kwarg.arg,
            'type': get_type_annotation(node.args.kwarg.annotation),
            'kind': 'kwarg'
        })
    
    # Handle keyword-only arguments
    for arg in node.args.kwonlyargs:
        param_info = {
            'name': arg.arg,
            'type': get_type_annotation(arg.annotation),
            'kind': 'keyword_only'
        }
        params.append(param_info)
    
    # Handle default values
    defaults = node.args.defaults
    if defaults:
        # Map defaults to their corresponding parameters
        default_offset = len(node.args.args) - len(defaults)
        for i, default in enumerate(defaults):
            param_index = default_offset + i
            if param_index < len(params):
                try:
                    params[param_index]['default'] = ast.unparse(default)
                except:
                    params[param_index]['default'] = '<complex default>'
    
    return {
        'name': node.name,
        'type': 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
        'docstring': extract_docstring(node),
        'parameters': params,
        'return_type': get_type_annotation(node.returns),
        'decorators': [ast.unparse(dec) for dec in node.decorator_list],
        'line_number': node.lineno
    }


def extract_class_info(node: ast.ClassDef) -> Dict[str, Any]:
    """Extract detailed information from a class AST node."""
    methods = []
    properties = []
    class_variables = []
    
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_info = extract_function_info(item)
            method_info['visibility'] = 'private' if item.name.startswith('_') else 'public'
            
            # Check if it's a property
            is_property = any(
                (isinstance(dec, ast.Name) and dec.id == 'property') or
                (isinstance(dec, ast.Attribute) and dec.attr == 'property')
                for dec in item.decorator_list
            )
            
            if is_property:
                properties.append(method_info)
            else:
                methods.append(method_info)
                
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            # Class variable with type annotation
            class_variables.append({
                'name': item.target.id,
                'type': get_type_annotation(item.annotation),
                'line_number': item.lineno
            })
        elif isinstance(item, ast.Assign):
            # Class variable without type annotation
            for target in item.targets:
                if isinstance(target, ast.Name):
                    class_variables.append({
                        'name': target.id,
                        'type': None,
                        'line_number': item.lineno
                    })
    
    # Extract base classes
    base_classes = []
    for base in node.bases:
        try:
            base_classes.append(ast.unparse(base))
        except:
            base_classes.append('<complex base>')
    
    return {
        'name': node.name,
        'type': 'class',
        'docstring': extract_docstring(node),
        'base_classes': base_classes,
        'methods': methods,
        'properties': properties,
        'class_variables': class_variables,
        'decorators': [ast.unparse(dec) for dec in node.decorator_list],
        'line_number': node.lineno
    }


def parse_module_ast(file_path: str) -> Dict[str, Any]:
    """Parse a Python module and extract all classes and functions."""
    tree = get_ast_from_file(file_path)
    if isinstance(tree, dict):  # Error occurred
        return tree
    
    module_info = {
        'file_path': file_path,
        'docstring': extract_docstring(tree),
        'classes': [],
        'functions': [],
        'imports': [],
        'variables': []
    }
    
        # First, collect all top-level nodes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            module_info['classes'].append(extract_class_info(node))
            
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_info['functions'].append(extract_function_info(node))
    
    # Then walk the entire tree for imports and variables
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_info['imports'].append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'line_number': node.lineno
                })
                
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                module_info['imports'].append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line_number': node.lineno
                })
    
    return module_info


def get_function_details(file_path: str, function_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific function."""
    tree = get_ast_from_file(file_path)
    if isinstance(tree, dict):  # Error occurred
        return tree
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return extract_function_info(node)
    
    return {'error': f'Function "{function_name}" not found in {file_path}'}


def get_class_details(file_path: str, class_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific class."""
    tree = get_ast_from_file(file_path)
    if isinstance(tree, dict):  # Error occurred
        return tree
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return extract_class_info(node)
    
    return {'error': f'Class "{class_name}" not found in {file_path}'}


def find_imports_in_file(file_path: str) -> Dict[str, Any]:
    """Find all imports in a Python file."""
    tree = get_ast_from_file(file_path)
    if isinstance(tree, dict):  # Error occurred
        return tree
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'line_number': node.lineno
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line_number': node.lineno
                })
    
    return {
        'file_path': file_path,
        'imports': imports,
        'total_imports': len(imports)
    }


def get_type_hints_from_file(file_path: str) -> Dict[str, Any]:
    """Extract type hints from a Python file."""
    tree = get_ast_from_file(file_path)
    if isinstance(tree, dict):  # Error occurred
        return tree
    
    functions = []
    classes = []
    variables = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                'name': node.name,
                'return_type': get_type_annotation(node.returns),
                'parameters': []
            }
            
            for arg in node.args.args:
                if arg.annotation:
                    func_info['parameters'].append({
                        'name': arg.arg,
                        'type': get_type_annotation(arg.annotation)
                    })
            
            if node.args.vararg and node.args.vararg.annotation:
                func_info['vararg_type'] = get_type_annotation(node.args.vararg.annotation)
            
            if node.args.kwarg and node.args.kwarg.annotation:
                func_info['kwarg_type'] = get_type_annotation(node.args.kwarg.annotation)
            
            if func_info['return_type'] or func_info['parameters']:
                functions.append(func_info)
        
        elif isinstance(node, ast.ClassDef):
            # Extract class variables with type hints
            class_variables = []
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    class_variables.append({
                        'name': child.target.id,
                        'type': get_type_annotation(child.annotation),
                        'line_number': child.lineno
                    })
            
            # Include all classes, not just those with base classes
            classes.append({
                'name': node.name,
                'base_classes': [ast.unparse(base) for base in node.bases] if node.bases else [],
                'class_variables': class_variables
            })
        
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            variables.append({
                'name': node.target.id,
                'type': get_type_annotation(node.annotation),
                'line_number': node.lineno
            })
    
    return {
        'file_path': file_path,
        'functions': functions,
        'classes': classes,
        'variables': variables
    }


# Test Analysis Functions

def find_test_files(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Find all test files in the project."""
    if directory is None:
        directory = config.project_root
    
    test_files = []
    test_patterns = ['test_*.py', '*_test.py', 'tests.py']
    
    try:
        for pattern in test_patterns:
            for path in directory.rglob(pattern):
                if path.is_file() and not is_ignored_path(path):
                    rel_path = get_relative_path(path)
                    stat = path.stat()
                    test_files.append({
                        'path': rel_path,
                        'name': path.name,
                        'size': stat.st_size,
                        'directory': get_relative_path(path.parent),
                        'modified': stat.st_mtime,
                        'is_test': True
                    })
    except Exception as e:
        print(f"Error finding test files: {e}", file=sys.stderr)
    
    return sorted(test_files, key=lambda x: x['path'])


def get_relative_path(path: Path) -> str:
    """Get path relative to project root."""
    try:
        return str(path.relative_to(config.project_root))
    except ValueError:
        return str(path)


def is_ignored_path(path: Path) -> bool:
    """Check if path should be ignored."""
    return config.is_ignored_path(path)


def find_python_files(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Find all Python files in the project."""
    if directory is None:
        directory = config.project_root
    
    python_files = []
    
    try:
        for path in directory.rglob('*'):
            if path.is_file() and config.is_python_file(path):
                if not is_ignored_path(path):
                    rel_path = get_relative_path(path)
                    stat = path.stat()
                    python_files.append({
                        'path': rel_path,
                        'name': path.name,
                        'size': stat.st_size,
                        'directory': get_relative_path(path.parent),
                        'modified': stat.st_mtime,
                        'is_test': config.is_test_file(path)
                    })
    except Exception as e:
        print(f"Error finding Python files: {e}", file=sys.stderr)
    
    return sorted(python_files, key=lambda x: x['path'])


def read_file_content(file_path: str, max_size: int = None) -> Dict[str, Any]:
    """Read file content with size limits."""
    if max_size is None:
        max_size = config.max_file_size
    
    try:
        full_path = config.project_root / file_path
        
        # Check if file should be ignored
        if config.is_ignored_path(full_path):
            return {'error': f'File is ignored: {file_path}'}
        
        # Check if file exists and is actually a file
        if not full_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        if not full_path.is_file():
            return {'error': f'Path is not a file: {file_path}'}
        
        file_size = full_path.stat().st_size
        truncated = file_size > max_size
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            if truncated:
                # Read only up to max_size
                content = f.read(max_size)
                # Try to end at a reasonable boundary
                if len(content) == max_size:
                    # Find the last newline to avoid cutting in the middle of a line
                    last_newline = content.rfind('\n')
                    if last_newline > max_size * 0.9:  # Only if we don't lose too much
                        content = content[:last_newline + 1]
            else:
                content = f.read()
        
        return {
            'path': file_path,
            'content': content,
            'size': file_size,
            'lines': len(content.splitlines()),
            'is_python': config.is_python_file(full_path),
            'is_text': config.is_text_file(full_path),
            'truncated': truncated
        }
    except PermissionError:
        return {'error': f'Permission denied: {file_path}'}
    except Exception as e:
        return {'error': f'Error reading file: {str(e)}'}


def get_directory_structure(directory: Optional[Path] = None, max_depth: int = None) -> Dict[str, Any]:
    """Get directory structure as a tree."""
    if directory is None:
        directory = config.project_root
    if max_depth is None:
        max_depth = config.max_directory_depth
    
    # Check if directory exists
    if not directory.exists():
        return None
    
    def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
        if current_depth > max_depth or is_ignored_path(path):
            return None
        
        if not path.exists():
            return None
        
        result = {
            'name': path.name,
            'type': 'directory' if path.is_dir() else 'file',
            'path': get_relative_path(path)
        }
        
        if path.is_dir():
            children = []
            try:
                for child in sorted(path.iterdir()):
                    if not is_ignored_path(child):
                        child_tree = build_tree(child, current_depth + 1)
                        if child_tree:
                            children.append(child_tree)
            except PermissionError:
                pass
            
            result['children'] = children
        else:
            try:
                stat = path.stat()
                result['size'] = stat.st_size
                result['modified'] = stat.st_mtime
                result['is_python'] = config.is_python_file(path)
                result['is_text'] = config.is_text_file(path)
            except (OSError, PermissionError):
                # Handle cases where stat fails
                result['size'] = 0
                result['modified'] = 0
                result['is_python'] = False
                result['is_text'] = False
        
        return result
    
    return build_tree(directory)


def get_project_info() -> Dict[str, Any]:
    """Get comprehensive project information."""
    info = config.get_project_info()
    
    # Count files by type
    file_counts = {'python': 0, 'test': 0, 'doc': 0, 'other': 0}
    total_size = 0
    
    for path in config.project_root.rglob('*'):
        if path.is_file() and not is_ignored_path(path):
            size = path.stat().st_size
            total_size += size
            
            if config.is_python_file(path):
                if config.is_test_file(path):
                    file_counts['test'] += 1
                else:
                    file_counts['python'] += 1
            elif path.suffix in {'.rst', '.md'}:
                file_counts['doc'] += 1
            else:
                file_counts['other'] += 1
    
    info['file_counts'] = file_counts
    info['total_size'] = total_size
    
    # Main directories
    main_dirs = []
    for item in config.project_root.iterdir():
        if item.is_dir() and not is_ignored_path(item):
            main_dirs.append({
                'name': item.name,
                'path': get_relative_path(item)
            })
    
    info['main_directories'] = sorted(main_dirs, key=lambda x: x['name'])
    
    # Read pyproject.toml if it exists
    pyproject_path = config.project_root / 'pyproject.toml'
    if pyproject_path.exists():
        try:
            with open(pyproject_path, 'r') as f:
                content = f.read()
                # Extract basic info from pyproject.toml
                lines = content.splitlines()
                for line in lines:
                    if line.startswith('name = '):
                        info['package_name'] = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('description = '):
                        info['description'] = line.split('=')[1].strip().strip('"\'')
                    elif line.startswith('requires-python = '):
                        info['python_requirement'] = line.split('=')[1].strip().strip('"\'')
        except Exception as e:
            print(f"Error reading pyproject.toml: {e}", file=sys.stderr)
    
    # Key files
    key_files = []
    for filename in ['README.md', 'CONTRIBUTING.md', 'LICENSE', 'pyproject.toml', 'tasks.py']:
        path = config.project_root / filename
        if path.exists():
            key_files.append({
                'name': filename,
                'path': get_relative_path(path),
                'size': path.stat().st_size
            })
    
    info['key_files'] = key_files
    
    # Get file lists
    info['python_files'] = find_python_files()
    info['test_files'] = find_test_files()
    
    # Add total count to file_counts
    info['file_counts']['total'] = sum(file_counts.values())
    
    # Calculate total lines of code
    total_lines = 0
    for file_info in info['python_files']:
        try:
            file_path = config.project_root / file_info['path']
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines += len(f.readlines())
        except Exception:
            pass  # Skip files that can't be read
    
    info['total_lines'] = total_lines
    
    # Configuration information
    info['configuration'] = {
        'max_file_size': config.max_file_size,
        'max_directory_depth': config.max_directory_depth,
        'python_extensions': list(config.python_extensions),
        'text_extensions': list(config.text_extensions),
        'ignore_dirs': list(config.ignore_dirs),
        'ignore_files': list(config.ignore_files),
        'debug': config.debug,
        'log_level': config.log_level
    }
    
    return info


def analyze_test_files(directory: Optional[str] = None) -> Dict[str, Any]:
    """Analyze test files and extract test structure including unittest and pytest patterns."""
    if directory:
        search_dir = config.project_root / directory
        if not search_dir.exists():
            return {'error': f'Directory not found: {directory}'}
    else:
        search_dir = None
    
    test_files = find_test_files(search_dir)
    
    analysis = {
        'total_test_files': len(test_files),
        'test_files': [],
        'test_classes': [],
        'test_functions': [],
        'fixtures': [],
        'markers': [],
        'imports': [],
        'unittest_classes': [],
        'pytest_fixtures': [],
        'setup_teardown_methods': [],
        'assertion_patterns': [],
        'mock_usage': []
    }
    
    for test_file in test_files:
        file_path = test_file['path']
        tree = get_ast_from_file(file_path)
        
        if isinstance(tree, dict):  # Error occurred
            continue
        
        file_analysis = {
            'file_path': file_path,
            'classes': [],
            'functions': [],
            'fixtures': [],
            'markers': [],
            'unittest_classes': [],
            'setup_teardown_methods': [],
            'assertion_patterns': [],
            'mock_usage': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = extract_class_info(node)
                file_analysis['classes'].append(class_info)
                
                # Check if it's a unittest TestCase
                is_unittest_class = any(
                    'TestCase' in base or 'unittest' in base 
                    for base in class_info['base_classes']
                )
                
                test_class_info = {
                    'name': node.name,
                    'file_path': file_path,
                    'line_number': node.lineno,
                    'methods': len(class_info['methods']),
                    'docstring': class_info['docstring'],
                    'framework': 'unittest' if is_unittest_class else 'pytest',
                    'base_classes': class_info['base_classes']
                }
                
                analysis['test_classes'].append(test_class_info)
                
                if is_unittest_class:
                    file_analysis['unittest_classes'].append(test_class_info)
                    analysis['unittest_classes'].append(test_class_info)
                
                # Analyze methods for setup/teardown patterns
                for method in class_info['methods']:
                    method_name = method['name']
                    if method_name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass', 
                                     'setUpModule', 'tearDownModule', 'setup_method', 'teardown_method',
                                     'setup_class', 'teardown_class']:
                        setup_teardown_info = {
                            'name': method_name,
                            'class_name': node.name,
                            'file_path': file_path,
                            'line_number': method.get('line_number', node.lineno),
                            'framework': 'unittest' if method_name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass'] else 'pytest',
                            'scope': 'class' if 'Class' in method_name or 'class' in method_name else 'method'
                        }
                        file_analysis['setup_teardown_methods'].append(setup_teardown_info)
                        analysis['setup_teardown_methods'].append(setup_teardown_info)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = extract_function_info(node)
                file_analysis['functions'].append(func_info)
                
                # Check if it's a test function (pytest or unittest style)
                is_test_func = (node.name.startswith('test_') or 
                               any('test' in dec for dec in func_info['decorators']))
                
                # Check if it's a pytest fixture
                is_fixture = any('fixture' in dec for dec in func_info['decorators'])
                
                if is_fixture:
                    fixture_info = {
                        'name': node.name,
                        'file_path': file_path,
                        'line_number': node.lineno,
                        'scope': 'function',  # default
                        'params': func_info['parameters'],
                        'docstring': func_info['docstring'],
                        'autouse': False
                    }
                    
                    # Extract fixture scope and other parameters from decorators
                    for dec in func_info['decorators']:
                        if 'scope=' in dec:
                            import re
                            scope_match = re.search(r'scope=[\'"](.*?)[\'"]', dec)
                            if scope_match:
                                fixture_info['scope'] = scope_match.group(1)
                        if 'autouse=' in dec:
                            autouse_match = re.search(r'autouse=(True|False)', dec)
                            if autouse_match:
                                fixture_info['autouse'] = autouse_match.group(1) == 'True'
                    
                    file_analysis['fixtures'].append(fixture_info)
                    analysis['fixtures'].append(fixture_info)
                    analysis['pytest_fixtures'].append(fixture_info)
                
                if is_test_func:
                    test_func_info = {
                        'name': node.name,
                        'file_path': file_path,
                        'line_number': node.lineno,
                        'parameters': func_info['parameters'],
                        'docstring': func_info['docstring'],
                        'decorators': func_info['decorators'],
                        'framework': 'pytest' if any('pytest' in dec for dec in func_info['decorators']) else 'unittest'
                    }
                    analysis['test_functions'].append(test_func_info)
            
            # Extract pytest markers
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'mark':
                    marker_info = {
                        'name': getattr(node.func.value, 'id', 'unknown'),
                        'file_path': file_path,
                        'line_number': node.lineno
                    }
                    file_analysis['markers'].append(marker_info)
                    analysis['markers'].append(marker_info)
            
            # Extract assertion patterns
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith('assert') or node.func.attr in ['assertEqual', 'assertTrue', 'assertFalse', 'assertRaises']:
                    assertion_info = {
                        'method': node.func.attr,
                        'file_path': file_path,
                        'line_number': node.lineno,
                        'framework': 'unittest' if node.func.attr.startswith('assert') and len(node.func.attr) > 6 else 'pytest'
                    }
                    file_analysis['assertion_patterns'].append(assertion_info)
                    analysis['assertion_patterns'].append(assertion_info)
            
            # Extract mock usage patterns
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and 'mock' in node.func.attr.lower():
                    mock_info = {
                        'method': node.func.attr,
                        'file_path': file_path,
                        'line_number': node.lineno
                    }
                    file_analysis['mock_usage'].append(mock_info)
                    analysis['mock_usage'].append(mock_info)
                elif isinstance(node.func, ast.Name) and 'Mock' in node.func.id:
                    mock_info = {
                        'method': node.func.id,
                        'file_path': file_path,
                        'line_number': node.lineno
                    }
                    file_analysis['mock_usage'].append(mock_info)
                    analysis['mock_usage'].append(mock_info)
        
        # Extract imports
        imports = find_imports_in_file(file_path)
        if 'imports' in imports:
            file_analysis['imports'] = imports['imports']
            analysis['imports'].extend(imports['imports'])
        
        analysis['test_files'].append(file_analysis)
    
    return analysis


def get_test_patterns(directory: Optional[str] = None) -> Dict[str, Any]:
    """Identify common testing patterns used in the project (pytest, unittest, mocking, etc.)."""
    analysis = analyze_test_files(directory)
    
    if 'error' in analysis:
        return analysis
    
    patterns = {
        'testing_frameworks': set(),
        'framework_usage': {
            'pytest': 0,
            'unittest': 0,
            'mixed': 0
        },
        'common_fixtures': {},
        'common_markers': {},
        'mocking_patterns': [],
        'assertion_patterns': {},
        'setup_patterns': [],
        'parametrization_patterns': [],
        'unittest_patterns': {
            'test_case_classes': len(analysis.get('unittest_classes', [])),
            'setup_teardown_methods': []
        },
        'pytest_patterns': {
            'fixtures': len(analysis.get('pytest_fixtures', [])),
            'markers': len(analysis.get('markers', [])),
            'parametrized_tests': 0
        }
    }
    
    # Analyze imports to detect testing frameworks
    for import_info in analysis['imports']:
        module = import_info.get('module', '')
        if module in ['pytest', 'unittest', 'nose', 'nose2']:
            patterns['testing_frameworks'].add(module)
        elif module.startswith('pytest'):
            patterns['testing_frameworks'].add('pytest')
        elif module.startswith('unittest'):
            patterns['testing_frameworks'].add('unittest')
        elif 'mock' in module.lower():
            patterns['mocking_patterns'].append(import_info)
    
    # Count framework usage
    for test_func in analysis['test_functions']:
        framework = test_func.get('framework', 'unknown')
        if framework == 'pytest':
            patterns['framework_usage']['pytest'] += 1
        elif framework == 'unittest':
            patterns['framework_usage']['unittest'] += 1
    
    # Analyze fixtures
    for fixture in analysis['fixtures']:
        fixture_name = fixture['name']
        if fixture_name not in patterns['common_fixtures']:
            patterns['common_fixtures'][fixture_name] = []
        patterns['common_fixtures'][fixture_name].append(fixture)
    
    # Analyze markers
    for marker in analysis['markers']:
        marker_name = marker['name']
        if marker_name not in patterns['common_markers']:
            patterns['common_markers'][marker_name] = 0
        patterns['common_markers'][marker_name] += 1
    
    # Analyze assertion patterns
    assertion_counts = {}
    for assertion in analysis.get('assertion_patterns', []):
        method = assertion['method']
        framework = assertion['framework']
        key = f"{framework}:{method}"
        assertion_counts[key] = assertion_counts.get(key, 0) + 1
    patterns['assertion_patterns'] = assertion_counts
    
    # Analyze setup/teardown patterns
    for setup_teardown in analysis.get('setup_teardown_methods', []):
        patterns['setup_patterns'].append({
            'method': setup_teardown['name'],
            'class_name': setup_teardown.get('class_name', ''),
            'file_path': setup_teardown['file_path'],
            'framework': setup_teardown['framework'],
            'scope': setup_teardown['scope']
        })
        
        if setup_teardown['framework'] == 'unittest':
            patterns['unittest_patterns']['setup_teardown_methods'].append(setup_teardown)
    
    # Analyze test functions for patterns
    for test_func in analysis['test_functions']:
        decorators = test_func.get('decorators', [])
        
        # Check for parametrization
        for decorator in decorators:
            if 'parametrize' in decorator:
                patterns['parametrization_patterns'].append({
                    'function': test_func['name'],
                    'file_path': test_func['file_path'],
                    'decorator': decorator
                })
                patterns['pytest_patterns']['parametrized_tests'] += 1
        
        # Check for setup patterns (function-level)
        if test_func['name'].startswith('setup_') or test_func['name'].startswith('teardown_'):
            patterns['setup_patterns'].append({
                'function': test_func['name'],
                'file_path': test_func['file_path'],
                'type': 'setup' if test_func['name'].startswith('setup_') else 'teardown',
                'framework': 'pytest',
                'scope': 'function'
            })
    
    # Analyze mock usage
    mock_counts = {}
    for mock_usage in analysis.get('mock_usage', []):
        method = mock_usage['method']
        mock_counts[method] = mock_counts.get(method, 0) + 1
    patterns['mock_usage_summary'] = mock_counts
    
    # Determine if project uses mixed frameworks
    frameworks_used = len(patterns['testing_frameworks'])
    pytest_usage = patterns['framework_usage']['pytest']
    unittest_usage = patterns['framework_usage']['unittest']
    
    if frameworks_used > 1 or (pytest_usage > 0 and unittest_usage > 0):
        patterns['framework_usage']['mixed'] = 1
    
    # Convert sets to lists for JSON serialization
    patterns['testing_frameworks'] = list(patterns['testing_frameworks'])
    
    return patterns


def find_untested_code(source_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compare source files with test files to find untested functions/classes."""
    if source_dir:
        source_path = config.project_root / source_dir
        if not source_path.exists():
            return {'error': f'Source directory not found: {source_dir}'}
    else:
        source_path = config.project_root
    
    if test_dir:
        test_path = config.project_root / test_dir
        if not test_path.exists():
            return {'error': f'Test directory not found: {test_dir}'}
    else:
        test_path = config.project_root / 'tests'
    
    # Find all source files (excluding test files)
    source_files = []
    for path in source_path.rglob('*.py'):
        if path.is_file() and not is_ignored_path(path):
            rel_path = get_relative_path(path)
            # Skip test files
            if not ('test' in rel_path or rel_path.startswith('tests/')):
                source_files.append(rel_path)
    
    # Analyze test files
    test_analysis = analyze_test_files(str(test_path.relative_to(config.project_root)))
    if 'error' in test_analysis:
        return test_analysis
    
    # Extract tested code references from test files
    tested_references = set()
    for test_file in test_analysis['test_files']:
        for import_info in test_file.get('imports', []):
            if import_info['type'] == 'from_import':
                tested_references.add(f"{import_info['module']}.{import_info['name']}")
            elif import_info['type'] == 'import':
                tested_references.add(import_info['module'])
    
    # Analyze source files
    untested_items = {
        'untested_functions': [],
        'untested_classes': [],
        'untested_files': [],
        'analysis_summary': {
            'total_source_files': len(source_files),
            'total_test_files': test_analysis['total_test_files'],
            'tested_references': len(tested_references)
        }
    }
    
    for source_file in source_files:
        module_info = parse_module_ast(source_file)
        if 'error' in module_info:
            continue
        
        # Check if file has any tests
        module_name = source_file.replace('/', '.').replace('.py', '')
        has_tests = any(module_name in ref for ref in tested_references)
        
        if not has_tests:
            untested_items['untested_files'].append({
                'file_path': source_file,
                'functions': len(module_info['functions']),
                'classes': len(module_info['classes'])
            })
        
        # Check functions
        for func in module_info['functions']:
            func_ref = f"{module_name}.{func['name']}"
            if func_ref not in tested_references and not func['name'].startswith('_'):
                untested_items['untested_functions'].append({
                    'name': func['name'],
                    'file_path': source_file,
                    'line_number': func['line_number'],
                    'docstring': func['docstring'],
                    'parameters': func['parameters']
                })
        
        # Check classes
        for cls in module_info['classes']:
            cls_ref = f"{module_name}.{cls['name']}"
            if cls_ref not in tested_references and not cls['name'].startswith('_'):
                untested_items['untested_classes'].append({
                    'name': cls['name'],
                    'file_path': source_file,
                    'line_number': cls['line_number'],
                    'docstring': cls['docstring'],
                    'methods': len(cls['methods']),
                    'public_methods': len([m for m in cls['methods'] if not m['name'].startswith('_')])
                })
    
    return untested_items


def suggest_test_cases(file_path: str, function_name: Optional[str] = None, class_name: Optional[str] = None, 
                      framework: Optional[str] = None) -> Dict[str, Any]:
    """Suggest test cases based on function signatures and docstrings for both pytest and unittest frameworks."""
    module_info = parse_module_ast(file_path)
    if 'error' in module_info:
        return module_info
    
    # Detect likely testing framework if not specified
    if framework is None:
        # Check for existing test patterns in the project
        test_patterns = get_test_patterns()
        frameworks = test_patterns.get('testing_frameworks', [])
        
        if 'pytest' in frameworks:
            framework = 'pytest'
        elif 'unittest' in frameworks:
            framework = 'unittest'
        else:
            framework = 'pytest'  # Default to pytest
    
    suggestions = {
        'file_path': file_path,
        'framework': framework,
        'recommended_framework': framework,
        'test_suggestions': []
    }
    
    def generate_function_tests(func_info: Dict[str, Any], framework: str) -> List[Dict[str, Any]]:
        """Generate test suggestions for a function based on the testing framework."""
        test_cases = []
        
        # Basic test case
        test_cases.append({
            'test_name': f"test_{func_info['name']}_basic",
            'description': f"Test basic functionality of {func_info['name']}",
            'test_type': 'positive',
            'priority': 'high',
            'framework': framework,
            'suggested_assertions': _get_suggested_assertions(func_info, framework)
        })
        
        # Parameter-based tests
        for param in func_info['parameters']:
            if param['kind'] == 'positional':
                # None/null test
                test_cases.append({
                    'test_name': f"test_{func_info['name']}_with_none_{param['name']}",
                    'description': f"Test {func_info['name']} with None value for {param['name']}",
                    'test_type': 'negative',
                    'priority': 'medium',
                    'framework': framework,
                    'suggested_assertions': ['raises exception' if framework == 'pytest' else 'assertRaises']
                })
                
                # Type-based tests
                if param['type']:
                    if 'str' in param['type']:
                        test_cases.append({
                            'test_name': f"test_{func_info['name']}_with_empty_string_{param['name']}",
                            'description': f"Test {func_info['name']} with empty string for {param['name']}",
                            'test_type': 'edge_case',
                            'priority': 'medium',
                            'framework': framework,
                            'suggested_assertions': _get_suggested_assertions(func_info, framework)
                        })
                    elif 'int' in param['type']:
                        test_cases.append({
                            'test_name': f"test_{func_info['name']}_with_zero_{param['name']}",
                            'description': f"Test {func_info['name']} with zero value for {param['name']}",
                            'test_type': 'edge_case',
                            'priority': 'medium',
                            'framework': framework,
                            'suggested_assertions': _get_suggested_assertions(func_info, framework)
                        })
                        test_cases.append({
                            'test_name': f"test_{func_info['name']}_with_negative_{param['name']}",
                            'description': f"Test {func_info['name']} with negative value for {param['name']}",
                            'test_type': 'edge_case',
                            'priority': 'medium',
                            'framework': framework,
                            'suggested_assertions': _get_suggested_assertions(func_info, framework)
                        })
                    elif 'list' in param['type']:
                        test_cases.append({
                            'test_name': f"test_{func_info['name']}_with_empty_list_{param['name']}",
                            'description': f"Test {func_info['name']} with empty list for {param['name']}",
                            'test_type': 'edge_case',
                            'priority': 'medium',
                            'framework': framework,
                            'suggested_assertions': _get_suggested_assertions(func_info, framework)
                        })
                    elif 'dict' in param['type']:
                        test_cases.append({
                            'test_name': f"test_{func_info['name']}_with_empty_dict_{param['name']}",
                            'description': f"Test {func_info['name']} with empty dict for {param['name']}",
                            'test_type': 'edge_case',
                            'priority': 'medium',
                            'framework': framework,
                            'suggested_assertions': _get_suggested_assertions(func_info, framework)
                        })
        
        # Return type-based tests
        if func_info['return_type']:
            test_cases.append({
                'test_name': f"test_{func_info['name']}_return_type",
                'description': f"Test that {func_info['name']} returns correct type: {func_info['return_type']}",
                'test_type': 'type_check',
                'priority': 'low',
                'framework': framework,
                'suggested_assertions': ['assert isinstance' if framework == 'pytest' else 'assertIsInstance']
            })
        
        # Docstring-based tests
        if func_info['docstring']:
            docstring = func_info['docstring'].lower()
            if 'raise' in docstring or 'exception' in docstring:
                test_cases.append({
                    'test_name': f"test_{func_info['name']}_raises_exception",
                    'description': f"Test that {func_info['name']} raises appropriate exceptions",
                    'test_type': 'exception',
                    'priority': 'high',
                    'framework': framework,
                    'suggested_assertions': ['pytest.raises' if framework == 'pytest' else 'assertRaises']
                })
            
            if 'async' in docstring or func_info['type'] == 'async_function':
                test_cases.append({
                    'test_name': f"test_{func_info['name']}_async",
                    'description': f"Test async behavior of {func_info['name']}",
                    'test_type': 'async',
                    'priority': 'high',
                    'framework': framework,
                    'suggested_assertions': ['await' if framework == 'pytest' else 'asyncio.run']
                })
        
        return test_cases
    
    def _get_suggested_assertions(func_info: Dict[str, Any], framework: str) -> List[str]:
        """Get suggested assertion methods based on the function and framework."""
        assertions = []
        
        if framework == 'pytest':
            assertions.extend(['assert', 'assert ==', 'assert !=', 'assert is', 'assert is not'])
        else:  # unittest
            assertions.extend(['assertEqual', 'assertNotEqual', 'assertTrue', 'assertFalse', 'assertIs', 'assertIsNot'])
        
        return assertions
    
    # Generate suggestions for specific function
    if function_name:
        for func in module_info['functions']:
            if func['name'] == function_name:
                suggestions['test_suggestions'].extend(generate_function_tests(func, framework))
                break
        else:
            return {'error': f'Function "{function_name}" not found in {file_path}'}
    
    # Generate suggestions for specific class
    elif class_name:
        for cls in module_info['classes']:
            if cls['name'] == class_name:
                # Class-level tests
                test_class_suggestion = {
                    'test_name': f"test_{cls['name']}_instantiation",
                    'description': f"Test that {cls['name']} can be instantiated",
                    'test_type': 'instantiation',
                    'priority': 'high',
                    'framework': framework,
                    'suggested_assertions': _get_suggested_assertions({'name': cls['name'], 'return_type': None}, framework)
                }
                
                # Add unittest-specific test class suggestion if needed
                if framework == 'unittest':
                    test_class_suggestion['test_class_name'] = f"Test{cls['name']}"
                    test_class_suggestion['inherits_from'] = 'unittest.TestCase'
                    test_class_suggestion['setup_methods'] = ['setUp', 'tearDown']
                
                suggestions['test_suggestions'].append(test_class_suggestion)
                
                # Method tests
                for method in cls['methods']:
                    if not method['name'].startswith('_') or method['name'] == '__init__':
                        suggestions['test_suggestions'].extend(generate_function_tests(method, framework))
                break
        else:
            return {'error': f'Class "{class_name}" not found in {file_path}'}
    
    # Generate suggestions for all functions and classes
    else:
        for func in module_info['functions']:
            if not func['name'].startswith('_'):
                suggestions['test_suggestions'].extend(generate_function_tests(func, framework))
        
        for cls in module_info['classes']:
            if not cls['name'].startswith('_'):
                test_class_suggestion = {
                    'test_name': f"test_{cls['name']}_instantiation",
                    'description': f"Test that {cls['name']} can be instantiated",
                    'test_type': 'instantiation',
                    'priority': 'high',
                    'framework': framework,
                    'suggested_assertions': _get_suggested_assertions({'name': cls['name'], 'return_type': None}, framework)
                }
                
                # Add unittest-specific test class suggestion if needed
                if framework == 'unittest':
                    test_class_suggestion['test_class_name'] = f"Test{cls['name']}"
                    test_class_suggestion['inherits_from'] = 'unittest.TestCase'
                    test_class_suggestion['setup_methods'] = ['setUp', 'tearDown']
                
                suggestions['test_suggestions'].append(test_class_suggestion)
                
                for method in cls['methods']:
                    if not method['name'].startswith('_') or method['name'] == '__init__':
                        suggestions['test_suggestions'].extend(generate_function_tests(method, framework))
    
    return suggestions


def get_test_coverage_info(coverage_file: Optional[str] = None) -> Dict[str, Any]:
    """Parse and show coverage information from pytest-cov data."""
    if coverage_file is None:
        # Try common coverage file locations
        coverage_files = ['.coverage', 'coverage.xml', 'htmlcov/index.html']
        for cf in coverage_files:
            coverage_path = config.project_root / cf
            if coverage_path.exists():
                coverage_file = cf
                break
    
    if coverage_file is None:
        return {'error': 'No coverage file found. Run tests with coverage first: pytest --cov=your_package --cov-report=xml'}
    
    coverage_path = config.project_root / coverage_file
    if not coverage_path.exists():
        return {'error': f'Coverage file not found: {coverage_file}'}
    
    coverage_info = {
        'coverage_file': coverage_file,
        'coverage_data': {},
        'summary': {},
        'uncovered_lines': [],
        'coverage_gaps': []
    }
    
    try:
        if coverage_file.endswith('.xml'):
            # Parse XML coverage report
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_path)
            root = tree.getroot()
            
            total_lines = 0
            covered_lines = 0
            
            for package in root.findall('.//package'):
                for class_elem in package.findall('.//class'):
                    filename = class_elem.get('filename', '')
                    if filename:
                        rel_path = get_relative_path(Path(filename))
                        
                        file_coverage = {
                            'filename': rel_path,
                            'lines': [],
                            'covered': [],
                            'missed': []
                        }
                        
                        for line in class_elem.findall('.//line'):
                            line_num = int(line.get('number', 0))
                            hits = int(line.get('hits', 0))
                            
                            file_coverage['lines'].append(line_num)
                            if hits > 0:
                                file_coverage['covered'].append(line_num)
                                covered_lines += 1
                            else:
                                file_coverage['missed'].append(line_num)
                            
                            total_lines += 1
                        
                        coverage_info['coverage_data'][rel_path] = file_coverage
                        
                        if file_coverage['missed']:
                            coverage_info['coverage_gaps'].append({
                                'file': rel_path,
                                'uncovered_lines': file_coverage['missed'],
                                'coverage_percentage': (len(file_coverage['covered']) / len(file_coverage['lines'])) * 100 if file_coverage['lines'] else 0
                            })
            
            coverage_info['summary'] = {
                'total_lines': total_lines,
                'covered_lines': covered_lines,
                'coverage_percentage': (covered_lines / total_lines) * 100 if total_lines > 0 else 0
            }
        
        elif coverage_file == '.coverage':
            # Try to use coverage library to read binary coverage file
            try:
                import coverage
                cov = coverage.Coverage(data_file=str(coverage_path))
                cov.load()
                
                files = cov.get_data().measured_files()
                total_lines = 0
                covered_lines = 0
                
                for file_path in files:
                    try:
                        rel_path = get_relative_path(Path(file_path))
                        analysis = cov.analysis2(file_path)
                        
                        if analysis:
                            statements = analysis[1]
                            missing = analysis[3]
                            covered = [line for line in statements if line not in missing]
                            
                            coverage_info['coverage_data'][rel_path] = {
                                'filename': rel_path,
                                'lines': list(statements),
                                'covered': covered,
                                'missed': list(missing)
                            }
                            
                            total_lines += len(statements)
                            covered_lines += len(covered)
                            
                            if missing:
                                coverage_info['coverage_gaps'].append({
                                    'file': rel_path,
                                    'uncovered_lines': list(missing),
                                    'coverage_percentage': (len(covered) / len(statements)) * 100 if statements else 0
                                })
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
                
                coverage_info['summary'] = {
                    'total_lines': total_lines,
                    'covered_lines': covered_lines,
                    'coverage_percentage': (covered_lines / total_lines) * 100 if total_lines > 0 else 0
                }
                
            except ImportError:
                return {'error': 'coverage library not available. Install with: pip install coverage'}
        
        else:
            return {'error': f'Unsupported coverage file format: {coverage_file}'}
    
    except Exception as e:
        return {'error': f'Error parsing coverage file: {str(e)}'}
    
    return coverage_info


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="find_python_files",
            description="List all Python files in the project",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (relative to project root). If not specified, searches entire project."
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="read_file",
            description="Read the contents of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to project root)"
                    },
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum file size to read in bytes (default: 1MB)",
                        "default": 1048576
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_directory_structure",
            description="Show the directory structure of the project",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to show structure for (relative to project root). If not specified, shows entire project."
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (default: 3)",
                        "default": 3
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_project_info",
            description="Get comprehensive information about the project",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="parse_module",
            description="Parse a Python file and return classes, functions, and their signatures",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file to parse (relative to project root)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_function_details",
            description="Get detailed information about a specific function (params, return type, docstring)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file containing the function (relative to project root)"
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to analyze"
                    }
                },
                "required": ["file_path", "function_name"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_class_details",
            description="Get class methods, properties, and inheritance information",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file containing the class (relative to project root)"
                    },
                    "class_name": {
                        "type": "string",
                        "description": "Name of the class to analyze"
                    }
                },
                "required": ["file_path", "class_name"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="find_imports",
            description="Show what modules/packages a file imports",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file to analyze imports (relative to project root)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_type_hints",
            description="Extract type annotations from functions/methods in a Python file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file to extract type hints from (relative to project root)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="analyze_test_files",
            description="Find and parse existing test files, show test structure including fixtures, markers, and test functions",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for test files (relative to project root). If not specified, searches entire project."
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_test_patterns",
            description="Identify common testing patterns used in the project (fixtures, mocks, frameworks, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to analyze for test patterns (relative to project root). If not specified, analyzes entire project."
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="find_untested_code",
            description="Compare source files with test files to find untested functions/classes",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_dir": {
                        "type": "string",
                        "description": "Source directory to analyze (relative to project root). If not specified, analyzes entire project."
                    },
                    "test_dir": {
                        "type": "string",
                        "description": "Test directory to use for comparison (relative to project root). If not specified, uses 'tests' directory."
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="suggest_test_cases",
            description="Based on function signatures and docstrings, suggest what test cases should exist for both pytest and unittest frameworks",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python file to analyze (relative to project root)"
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Name of specific function to generate test suggestions for. If not specified, generates suggestions for all functions and classes."
                    },
                    "class_name": {
                        "type": "string",
                        "description": "Name of specific class to generate test suggestions for. If not specified, generates suggestions for all functions and classes."
                    },
                    "framework": {
                        "type": "string",
                        "description": "Testing framework to target ('pytest' or 'unittest'). If not specified, auto-detects from project patterns.",
                        "enum": ["pytest", "unittest"]
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_test_coverage_info",
            description="Parse and show coverage information from pytest-cov data (supports .coverage, coverage.xml)",
            inputSchema={
                "type": "object",
                "properties": {
                    "coverage_file": {
                        "type": "string",
                        "description": "Path to coverage file (relative to project root). If not specified, searches for common coverage files."
                    }
                },
                "additionalProperties": False
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls."""
    
    try:
        if name == "find_python_files":
            directory = arguments.get("directory")
            search_dir = config.project_root / directory if directory else None
            
            if directory and not search_dir.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Directory not found: {directory}"
                )]
            
            files = find_python_files(search_dir)
            result = files
            
        elif name == "read_file":
            file_path = arguments.get("file_path")
            max_size = arguments.get("max_size", config.max_file_size)
            
            if not file_path:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path is required"}, indent=2)
                )]
            
            result = read_file_content(file_path, max_size)
            
        elif name == "get_directory_structure":
            directory = arguments.get("directory")
            max_depth = arguments.get("max_depth", config.max_directory_depth)
            
            search_dir = config.project_root / directory if directory else None
            
            if directory and not search_dir.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Directory not found: {directory}"
                )]
            
            result = get_directory_structure(search_dir, max_depth)
            
        elif name == "get_project_info":
            result = get_project_info()
            
        elif name == "parse_module":
            file_path = arguments.get("file_path")
            
            if not file_path:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path is required"}, indent=2)
                )]
            
            result = parse_module_ast(file_path)
            
        elif name == "get_function_details":
            file_path = arguments.get("file_path")
            function_name = arguments.get("function_name")
            
            if not file_path or not function_name:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path and function_name are required"}, indent=2)
                )]
            
            result = get_function_details(file_path, function_name)
            
        elif name == "get_class_details":
            file_path = arguments.get("file_path")
            class_name = arguments.get("class_name")
            
            if not file_path or not class_name:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path and class_name are required"}, indent=2)
                )]
            
            result = get_class_details(file_path, class_name)
            
        elif name == "find_imports":
            file_path = arguments.get("file_path")
            
            if not file_path:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path is required"}, indent=2)
                )]
            
            result = find_imports_in_file(file_path)
            
        elif name == "get_type_hints":
            file_path = arguments.get("file_path")
            
            if not file_path:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path is required"}, indent=2)
                )]
            
            result = get_type_hints_from_file(file_path)
            
        elif name == "analyze_test_files":
            directory = arguments.get("directory")
            search_dir = config.project_root / directory if directory else None
            
            if directory and not search_dir.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Directory not found: {directory}"
                )]
            
            result = analyze_test_files(str(search_dir.relative_to(config.project_root)) if search_dir else None)
            
        elif name == "get_test_patterns":
            directory = arguments.get("directory")
            search_dir = config.project_root / directory if directory else None
            
            if directory and not search_dir.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Directory not found: {directory}"
                )]
            
            result = get_test_patterns(str(search_dir.relative_to(config.project_root)) if search_dir else None)
            
        elif name == "find_untested_code":
            source_dir = arguments.get("source_dir")
            test_dir = arguments.get("test_dir")
            
            source_path = config.project_root / source_dir if source_dir else None
            test_path = config.project_root / test_dir if test_dir else None
            
            if source_dir and not source_path.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Source directory not found: {source_dir}"
                )]
            if test_dir and not test_path.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Test directory not found: {test_dir}"
                )]
            
            result = find_untested_code(source_dir, test_dir)
            
        elif name == "suggest_test_cases":
            file_path = arguments.get("file_path")
            function_name = arguments.get("function_name")
            class_name = arguments.get("class_name")
            framework = arguments.get("framework")
            
            if not file_path:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "file_path is required"}, indent=2)
                )]
            
            if function_name and class_name:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": "Please provide only one of function_name or class_name."}, indent=2)
                )]
            
            if function_name:
                result = suggest_test_cases(file_path, function_name, framework=framework)
            elif class_name:
                result = suggest_test_cases(file_path, class_name=class_name, framework=framework)
            else:
                result = suggest_test_cases(file_path, framework=framework)
            
        elif name == "get_test_coverage_info":
            coverage_file = arguments.get("coverage_file")
            
            if not coverage_file:
                result = get_test_coverage_info()
            else:
                result = get_test_coverage_info(coverage_file)
            
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        # Try to serialize the result to JSON
        try:
            json_text = json.dumps(result, indent=2)
        except (TypeError, ValueError) as e:
            # Handle JSON serialization errors (e.g., circular references)
            json_text = json.dumps({"error": f"JSON serialization error: {str(e)}"}, indent=2)
        
        return [types.TextContent(
            type="text",
            text=json_text
        )]
    
    except Exception as e:
        # Handle any other exceptions
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Tool execution error: {str(e)}"}, indent=2)
        )]


async def main():
    """Run the MCP server."""
    if config.debug:
        print(f"Starting MCP server: {config.server_name} v{config.server_version}", file=sys.stderr)
        print(f"Project root: {config.project_root}", file=sys.stderr)
    
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=config.server_name,
                server_version=config.server_version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{config.server_name} MCP Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-file-size", type=int, default=config.max_file_size, 
                        help="Maximum file size to read in bytes")
    parser.add_argument("--max-depth", type=int, default=config.max_directory_depth, 
                        help="Maximum directory depth to traverse")
    args = parser.parse_args()
    
    # Update configuration from command line arguments
    if args.debug:
        config.debug = True
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    if args.max_file_size:
        config.max_file_size = args.max_file_size
    
    if args.max_depth:
        config.max_directory_depth = args.max_depth
    
    asyncio.run(main()) 