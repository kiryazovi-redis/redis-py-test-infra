# Redis-py MCP Server

A Model Context Protocol (MCP) server for the redis-py client library project. This server provides tools to help with test generation, code exploration, and project analysis.

## Features

The MCP server provides the following tools:

### File System Tools
- **`list_python_files`**: List all Python files in the project with metadata
- **`read_file`**: Read file contents with size limits and type detection
- **`show_directory_structure`**: Display project directory structure as a tree
- **`get_project_info`**: Get comprehensive project information including file counts and structure

### AST Parsing Tools
- **`parse_module`**: Parse a Python file and return classes, functions, and their signatures
- **`get_function_info`**: Get detailed information about a specific function (params, return type, docstring)
- **`get_class_info`**: Get class methods, properties, and inheritance information
- **`find_imports`**: Show what modules/packages a file imports
- **`get_type_hints`**: Extract type annotations from functions/methods in a Python file

### Test Analysis Tools
- **`analyze_test_files`**: Find and parse existing test files, show comprehensive test structure including pytest and unittest patterns
- **`get_test_patterns`**: Identify common testing patterns used in the project (fixtures, mocks, frameworks, setup/teardown methods)
- **`find_untested_code`**: Compare source files with test files to find untested functions/classes
- **`suggest_test_cases`**: Based on function signatures and docstrings, suggest what test cases should exist for both pytest and unittest frameworks
- **`get_test_coverage_info`**: Parse and show coverage information from pytest-cov data (supports .coverage, coverage.xml)

## Installation

### Prerequisites

- Python 3.9 or later
- The redis-py project (this MCP server is designed to work with it)

### Install Dependencies

Navigate to the MCP server directory and install the required dependencies:

```bash
cd tools/mcp_server
pip install -r requirements.txt
```

### Recommended: Use Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The server is configured through the `config.py` file. Key configuration options include:

- **File size limits**: Maximum file size to read (default: 1MB)
- **Directory depth**: Maximum depth for directory traversal (default: 3)
- **File type detection**: Automatic detection of Python and text files
- **Ignore patterns**: Configurable patterns for files/directories to ignore

### Environment Variables

You can configure the server using environment variables:

- `MCP_DEBUG`: Set to `true` to enable debug logging
- `MCP_LOG_LEVEL`: Set logging level (default: `INFO`)

## Usage

### Running the Server

#### Basic Usage

```bash
cd tools/mcp_server
python main.py
```

#### With Debug Logging

```bash
python main.py --debug
```

#### With Custom Configuration

```bash
python main.py --max-file-size 2097152 --max-depth 5 --debug
```

### Command Line Options

- `--debug`: Enable debug logging
- `--max-file-size INT`: Maximum file size to read in bytes (default: 1048576)
- `--max-depth INT`: Maximum directory depth to traverse (default: 3)

### Integration with MCP Clients

#### Claude Desktop Configuration

Add the following to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "redis-py-test-infra": {
      "command": "python",
      "args": ["path/to/redis-py-test-infra/tools/mcp_server/main.py"],
      "cwd": "path/to/redis-py-test-infra"
    }
  }
}
```

#### Using with MCP Client Libraries

```python
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

async def use_redis_py_mcp():
    # Connect to the MCP server
    async with stdio_client([
        "python", 
        "tools/mcp_server/main.py"
    ]) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools])
            
            # Get project info
            result = await session.call_tool("get_project_info", {})
            print("Project info:", result)
            
            # List Python files
            files = await session.call_tool("list_python_files", {})
            print("Python files:", files)

# Run the example
asyncio.run(use_redis_py_mcp())
```

## Available Tools

### 1. `list_python_files`

Lists all Python files in the project with metadata.

**Parameters:**
- `directory` (optional): Directory to search in (relative to project root)

**Returns:**
```json
[
  {
    "path": "redis/client.py",
    "name": "client.py",
    "size": 45678,
    "directory": "redis"
  }
]
```

### 2. `read_file`

Reads file contents with size limits and type detection.

**Parameters:**
- `file_path` (required): Path to the file to read
- `max_size` (optional): Maximum file size in bytes

**Returns:**
```json
{
  "path": "redis/client.py",
  "content": "# File content here...",
  "size": 45678,
  "lines": 1234,
  "is_python": true,
  "is_text": true
}
```

### 3. `show_directory_structure`

Shows the directory structure as a tree.

**Parameters:**
- `directory` (optional): Directory to show structure for
- `max_depth` (optional): Maximum depth to traverse

**Returns:**
```json
{
  "name": "redis",
  "type": "directory",
  "path": "redis",
  "children": [
    {
      "name": "client.py",
      "type": "file",
      "path": "redis/client.py",
      "size": 45678,
      "is_python": true,
      "is_text": true
    }
  ]
}
```

### 4. `get_project_info`

Gets comprehensive project information.

**Parameters:** None

**Returns:**
```json
{
  "name": "redis-py-test-infra",
  "version": "1.0.0",
  "project_root": "/path/to/project",
  "debug": false,
  "max_file_size": 1048576,
  "supported_extensions": {
    "python": [".py", ".pyi"],
    "text": [".txt", ".md", ".rst", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini"]
  },
  "file_counts": {
    "python": 150,
    "test": 85,
    "doc": 25,
    "other": 40
  },
  "total_size": 2500000,
  "main_directories": [
    {"name": "benchmarks", "path": "benchmarks"},
    {"name": "redis", "path": "redis"},
    {"name": "tests", "path": "tests"}
  ],
  "key_files": [
    {"name": "README.md", "path": "README.md", "size": 5432},
    {"name": "pyproject.toml", "path": "pyproject.toml", "size": 1234}
  ]
}
```

### 5. `parse_module`

Parses a Python file and returns all classes, functions, and their signatures.

**Parameters:**
- `file_path` (required): Path to the Python file to parse

**Returns:**
```json
{
  "file_path": "redis/client.py",
  "docstring": "Redis client implementation...",
  "classes": [
    {
      "name": "Redis",
      "type": "class",
      "docstring": "Redis client class...",
      "base_classes": ["object"],
      "methods": [
        {
          "name": "__init__",
          "type": "function",
          "docstring": "Initialize Redis client...",
          "parameters": [
            {
              "name": "self",
              "type": null,
              "kind": "positional"
            },
            {
              "name": "host",
              "type": "str",
              "kind": "positional",
              "default": "'localhost'"
            }
          ],
          "return_type": null,
          "decorators": [],
          "line_number": 45
        }
      ],
      "properties": [],
      "class_variables": [],
      "decorators": [],
      "line_number": 40
    }
  ],
  "functions": [
    {
      "name": "from_url",
      "type": "function",
      "docstring": "Create Redis client from URL...",
      "parameters": [
        {
          "name": "url",
          "type": "str",
          "kind": "positional"
        }
      ],
      "return_type": "Redis",
      "decorators": ["@classmethod"],
      "line_number": 100
    }
  ],
  "imports": [
    {
      "type": "import",
      "module": "socket",
      "alias": null,
      "line_number": 1
    },
    {
      "type": "from_import",
      "module": "typing",
      "name": "Optional",
      "alias": null,
      "line_number": 2
    }
  ],
  "variables": []
}
```

### 6. `get_function_info`

Gets detailed information about a specific function.

**Parameters:**
- `file_path` (required): Path to the Python file containing the function
- `function_name` (required): Name of the function to analyze

**Returns:**
```json
{
  "name": "connect",
  "type": "async_function",
  "docstring": "Establish connection to Redis server...",
  "parameters": [
    {
      "name": "self",
      "type": null,
      "kind": "positional"
    },
    {
      "name": "host",
      "type": "str",
      "kind": "positional",
      "default": "'localhost'"
    },
    {
      "name": "port",
      "type": "int",
      "kind": "positional",
      "default": "6379"
    },
    {
      "name": "timeout",
      "type": "Optional[float]",
      "kind": "keyword_only",
      "default": "None"
    }
  ],
  "return_type": "None",
  "decorators": [],
  "line_number": 150
}
```

### 7. `get_class_info`

Gets class methods, properties, and inheritance information.

**Parameters:**
- `file_path` (required): Path to the Python file containing the class
- `class_name` (required): Name of the class to analyze

**Returns:**
```json
{
  "name": "RedisCluster",
  "type": "class",
  "docstring": "Redis cluster client implementation...",
  "base_classes": ["Redis"],
  "methods": [
    {
      "name": "__init__",
      "type": "function",
      "visibility": "public",
      "docstring": "Initialize cluster client...",
      "parameters": [
        {
          "name": "self",
          "type": null,
          "kind": "positional"
        },
        {
          "name": "startup_nodes",
          "type": "List[dict]",
          "kind": "positional"
        }
      ],
      "return_type": null,
      "decorators": [],
      "line_number": 25
    },
    {
      "name": "_get_node",
      "type": "function",
      "visibility": "private",
      "docstring": "Get node for key...",
      "parameters": [
        {
          "name": "self",
          "type": null,
          "kind": "positional"
        },
        {
          "name": "key",
          "type": "str",
          "kind": "positional"
        }
      ],
      "return_type": "dict",
      "decorators": [],
      "line_number": 45
    }
  ],
  "properties": [
    {
      "name": "nodes",
      "type": "function",
      "visibility": "public",
      "docstring": "Get all cluster nodes...",
      "parameters": [
        {
          "name": "self",
          "type": null,
          "kind": "positional"
        }
      ],
      "return_type": "List[dict]",
      "decorators": ["@property"],
      "line_number": 65
    }
  ],
  "class_variables": [
    {
      "name": "DEFAULT_PORT",
      "type": "int",
      "line_number": 15
    }
  ],
  "decorators": [],
  "line_number": 10
}
```

### 8. `find_imports`

Shows what modules/packages a file imports.

**Parameters:**
- `file_path` (required): Path to the Python file to analyze imports

**Returns:**
```json
{
  "file_path": "redis/client.py",
  "imports": [
    {
      "type": "import",
      "module": "socket",
      "alias": null,
      "line_number": 1
    },
    {
      "type": "import",
      "module": "json",
      "alias": null,
      "line_number": 2
    },
    {
      "type": "from_import",
      "module": "typing",
      "name": "Optional",
      "alias": null,
      "line_number": 3
    },
    {
      "type": "from_import",
      "module": "typing",
      "name": "Dict",
      "alias": null,
      "line_number": 3
    },
    {
      "type": "from_import",
      "module": "redis.connection",
      "name": "Connection",
      "alias": "RedisConnection",
      "line_number": 4
    }
  ],
  "total_imports": 5
}
```

### 9. `get_type_hints`

Extracts type annotations from functions/methods in a Python file.

**Parameters:**
- `file_path` (required): Path to the Python file to extract type hints from

**Returns:**
```json
{
  "file_path": "redis/client.py",
  "functions": [
    {
      "name": "connect",
      "type": "async_function",
      "return_type": "None",
      "parameters": [
        {
          "name": "host",
          "type": "str"
        },
        {
          "name": "port",
          "type": "int"
        },
        {
          "name": "timeout",
          "type": "Optional[float]"
        }
      ]
    }
  ],
  "classes": [
    {
      "name": "Redis",
      "methods": [
        {
          "name": "__init__",
          "return_type": "None",
          "parameters": [
            {
              "name": "host",
              "type": "str"
            },
            {
              "name": "port",
              "type": "int"
            }
          ]
        },
        {
          "name": "get",
          "return_type": "Optional[bytes]",
          "parameters": [
            {
              "name": "key",
              "type": "str"
            }
          ]
        }
      ],
      "variables": [
        {
          "name": "connection",
          "type": "Connection"
        }
      ]
    }
  ],
  "variables": [
    {
      "name": "DEFAULT_HOST",
      "type": "str"
    },
    {
      "name": "DEFAULT_PORT",
      "type": "int"
    }
  ]
}
```

## Development

### Extending the Server

To add new tools to the MCP server:

1. Add the tool definition to the `handle_list_tools()` function
2. Add the tool handler to the `handle_call_tool()` function
3. Implement the tool's functionality as a separate function

Example:

```python
# Add to handle_list_tools()
types.Tool(
    name="my_new_tool",
    description="Description of what the tool does",
    inputSchema={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of parameter"
            }
        },
        "required": ["param1"]
    }
)

# Add to handle_call_tool()
elif name == "my_new_tool":
    param1 = arguments.get("param1")
    result = my_new_tool_function(param1)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
```

### Testing

You can test the server using the MCP inspector:

```bash
# Install the MCP inspector
pip install mcp-inspector

# Run the inspector
mcp-inspector tools/mcp_server/main.py
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the correct directory and all dependencies are installed.

2. **File not found errors**: Ensure the project root is correctly detected. The server automatically detects the project root based on the script location.

3. **Permission errors**: Make sure the server has read access to the project files.

### Debug Mode

Run with `--debug` flag to see detailed logging:

```bash
python main.py --debug
```

This will show:
- Server startup information
- Project root detection
- Tool execution details
- Error messages with stack traces

## Test Analysis Tools Documentation

### 6. `analyze_test_files`

Find and parse existing test files, showing comprehensive test structure including pytest and unittest patterns.

**Parameters:**
- `directory` (optional): Directory to search for test files (relative to project root)

**Returns:**
```json
{
  "total_test_files": 25,
  "test_files": [
    {
      "file_path": "tests/test_client.py",
      "classes": [...],
      "functions": [...],
      "fixtures": [...],
      "markers": [...]
    }
  ],
  "test_classes": [
    {
      "name": "TestRedisClient",
      "file_path": "tests/test_client.py",
      "framework": "unittest",
      "base_classes": ["unittest.TestCase"]
    }
  ],
  "test_functions": [
    {
      "name": "test_connect",
      "file_path": "tests/test_client.py",
      "framework": "pytest",
      "decorators": ["pytest.mark.slow"]
    }
  ],
  "unittest_classes": [...],
  "pytest_fixtures": [
    {
      "name": "client_fixture",
      "scope": "function",
      "autouse": false
    }
  ],
  "setup_teardown_methods": [
    {
      "name": "setUp",
      "class_name": "TestRedisClient",
      "framework": "unittest",
      "scope": "method"
    }
  ],
  "assertion_patterns": [
    {
      "method": "assertEqual",
      "framework": "unittest"
    }
  ],
  "mock_usage": [
    {
      "method": "patch",
      "file_path": "tests/test_client.py"
    }
  ]
}
```

### 7. `get_test_patterns`

Identify common testing patterns used in the project including frameworks, fixtures, mocks, and setup/teardown methods.

**Parameters:**
- `directory` (optional): Directory to analyze for test patterns (relative to project root)

**Returns:**
```json
{
  "testing_frameworks": ["pytest", "unittest"],
  "framework_usage": {
    "pytest": 45,
    "unittest": 12,
    "mixed": 1
  },
  "common_fixtures": {
    "client": [
      {
        "name": "client",
        "scope": "function",
        "file_path": "tests/conftest.py"
      }
    ]
  },
  "common_markers": {
    "slow": 8,
    "integration": 15
  },
  "mocking_patterns": [
    {
      "module": "unittest.mock",
      "type": "from_import",
      "name": "patch"
    }
  ],
  "assertion_patterns": {
    "pytest:assert": 123,
    "unittest:assertEqual": 45
  },
  "unittest_patterns": {
    "test_case_classes": 8,
    "setup_teardown_methods": [...]
  },
  "pytest_patterns": {
    "fixtures": 12,
    "markers": 25,
    "parametrized_tests": 8
  },
  "mock_usage_summary": {
    "patch": 15,
    "Mock": 8
  }
}
```

### 8. `find_untested_code`

Compare source files with test files to find untested functions and classes.

**Parameters:**
- `source_dir` (optional): Source directory to analyze (relative to project root)
- `test_dir` (optional): Test directory to use for comparison (relative to project root)

**Returns:**
```json
{
  "untested_functions": [
    {
      "name": "cleanup_connections",
      "file_path": "redis/client.py",
      "line_number": 150,
      "docstring": "Clean up all connections",
      "parameters": []
    }
  ],
  "untested_classes": [
    {
      "name": "ConnectionPool",
      "file_path": "redis/connection.py",
      "line_number": 45,
      "methods": 8,
      "public_methods": 5
    }
  ],
  "untested_files": [
    {
      "file_path": "redis/utils.py",
      "functions": 5,
      "classes": 2
    }
  ],
  "analysis_summary": {
    "total_source_files": 25,
    "total_test_files": 18,
    "tested_references": 145
  }
}
```

### 9. `suggest_test_cases`

Based on function signatures and docstrings, suggest what test cases should exist for both pytest and unittest frameworks.

**Parameters:**
- `file_path` (required): Path to the Python file to analyze (relative to project root)
- `function_name` (optional): Name of specific function to generate test suggestions for
- `class_name` (optional): Name of specific class to generate test suggestions for
- `framework` (optional): Testing framework to target ('pytest' or 'unittest'). Auto-detects if not specified.

**Returns:**
```json
{
  "file_path": "redis/client.py",
  "framework": "pytest",
  "test_suggestions": [
    {
      "test_name": "test_connect_basic",
      "description": "Test basic functionality of connect",
      "test_type": "positive",
      "priority": "high",
      "framework": "pytest",
      "suggested_assertions": ["assert", "assert ==", "assert !="]
    },
    {
      "test_name": "test_connect_with_none_host",
      "description": "Test connect with None value for host",
      "test_type": "negative",
      "priority": "medium",
      "framework": "pytest",
      "suggested_assertions": ["raises exception"]
    },
    {
      "test_name": "test_connect_raises_exception",
      "description": "Test that connect raises appropriate exceptions",
      "test_type": "exception",
      "priority": "high",
      "framework": "pytest",
      "suggested_assertions": ["pytest.raises"]
    }
  ]
}
```

### 10. `get_test_coverage_info`

Parse and show coverage information from pytest-cov data (supports .coverage, coverage.xml).

**Parameters:**
- `coverage_file` (optional): Path to coverage file (relative to project root). Auto-detects if not specified.

**Returns:**
```json
{
  "coverage_file": "coverage.xml",
  "coverage_data": {
    "redis/client.py": {
      "filename": "redis/client.py",
      "lines": [1, 2, 3, 4, 5, 10, 15, 20],
      "covered": [1, 2, 3, 5, 10, 15],
      "missed": [4, 20]
    }
  },
  "summary": {
    "total_lines": 1250,
    "covered_lines": 1100,
    "coverage_percentage": 88.0
  },
  "coverage_gaps": [
    {
      "file": "redis/client.py",
      "uncovered_lines": [4, 20],
      "coverage_percentage": 75.0
    }
  ]
}
```

## Framework Support

The MCP server provides comprehensive support for both **pytest** and **unittest** testing frameworks:

### pytest Support
- **Fixtures**: Detection of fixtures with scope, autouse, and parameters
- **Markers**: Identification of pytest markers (slow, integration, etc.)
- **Parametrization**: Detection of parametrized tests 
- **Async Tests**: Support for async test functions
- **Assertion Styles**: pytest-style assert statements

### unittest Support
- **TestCase Classes**: Detection of unittest.TestCase subclasses
- **Setup/Teardown**: setUp, tearDown, setUpClass, tearDownClass methods
- **Assertion Methods**: assertEqual, assertTrue, assertRaises, etc.
- **Test Discovery**: Automatic detection of test methods (test_*)

### Mixed Framework Projects
The server can analyze projects that use both frameworks and provide:
- Framework usage statistics
- Migration suggestions
- Pattern analysis across different testing styles

## Usage Examples

### Analyzing Test Structure
```python
# Analyze all test files
result = analyze_test_files()

# Analyze specific test directory
result = analyze_test_files(directory="tests/integration")
```

### Getting Test Patterns
```python
# Get patterns for entire project
patterns = get_test_patterns()

# Get patterns for specific directory
patterns = get_test_patterns(directory="tests/unit")
```

### Finding Untested Code
```python
# Find untested code in entire project
untested = find_untested_code()

# Find untested code in specific directories
untested = find_untested_code(source_dir="redis", test_dir="tests")
```

### Suggesting Test Cases
```python
# Suggest tests for entire file
suggestions = suggest_test_cases("redis/client.py")

# Suggest tests for specific function with pytest
suggestions = suggest_test_cases("redis/client.py", function_name="connect", framework="pytest")

# Suggest tests for specific class with unittest
suggestions = suggest_test_cases("redis/client.py", class_name="Redis", framework="unittest")
```

### Getting Coverage Information
```python
# Auto-detect coverage file
coverage = get_test_coverage_info()

# Use specific coverage file
coverage = get_test_coverage_info(coverage_file="coverage.xml")
```

## License

This MCP server is part of the redis-py project and follows the same licensing terms.

## Contributing

Contributions are welcome! Please:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tools return well-structured JSON responses

## Related Projects

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [redis-py](https://github.com/redis/redis-py) - The main project this server supports 