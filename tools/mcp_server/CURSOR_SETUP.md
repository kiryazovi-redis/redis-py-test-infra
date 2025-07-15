# MCP Server Setup for Cursor

## Quick Setup

1. **Install dependencies:**
   ```bash
   cd tools/mcp_server
   pip install -r requirements.txt
   ```

2. **Add to Cursor setting by searching for MCP in them, making sure your paths are full.**
   ```json
  {
    "mcpServers": {
      "redis-py-test-infra": {
        "command": "/Users/ivaylo.kiryazov/redis-py-test-infra/.venv/bin/python",
        "args": [
          "/Users/ivaylo.kiryazov/redis-py-test-infra/tools/mcp_server/main.py",
          "--debug"
        ],
        "cwd": "/Users/ivaylo.kiryazov/redis-py-test-infra"
      }
    }
  }
   ```

3. **Restart Cursor**

## Test Generation Workflows

### 1. Find Missing Tests
```
"Use the MCP tools to find untested code in the redis module and suggest test cases for the most important functions"
```

### 2. Analyze Test Patterns
```
"Analyze the existing test files to understand the testing patterns, then suggest tests for redis/client.py using the same patterns"
```

### 3. Generate Specific Tests
```
"Parse redis/client.py, find the Redis class, and generate comprehensive pytest test cases for its connection methods"
```

## Available Tools

- `get_project_info` - Project overview
- `parse_module` - Analyze Python files
- `find_untested_code` - Find missing tests
- `suggest_test_cases` - Generate test suggestions
- `analyze_test_files` - Analyze existing tests
- `get_test_patterns` - Identify testing patterns

## Debugging

If connection fails:
1. Check Python path in configuration
2. Verify dependencies are installed
3. Check Cursor Developer Tools console 