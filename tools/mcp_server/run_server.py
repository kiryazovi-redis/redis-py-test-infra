#!/usr/bin/env python3
"""
Redis-py MCP Server Launcher

This script provides a convenient way to launch the MCP server with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import mcp
        print("✓ MCP SDK installed")
        return True
    except ImportError:
        print("✗ MCP SDK not found. Please install dependencies:")
        print("  cd tools/mcp_server && pip install -r requirements.txt")
        return False


def run_server(debug=False, max_file_size=None, max_depth=None, test_mode=False):
    """Run the MCP server with specified options."""
    script_dir = Path(__file__).parent
    
    if test_mode:
        # Run in test mode (just validate and exit)
        print("Running in test mode...")
        test_script = script_dir / "test_server.py"
        if test_script.exists():
            return subprocess.run([sys.executable, str(test_script)]).returncode
        else:
            print("✗ Test script not found")
            return 1
    
    # Check main script exists only when not in test mode
    main_script = script_dir / "main.py"
    if not main_script.exists():
        print(f"✗ Main script not found: {main_script}")
        return 1
    
    # Prepare command
    cmd = [sys.executable, str(main_script)]
    
    if debug:
        cmd.append("--debug")
    
    if max_file_size:
        cmd.extend(["--max-file-size", str(max_file_size)])
    
    if max_depth:
        cmd.extend(["--max-depth", str(max_depth)])
    
    print(f"Starting MCP server...")
    print(f"Command: {' '.join(cmd)}")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        # Run the server
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
        return 0
    except Exception as e:
        print(f"✗ Error running server: {e}")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Redis-py MCP Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                    # Run with default settings
  python run_server.py --debug            # Run with debug logging
  python run_server.py --test             # Run tests and exit
  python run_server.py --max-file-size 2M # Allow larger files
        """
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--max-file-size", 
        type=str, 
        help="Maximum file size (e.g., 1M, 2048K, 1048576)"
    )
    
    parser.add_argument(
        "--max-depth", 
        type=int, 
        help="Maximum directory depth to traverse"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run tests and exit"
    )
    
    parser.add_argument(
        "--check-deps", 
        action="store_true", 
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    # Check dependencies before running
    if not check_dependencies():
        return 1
    
    # Parse file size
    max_file_size = None
    if args.max_file_size:
        size_str = args.max_file_size.upper()
        if size_str.endswith('K'):
            max_file_size = int(size_str[:-1]) * 1024
        elif size_str.endswith('M'):
            max_file_size = int(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith('G'):
            max_file_size = int(size_str[:-1]) * 1024 * 1024 * 1024
        else:
            max_file_size = int(size_str)
    
    # Run the server
    return run_server(
        debug=args.debug,
        max_file_size=max_file_size,
        max_depth=args.max_depth,
        test_mode=args.test
    )


if __name__ == "__main__":
    sys.exit(main()) 