#!/usr/bin/env python3
"""
Test runner for the Redis-py MCP Server

This script provides an easy way to run all tests with various options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, parallel=False, marker=None):
    """Run tests with specified options."""
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths based on test type
    if test_type == "all":
        cmd.append("tests/")
    elif test_type == "unit":
        cmd.extend([
            "tests/test_config.py",
            "tests/test_ast_parsing.py",
            "tests/test_file_operations.py",
            "tests/test_analysis.py",
            "tests/test_run_server.py"
        ])
    elif test_type == "integration":
        cmd.append("tests/test_integration.py")
    elif test_type == "server":
        cmd.append("tests/test_server_tools.py")
    else:
        cmd.append(f"tests/test_{test_type}.py")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=config",
            "--cov=main", 
            "--cov=run_server",
            "--cov-report=term-missing",
            "--cov-report=html:test_coverage_html"
        ])
    
    # Add parallel execution
    if parallel:
        try:
            import pytest_xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("Warning: pytest-xdist not installed, running tests sequentially")
    
    # Add marker filtering
    if marker:
        cmd.extend(["-m", marker])
    
    # Add additional options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def check_test_dependencies():
    """Check if test dependencies are installed."""
    required_packages = ["pytest", "pytest-asyncio", "pytest-cov", "pytest-mock"]
    optional_packages = ["pytest-xdist"]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} is installed")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} is missing")
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} is installed (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"○ {package} is missing (optional)")
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install pytest-xdist")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Redis-py MCP Server Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  all         Run all tests (default)
  unit        Run unit tests only
  integration Run integration tests only
  server      Run server tool tests only
  config      Run configuration tests only
  ast         Run AST parsing tests only
  file        Run file operations tests only
  analysis    Run analysis tests only
  run_server  Run run_server tests only

Markers:
  unit        Unit tests
  integration Integration tests
  ast         AST parsing related tests
  filesystem  File system related tests
  server      Server functionality tests
  slow        Slow tests (can be excluded with '-m "not slow"')

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --verbose          # Run with verbose output
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --parallel         # Run tests in parallel
  python run_tests.py unit               # Run only unit tests
  python run_tests.py -m "not slow"      # Exclude slow tests
  python run_tests.py --check-deps       # Check test dependencies
        """
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "-m", "--marker",
        help="Run tests with specific marker"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check test dependencies and exit"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        return 0 if check_test_dependencies() else 1
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✓ Dependencies installed successfully")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return 1
    
    # Check dependencies before running tests
    if not check_test_dependencies():
        print("\nUse --install-deps to install missing dependencies")
        return 1
    
    print("\n" + "=" * 60)
    print("Redis-py MCP Server - Test Suite")
    print("=" * 60)
    
    # Run tests
    return run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        marker=args.marker
    )


if __name__ == "__main__":
    sys.exit(main()) 