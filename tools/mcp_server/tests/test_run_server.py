#!/usr/bin/env python3
"""
Comprehensive unit tests for run_server.py module
"""

import os
import sys
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_server import (
    check_dependencies,
    run_server,
    main
)


class TestCheckDependencies:
    """Test the check_dependencies function"""
    
    def test_check_dependencies_success(self):
        """Test successful dependency check"""
        with patch('builtins.print') as mock_print:
            result = check_dependencies()
            
            assert result is True
            mock_print.assert_called_with("✓ MCP SDK installed")
    
    def test_check_dependencies_failure(self):
        """Test failed dependency check"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mcp'")):
            with patch('builtins.print') as mock_print:
                result = check_dependencies()
                
                assert result is False
                mock_print.assert_any_call("✗ MCP SDK not found. Please install dependencies:")
                mock_print.assert_any_call("  cd tools/mcp_server && pip install -r requirements.txt")
    
    def test_check_dependencies_import_error_handling(self):
        """Test handling of import errors"""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with patch('builtins.print') as mock_print:
                result = check_dependencies()
                
                assert result is False
                assert any("MCP SDK not found" in str(call) for call in mock_print.call_args_list)


class TestRunServer:
    """Test the run_server function"""
    
    def test_run_server_basic(self):
        """Test basic server run"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server()
            
            assert result == 0
            mock_run.assert_called_once()
            
            # Check that the command includes the main script
            call_args = mock_run.call_args[0][0]
            assert any('main.py' in str(arg) for arg in call_args)
    
    def test_run_server_with_debug(self):
        """Test server run with debug flag"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server(debug=True)
            
            assert result == 0
            call_args = mock_run.call_args[0][0]
            assert '--debug' in call_args
    
    def test_run_server_with_max_file_size(self):
        """Test server run with max file size"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server(max_file_size=2048)
            
            assert result == 0
            call_args = mock_run.call_args[0][0]
            assert '--max-file-size' in call_args
            assert '2048' in call_args
    
    def test_run_server_with_max_depth(self):
        """Test server run with max depth"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server(max_depth=5)
            
            assert result == 0
            call_args = mock_run.call_args[0][0]
            assert '--max-depth' in call_args
            assert '5' in call_args
    
    def test_run_server_with_all_options(self):
        """Test server run with all options"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server(debug=True, max_file_size=1024, max_depth=3)
            
            assert result == 0
            call_args = mock_run.call_args[0][0]
            assert '--debug' in call_args
            assert '--max-file-size' in call_args
            assert '1024' in call_args
            assert '--max-depth' in call_args
            assert '3' in call_args
    
    def test_run_server_test_mode(self):
        """Test server run in test mode"""
        with patch('run_server.subprocess.run') as mock_run:
            with patch('run_server.Path.exists', return_value=True):
                mock_run.return_value = MagicMock(returncode=0)
                
                result = run_server(test_mode=True)
                
                assert result == 0
                # Should run test_server.py instead of main.py
                call_args = mock_run.call_args[0][0]
                assert any('test_server.py' in str(arg) for arg in call_args)
    
    def test_run_server_test_mode_no_test_file(self):
        """Test server run in test mode when test file doesn't exist"""
        with patch('run_server.Path.exists', return_value=False):
            with patch('builtins.print') as mock_print:
                result = run_server(test_mode=True)
                
                assert result == 1
                mock_print.assert_any_call("✗ Test script not found")
    
    def test_run_server_main_script_not_found(self):
        """Test server run when main script doesn't exist"""
        with patch('run_server.Path.exists', return_value=False):
            with patch('builtins.print') as mock_print:
                result = run_server()
                
                assert result == 1
                assert any("Main script not found" in str(call) for call in mock_print.call_args_list)
    
    def test_run_server_keyboard_interrupt(self):
        """Test server run interrupted by keyboard"""
        with patch('run_server.subprocess.run', side_effect=KeyboardInterrupt()):
            with patch('builtins.print') as mock_print:
                result = run_server()
                
                assert result == 0
                mock_print.assert_any_call("\n✓ Server stopped by user")
    
    def test_run_server_exception(self):
        """Test server run with exception"""
        with patch('run_server.subprocess.run', side_effect=Exception("Server error")):
            with patch('builtins.print') as mock_print:
                result = run_server()
                
                assert result == 1
                assert any("Error running server" in str(call) for call in mock_print.call_args_list)
    
    def test_run_server_subprocess_error(self):
        """Test server run with subprocess error"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            
            result = run_server()
            
            assert result == 1
    
    def test_run_server_prints_command(self):
        """Test that server prints command before running"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch('builtins.print') as mock_print:
                
                run_server()
                
                # Check that command is printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Starting MCP server" in call for call in print_calls)
                assert any("Command:" in call for call in print_calls)


class TestMain:
    """Test the main function"""
    
    def test_main_default_arguments(self):
        """Test main function with default arguments"""
        test_args = ['run_server.py']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=None,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_debug_flag(self):
        """Test main function with debug flag"""
        test_args = ['run_server.py', '--debug']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=True,
                        max_file_size=None,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_max_file_size(self):
        """Test main function with max file size"""
        test_args = ['run_server.py', '--max-file-size', '2M']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=2 * 1024 * 1024,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_max_file_size_kb(self):
        """Test main function with max file size in KB"""
        test_args = ['run_server.py', '--max-file-size', '1024K']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=1024 * 1024,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_max_file_size_gb(self):
        """Test main function with max file size in GB"""
        test_args = ['run_server.py', '--max-file-size', '1G']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=1024 * 1024 * 1024,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_max_file_size_bytes(self):
        """Test main function with max file size in bytes"""
        test_args = ['run_server.py', '--max-file-size', '1048576']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=1048576,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_max_depth(self):
        """Test main function with max depth"""
        test_args = ['run_server.py', '--max-depth', '5']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=None,
                        max_depth=5,
                        test_mode=False
                    )
    
    def test_main_with_test_flag(self):
        """Test main function with test flag"""
        test_args = ['run_server.py', '--test']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=None,
                        max_depth=None,
                        test_mode=True
                    )
    
    def test_main_with_check_deps_flag(self):
        """Test main function with check dependencies flag"""
        test_args = ['run_server.py', '--check-deps']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True) as mock_check:
                result = main()
                
                assert result == 0
                mock_check.assert_called_once()
    
    def test_main_with_check_deps_flag_failure(self):
        """Test main function with check dependencies flag failure"""
        test_args = ['run_server.py', '--check-deps']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=False) as mock_check:
                result = main()
                
                assert result == 1
                mock_check.assert_called_once()
    
    def test_main_with_all_flags(self):
        """Test main function with all flags"""
        test_args = ['run_server.py', '--debug', '--max-file-size', '1M', '--max-depth', '3']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=True,
                        max_file_size=1024 * 1024,
                        max_depth=3,
                        test_mode=False
                    )
    
    def test_main_dependency_check_failure(self):
        """Test main function when dependency check fails"""
        test_args = ['run_server.py']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=False):
                result = main()
                
                assert result == 1
    
    def test_main_help_flag(self):
        """Test main function with help flag"""
        test_args = ['run_server.py', '--help']
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
    
    def test_main_invalid_argument(self):
        """Test main function with invalid argument"""
        test_args = ['run_server.py', '--invalid-arg']
        
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 2  # argparse error code


class TestMainIntegration:
    """Integration tests for main function"""
    
    def test_main_integration_success(self):
        """Test full integration success scenario"""
        test_args = ['run_server.py', '--debug', '--max-file-size', '512K']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once()
                    
                    # Verify the command includes all expected arguments
                    call_args = mock_run.call_args[0][0]
                    assert '--debug' in call_args
                    assert '--max-file-size' in call_args
                    assert '524288' in call_args  # 512K in bytes
    
    def test_main_integration_server_failure(self):
        """Test integration when server fails"""
        test_args = ['run_server.py']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=1)
                    
                    result = main()
                    
                    assert result == 1
    
    def test_main_integration_with_test_mode(self):
        """Test integration with test mode"""
        test_args = ['run_server.py', '--test']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    with patch('run_server.Path.exists', return_value=True):
                        mock_run.return_value = MagicMock(returncode=0)
                        
                        result = main()
                        
                        assert result == 0
                        # Should run test_server.py
                        call_args = mock_run.call_args[0][0]
                        assert any('test_server.py' in str(arg) for arg in call_args)
    
    def test_main_integration_with_test_mode_no_test_file(self):
        """Test integration with test mode when test file doesn't exist"""
        test_args = ['run_server.py', '--test']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    with patch('run_server.Path.exists', return_value=False):
                        result = main()
                        
                        assert result == 1
    
    def test_main_integration_server_failure(self):
        """Test integration when server fails"""
        test_args = ['run_server.py']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=1)
                    
                    result = main()
                    
                    assert result == 1
    
    def test_main_integration_with_test_mode(self):
        """Test integration with test mode"""
        test_args = ['run_server.py', '--test']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.subprocess.run') as mock_run:
                    with patch('run_server.Path.exists', return_value=True):
                        mock_run.return_value = MagicMock(returncode=0)
                        
                        result = main()
                        
                        assert result == 0
                        # Should run test_server.py
                        call_args = mock_run.call_args[0][0]
                        assert any('test_server.py' in str(arg) for arg in call_args)


class TestRunServerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_run_server_with_none_values(self):
        """Test run_server with None values"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            result = run_server(
                debug=False,
                max_file_size=None,
                max_depth=None,
                test_mode=False
            )
            
            assert result == 0
            call_args = mock_run.call_args[0][0]
            assert '--debug' not in call_args
            assert '--max-file-size' not in call_args
            assert '--max-depth' not in call_args
    
    def test_main_with_invalid_file_size_format(self):
        """Test main function with invalid file size format"""
        test_args = ['run_server.py', '--max-file-size', 'invalid']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with pytest.raises(ValueError):
                    main()
    
    def test_main_with_zero_file_size(self):
        """Test main function with zero file size"""
        test_args = ['run_server.py', '--max-file-size', '0']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=0,
                        max_depth=None,
                        test_mode=False
                    )
    
    def test_main_with_negative_depth(self):
        """Test main function with negative depth"""
        test_args = ['run_server.py', '--max-depth', '-1']
        
        with patch('sys.argv', test_args):
            with patch('run_server.check_dependencies', return_value=True):
                with patch('run_server.run_server', return_value=0) as mock_run:
                    result = main()
                    
                    assert result == 0
                    mock_run.assert_called_once_with(
                        debug=False,
                        max_file_size=None,
                        max_depth=-1,
                        test_mode=False
                    )
    
    def test_run_server_path_resolution(self):
        """Test that run_server correctly resolves script paths"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            run_server()
            
            call_args = mock_run.call_args[0][0]
            # Check that the path is properly resolved
            assert len(call_args) >= 2
            assert call_args[0] == sys.executable
            assert call_args[1].endswith('main.py')
    
    def test_run_server_environment_preservation(self):
        """Test that run_server preserves environment variables"""
        with patch('run_server.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            # Set a test environment variable
            os.environ['TEST_VAR'] = 'test_value'
            
            run_server()
            
            # subprocess.run should be called without env parameter,
            # so it inherits the current environment
            call_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
            assert 'env' not in call_kwargs or call_kwargs['env'] is None
    
    def test_check_dependencies_import_success_after_failure(self):
        """Test check_dependencies recovery after initial failure"""
        import_calls = [ImportError("Not found"), None]  # Fail first, succeed second
        
        with patch('builtins.__import__', side_effect=import_calls):
            with patch('builtins.print'):
                # First call should fail
                result1 = check_dependencies()
                assert result1 is False
                
                # Second call should succeed (if __import__ is reset)
                # This tests the resilience of the dependency check
                with patch('builtins.__import__', return_value=MagicMock()):
                    result2 = check_dependencies()
                    assert result2 is True 