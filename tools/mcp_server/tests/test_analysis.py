#!/usr/bin/env python3
"""
Comprehensive unit tests for test analysis functions in main.py
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    analyze_test_files,
    get_test_patterns,
    find_untested_code,
    suggest_test_cases,
    get_test_coverage_info
)


class TestAnalyzeTestFiles:
    """Test the analyze_test_files function"""
    
    def test_analyze_test_files_with_directory(self, temp_project_dir):
        """Test analyzing test files in a directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            assert 'error' not in result
            assert 'total_test_files' in result
            assert 'test_classes' in result
            assert 'test_functions' in result
            assert 'fixtures' in result
            assert 'markers' in result
            assert 'imports' in result
            assert 'unittest_classes' in result
            assert 'pytest_fixtures' in result
            assert 'setup_teardown_methods' in result
            assert 'assertion_patterns' in result
            assert 'mock_usage' in result
            
            assert result['total_test_files'] >= 0
            assert isinstance(result['test_classes'], list)
            assert isinstance(result['test_functions'], list)
            assert isinstance(result['fixtures'], list)
    
    def test_analyze_test_files_default_directory(self, temp_project_dir):
        """Test analyzing test files in default directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files()
            
            assert 'error' not in result
            assert result['total_test_files'] > 0
    
    def test_analyze_test_files_nonexistent_directory(self, temp_project_dir):
        """Test analyzing test files in non-existent directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("nonexistent")
            
            assert 'error' in result
            assert 'not found' in result['error']
    
    def test_analyze_test_files_structure(self, temp_project_dir, temp_test_file):
        """Test the structure of test analysis results"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            # Check test functions structure
            if result['test_functions']:
                test_func = result['test_functions'][0]
                assert 'name' in test_func
                assert 'file_path' in test_func
                assert 'line_number' in test_func
                assert 'parameters' in test_func
                assert 'decorators' in test_func
                assert 'framework' in test_func
            
            # Check test classes structure
            if result['test_classes']:
                test_class = result['test_classes'][0]
                assert 'name' in test_class
                assert 'file_path' in test_class
                assert 'line_number' in test_class
                assert 'methods' in test_class
                assert 'framework' in test_class
    
    def test_analyze_test_files_frameworks_detection(self, temp_project_dir, temp_test_file):
        """Test detection of testing frameworks"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            # Check that test functions have framework detection
            if result['test_functions']:
                frameworks = [func.get('framework') for func in result['test_functions']]
                assert 'pytest' in frameworks or 'unittest' in frameworks
    
    def test_analyze_test_files_fixtures_detection(self, temp_project_dir, temp_test_file):
        """Test detection of test fixtures"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            fixtures = result['fixtures']
            if fixtures:
                fixture = fixtures[0]
                assert 'name' in fixture
                assert 'file_path' in fixture
                assert 'scope' in fixture
    
    def test_analyze_test_files_mocking_detection(self, temp_project_dir, temp_test_file):
        """Test detection of mocking usage"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            mocking = result['mock_usage']
            assert isinstance(mocking, list)
            # Check structure if mocking found
            if mocking:
                mock_info = mocking[0]
                assert 'method' in mock_info
                assert 'file_path' in mock_info
    
    def test_analyze_test_files_parametrized_tests(self, temp_project_dir, temp_test_file):
        """Test detection of parametrized tests"""
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files("tests")
            
            # Check test functions for parametrized decorators
            if result['test_functions']:
                for func in result['test_functions']:
                    if func.get('decorators'):
                        for decorator in func['decorators']:
                            if 'parametrize' in decorator:
                                # Found parametrized test
                                assert 'name' in func
                                assert 'file_path' in func
                                break


class TestGetTestPatterns:
    """Test the get_test_patterns function"""
    
    def test_get_test_patterns_basic(self, temp_project_dir):
        """Test basic test patterns analysis"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            assert 'error' not in result
            assert 'testing_frameworks' in result
            assert 'common_fixtures' in result
            assert 'mocking_patterns' in result
            assert 'parametrization_patterns' in result
            assert 'assertion_patterns' in result
            assert 'setup_patterns' in result
            assert 'framework_usage' in result
    
    def test_get_test_patterns_frameworks(self, temp_project_dir):
        """Test framework detection in test patterns"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            frameworks = result['testing_frameworks']
            assert isinstance(frameworks, list)
            assert 'pytest' in frameworks
    
    def test_get_test_patterns_fixtures(self, temp_project_dir):
        """Test fixture patterns detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            fixtures = result['common_fixtures']
            assert isinstance(fixtures, dict)
            
            # Check structure of fixture information
            for fixture_name, usage_list in fixtures.items():
                assert isinstance(usage_list, list)
                if usage_list:
                    usage = usage_list[0]
                    assert 'file_path' in usage
                    assert 'line_number' in usage
    
    def test_get_test_patterns_mocking(self, temp_project_dir):
        """Test mocking patterns detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            mocking = result['mocking_patterns']
            assert isinstance(mocking, list)
            # Check mock usage summary
            if 'mock_usage_summary' in result:
                assert isinstance(result['mock_usage_summary'], dict)
    
    def test_get_test_patterns_parametrization(self, temp_project_dir):
        """Test parametrization patterns detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            parametrization = result['parametrization_patterns']
            assert isinstance(parametrization, list)
            # Check pytest patterns
            if 'pytest_patterns' in result:
                assert 'parametrized_tests' in result['pytest_patterns']
    
    def test_get_test_patterns_assertions(self, temp_project_dir):
        """Test assertion patterns detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            assertions = result['assertion_patterns']
            assert isinstance(assertions, dict)
            # Check that assertion counts are present
            if assertions:
                for key, value in assertions.items():
                    assert isinstance(value, int)
                    assert ':' in key  # Should have format like 'pytest:assert' or 'unittest:assertEqual'
    
    def test_get_test_patterns_organization(self, temp_project_dir):
        """Test test organization patterns"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("tests")
            
            # Check framework usage organization
            framework_usage = result['framework_usage']
            assert isinstance(framework_usage, dict)
            assert 'pytest' in framework_usage
            assert 'unittest' in framework_usage
            assert 'mixed' in framework_usage
    
    def test_get_test_patterns_nonexistent_directory(self, temp_project_dir):
        """Test test patterns for non-existent directory"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_patterns("nonexistent")
            
            assert 'error' in result
            assert 'not found' in result['error']


class TestFindUntestedCode:
    """Test the find_untested_code function"""
    
    def test_find_untested_code_basic(self, temp_project_dir):
        """Test basic untested code detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("src", "tests")
            
            assert 'error' not in result
            assert 'analysis_summary' in result
            assert 'untested_functions' in result
            assert 'untested_classes' in result
            assert 'untested_files' in result
    
    def test_find_untested_code_summary(self, temp_project_dir):
        """Test analysis summary structure"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("src", "tests")
            
            summary = result['analysis_summary']
            assert 'total_source_files' in summary
            assert 'total_test_files' in summary
            assert 'tested_references' in summary
            
            assert isinstance(summary['total_source_files'], int)
            assert isinstance(summary['total_test_files'], int)
            assert isinstance(summary['tested_references'], int)
    
    def test_find_untested_code_untested_functions(self, temp_project_dir):
        """Test untested functions detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("src", "tests")
            
            untested_functions = result['untested_functions']
            assert isinstance(untested_functions, list)
            
            # Check structure of untested function info
            for func_info in untested_functions:
                assert 'name' in func_info
                assert 'file_path' in func_info
                assert 'line_number' in func_info
                assert 'docstring' in func_info
                assert 'parameters' in func_info
    
    def test_find_untested_code_untested_classes(self, temp_project_dir):
        """Test untested classes detection"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("src", "tests")
            
            untested_classes = result['untested_classes']
            assert isinstance(untested_classes, list)
            
            # Check structure of untested class info
            for class_info in untested_classes:
                assert 'name' in class_info
                assert 'file_path' in class_info
                assert 'line_number' in class_info
                assert 'methods' in class_info
    
    def test_find_untested_code_suggestions(self, temp_project_dir):
        """Test test coverage analysis"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("src", "tests")
            
            # Test that basic analysis structure is present
            assert 'untested_functions' in result
            assert 'untested_classes' in result
            assert 'untested_files' in result
            assert 'analysis_summary' in result
    
    def test_find_untested_code_default_directories(self, temp_project_dir):
        """Test with default directories"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code()
            
            assert 'error' not in result
            assert 'analysis_summary' in result
    
    def test_find_untested_code_nonexistent_directories(self, temp_project_dir):
        """Test with non-existent directories"""
        with patch('config.config.project_root', temp_project_dir):
            result = find_untested_code("nonexistent_src", "nonexistent_tests")
            
            assert 'error' in result
            assert 'not found' in result['error']


class TestSuggestTestCases:
    """Test the suggest_test_cases function"""
    
    def test_suggest_test_cases_file_only(self, temp_python_file):
        """Test suggesting test cases for entire file"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name)
            
            assert 'error' not in result
            assert 'file_path' in result
            assert 'test_suggestions' in result
            assert 'framework' in result
            
            assert result['file_path'] == temp_python_file.name
            assert isinstance(result['test_suggestions'], list)
    
    def test_suggest_test_cases_specific_function(self, temp_python_file):
        """Test suggesting test cases for specific function"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, function_name="simple_function")
            
            assert 'error' not in result
            assert len(result['test_suggestions']) > 0
            
            # Check that suggestions are for the specific function
            for suggestion in result['test_suggestions']:
                assert 'test_name' in suggestion
                assert 'test_type' in suggestion
                assert 'description' in suggestion
                assert 'priority' in suggestion
                assert 'framework' in suggestion
                assert 'suggested_assertions' in suggestion
    
    def test_suggest_test_cases_specific_class(self, temp_python_file):
        """Test suggesting test cases for specific class"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, class_name="SampleClass")
            
            assert 'error' not in result
            assert len(result['test_suggestions']) > 0
            
            # Check that suggestions exist - may reference class name or methods
            class_found = False
            for suggestion in result['test_suggestions']:
                if 'SampleClass' in suggestion['test_name'] or 'SampleClass' in suggestion['description'] or 'instantiation' in suggestion['test_name']:
                    class_found = True
                    break
            assert class_found
    
    def test_suggest_test_cases_with_framework(self, temp_python_file):
        """Test suggesting test cases with specific framework"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, framework="pytest")
            
            assert 'error' not in result
            assert result['framework'] == "pytest"
            
            # Check that test suggestions use pytest conventions
            for suggestion in result['test_suggestions']:
                assert suggestion['framework'] == 'pytest'
                # Check that suggested assertions are present
                if suggestion.get('suggested_assertions'):
                    assertions = suggestion['suggested_assertions']
                    assert len(assertions) > 0
    
    def test_suggest_test_cases_unittest_framework(self, temp_python_file):
        """Test suggesting test cases with unittest framework"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, framework="unittest")
            
            assert 'error' not in result
            assert result['framework'] == "unittest"
            
            # Check that test suggestions use unittest conventions
            for suggestion in result['test_suggestions']:
                assert suggestion['framework'] == 'unittest'
                # Check that suggested assertions are present
                if suggestion.get('suggested_assertions'):
                    assertions = suggestion['suggested_assertions']
                    assert len(assertions) > 0
    
    def test_suggest_test_cases_suggestion_types(self, temp_python_file):
        """Test different types of test suggestions"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name)
            
            # Check that we have different types of suggestions
            suggestion_types = [s['test_type'] for s in result['test_suggestions']]
            valid_types = ['positive', 'negative', 'edge_case', 'type_check', 'exception', 'async', 'instantiation']
            assert any(t in valid_types for t in suggestion_types)
            
            # Check priority levels
            priorities = [s['priority'] for s in result['test_suggestions']]
            assert 'high' in priorities or 'medium' in priorities or 'low' in priorities
    
    def test_suggest_test_cases_nonexistent_file(self, temp_project_dir):
        """Test suggesting test cases for non-existent file"""
        with patch('config.config.project_root', temp_project_dir):
            result = suggest_test_cases("nonexistent.py")
            
            assert 'error' in result
            assert 'not found' in result['error'].lower()
    
    def test_suggest_test_cases_nonexistent_function(self, temp_python_file):
        """Test suggesting test cases for non-existent function"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, function_name="nonexistent_function")
            
            assert 'error' in result
            assert 'not found' in result['error'].lower()
    
    def test_suggest_test_cases_nonexistent_class(self, temp_python_file):
        """Test suggesting test cases for non-existent class"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name, class_name="NonexistentClass")
            
            assert 'error' in result
            assert 'not found' in result['error'].lower()
    
    def test_suggest_test_cases_analysis_summary(self, temp_python_file):
        """Test basic structure of test suggestions"""
        with patch('config.config.project_root', temp_python_file.parent):
            result = suggest_test_cases(temp_python_file.name)
            
            assert 'file_path' in result
            assert 'framework' in result
            assert 'test_suggestions' in result
            
            # Check that we have test suggestions
            assert isinstance(result['test_suggestions'], list)
            if result['test_suggestions']:
                suggestion = result['test_suggestions'][0]
                assert 'test_name' in suggestion
                assert 'description' in suggestion
                assert 'test_type' in suggestion
                assert 'priority' in suggestion


class TestGetTestCoverageInfo:
    """Test the get_test_coverage_info function"""
    
    def test_get_test_coverage_info_no_file(self, temp_project_dir):
        """Test getting coverage info when no coverage file exists"""
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_coverage_info()
            
            assert 'error' in result
            assert 'no coverage file found' in result['error'].lower()
    
    def test_get_test_coverage_info_with_file(self, temp_project_dir):
        """Test getting coverage info with existing coverage file"""
        # Create a mock coverage file
        coverage_file = temp_project_dir / ".coverage"
        coverage_file.write_text("mock coverage data")
        
        with patch('config.config.project_root', temp_project_dir):
            with patch('coverage.Coverage') as mock_coverage:
                mock_cov = MagicMock()
                mock_coverage.return_value = mock_cov
                mock_cov.get_data.return_value = MagicMock()
                mock_cov.report.return_value = 85.5
                
                result = get_test_coverage_info()
                
                assert 'error' not in result
                assert 'coverage_file' in result
                assert 'summary' in result
                assert 'coverage_gaps' in result
                assert 'coverage_data' in result
    
    def test_get_test_coverage_info_custom_file(self, temp_project_dir):
        """Test getting coverage info with custom coverage file"""
        custom_file = temp_project_dir / "custom_coverage"
        custom_file.write_text("custom coverage data")
        
        with patch('config.config.project_root', temp_project_dir):
            result = get_test_coverage_info(str(custom_file))
            
            # This should error for unsupported format
            assert 'error' in result
            assert 'unsupported' in result['error'].lower() or 'format' in result['error'].lower()
    
    def test_get_test_coverage_info_structure(self, temp_project_dir):
        """Test structure of coverage info"""
        coverage_file = temp_project_dir / ".coverage"
        coverage_file.write_text("mock coverage data")
        
        with patch('config.config.project_root', temp_project_dir):
            with patch('coverage.Coverage') as mock_coverage:
                mock_cov = MagicMock()
                mock_coverage.return_value = mock_cov
                mock_cov.get_data.return_value = MagicMock()
                mock_cov.report.return_value = 85.5
                
                result = get_test_coverage_info()
                
                summary = result['summary']
                assert 'total_lines' in summary
                assert 'covered_lines' in summary
                assert 'coverage_percentage' in summary
                
                assert isinstance(result['coverage_gaps'], list)
                assert 'coverage_data' in result
    
    def test_get_test_coverage_info_import_error(self, temp_project_dir):
        """Test handling of coverage import error"""
        coverage_file = temp_project_dir / ".coverage"
        coverage_file.write_text("mock coverage data")
        
        with patch('config.config.project_root', temp_project_dir):
            with patch('coverage.Coverage', side_effect=ImportError("No coverage module")):
                result = get_test_coverage_info()
                
                assert 'error' in result
                assert 'coverage library not available' in result['error'].lower()
    
    def test_get_test_coverage_info_data_error(self, temp_project_dir):
        """Test handling of coverage data errors"""
        coverage_file = temp_project_dir / ".coverage"
        coverage_file.write_text("invalid coverage data")
        
        with patch('config.config.project_root', temp_project_dir):
            with patch('coverage.Coverage') as mock_coverage:
                mock_cov = MagicMock()
                mock_coverage.return_value = mock_cov
                mock_cov.get_data.side_effect = Exception("Data error")
                
                result = get_test_coverage_info()
                
                assert 'error' in result
                assert 'error parsing coverage' in result['error'].lower()


class TestAnalysisEdgeCases:
    """Test edge cases and error conditions for analysis functions"""
    
    def test_analysis_with_empty_files(self, temp_project_dir):
        """Test analysis with empty Python files"""
        empty_file = temp_project_dir / "empty.py"
        empty_file.write_text("")
        
        with patch('config.config.project_root', temp_project_dir):
            result = suggest_test_cases("empty.py")
            
            assert 'error' not in result
            assert len(result['test_suggestions']) == 0
    
    def test_analysis_with_syntax_error_files(self, temp_project_dir, invalid_python_file):
        """Test analysis with files containing syntax errors"""
        with patch('config.config.project_root', temp_project_dir):
            result = suggest_test_cases(invalid_python_file.name)
            
            assert 'error' in result
            assert 'syntax error' in result['error'].lower()
    
    def test_analysis_with_large_files(self, temp_project_dir):
        """Test analysis with very large files"""
        large_file = temp_project_dir / "large.py"
        large_content = ["def func_{}(): pass".format(i) for i in range(1000)]
        large_file.write_text('\n'.join(large_content))
        
        with patch('config.config.project_root', temp_project_dir):
            result = suggest_test_cases("large.py")
            
            # This should work with proper syntax
            if 'error' in result:
                # If there's an error, it should be about syntax
                assert 'syntax' in result['error'].lower()
            else:
                assert len(result['test_suggestions']) > 0
    
    def test_analysis_with_complex_decorators(self, temp_project_dir):
        """Test analysis with complex decorators"""
        complex_file = temp_project_dir / "complex.py"
        complex_content = """
import functools

@functools.lru_cache(maxsize=128)
@property
def complex_function():
    pass

class ComplexClass:
    @classmethod
    @functools.wraps(some_func)
    def complex_method(cls):
        pass
"""
        complex_file.write_text(complex_content)
        
        with patch('config.config.project_root', temp_project_dir):
            result = suggest_test_cases("complex.py")
            
            assert 'error' not in result or 'not found' in result['error']
    
    def test_analysis_performance_with_many_files(self, temp_project_dir):
        """Test analysis performance with many test files"""
        # Create many test files
        for i in range(50):
            test_file = temp_project_dir / f"test_{i}.py"
            test_file.write_text(f"def test_function_{i}(): pass")
        
        with patch('config.config.project_root', temp_project_dir):
            result = analyze_test_files()
            
            assert 'error' not in result
            assert result['total_test_files'] >= 50 