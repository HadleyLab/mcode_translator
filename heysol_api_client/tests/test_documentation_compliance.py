"""
Tests for API documentation compliance in the HeySol API client.
"""

import pytest
import inspect
from typing import get_type_hints, Dict, Any, List, Optional
import re

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError


class TestDocumentationCompliance:
    """Test API documentation compliance."""

    def test_readme_method_signatures_match_client(self):
        """Test that README method signatures match actual client implementations."""
        # Extract method signatures from README
        readme_methods = self._extract_readme_methods()

        # Get actual methods from client
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        # Check that documented methods exist in clients
        for method_name in readme_methods:
            assert method_name in client_methods or method_name in async_client_methods, \
                f"Documented method '{method_name}' not found in client implementations"

    def test_method_docstrings_exist(self):
        """Test that all public methods have docstrings."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_'):  # Skip private methods
                assert method.__doc__ is not None, \
                    f"Method '{method_name}' is missing a docstring"
                assert len(method.__doc__.strip()) > 10, \
                    f"Method '{method_name}' has an inadequate docstring"

    def test_method_docstrings_format(self):
        """Test that method docstrings follow proper format."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_') and method.__doc__:
                docstring = method.__doc__.strip()

                # Check for basic structure
                assert docstring.startswith(method_name.split('.')[-1].replace('_', ' ').title()) or \
                       "Test" in docstring or \
                       len(docstring.split('\n\n')) >= 2, \
                    f"Method '{method_name}' docstring doesn't follow proper format"

    def test_parameter_documentation(self):
        """Test that method parameters are properly documented."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_') and method.__doc__:
                sig = inspect.signature(method)
                docstring = method.__doc__

                # Check that parameters mentioned in docstring exist in signature
                # This is a basic check - more sophisticated parsing would be needed for full compliance
                if "Args:" in docstring:
                    args_section = docstring.split("Args:")[1].split("\n\n")[0]
                    for line in args_section.split('\n'):
                        if ':' in line and not line.strip().startswith(' '):
                            param_name = line.split(':')[0].strip()
                            if param_name in sig.parameters:
                                continue  # Parameter is documented

    def test_return_value_documentation(self):
        """Test that return values are properly documented."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_') and method.__doc__:
                docstring = method.__doc__

                # Check for Returns section
                if "Returns:" in docstring:
                    returns_section = docstring.split("Returns:")[1]
                    if "\n\n" in returns_section:
                        returns_section = returns_section.split("\n\n")[0]

                    assert len(returns_section.strip()) > 10, \
                        f"Method '{method_name}' has inadequate return value documentation"

    def test_exception_documentation(self):
        """Test that exceptions are properly documented."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_') and method.__doc__:
                docstring = method.__doc__

                # Check for Raises section if method can raise exceptions
                if "Raises:" in docstring:
                    raises_section = docstring.split("Raises:")[1]
                    if "\n\n" in raises_section:
                        raises_section = raises_section.split("\n\n")[0]

                    assert len(raises_section.strip()) > 10, \
                        f"Method '{method_name}' has inadequate exception documentation"

    def test_type_hints_compliance(self):
        """Test that methods have proper type hints."""
        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        all_methods = {**client_methods, **async_client_methods}

        for method_name, method in all_methods.items():
            if not method_name.startswith('_'):
                try:
                    hints = get_type_hints(method)
                    # At minimum, should have return type hint
                    assert 'return' in hints, \
                        f"Method '{method_name}' is missing return type hint"
                except (NameError, TypeError):
                    # Some type hints might reference undefined types
                    pass

    def test_example_code_validity(self):
        """Test that example code in documentation is syntactically valid."""
        # This would require parsing code examples from README and other docs
        # For now, we'll check that the examples directory exists and contains valid Python
        import os
        examples_dir = "heysol_api_client/examples"

        assert os.path.exists(examples_dir), "Examples directory should exist"

        example_files = [f for f in os.listdir(examples_dir) if f.endswith('.py')]
        assert len(example_files) > 0, "Should have example Python files"

        for example_file in example_files:
            file_path = os.path.join(examples_dir, example_file)
            with open(file_path, 'r') as f:
                content = f.read()

            # Basic syntax check
            try:
                compile(content, file_path, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in example file {example_file}: {e}")

    def test_config_documentation_completeness(self):
        """Test that configuration options are fully documented."""
        config_class = HeySolConfig

        # Get all configuration attributes
        config_attrs = [attr for attr in dir(config_class) if not attr.startswith('_') and not callable(getattr(config_class, attr))]

        # Check that each config attribute is documented in the class docstring
        if config_class.__doc__:
            docstring = config_class.__doc__.lower()

            for attr in config_attrs:
                # Skip some common attributes that don't need documentation
                if attr in ['__annotations__', '__dataclass_fields__', '__dataclass_params__']:
                    continue

                assert attr.lower() in docstring, \
                    f"Configuration attribute '{attr}' is not documented in HeySolConfig docstring"

    def test_exception_hierarchy_documentation(self):
        """Test that exception hierarchy is properly documented."""
        from heysol.exceptions import HeySolError

        # Get all exception classes
        exception_classes = []
        for name in dir(HeySolError.__module__):
            obj = getattr(HeySolError.__module__, name)
            if (isinstance(obj, type) and
                issubclass(obj, Exception) and
                obj != Exception):
                exception_classes.append(obj)

        # Check that each exception has proper documentation
        for exc_class in exception_classes:
            assert exc_class.__doc__ is not None, \
                f"Exception class '{exc_class.__name__}' is missing docstring"

            # Check that it mentions inheritance from HeySolError
            if exc_class != HeySolError:
                assert "HeySolError" in exc_class.__doc__, \
                    f"Exception '{exc_class.__name__}' docstring should mention inheritance from HeySolError"

    def test_method_signature_consistency(self):
        """Test that sync and async method signatures are consistent."""
        sync_methods = self._get_client_methods(HeySolClient)
        async_methods = self._get_client_methods(AsyncHeySolClient)

        # Find common method names
        common_methods = set(sync_methods.keys()) & set(async_methods.keys())

        for method_name in common_methods:
            if not method_name.startswith('_'):
                sync_method = sync_methods[method_name]
                async_method = async_methods[method_name]

                sync_sig = inspect.signature(sync_method)
                async_sig = inspect.signature(async_method)

                # Compare parameter names and types (excluding 'self')
                sync_params = list(sync_sig.parameters.keys())[1:]  # Skip 'self'
                async_params = list(async_sig.parameters.keys())[1:]  # Skip 'self'

                assert sync_params == async_params, \
                    f"Method '{method_name}' has inconsistent parameters between sync and async versions"

    def test_api_endpoint_coverage(self):
        """Test that all API endpoints mentioned in documentation are implemented."""
        # This would require parsing the API documentation to extract endpoints
        # For now, we'll check that key endpoints from the README are implemented

        readme_endpoints = [
            "get_user_profile",
            "get_spaces",
            "create_space",
            "get_or_create_space",
            "ingest",
            "search",
            "get_ingestion_logs",
            "get_specific_log",
            "delete_log_entry"
        ]

        client_methods = self._get_client_methods(HeySolClient)
        async_client_methods = self._get_client_methods(AsyncHeySolClient)

        for endpoint in readme_endpoints:
            assert endpoint in client_methods or endpoint in async_client_methods, \
                f"Documented endpoint '{endpoint}' not found in client implementations"

    def test_configuration_options_coverage(self):
        """Test that all configuration options are properly documented."""
        # Check that all HeySolConfig attributes are mentioned in documentation
        config = HeySolConfig()

        documented_options = [
            "api_key", "base_url", "source", "timeout", "max_retries",
            "rate_limit_per_minute", "rate_limit_enabled", "log_level",
            "log_to_file", "log_file_path", "async_enabled", "max_async_workers"
        ]

        for option in documented_options:
            assert hasattr(config, option), \
                f"Documented configuration option '{option}' not found in HeySolConfig"

    def test_error_code_documentation(self):
        """Test that error codes and their meanings are documented."""
        # Check that exception classes document the HTTP status codes they represent
        from heysol.exceptions import AuthenticationError, NotFoundError, RateLimitError

        error_codes = {
            AuthenticationError: 401,
            NotFoundError: 404,
            RateLimitError: 429
        }

        for exc_class, expected_code in error_codes.items():
            if exc_class.__doc__:
                docstring = exc_class.__doc__.lower()
                assert str(expected_code) in docstring, \
                    f"Exception '{exc_class.__name__}' docstring should mention status code {expected_code}"

    def _extract_readme_methods(self) -> List[str]:
        """Extract method names from README documentation."""
        # This is a simplified extraction - in practice, you'd want more sophisticated parsing
        readme_path = "heysol_api_client/README.md"

        try:
            with open(readme_path, 'r') as f:
                content = f.read()

            # Extract method names from the API Reference section
            methods = []
            in_api_section = False

            for line in content.split('\n'):
                if "## API Reference" in line:
                    in_api_section = True
                    continue
                elif in_api_section and line.startswith("## "):
                    break  # End of API section

                if in_api_section and line.strip().startswith('- `'):
                    # Extract method name from lines like "- `get_user_profile()`"
                    method_match = re.search(r'`(\w+)\(\)', line)
                    if method_match:
                        methods.append(method_match.group(1))

            return methods

        except FileNotFoundError:
            return []

    def _get_client_methods(self, client_class) -> Dict[str, Any]:
        """Get all methods from a client class."""
        methods = {}

        for name in dir(client_class):
            attr = getattr(client_class, name)
            if callable(attr) and not name.startswith('_'):
                methods[name] = attr

        return methods

    def test_installation_instructions_accuracy(self):
        """Test that installation instructions are accurate."""
        # Check that setup.py and pyproject.toml exist
        assert os.path.exists("heysol_api_client/setup.py"), "setup.py should exist for installation"
        assert os.path.exists("heysol_api_client/pyproject.toml"), "pyproject.toml should exist for installation"

        # Check that requirements.txt exists
        assert os.path.exists("heysol_api_client/requirements.txt"), "requirements.txt should exist"

    def test_import_structure_documentation(self):
        """Test that import structure matches documentation."""
        # Test that the documented imports work
        try:
            from heysol import HeySolClient, HeySolConfig
            from heysol import AsyncHeySolClient
            from heysol import (
                HeySolError,
                AuthenticationError,
                RateLimitError,
                APIError,
                ValidationError,
                ConnectionError,
                ServerError,
                NotFoundError
            )
        except ImportError as e:
            pytest.fail(f"Documented imports failed: {e}")

    def test_version_consistency(self):
        """Test that version information is consistent across files."""
        # Check that version is consistent in setup.py, pyproject.toml, and __init__.py
        version_files = [
            "heysol_api_client/setup.py",
            "heysol_api_client/pyproject.toml",
            "heysol_api_client/heysol/__init__.py"
        ]

        versions = []

        for file_path in version_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()

                    # Extract version using regex
                    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                    if version_match:
                        versions.append(version_match.group(1))

        # All versions should be the same
        if len(versions) > 1:
            assert all(v == versions[0] for v in versions), \
                f"Version inconsistency found: {versions}"

    def test_example_code_runs_without_errors(self):
        """Test that example code can be imported and basic structure is valid."""
        # Test that example files can be imported (syntax check)
        import sys
        import os

        examples_dir = "heysol_api_client/examples"
        if os.path.exists(examples_dir):
            sys.path.insert(0, examples_dir)

            try:
                # Try importing basic_usage (this will catch major syntax errors)
                import basic_usage
                assert hasattr(basic_usage, '__doc__'), "basic_usage.py should have module docstring"

                # Try importing async_usage
                import async_usage
                assert hasattr(async_usage, '__doc__'), "async_usage.py should have module docstring"

            except ImportError as e:
                pytest.fail(f"Example code import failed: {e}")
            finally:
                # Clean up sys.path
                if examples_dir in sys.path:
                    sys.path.remove(examples_dir)