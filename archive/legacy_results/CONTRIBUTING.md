# Contribution Guidelines

We welcome contributions to improve the mCODE Translator! Please follow these guidelines:

## Development Setup

1. Fork the repository
2. Clone your fork locally
3. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Contribution Process

1. Create a new branch for your feature/bugfix
2. Make your changes with tests
3. Run tests and linting:
   ```bash
   pytest
   flake8
   ```
4. Submit a pull request with:
   - Description of changes
   - Reference to related issues
   - Test results

## Documentation Standards

- Follow existing markdown style
- Use consistent headers and structure
- Include code examples where applicable
- Update all related documentation files

## Code Style

- Follow PEP 8 guidelines
- Type hints for all new code
- Docstrings for all public methods
- 100% test coverage for new features

## Reporting Issues

Include:
1. Expected vs actual behavior
2. Steps to reproduce
3. Environment details
4. Error logs if applicable