# Testing Structure

This directory contains all tests for the Mcode Translator project, organized by module and test type.

## Directory Structure

- `unit/` - Unit tests for individual functions and classes
  - `mock/` - Mocked unit tests that don't make external API calls
  - `live/` - Live unit tests that make actual API calls
  - `src/` - Unit tests organized by source module
    - `pipeline/` - Tests for pipeline modules
    - `optimization/` - Tests for optimization modules
    - `utils/` - Tests for utility modules
- `integration/` - Integration tests that test multiple components working together
- `e2e/` - End-to-end tests that test complete workflows
- `data/` - Test data files
- `runners/` - Test runners and execution scripts

## Test Organization

Tests are organized by source module to make it easy to find tests for specific functionality.
Mock tests are separated from live tests to allow for faster test execution and offline development.