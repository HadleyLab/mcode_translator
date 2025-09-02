# Style & Linting (Python)
- Formatting: **black**; Import order: **isort**; Linting: **ruff**; Type checking: **mypy** in **strict** mode.
- Target Python >= 3.10; prefer pattern matching where clear; avoid magic numbers.
- 120-char soft line length; no trailing whitespace; final newline enforced.
- Public APIs must have docstrings (Google or NumPy style) and type hints.
- Forbid wildcard imports and unused code. No implicit re-exports.
