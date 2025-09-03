# Environment-Scoped Execution (Conda `mcode_translator`)
- **All commands** for Python tooling **must run inside** the `mcode_translator` conda environment.
- Use the prefix form for every command:  
  `source activate mcode_translator && <command>`
- Examples (authoritative):
  - Python: `source activate mcode_translator && python -V`
  - Pip: `source activate mcode_translator && pip install -U pip`
  - Tests: `source activate mcode_translator && pytest -q`
  - Lint/Format: `source activate mcode_translator && ruff check . && black --check .`
  - Type check: `source activate mcode_translator && mypy --strict src`
- CI and local scripts must not invoke `python`, `pip`, or any tooling outside the activated environment.
- If the environment is missing or invalid, **abort immediately** with a clear error.
