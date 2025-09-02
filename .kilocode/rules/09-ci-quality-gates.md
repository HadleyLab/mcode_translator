# CI Quality Gates (block on failure)
- Format (black) + lint (ruff) + type-check (mypy --strict) must pass.
- Tests must pass with coverage thresholds; publish coverage report/summary.
- Security audit must have 0 high/critical issues or an explicit risk acceptance file included in the PR.
- Enforce branch protection: required checks, linear history, at least one approving review.
