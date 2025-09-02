# Dependencies & Build
- Prefer stdlib first, then well-maintained deps (active maintainers, docs, tests).
- Use **uv/poetry** or **pip-tools** to lock versions; commit lockfiles.
- Remove unused deps in the same PR when discovered.
- Build must be reproducible locally and in CI with a single command.
