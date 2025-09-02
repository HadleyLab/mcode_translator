# Testing
- Use **pytest**. All new or changed code requires tests.
- Coverage thresholds: units ≥ 80%, critical modules ≥ 90%; branch coverage reported.
- Prefer deterministic, isolated tests with fakes over network/filesystem.
- Add **property-based tests** with **hypothesis** for parsers/validators.
- Each bug fix must include a regression test (fails pre-fix, passes post-fix).
