# Security & Secrets
- **Never** commit secrets. Provide `.env.example`; validate required env vars on startup.
- Validate all untrusted inputs (type, range, length, whitelist); fail fast with clear exceptions.
- SQL access must use parameterized queries; forbid string-built SQL.
- Dependency audit in CI: `pip-audit` must report 0 high/critical issues or have explicit risk acceptance.
- Log sensitive data redacted/masked by default.
