# Using the AI Agent (Kilo Code)
- Before generating code, capture context: goal, constraints, interfaces, perf/security needs.
- When editing files, produce a diff summary and explain **why** changes are safe.
- Do not run shell commands that modify global system state without explicit approval.
- Prefer tests-first flow (write failing tests, then code, then docs update).
- If uncertain, present 2–3 implementation sketches with trade-offs.
