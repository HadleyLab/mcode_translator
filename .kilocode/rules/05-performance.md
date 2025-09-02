# Performance
- Favor **lean, allocation-aware** code: avoid unnecessary abstractions and convenience wrappers on hot paths.
- Complexity awareness: justify algorithms > O(n log n) on hot paths; include micro-benchmarks when changing them.
- Data handling: stream large files; avoid whole-file loads. Batch external calls; prevent N+1 patterns.
- Prefer vectorized **NumPy**/**pandas** ops where appropriate; avoid chained assignment in pandas.
- Provide simple before/after numbers for perf-impacting PRs.
