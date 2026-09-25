# Handoff

**State:** Phase 1 done and published after this push. Protocol frozen at `8d3df10`. Post-hoc (not a re-run of test): 5-seed TF-IDF curve at n=8/80/200/400/800, Holm on the 12 quant steps. Crossover is between 200 and 400 (interpolated ≈ 307). Cold pass is 22 min, 46% of warm+cold. `make smoke` is the fresh-clone check.
**Next:** Raspberry Pi 5 only if the owner asks. Results for another device go under `results/raw/<device>/test/`. Edit prose in `docs/*.tmpl.md`, then `uv run python -m etb.report`.
**Notes:** shell commands need full permissions (sandbox blocks the venv). New files sometimes need ~1 s before `python -m` sees them.
