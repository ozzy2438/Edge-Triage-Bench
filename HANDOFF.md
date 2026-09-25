# Handoff

**State:** Phase 1 complete. Protocol `8d3df10`; test run once per config; `results/summary.csv`, 5 figures, README.md and MEMO.md rendered by `uv run python -m etb.report` from `docs/*.tmpl.md`. 18 tests pass.
**Next (owner decides):** Phase 2 (judge, needs owner-labelled sheet) or Phase 3 (Pi 5, `docs/PI_SETUP.md` not yet written). Results for another device go to `results/raw/<device>/test/`; set `DEVICE` in `configs.py`.
**Notes:** run shell with full permissions (sandbox blocks venv). New files sometimes need ~1 s before `python -m` sees them. Edit prose in `docs/*.tmpl.md`, never in README.md/MEMO.md directly.
