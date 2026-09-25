# Handoff

**State:** All code written (data, quantise, run, measure, metrics, baselines, report); 16 tests pass. Dev runs done for qwen3 (0.6b, 1.7b) + baselines; prompt frozen (8-shot label words, see DECISIONS). PROTOCOL.md written. `make models` still producing smollm2-1.7b + qwen3-8b GGUFs.
**Next:** dev-run smollm2 + qwen3-8b, `uv run python -m etb.report dev`, commit PROTOCOL.md, STOP #2 (owner go-ahead). Then `make bench report`, write docs/README.tmpl.md + docs/MEMO.tmpl.md (placeholders from results/values.json), render.
**Notes:** run shell with full permissions (sandbox blocks venv). New files sometimes need ~1 s before `python -m` sees them.
