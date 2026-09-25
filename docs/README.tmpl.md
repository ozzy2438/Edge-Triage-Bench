# Edge Triage Bench

*Independent project using public data.*

**Question:** how small and how compressed can a language model get before it stops being useful for a real triage task, and can it tell when it is unsure? I routed 500 bank customer-service messages ([BANKING77](https://huggingface.co/datasets/PolyAI/banking77), 8 triage labels) with three small open models (0.6B–1.7B), each quantised from F16 down to Q3_K_M on a laptop CPU. They are compared against a TF-IDF classifier and an 8B reference. **Answer:** the best small LLM (`{best}`, 8-shot) reaches macro-F1 {qwen3_17b_Q8_0__macro_f1:.3f}. That beats a TF-IDF model trained on the same 8 examples by a wide margin, but a TF-IDF model overtakes it after about **{crossing_n:,.0f} labelled examples**. With the full training set, TF-IDF scores {baseline_tfidf_lr__macro_f1:.3f} in {baseline_tfidf_lr__peak_rss_mb:,.0f} MB of RAM, beating every LLM tested, including the 8B.

![Accuracy vs memory and latency](results/figures/headline.png)

![TF-IDF learning curve vs the best small LLM](results/figures/learning_curve.png)

## Results (test, n = 500, CPU-only)

{table}

*Invalid* is the share of outputs that could not be parsed as a label (a grammar restricts the output to label words). *Acc @ X% cov* is the accuracy when the model answers only its X% most confident items. *Warm* latency reuses the fixed prompt prefix from the KV cache. *Cold* latency processes the full prompt (first 50 test items). The full table is in `results/summary.csv`, per-class F1 in `results/per_class_f1_test.csv`, and every figure in `results/figures/`.

**Does quantisation level matter?** Paired bootstrap of the macro-F1 change at each step (1,000 resamples on the same items):

{pairs_table}

**TF-IDF learning curve vs `{best}`** (paired bootstrap):

{lc_table}

## Findings

- **How many labels does the LLM save you? About {crossing_n:,.0f}.** Trained on the 8 examples the LLMs see, TF-IDF scores {baseline_tfidf_lr_n8__macro_f1:.3f}; with 80 it scores {baseline_tfidf_lr_n80__macro_f1:.3f}, still {lc__80__vs_best_diff:+.3f} behind `{best}`. At 800 examples it is {lc__800__vs_best_diff:+.3f} ahead, and with the full training split it reaches {baseline_tfidf_lr__macro_f1:.3f}, ahead of the 8B reference ({qwen3_8b_Q4_K_M__macro_f1:.3f}) as well. A small LLM is a cold-start tool for this task, not the end state.
- **Q8_0 is free; Q3_K_M breaks every model.** F16 → Q8_0 is never significant (Qwen3-1.7B {pair__qwen3_17b__Q8_0:+.3f}, p = {pair__qwen3_17b__Q8_0__p:.2f}), and it halves memory. Q4_K_M → Q3_K_M is a significant drop for all three models ({pair__qwen3_17b__Q3_K_M:+.3f}, {pair__qwen3_06b__Q3_K_M:+.3f}, {pair__smollm2_17b__Q3_K_M:+.3f}). Between those two points the effect depends on the model: Qwen3-1.7B loses {pair__qwen3_17b__Q4_K_M:+.3f} at Q5_K_M → Q4_K_M, and Qwen3-0.6B at Q5_K_M is significantly *worse* than at Q4_K_M. Each quantised file has to be tested; bit count alone is not a reliable guide.
- **The confidence signal is usable for abstention even though the probabilities are not calibrated.** `{best}` goes from {qwen3_17b_Q8_0__accuracy:.3f} accuracy on everything to {qwen3_17b_Q8_0__sel_acc70:.3f} on its 70% most confident items and {qwen3_17b_Q8_0__sel_acc50:.3f} on the top 50%. But it is badly overconfident: {qwen3_17b_Q8_0__share_conf90:.0%} of items get confidence ≥ 0.9 (ECE {qwen3_17b_Q8_0__ece:.3f}). Use its confidence to rank items, not as a probability. SmolLM2 is better calibrated (ECE {smollm2_17b_Q8_0__ece:.3f}) but ranks worse (AURC {smollm2_17b_Q8_0__aurc:.3f} vs {qwen3_17b_Q8_0__aurc:.3f}).
- **Latency depends mostly on the prompt cache, not the quantisation level.** For `{best}`, warm p50 is {qwen3_17b_Q8_0__lat_p50_ms:.0f} ms and cold p50 (full prompt of about {qwen3_17b_Q8_0__cold_prompt_n:.0f} tokens) is {qwen3_17b_Q8_0__cold_p50_ms:,.0f} ms. The 8B reference takes {qwen3_8b_Q4_K_M__cold_p50_ms:,.0f} ms cold. Q3_K_M is *slower* than Q4_K_M (Qwen3-1.7B warm p50 {qwen3_17b_Q3_K_M__lat_p50_ms:.0f} vs {qwen3_17b_Q4_K_M__lat_p50_ms:.0f} ms): llama.cpp repacks Q4_K and Q8_0 weights for the ARM int8-matmul kernels but leaves q3_K on a generic path (see `DECISIONS.md`).
- **Per-class failures are concentrated.** `{best}` is weakest on `{qwen3_17b_Q8_0__worst_class}` (F1 {qwen3_17b_Q8_0__worst_f1:.3f}); its most common misroutes are to {qwen3_17b_Q8_0__worst_confused} (item counts). Lost-phone and passcode messages read like account issues, and card-not-arrived versus card-not-working is a thin line.

## Reproduce

```bash
make all      # llama.cpp build, data, download + quantise models, test bench, report
make test     # pytest: metric maths vs scikit-learn, label parsing, split integrity
```

Requirements: `uv`, `cmake`, git, and about 60 GB of free disk space (mostly the Qwen3-8B F16 intermediate). On the recorded machine the warm test pass across all configurations took {total_wall_min:.0f} min of inference. Model loads, warm-ups and the cold pass come on top of that, and so do download and quantisation time, which depend on the network.

- Machine: {machine}.
- Runtime: {runtime}.
- The protocol was frozen at commit **`{protocol_commit}`**, before any test run.

## Method in brief

- **Data.** BANKING77, with 77 intents mapped to 8 triage labels. `dev` has 100 items and `test` 500, both stratified (seed 42) from the official test split. TF-IDF trains on the official train split, with no overlap (checked by a test). See [DATA.md](DATA.md).
  - CFPB complaint narratives were the first choice but are no longer published (see [DECISIONS.md](DECISIONS.md)).
- **Models.** Qwen3-0.6B, Qwen3-1.7B and SmolLM2-1.7B-Instruct at F16, Q8_0, Q5_K_M, Q4_K_M and Q3_K_M, plus Qwen3-8B at Q4_K_M. All were converted and quantised locally from pinned Hugging Face revisions. See [MODELS.md](MODELS.md).
- **Inference.** `llama-server` on CPU only at temperature 0. The prompt uses 8 few-shot examples, one per label, taken from dev. A GBNF grammar restricts output to the label words. Confidence comes from first-token log-probabilities renormalised over the labels.
- **Metrics.** Bootstrap CIs, paired bootstrap tests, risk–coverage and AURC, ECE, peak RSS, and warm and cold latency. See [PROTOCOL.md](PROTOCOL.md).
- **Discipline.** All prompt work was done on dev only. Test was run once per configuration. Every number here is written by `src/etb/report.py`.

## Limitations

- **One task, one dataset.** Short English messages from a single UK-style digital bank. It is not complaint narratives, and the brief's CFPB data could not be used. Longer inputs would change both the accuracy and the latency picture.
- **Small sample.** 500 test items give macro-F1 CIs of about ±0.04. Differences smaller than that are not claimed unless the paired test supports them.
- **CPU-only, one machine.** An Apple M4 with 4 performance-core threads. GPU/Metal and the Raspberry Pi 5 are not measured yet.
- **One prompt, no fine-tuning.** Prompt tuning on dev was light (three variants). Fine-tuning the small models would likely move the crossing point.
- **Warm latency assumes a fixed, cached prompt prefix.** Cold numbers use a 50-item subset.
- **The 8B reference is Q4_K_M only.**
