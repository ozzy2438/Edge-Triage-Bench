# Protocol (frozen before any test run)

**Task.** Single-label classification of a bank customer-service message into 8 triage labels:
`card_setup, card_problem, card_payment, cash_atm, top_up, transfer, exchange, account` (mapping from the 77 BANKING77 intents in `DATA.md`).

**Data.** `data/processed/{train,dev,test}.jsonl`, hashes in `DATA.md`. `dev` (100) is the only split used for prompt/parsing work. `test` (500) is run once per configuration after this file is committed. TF-IDF trains on `train` only.

**Configurations.** Every model below is self-quantised from the pinned HF revision (`MODELS.md`) with llama.cpp `1ab7e5a`.
- Qwen3-0.6B and Qwen3-1.7B: F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M
- SmolLM2-1.7B-Instruct: F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M
- Reference: Qwen3-8B Q4_K_M
- Baselines: majority class, and TF-IDF (1–2-grams, sublinear tf, min_df 2; min_df 1 below 800 examples) + logistic regression (C = 10)
- **TF-IDF learning curve.** The same pipeline is trained on 8 examples (the exact 8 few-shot dev examples the LLMs see), then on 80 and 800 examples (stratified samples of `train`, seed 42), and on the full `train` split (9,997). The **headline question** is at what training size TF-IDF matches the best small LLM. The crossing point is interpolated linearly in log(n), and each point is compared with the best small LLM by paired bootstrap.

**Runtime.** llama.cpp `1ab7e5a` built as Release with `GGML_METAL=OFF` (no GPU backend compiled; `GGML_BLAS=ON` means Apple Accelerate on CPU, and `GGML_CPU_REPACK=ON`). Served by `llama-server` with `-ngl 0 -t 4 -tb 4 -c 1024 -np 1 -s 42 -lm none`, CPU only, one configuration at a time with no other benchmark running. The build flags and server arguments are written to every `results/raw/**/*.meta.json`. Before measurement, 3 dev items are run as a warm-up and discarded. Messages are capped at 128 tokens by each model's tokenizer.

**Prompt** (as `src/etb/run.py` at this commit). The system prompt lists the 8 label words (`NewCard, CardIssue, Payment, ATM, TopUp, Transfer, Exchange, Account`), each with a one-line description. It is followed by 8 few-shot chat turns, one dev example per label (the first by id), then `Customer message: <text>`. Each model uses its own chat template, and Qwen3 runs with thinking disabled. Decoding is temperature 0, seed 42, and a GBNF grammar that allows only the 8 label words. `n_probs` is 50.

**Confidence.** The top-50 first-token probabilities (pre-grammar) are attributed to the label each token uniquely prefixes, then renormalised over the labels. Confidence is the probability of the predicted label. For TF-IDF it is the maximum of `predict_proba`, and for majority it is the class prior.

**Metrics** (`src/etb/metrics.py`), all reported on `test` for every configuration:
- Quality: accuracy, macro-F1 and per-class F1, with 1,000-resample bootstrap 95% CIs (seed 42). Invalid outputs count as wrong.
- Paired comparisons: a paired bootstrap of the macro-F1 difference (1,000 resamples on the same items, seed 42) gives a 95% CI and a two-sided p-value. It is run for each adjacent quantisation step of each small model (F16→Q8_0→Q5_K_M→Q4_K_M→Q3_K_M) and for each learning-curve point against the best small LLM. A difference is only called real when its CI excludes 0.
- Abstention: using the log-prob confidence above, the full selective-accuracy vs coverage curve (`results/figures/selective_accuracy.png`), selective accuracy at 50%, 70% and 90% coverage, and AURC. These appear in `results/summary.csv` and the README table.
- Calibration: ECE over 15 equal-width bins, and a reliability diagram for the best small configuration.
- Cost: GGUF or joblib file size, load time, peak RSS sampled every 100 ms (10 ms for the baselines), prompt and generation tokens/s, and total wall time.
- Latency, reported two ways. **Warm** is p50/p95/max over all 500 items; the fixed system and few-shot prefix is reused from the KV cache, so only the message tokens are processed. **Cold** is p50/p95 over the first 50 test items (file order), each re-run in the same server with `cache_prompt=false`, which processes the full prompt. The cold pass also reports how often its prediction matches the warm prediction. The baselines have no prompt, so cold equals warm and is shown as "–".

**Primary outcome.** Macro-F1 on `test` against peak RSS and p50 latency, and the TF-IDF learning-curve crossing point.

**Re-run rule.** If a bug is found after the test run, it is logged in `DECISIONS.md` and every configuration is re-run.

## Post-hoc analyses

Added after the test run, once the results above were already in hand. Not part of the frozen protocol (commit `8d3df10`).

- **Learning curve.** Training sizes 200 and 400 were added, and every size in {8, 80, 200, 400, 800} was repeated over 5 stratified seeds (42–46). Reported as mean ± SD. The crossover quoted in the README is an interpolation on those means, between the two measured sizes that bracket the best small LLM. The original single-seed curve (8 prompt examples, 80, 800, full train) is unchanged.
- **Holm correction.** The 12 adjacent quantisation comparisons are reported with Holm-adjusted p-values. README claims of a quantisation difference require the adjusted p to be at most 0.05. The uncorrected intervals are still in the table.
