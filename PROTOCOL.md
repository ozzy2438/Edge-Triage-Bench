# Protocol (frozen before any test run)

**Task.** Single-label classification of a bank customer-service message into 8 triage labels:
`card_setup, card_problem, card_payment, cash_atm, top_up, transfer, exchange, account` (mapping from the 77 BANKING77 intents in `DATA.md`).

**Data.** `data/processed/{train,dev,test}.jsonl`, hashes in `DATA.md`. `dev` (100) is the only split used for prompt/parsing work. `test` (500) is run once per configuration after this file is committed. TF-IDF trains on `train` only.

**Configurations.** Every model below is self-quantised from the pinned HF revision (`MODELS.md`) with llama.cpp `1ab7e5a`.
- Qwen3-0.6B and Qwen3-1.7B: F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M
- SmolLM2-1.7B-Instruct: F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M
- Reference: Qwen3-8B Q4_K_M
- Baselines: majority class, and TF-IDF (1–2-grams, sublinear tf, min_df 2) + logistic regression (C = 10)

**Runtime.** `llama-server` with `-ngl 0 -t 4 -tb 4 -c 1024 -np 1 -s 42 -lm none`, CPU only, one configuration at a time with no other benchmark running. Before measurement, 3 dev items are run as a warm-up and discarded. Messages are capped at 128 tokens by each model's tokenizer.

**Prompt** (as `src/etb/run.py` at this commit). The system prompt lists the 8 label words (`NewCard, CardIssue, Payment, ATM, TopUp, Transfer, Exchange, Account`), each with a one-line description. It is followed by 8 few-shot chat turns, one dev example per label (the first by id), then `Customer message: <text>`. Each model uses its own chat template, and Qwen3 runs with thinking disabled. Decoding is temperature 0, seed 42, and a GBNF grammar that allows only the 8 label words. `n_probs` is 50.

**Confidence.** The top-50 first-token probabilities (pre-grammar) are attributed to the label each token uniquely prefixes, then renormalised over the labels. Confidence is the probability of the predicted label. For TF-IDF it is the maximum of `predict_proba`, and for majority it is the class prior.

**Metrics** (`src/etb/metrics.py`):
- Quality: accuracy, macro-F1 and per-class F1, with 1,000-resample bootstrap 95% CIs (seed 42). Invalid outputs count as wrong.
- Abstention: selective accuracy at 50%, 70% and 90% coverage, the full risk–coverage curve, and AURC.
- Calibration: ECE over 15 equal-width bins, and a reliability diagram.
- Cost: GGUF or joblib file size, load time, peak RSS sampled every 100 ms (10 ms for the baselines), latency p50/p95/max, prompt and generation tokens/s, and total wall time.

**Primary outcome.** Macro-F1 on `test` against peak RSS and p50 latency.

**Re-run rule.** If a bug is found after the test run, it is logged in `DECISIONS.md` and every configuration is re-run.
