# Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-25 | **Dataset: BANKING77 fallback instead of CFPB.** | CFPB Release 23 (July 2026) removed consumer complaint narratives; the downloaded `complaints.csv.zip` (2026-09-24 build) has no narrative column and the API returns none. Brief §3.2 fallback. |
| 2026-09-25 | 8 coarse labels (see `DATA.md`), no intents dropped. | Brief asks ~8; grouping follows the support queue a message would be routed to. |
| 2026-09-25 | dev/test sampled from the official BANKING77 test split; TF-IDF trained on the full official train split minus overlaps (9,997 rows), not a 5,000 sample. | Guarantees disjointness by construction; more training data makes the classical baseline stronger, which makes the comparison harder for the LLMs, not easier. |
| 2026-09-25 | Models: Qwen3-0.6B, Qwen3-1.7B (Qwen family), SmolLM2-1.7B-Instruct (SmolLM family); reference Qwen3-8B at Q4_K_M. All Apache-2.0, ungated. | Two families, ungated so no licence STOP; Qwen3.5 skipped (multimodal hybrid arch, riskier conversion). |
| 2026-09-25 | Quant levels F16, Q8_0, Q5_K_M, Q4_K_M, Q3_K_M (all self-quantised). | F16 is the unquantised anchor; Q3_K_M probes where it breaks. |
| 2026-09-25 | llama.cpp `1ab7e5a`, built with `-DGGML_METAL=OFF`; runs with `-ngl 0 -t 4`. | CPU-only headline; 4 = M4 performance cores. CPU matmuls may use Apple Accelerate BLAS (still CPU). |
| 2026-09-25 | Prompt, tuned on dev only (qwen3-0.6b Q8_0 dev accuracy): letters A–H 0.23 → label words 0.12 (model answered the last-listed label 96/100) → label words + 8-shot (one dev example per label as chat turns) **0.53**. Frozen at the last. | Small models fail the letter indirection and show strong recency bias; one example per class fixes most of it. Dev scores include the 8 shot items (diagnostic only; test is untouched). |
| 2026-09-25 | Output constrained by a GBNF grammar to the 8 label words, which have distinct first tokens; confidence = first-token top-50 probabilities (raw logits, pre-grammar) attributed to the single label each token prefixes, renormalised. `label_mass` = raw mass on label tokens is logged as a format-adherence signal. | One forward pass gives both the prediction and a proper distribution over labels. Invalid-output rate is measured but expected ~0 because of the grammar. |
| 2026-09-25 | Qwen3 run in non-thinking mode (`enable_thinking=false` in the chat template). | Thinking would cost hundreds of tokens per item; the task is single-label routing. |
| 2026-09-25 | System prompt + few-shot turns are a fixed prefix reused through llama-server's prompt cache; per-item latency covers only the new message tokens plus decoding. | This is how a deployed classifier with a fixed prompt would run; noted as a limitation when comparing to other harnesses. |
| 2026-09-25 | Model loaded with `-lm none` (no mmap). | Peak RSS then reflects the full weights in RAM, not only touched pages. |
| 2026-09-25 | Raw results stored as `results/raw/<device>/<split>/<config>.jsonl` (+ `.meta.json`). | Keeps dev/test and laptop/Pi separate for Phase 3 without changing the format. |
| 2026-09-25 | HF weights fetched with resumable `curl` instead of `huggingface_hub`. | `snapshot_download` stalled twice on this network. |
