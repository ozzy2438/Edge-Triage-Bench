# Memo: small quantised LLMs for message triage

*Independent benchmark on public data (BANKING77, 8 labels, 500 test items, CPU-only Apple M4). All numbers are produced by the pipeline.*

## Is this practical?

**Yes, but only as a cold start, with abstention switched on.** If the team has fewer than a few hundred labelled messages, take forward **`{best}`** on a laptop-class CPU:
- Macro-F1 {qwen3_17b_Q8_0__macro_f1:.3f} [{qwen3_17b_Q8_0__f1_lo:.3f}, {qwen3_17b_Q8_0__f1_hi:.3f}].
- {qwen3_17b_Q8_0__peak_rss_mb:,.0f} MB of RAM.
- {qwen3_17b_Q8_0__lat_p50_ms:.0f} ms per message with a cached prompt, {qwen3_17b_Q8_0__cold_p50_ms:,.0f} ms without.

If memory is tight, Q5_K_M is statistically indistinguishable from it ({pair__qwen3_17b__Q5_K_M:+.3f}, p = {pair__qwen3_17b__Q5_K_M__p:.2f}) and uses {qwen3_17b_Q5_K_M__peak_rss_mb:,.0f} MB.

**Once labelled examples are in the range {crossing_phrase}, switch to TF-IDF + logistic regression.** That range is a post-hoc interpolation (five seeds, added after the test run), not a measured cutoff. At 800 examples the seeded mean is already {lc__800__diff:+.3f} macro-F1 ahead of the LLM. With the full training split TF-IDF reaches {baseline_tfidf_lr__macro_f1:.3f}, using {baseline_tfidf_lr__peak_rss_mb:,.0f} MB and {baseline_tfidf_lr__lat_p50_ms:.1f} ms. For a closed label set with labelled history, the LLM is not the right tool.

## Where does it break?

- **Quantisation.** {quant_note} Q3_K_M is also slower than Q4_K_M on this CPU, so it saves memory and nothing else.
- **Model size.** Qwen3-0.6B tops out at {qwen3_06b_Q8_0__macro_f1:.3f}. That's too weak to route unsupervised.
- **Classes.** The weakest class for `{best}` is `{qwen3_17b_Q8_0__worst_class}` (F1 {qwen3_17b_Q8_0__worst_f1:.3f}). It is mostly misrouted to {qwen3_17b_Q8_0__worst_confused}.
- **Latency without a cache.** A cold full prompt costs about 10× a warm one, and the 8B reference needs {qwen3_8b_Q4_K_M__cold_p50_ms:,.0f} ms cold. Deployments must keep the few-shot prefix cached.

## What would need to change

- **Abstain and route.** Answer only the top 50–70% most confident messages. That lifts accuracy from {qwen3_17b_Q8_0__accuracy:.3f} to {qwen3_17b_Q8_0__sel_acc70:.3f}–{qwen3_17b_Q8_0__sel_acc50:.3f}. Send the rest to a person or to the 8B reference ({qwen3_8b_Q4_K_M__macro_f1:.3f} macro-F1, {qwen3_8b_Q4_K_M__sel_acc70:.3f} at 70% coverage). Set the threshold by rank, not by the raw probability: the model is overconfident (ECE {qwen3_17b_Q8_0__ece:.3f}).
- **Use the labels as they arrive.** Retrain TF-IDF weekly and switch over once it beats the LLM on a held-out set.
- **Label set.** Sharper descriptions for `card_problem` (for example, moving lost-phone and passcode intents to `account`) would target the largest single error source.

## What we would test next

1. **A cascade:** TF-IDF when it is confident, the LLM otherwise, measured at 80 and 800 labels.
2. **Fine-tuning** the 1.7B model (LoRA) on 800 examples, to see whether the crossing point moves.
3. **Raspberry Pi 5,** with the same frozen test set (Phase 3).
4. **Long-form complaint text,** such as CFPB narratives from the FOIA Reading Room, where TF-IDF's advantage may shrink.
