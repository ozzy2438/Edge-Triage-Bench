# Memo: small quantised LLMs for message triage

*Independent benchmark on public data (BANKING77, 8 labels, 500 test items, CPU-only Apple M4). All numbers are produced by the pipeline.*

## Is this practical?

**Yes, but only as a cold start, with abstention switched on.** If the team has fewer than a few hundred labelled messages, take forward **`qwen3-1.7b-Q8_0`** on a laptop-class CPU:
- Macro-F1 0.768 [0.730, 0.803].
- 2,779 MB of RAM.
- 137 ms per message with a cached prompt, 1,275 ms without.

If memory is tight, Q5_K_M is statistically indistinguishable from it (-0.009, p = 0.20) and uses 1,979 MB.

**Once labelled examples are in the range between 200 and 400 examples (interpolated ≈ 307), switch to TF-IDF + logistic regression.** That range is a post-hoc interpolation (five seeds, added after the test run), not a measured cutoff. At 800 examples the seeded mean is already +0.086 macro-F1 ahead of the LLM. With the full training split TF-IDF reaches 0.950, using 165 MB and 0.2 ms. For a closed label set with labelled history, the LLM is not the right tool.

## Where does it break?

- **Quantisation.** Of 12 adjacent steps, 7 stay significant after a Holm correction across all 12 (α = 0.05). F16 → Q8_0 is not significant for any model, and it halves memory. Q4_K_M → Q3_K_M stays a significant drop for all three (-0.143, -0.084, -0.165). Qwen3-1.7B drops 0.048 at Q5_K_M → Q4_K_M (Holm p <0.001). Qwen3-0.6B at Q5_K_M remains worse than at Q4_K_M after Holm (Q4 is 0.048 higher, p <0.001). Each quantised file still has to be tested; bit count alone is not a reliable guide. Q3_K_M is also slower than Q4_K_M on this CPU, so it saves memory and nothing else.
- **Model size.** Qwen3-0.6B tops out at 0.616. That's too weak to route unsupervised.
- **Classes.** The weakest class for `qwen3-1.7b-Q8_0` is `card_problem` (F1 0.652). It is mostly misrouted to `account` (10) and `card_setup` (9).
- **Latency without a cache.** A cold full prompt costs about 10× a warm one, and the 8B reference needs 7,679 ms cold. Deployments must keep the few-shot prefix cached.

## What would need to change

- **Abstain and route.** Answer only the top 50–70% most confident messages. That lifts accuracy from 0.766 to 0.871–0.908. Send the rest to a person or to the 8B reference (0.813 macro-F1, 0.917 at 70% coverage). Set the threshold by rank, not by the raw probability: the model is overconfident (ECE 0.215).
- **Use the labels as they arrive.** Retrain TF-IDF weekly and switch over once it beats the LLM on a held-out set.
- **Label set.** Sharper descriptions for `card_problem` (for example, moving lost-phone and passcode intents to `account`) would target the largest single error source.

## What we would test next

1. **A cascade:** TF-IDF when it is confident, the LLM otherwise, measured at 80 and 800 labels.
2. **Fine-tuning** the 1.7B model (LoRA) on 800 examples, to see whether the crossing point moves.
3. **Raspberry Pi 5,** with the same frozen test set (Phase 3).
4. **Long-form complaint text,** such as CFPB narratives from the FOIA Reading Room, where TF-IDF's advantage may shrink.
