# Memo: small quantised LLMs for message triage

*Independent benchmark on public data (BANKING77, 8 labels, 500 test items, CPU-only Apple M4). All numbers are produced by the pipeline.*

## Is this practical?

**Yes, but only as a cold start, with abstention switched on.** If the team has fewer than a few hundred labelled messages, take forward **`qwen3-1.7b-Q8_0`** on a laptop-class CPU:
- Macro-F1 0.768 [0.730, 0.803].
- 2,779 MB of RAM.
- 137 ms per message with a cached prompt, 1,275 ms without.

If memory is tight, Q5_K_M is statistically indistinguishable from it (-0.009, p = 0.20) and uses 1,979 MB.

**Once about 407 labelled examples exist, switch to TF-IDF + logistic regression.** It already beats the LLM at 800 examples (+0.101 macro-F1). With the full training split it reaches 0.950, using 165 MB and 0.2 ms. For a closed label set with labelled history, the LLM is not the right tool.

## Where does it break?

- **Quantisation.** Q3_K_M is a significant drop for every model (-0.084 for Qwen3-1.7B, -0.165 for SmolLM2). It is also slower than Q4_K_M on this CPU, so it saves memory and nothing else. Below Q8_0 the losses are model-specific, so every quantised file needs its own test.
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
