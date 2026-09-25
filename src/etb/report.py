"""Build summary CSV, figures, and render README/MEMO templates from pipeline numbers only."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from etb.baselines import CURVE, seeded_curve
from etb.configs import DEVICE, FIGS, MODELS, QUANTS, RAW, REFERENCE, RESULTS, ROOT, model_of
from etb.data import LABEL_NAMES
from etb.metrics import holm, paired_bootstrap, reliability, risk_coverage, summarise

BASE = {"baseline-majority": "Majority class", "baseline-tfidf-lr": "TF-IDF + LogReg (full train)",
        **{f"baseline-tfidf-lr-n{n}": f"TF-IDF + LogReg (n={n})" for n in CURVE}}
COLORS = dict(zip(MODELS, ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]))


def load(split: str, device: str = DEVICE):
    rows, recs = [], {}
    for p in sorted((RAW / device / split).glob("*.jsonl")):
        r = [json.loads(l) for l in open(p)]
        meta = json.loads(p.with_suffix(".meta.json").read_text())
        cfg = p.stem
        model, quant = ("baseline", cfg) if cfg in BASE else (model_of(cfg), cfg[len(model_of(cfg)) + 1:])
        recs[cfg] = sorted(r, key=lambda x: x["id"])
        rows.append({"config": cfg, "model": model, "quant": quant, "device": device,
                     **{k: meta.get(k, np.nan) for k in ("file_mb", "load_s", "peak_rss_mb", "wall_s", "cold_p50_ms",
                                                         "cold_p95_ms", "cold_prompt_n", "cold_warm_agree")},
                     **summarise(r)})
    return pd.DataFrame(rows).set_index("config"), recs, meta["machine"]


def pareto(df):
    front, best = [], -1
    for cfg, r in df.sort_values("peak_rss_mb").iterrows():
        if r.macro_f1 > best:
            front.append(cfg)
            best = r.macro_f1
    return front


def label(cfg):
    return BASE.get(cfg, cfg)


def figures(df, recs, best):
    FIGS.mkdir(parents=True, exist_ok=True)
    llm = df[df.model != "baseline"]
    # 1. headline: macro-F1 vs peak RSS and vs p50 latency
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    front = pareto(df)
    curve = [f"baseline-tfidf-lr-n{n}" for n in CURVE if f"baseline-tfidf-lr-n{n}" in df.index]
    for ax, x, xl in [(axes[0], "peak_rss_mb", "Peak RSS (MB, log)"), (axes[1], "lat_p50_ms", "Warm latency p50 per item (ms, log)")]:
        for m, g in llm.groupby("model"):
            ax.scatter(g[x], g.macro_f1, s=40, color=COLORS[m], label=m + (" (reference)" if m == REFERENCE else ""), zorder=3)
            if x == "peak_rss_mb":
                for cfg, r in g.iterrows():
                    ax.annotate(r.quant, (r[x], r.macro_f1), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        ax.scatter(df.loc[curve, x], df.loc[curve, "macro_f1"], marker="*", s=70, color="grey",
                   label="TF-IDF + LogReg (n=" + "/".join(c.rsplit("n", 1)[1] for c in curve) + ")", zorder=4)
        for cfg, mk in [("baseline-tfidf-lr", "*"), ("baseline-majority", "X")]:
            if cfg in df.index:
                r = df.loc[cfg]
                ax.scatter(max(r[x], 0.1), r.macro_f1, marker=mk, s=140, color="black", label=label(cfg), zorder=4)
        if x == "peak_rss_mb":
            f = df.loc[front].sort_values(x)
            ax.step(f[x], f.macro_f1, where="post", color="orange", lw=1, ls="--", label="Pareto frontier", zorder=2)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[1].set_xlim(0.05, None)
    axes[1].annotate("majority plotted at 0.1 ms (actual ~0)", (0.1, 0.03), fontsize=6.5, xytext=(6, 4), textcoords="offset points")
    axes[0].set_ylabel("Macro-F1 (test, n=500)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Accuracy vs memory and latency (CPU-only, 4 threads, Apple M4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "headline.png", dpi=150)
    # 2. selective accuracy vs coverage: best quant per model + baselines
    fig, ax = plt.subplots(figsize=(6, 4))
    picks = list(llm.sort_values("macro_f1").groupby("model").tail(1).index) + ["baseline-tfidf-lr"]
    for cfg in picks:
        c = [r["conf"] for r in recs[cfg]]
        ok = [r["pred"] == r["label"] for r in recs[cfg]]
        cov, acc = risk_coverage(c, ok)
        ax.plot(cov * 100, acc, label=f"{label(cfg)} (AURC {df.loc[cfg].aurc:.3f})",
                color=COLORS.get(model_of(cfg) if cfg not in BASE else "", "black"))
    ax.set_xlabel("Coverage (% of items answered, most confident first)")
    ax.set_ylabel("Selective accuracy")
    ax.set_xlim(5, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "selective_accuracy.png", dpi=150)
    # 3. accuracy by quant level
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in llm[llm.model != REFERENCE].groupby("model"):
        g = g.set_index("quant").reindex(QUANTS)
        ax.errorbar(QUANTS, g.accuracy, yerr=[g.accuracy - g.acc_lo, g.acc_hi - g.accuracy], marker="o", capsize=3,
                    color=COLORS[m], label=m)
    for cfg, ls in [("baseline-tfidf-lr", "--"), (f"{REFERENCE}-Q4_K_M", ":")]:
        if cfg in df.index:
            ax.axhline(df.loc[cfg].accuracy, color="black", ls=ls, lw=1, label=label(cfg))
    ax.set_xlabel("Quantisation level (bits decrease to the right)")
    ax.set_ylabel("Accuracy (95% bootstrap CI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_by_quant.png", dpi=150)
    # 4. reliability diagram, best small config
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    rel = reliability([r["conf"] for r in recs[best]], [r["pred"] == r["label"] for r in recs[best]], bins=10)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.plot([c for c, _, _ in rel], [a for _, a, _ in rel], marker="o")
    for c, a, n in rel:
        ax.annotate(str(n), (c, a), fontsize=7, xytext=(3, -9), textcoords="offset points")
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(f"Reliability: {best} (ECE {df.loc[best].ece:.3f})", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "reliability.png", dpi=150)


def paired(recs, a, b) -> dict:
    y = [r["label"] for r in recs[a]]
    assert y == [r["label"] for r in recs[b]]
    return paired_bootstrap(y, [r["pred"] or "INVALID" for r in recs[a]], [r["pred"] or "INVALID" for r in recs[b]])


def quant_pairs(df, recs) -> pd.DataFrame:
    """Paired bootstrap of macro-F1 between adjacent quant levels of each small model."""
    rows = []
    for m in MODELS:
        if m == REFERENCE:
            continue
        for hi, lo in zip(QUANTS, QUANTS[1:]):
            a, b = f"{m}-{hi}", f"{m}-{lo}"
            if a in recs and b in recs:
                rows.append({"model": m, "from": hi, "to": lo, **paired(recs, b, a)})
    qp = pd.DataFrame(rows)
    qp["p_holm"] = holm(qp["p"]) if len(qp) else []
    return qp


def learning_curve(df, best):
    """Post-hoc curve: mean ± SD over 5 stratified seeds. Crossing is interpolated on the means."""
    seeds = seeded_curve()
    rows = []
    prompt = seeds[seeds.kind == "prompt"].iloc[0]
    rows.append({"n": 8, "name": "8 (the prompt examples)", "k": 1, "macro_f1": prompt.macro_f1, "sd": np.nan})
    for n, g in seeds[seeds.kind == "stratified"].groupby("n"):
        rows.append({"n": int(n), "name": f"{int(n)}", "k": len(g), "macro_f1": g.macro_f1.mean(), "sd": g.macro_f1.std(ddof=1)})
    full = seeds[seeds.kind == "full"].iloc[0]
    rows.append({"n": int(full.n), "name": f"{int(full.n):,} (full train)", "k": 1, "macro_f1": full.macro_f1, "sd": np.nan})
    lc = pd.DataFrame(rows)
    target = df.loc[best].macro_f1
    curve = lc[~lc.name.str.contains("prompt")]
    cross = lo = hi = None
    pts = list(curve[["n", "macro_f1"]].itertuples(index=False))
    for (n0, f0), (n1, f1) in zip(pts, pts[1:]):
        if f0 < target <= f1:
            lo, hi = int(n0), int(n1)
            cross = float(np.exp(np.log(n0) + (target - f0) / (f1 - f0) * (np.log(n1) - np.log(n0))))
            break
    return lc, cross, lo, hi


def crossing_phrase(cross, lo, hi) -> str:
    if cross is None:
        return "at a training size the curve does not bracket"
    return f"between {lo:,} and {hi:,} examples (interpolated ≈ {cross:,.0f})"


def lc_figure(df, lc, best, cross):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    seeded = lc[lc.k > 1]
    ax.errorbar(seeded.n, seeded.macro_f1, yerr=seeded.sd, marker="o", color="black", capsize=3,
                label="TF-IDF, mean ± SD (5 seeds)")
    single = lc[lc.k == 1]
    ax.scatter(single.n, single.macro_f1, marker="*", s=90, color="black", label="TF-IDF, single training set", zorder=3)
    for cfg, ls in [(best, "-"), (f"{REFERENCE}-Q4_K_M", ":")]:
        if cfg in df.index:
            ax.axhline(df.loc[cfg].macro_f1, color=COLORS[model_of(cfg)], ls=ls, label=f"{cfg} (8-shot)")
    if cross:
        ax.axvline(cross, color="orange", ls="--", lw=1, label=f"interpolated ≈ {cross:,.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("TF-IDF training examples (log scale)")
    ax.set_ylabel("Macro-F1 on the frozen test set")
    ax.set_title("Post-hoc, added after the test run", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curve.png", dpi=150)


def md(df: pd.DataFrame, fmt: dict) -> str:
    L = ["| " + " | ".join(fmt) + " |", "|" + "---|" * len(fmt)]
    L += ["| " + " | ".join(f(r) for f in fmt.values()) + " |" for _, r in df.iterrows()]
    return "\n".join(L)


def ms(x) -> str:
    return "–" if pd.isna(x) else f"{x:,.1f}"


def table(df) -> str:
    # single-n TF-IDF rows are the learning curve, reported there as mean ± SD
    df = df.loc[~df.index.str.startswith("baseline-tfidf-lr-n")]
    return md(df.sort_values("macro_f1", ascending=False).reset_index(), {
        "Config": lambda r: label(r.config),
        "Macro-F1 [95% CI]": lambda r: f"{r.macro_f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]",
        "Acc": lambda r: f"{r.accuracy:.3f}",
        "Acc @50/70/90% cov": lambda r: f"{r['sel_acc@50']:.3f} / {r['sel_acc@70']:.3f} / {r['sel_acc@90']:.3f}",
        "AURC": lambda r: f"{r.aurc:.3f}", "ECE": lambda r: f"{r.ece:.3f}", "Invalid": lambda r: f"{r.invalid_rate:.1%}",
        "File MB": lambda r: f"{r.file_mb:,.1f}", "Peak RSS MB": lambda r: f"{r.peak_rss_mb:,.0f}",
        "Warm p50 / p95 ms": lambda r: f"{ms(r.lat_p50_ms)} / {ms(r.lat_p95_ms)}",
        "Cold p50 / p95 ms": lambda r: f"{ms(r.cold_p50_ms)} / {ms(r.cold_p95_ms)}"})


def fmt_p(p) -> str:
    return "<0.001" if p == 0 else f"{p:.3f}"


def pairs_table(qp) -> str:
    return md(qp, {"Model": lambda r: r.model, "Step": lambda r: f"{r['from']} → {r['to']}",
                   "ΔMacro-F1 [95% CI]": lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]",
                   "p": lambda r: fmt_p(r.p), "p Holm": lambda r: fmt_p(r.p_holm),
                   "Holds after Holm": lambda r: "yes" if r.p_holm <= 0.05 else "no"})


def lc_table(lc, best, best_f1) -> str:
    def sd(r):
        return "–" if pd.isna(r.sd) else f"{r.sd:.3f}"
    return md(lc, {"Train examples": lambda r: str(r["name"]), "Seeds": lambda r: str(int(r.k)),
                   "Macro-F1 mean": lambda r: f"{r.macro_f1:.3f}", "SD": sd,
                   f"Mean − {best}": lambda r: f"{r.macro_f1 - best_f1:+.3f}"})


def quant_note(qp) -> str:
    """Prose that follows the Holm results, so a contrast is only called real when it survives."""
    def step(model, to):
        return qp[(qp.model == model) & (qp["to"] == to)].iloc[0]

    q3 = qp[qp["to"] == "Q3_K_M"]
    q3_txt = ", ".join(f"{r['diff']:+.3f}" for _, r in q3.iterrows())
    parts = [f"Of {len(qp)} adjacent steps, {(qp.p_holm <= 0.05).sum()} stay significant after a Holm correction "
             f"across all {len(qp)} (α = 0.05)."]
    if (qp[qp["to"] == "Q8_0"].p_holm > 0.05).all():
        parts.append("F16 → Q8_0 is not significant for any model, and it halves memory.")
    if (q3.p_holm <= 0.05).all() and (q3["diff"] < 0).all():
        parts.append(f"Q4_K_M → Q3_K_M stays a significant drop for all three ({q3_txt}).")
    else:
        parts.append("Q4_K_M → Q3_K_M is not significant for every model after Holm; see the table.")
    mid = step("qwen3-1.7b", "Q4_K_M")
    rev = step("qwen3-0.6b", "Q4_K_M")
    if mid.p_holm <= 0.05:
        parts.append(f"Qwen3-1.7B drops {abs(mid['diff']):.3f} at Q5_K_M → Q4_K_M (Holm p {fmt_p(mid.p_holm)}).")
    else:
        parts.append(f"Qwen3-1.7B's Q5_K_M → Q4_K_M change ({mid['diff']:+.3f}, uncorrected p {fmt_p(mid.p)}) "
                     f"does not survive Holm (p {fmt_p(mid.p_holm)}).")
    if rev.p_holm <= 0.05:
        parts.append(f"Qwen3-0.6B at Q5_K_M remains worse than at Q4_K_M after Holm "
                     f"(Q4 is {abs(rev['diff']):.3f} higher, p {fmt_p(rev.p_holm)}).")
    else:
        parts.append(f"Qwen3-0.6B Q5_K_M vs Q4_K_M ({rev['diff']:+.3f}, uncorrected p {fmt_p(rev.p)}) "
                     f"does not survive Holm (p {fmt_p(rev.p_holm)}); that reversal is suggestive only.")
    parts.append("Each quantised file still has to be tested; bit count alone is not a reliable guide.")
    return " ".join(parts)


def values(df, recs, machine, best, qp, lc, cross, lo, hi) -> dict:
    """Named numbers available to the README/MEMO templates."""
    b = machine.get("llama_cpp_build", {})
    best_f1 = float(df.loc[best].macro_f1)
    llm = df[df.model != "baseline"]
    phrase = crossing_phrase(cross, lo, hi)
    smoke = json.loads((RESULTS / "smoke.json").read_text()) if (RESULTS / "smoke.json").exists() else None
    v = {"table": table(df), "pairs_table": pairs_table(qp), "lc_table": lc_table(lc, best, best_f1),
         "best": best, "crossing_phrase": phrase, "quant_note": quant_note(qp),
         "headline": (f"An 8-shot 1.7B model (macro-F1 {best_f1:.3f}) beats TF-IDF trained on the same 8 examples, "
                      f"but TF-IDF overtakes it {phrase}."),
         # pinned: the commit that froze PROTOCOL.md before the test run. Later edits are marked post-hoc.
         "protocol_commit": "8d3df10",
         "machine": f"{machine['cpu']}, {machine['cores_logical']} cores, {machine['ram_gb']} GB RAM, {machine['os']}",
         "runtime": f"llama.cpp `{machine['llama_cpp_commit']}` ({b.get('CMAKE_BUILD_TYPE')}, GGML_METAL={b.get('GGML_METAL')}, "
                    f"GGML_BLAS={b.get('GGML_BLAS')} ({b.get('GGML_BLAS_VENDOR')}), GGML_CPU_REPACK={b.get('GGML_CPU_REPACK')}), "
                    f"`llama-server -ngl 0 -t {machine['threads']} -tb {machine['threads']}`",
         "warm_min": float(llm.wall_s.sum()) / 60,
         "cold_min": float((llm.cold_p50_ms / 1000 * 50).sum()) / 60,
         "cold_share": float((llm.cold_p50_ms / 1000 * 50).sum()) / float(llm.wall_s.sum() + (llm.cold_p50_ms / 1000 * 50).sum()),
         "smoke_line": (f"ran from a fresh clone: {smoke['config']} on {smoke['n']} dev items "
                        f"(macro-F1 {smoke['llm_macro_f1']:.3f}, median {smoke['llm_lat_p50_ms']:.0f} ms) and TF-IDF "
                        f"on the same items (macro-F1 {smoke['tfidf_macro_f1']:.3f})." if smoke else "not run yet.")}
    for _, r in qp.iterrows():
        v[f"pair__{r.model.replace('-', '_').replace('.', '')}__{r['to']}"] = r["diff"]
        v[f"pair__{r.model.replace('-', '_').replace('.', '')}__{r['to']}__p"] = r.p
    for _, r in lc.iterrows():
        key = "prompt" if "prompt" in str(r["name"]) else ("full" if "full" in str(r["name"]) else str(int(r.n)))
        v[f"lc__{key}__f1"] = float(r.macro_f1)
        v[f"lc__{key}__diff"] = float(r.macro_f1 - best_f1)
    for cfg, r in df.iterrows():
        k = cfg.replace("-", "_").replace(".", "")
        for m in ["macro_f1", "accuracy", "sel_acc@50", "sel_acc@70", "sel_acc@90", "ece", "aurc", "peak_rss_mb",
                  "lat_p50_ms", "lat_p95_ms", "file_mb", "invalid_rate", "wall_s", "f1_lo", "f1_hi", "label_mass",
                  "cold_p50_ms", "cold_p95_ms", "cold_prompt_n", "cold_warm_agree", "load_s"]:
            if m in r and pd.notna(r[m]):
                v[f"{k}__{m.replace('@', '')}"] = r[m]
        f1, worst = min((r[f"f1_{c}"], c) for c in LABEL_NAMES)
        v[f"{k}__worst_class"], v[f"{k}__worst_f1"] = worst, f1
        v[f"{k}__share_conf90"] = float(np.mean([x["conf"] >= 0.9 for x in recs[cfg]]))
        wrong = pd.Series([x["pred"] for x in recs[cfg] if x["label"] == worst and x["pred"] != worst]).value_counts()
        v[f"{k}__worst_confused"] = " and ".join(f"`{c}` ({n})" for c, n in wrong.head(2).items())
    return v


def render(v: dict) -> None:
    for name in ("README", "MEMO"):
        tmpl = ROOT / "docs" / f"{name}.tmpl.md"
        if tmpl.exists():
            (ROOT / f"{name}.md").write_text(tmpl.read_text().format_map(v))


def main(split="test"):
    df, recs, machine = load(split)
    small = df[(df.model != "baseline") & (df.model != REFERENCE)]
    best = small.macro_f1.idxmax()
    out = RESULTS / ("summary.csv" if split == "test" else f"summary_{split}.csv")
    per_class = [f"f1_{c}" for c in LABEL_NAMES]
    df.drop(columns=per_class).round(4).to_csv(out)
    df[per_class].round(4).to_csv(RESULTS / f"per_class_f1_{split}.csv")
    qp = quant_pairs(df, recs)
    lc, cross, lo, hi = learning_curve(df, best) if split == "test" else (None, None, None, None)
    sfx = "" if split == "test" else f"_{split}"
    qp.round(4).to_csv(RESULTS / f"paired_quant{sfx}.csv", index=False)
    if split == "test":
        lc.round(4).to_csv(RESULTS / "learning_curve.csv", index=False)
        figures(df, recs, best)
        lc_figure(df, lc, best, cross)
        v = values(df, recs, machine, best, qp, lc, cross, lo, hi)
        (RESULTS / "values.json").write_text(json.dumps({k: x for k, x in v.items() if not k.endswith("table")},
                                                         indent=1, default=float))
        render(v)
        print(lc_table(lc, best, df.loc[best].macro_f1), quant_note(qp), v["crossing_phrase"],
              f"cold {v['cold_min']:.0f} min, {v['cold_share']:.0%} of warm+cold", sep="\n\n")
    print(table(df), pairs_table(qp), sep="\n\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
