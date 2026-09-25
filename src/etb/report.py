"""Build summary CSV, figures, and render README/MEMO templates from pipeline numbers only."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import subprocess

import numpy as np

from etb.baselines import CURVE
from etb.configs import DEVICE, FIGS, MODELS, QUANTS, RAW, REFERENCE, RESULTS, ROOT, model_of
from etb.data import LABEL_NAMES
from etb.metrics import paired_bootstrap, reliability, risk_coverage, summarise
from etb.run import load_split

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
    for ax, x, xl in [(axes[0], "peak_rss_mb", "Peak RSS (MB, log)"), (axes[1], "lat_p50_ms", "Latency p50 per item (ms, log)")]:
        for m, g in llm.groupby("model"):
            ax.scatter(g[x], g.macro_f1, s=40, color=COLORS[m], label=m + (" (reference)" if m == REFERENCE else ""), zorder=3)
            for cfg, r in g.iterrows():
                ax.annotate(r.quant, (r[x], r.macro_f1), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        for cfg, mk in [("baseline-tfidf-lr", "*"), ("baseline-majority", "X")]:
            if cfg in df.index:
                r = df.loc[cfg]
                ax.scatter(max(r[x], 1e-3), r.macro_f1, marker=mk, s=140, color="black", label=label(cfg), zorder=4)
        if x == "peak_rss_mb":
            f = df.loc[front].sort_values(x)
            ax.step(f[x], f.macro_f1, where="post", color="orange", lw=1, ls="--", label="Pareto frontier", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
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
    return pd.DataFrame(rows)


def learning_curve(df, recs, best) -> tuple[pd.DataFrame, float | None]:
    """TF-IDF macro-F1 by training size, paired against the best small LLM; interpolated crossing point (log n)."""
    n_full = len(load_split("train"))
    pts = [(n, f"baseline-tfidf-lr-n{n}") for n in CURVE] + [(n_full, "baseline-tfidf-lr")]
    rows = [{"n": n, "config": c, "macro_f1": df.loc[c].macro_f1, "f1_lo": df.loc[c].f1_lo, "f1_hi": df.loc[c].f1_hi,
             **{f"vs_best_{k}": v for k, v in paired(recs, c, best).items()}} for n, c in pts if c in df.index]
    lc = pd.DataFrame(rows)
    target, cross = df.loc[best].macro_f1, None
    for (n0, f0), (n1, f1) in zip(lc[["n", "macro_f1"]].values, lc[["n", "macro_f1"]].values[1:]):
        if f0 < target <= f1:
            cross = float(np.exp(np.log(n0) + (target - f0) / (f1 - f0) * (np.log(n1) - np.log(n0))))
            break
    if cross is None and len(lc) and lc.macro_f1.iloc[0] >= target:
        cross = float(lc.n.iloc[0])
    return lc, cross


def lc_figure(df, lc, best, cross):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(lc.n, lc.macro_f1, yerr=[lc.macro_f1 - lc.f1_lo, lc.f1_hi - lc.macro_f1], marker="*", ms=9,
                color="black", capsize=3, label="TF-IDF + LogReg")
    for cfg, ls in [(best, "-"), (f"{REFERENCE}-Q4_K_M", ":")]:
        if cfg in df.index:
            r = df.loc[cfg]
            ax.axhline(r.macro_f1, color=COLORS[model_of(cfg)], ls=ls, label=f"{cfg} (8-shot)")
            ax.axhspan(r.f1_lo, r.f1_hi, color=COLORS[model_of(cfg)], alpha=0.1)
    if cross:
        ax.axvline(cross, color="orange", ls="--", lw=1, label=f"crossing = {cross:,.0f} examples")
    ax.set_xscale("log")
    ax.set_xlabel("TF-IDF training examples (log; n=8 = the LLMs' few-shot examples)")
    ax.set_ylabel("Macro-F1 (test, 95% CI)")
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
    return md(df.sort_values("macro_f1", ascending=False).reset_index(), {
        "Config": lambda r: label(r.config),
        "Macro-F1 [95% CI]": lambda r: f"{r.macro_f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]",
        "Acc": lambda r: f"{r.accuracy:.3f}",
        "Acc @50/70/90% cov": lambda r: f"{r['sel_acc@50']:.3f} / {r['sel_acc@70']:.3f} / {r['sel_acc@90']:.3f}",
        "AURC": lambda r: f"{r.aurc:.3f}", "ECE": lambda r: f"{r.ece:.3f}", "Invalid": lambda r: f"{r.invalid_rate:.1%}",
        "File MB": lambda r: f"{r.file_mb:,.1f}", "Peak RSS MB": lambda r: f"{r.peak_rss_mb:,.0f}",
        "Warm p50 / p95 ms": lambda r: f"{ms(r.lat_p50_ms)} / {ms(r.lat_p95_ms)}",
        "Cold p50 / p95 ms": lambda r: f"{ms(r.cold_p50_ms)} / {ms(r.cold_p95_ms)}"})


def pairs_table(qp) -> str:
    return md(qp, {"Model": lambda r: r.model, "Step": lambda r: f"{r['from']} → {r['to']}",
                   "ΔMacro-F1 [95% CI]": lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]",
                   "p": lambda r: f"{r.p:.3f}", "Significant (CI excludes 0)": lambda r: "yes" if r.lo > 0 or r.hi < 0 else "no"})


def lc_table(lc, best) -> str:
    return md(lc, {"Train examples": lambda r: f"{int(r.n):,}", "Macro-F1 [95% CI]": lambda r: f"{r.macro_f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]",
                   f"Δ vs {best} [95% CI]": lambda r: f"{r.vs_best_diff:+.3f} [{r.vs_best_lo:+.3f}, {r.vs_best_hi:+.3f}]",
                   "p": lambda r: f"{r.vs_best_p:.3f}"})


def values(df, recs, machine, best, qp, lc, cross) -> dict:
    """Named numbers available to the README/MEMO templates."""
    b = machine.get("llama_cpp_build", {})
    v = {"table": table(df), "pairs_table": pairs_table(qp), "lc_table": lc_table(lc, best), "best": best,
         "crossing_n": cross if cross is not None else float("nan"),
         "protocol_commit": subprocess.run(["git", "log", "-1", "--format=%h", "--", "PROTOCOL.md"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
         "machine": f"{machine['cpu']}, {machine['cores_logical']} cores, {machine['ram_gb']} GB RAM, {machine['os']}",
         "runtime": f"llama.cpp `{machine['llama_cpp_commit']}` ({b.get('CMAKE_BUILD_TYPE')}, GGML_METAL={b.get('GGML_METAL')}, "
                    f"GGML_BLAS={b.get('GGML_BLAS')} ({b.get('GGML_BLAS_VENDOR')}), GGML_CPU_REPACK={b.get('GGML_CPU_REPACK')}), "
                    f"`llama-server -ngl 0 -t {machine['threads']} -tb {machine['threads']}`",
         "total_wall_min": df.wall_s.sum() / 60, "n_sig_pairs": int(((qp.lo > 0) | (qp.hi < 0)).sum()), "n_pairs": len(qp)}
    for _, r in qp.iterrows():
        v[f"pair__{r.model.replace('-', '_').replace('.', '')}__{r['to']}"] = r["diff"]
        v[f"pair__{r.model.replace('-', '_').replace('.', '')}__{r['to']}__p"] = r.p
    for _, r in lc.iterrows():
        v[f"lc__{int(r.n)}__vs_best_p"] = r.vs_best_p
        v[f"lc__{int(r.n)}__vs_best_diff"] = r.vs_best_diff
    for cfg, r in df.iterrows():
        k = cfg.replace("-", "_").replace(".", "")
        for m in ["macro_f1", "accuracy", "sel_acc@50", "sel_acc@70", "sel_acc@90", "ece", "aurc", "peak_rss_mb",
                  "lat_p50_ms", "lat_p95_ms", "file_mb", "invalid_rate", "wall_s", "f1_lo", "f1_hi", "label_mass",
                  "cold_p50_ms", "cold_p95_ms", "cold_prompt_n", "cold_warm_agree", "load_s"]:
            if m in r and pd.notna(r[m]):
                v[f"{k}__{m.replace('@', '')}"] = r[m]
        f1, worst = min((r[f"f1_{c}"], c) for c in LABEL_NAMES)
        v[f"{k}__worst_class"], v[f"{k}__worst_f1"] = worst, f1
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
    lc, cross = learning_curve(df, recs, best)
    sfx = "" if split == "test" else f"_{split}"
    qp.round(4).to_csv(RESULTS / f"paired_quant{sfx}.csv", index=False)
    lc.round(4).to_csv(RESULTS / f"learning_curve{sfx}.csv", index=False)
    if split == "test":
        figures(df, recs, best)
        lc_figure(df, lc, best, cross)
        v = values(df, recs, machine, best, qp, lc, cross)
        (RESULTS / "values.json").write_text(json.dumps({k: x for k, x in v.items() if not k.endswith("table")},
                                                         indent=1, default=float))
        render(v)
    print(table(df), pairs_table(qp), lc_table(lc, best), f"best small LLM: {best}; crossing n = {cross}", sep="\n\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
