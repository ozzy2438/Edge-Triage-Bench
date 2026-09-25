"""Download original HF weights, convert to F16 GGUF, quantise with llama-quantize, record sizes + hashes."""
import hashlib
import json
import os
import subprocess
import sys

from huggingface_hub import list_repo_files

from etb.configs import LLAMA, LLAMA_BIN, MODELS, MODELS_DIR, QUANTS, REFERENCE, REFERENCE_QUANTS, ROOT, gguf_path

MANIFEST = MODELS_DIR / "manifest.json"


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build(name: str) -> None:
    repo, rev, _, _ = MODELS[name]
    hf_dir = MODELS_DIR / "hf" / name
    hf_dir.mkdir(parents=True, exist_ok=True)
    # curl with resume: huggingface_hub downloads stalled repeatedly on this network
    for f in list_repo_files(repo, revision=rev):
        if f.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")) and not (hf_dir / f).exists():
            part = hf_dir / (f + ".part")
            subprocess.run(["curl", "-sfL", "--retry", "20", "--retry-all-errors", "-C", "-", "-o", part,
                            f"https://huggingface.co/{repo}/resolve/{rev}/{f}"], check=True)
            part.rename(hf_dir / f)
    f16 = gguf_path(f"{name}-F16")
    if not f16.exists():
        env = {**os.environ, "PYTHONPATH": str(LLAMA / "gguf-py")}
        subprocess.run([sys.executable, LLAMA / "convert_hf_to_gguf.py", hf_dir,
                        "--outtype", "f16", "--outfile", f16], check=True, env=env)
    quants = REFERENCE_QUANTS if name == REFERENCE else QUANTS
    for q in quants:
        out = gguf_path(f"{name}-{q}")
        if q != "F16" and not out.exists():
            subprocess.run([LLAMA_BIN / "llama-quantize", f16, out, q], check=True,
                           stdout=subprocess.DEVNULL)


def write_manifest() -> None:
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        size = p.stat().st_size
        if old.get(p.stem, {}).get("bytes") != size:
            old[p.stem] = {"bytes": size, "sha256": sha256(p)}
    MANIFEST.write_text(json.dumps(old, indent=1))
    commit = subprocess.run(["git", "-C", LLAMA, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines = ["# Models", "", f"All GGUF files produced locally with llama.cpp `{commit}` "
             "(`convert_hf_to_gguf.py --outtype f16`, then `llama-quantize`). No pre-quantised files used.", "",
             "| Model | HF repo | Revision | Params | Licence | Gated |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | [{r}](https://huggingface.co/{r}) | `{rev[:12]}` | {p} | Apache-2.0 | no |"
              for n, (r, rev, _, p) in MODELS.items()]
    lines += ["", "| File | Size (MB) | SHA-256 |", "|---|---|---|"]
    lines += [f"| {k}.gguf | {v['bytes'] / 1e6:.0f} | `{v['sha256']}` |" for k, v in sorted(old.items())]
    (ROOT / "MODELS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    for name in sys.argv[1:] or MODELS:
        build(name)
    write_manifest()
