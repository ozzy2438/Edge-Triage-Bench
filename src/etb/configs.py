"""Single source of truth for paths, models, quant levels and run settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FIGS = RESULTS / "figures"
LLAMA = ROOT / "vendor" / "llama.cpp"
LLAMA_BIN = LLAMA / "build" / "bin"

SEED = 42
THREADS = 4          # M4 performance cores
CTX = 1024
MAX_NARRATIVE_TOKENS = 384
DEVICE = "mac-mini-m4-cpu"

# name -> (hf repo, pinned revision, family, params)
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "qwen", "0.6B"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "qwen", "1.7B"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", "smollm", "1.7B"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen", "8B"),
}
REFERENCE = "qwen3-8b"
QUANTS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
REFERENCE_QUANTS = ["Q4_K_M"]


def configs() -> list[str]:
    out = [f"{m}-{q}" for m in MODELS if m != REFERENCE for q in QUANTS]
    return out + [f"{REFERENCE}-{q}" for q in REFERENCE_QUANTS]


def gguf_path(config: str) -> Path:
    return MODELS_DIR / f"{config}.gguf"


def model_of(config: str) -> str:
    return next(m for m in MODELS if config.startswith(m + "-"))
