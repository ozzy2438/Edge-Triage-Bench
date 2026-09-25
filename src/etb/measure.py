"""Peak-RSS sampler and machine description."""
import platform
import subprocess
import threading

import psutil

from etb.configs import LLAMA, THREADS


class PeakRSS:
    """Samples RSS of a process every `interval` s in a background thread."""

    def __init__(self, pid: int, interval: float = 0.1):
        self.proc, self.interval, self.peak = psutil.Process(pid), interval, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()

    @property
    def peak_mb(self) -> float:
        return self.peak / 1e6


def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


BUILD_KEYS = ("CMAKE_BUILD_TYPE", "GGML_METAL", "GGML_BLAS", "GGML_BLAS_VENDOR", "GGML_CPU_REPACK", "GGML_NATIVE")


def build_flags() -> dict:
    cache = LLAMA / "build" / "CMakeCache.txt"
    kv = dict(l.split(":", 1)[0:1] + l.split("=", 1)[1:] for l in cache.read_text().splitlines()
              if "=" in l and ":" in l.split("=", 1)[0]) if cache.exists() else {}
    return {k: kv.get(k) for k in BUILD_KEYS}


def machine_info() -> dict:
    cpu = _sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor()
    return {"cpu": cpu, "cores_physical": psutil.cpu_count(logical=False), "cores_logical": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30), "os": f"{platform.system()} {platform.release()}",
            "threads": THREADS, "llama_cpp_commit": _sh("git", "-C", str(LLAMA), "rev-parse", "--short", "HEAD"),
            "llama_cpp_build": build_flags()}
