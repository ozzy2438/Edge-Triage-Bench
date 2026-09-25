LLAMA_COMMIT := 1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865
PY := uv run python -m

.PHONY: all llama data models bench-dev bench report test smoke

all: llama data models bench report

llama:
	test -d vendor/llama.cpp || git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
	cd vendor/llama.cpp && git fetch -q origin $(LLAMA_COMMIT) && git checkout -q $(LLAMA_COMMIT)
	cd vendor/llama.cpp && cmake -B build -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
	cd vendor/llama.cpp && cmake --build build -j 8 --target llama-server llama-quantize \
	  || cmake --build build -j 8 --target llama-server llama-quantize

data:
	$(PY) etb.data

models:
	$(PY) etb.quantise

bench-dev:
	$(PY) etb.run dev
	$(PY) etb.baselines dev

bench:
	$(PY) etb.run test
	$(PY) etb.baselines test

report:
	$(PY) etb.report

test:
	uv run pytest -q

smoke: llama data
	$(PY) etb.quantise qwen3-0.6b:Q4_K_M
	$(PY) etb.smoke
