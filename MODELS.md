# Models

All GGUF files produced locally with llama.cpp `1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865` (`convert_hf_to_gguf.py --outtype f16`, then `llama-quantize`). No pre-quantised files used.

| Model | HF repo | Revision | Params | Licence | Gated |
|---|---|---|---|---|---|
| qwen3-0.6b | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a0` | 0.6B | Apache-2.0 | no |
| qwen3-1.7b | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `70d244cc86cc` | 1.7B | Apache-2.0 | no |
| smollm2-1.7b | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | `31b70e2e869a` | 1.7B | Apache-2.0 | no |
| qwen3-8b | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46` | 8B | Apache-2.0 | no |

| File | Size (MB) | SHA-256 |
|---|---|---|
| qwen3-0.6b-F16.gguf | 1509 | `e75dd5684fdd21c37aaf65043a1339afc7c6589ea3889d538dd42b04fe69c291` |
| qwen3-0.6b-Q3_K_M.gguf | 414 | `0ada84440456ede3e5f14856c403dbdf260de1ad0553ca1137895433a05203ce` |
| qwen3-0.6b-Q4_K_M.gguf | 484 | `182098b9a6e1ecb456cacd80d1001720f1567909c0e173927f74a6128ccbdebd` |
| qwen3-0.6b-Q5_K_M.gguf | 551 | `d31fa66a4fc15ea2b7bc2bf6995fa35999c864521ee080d655e64e3787eaa48c` |
| qwen3-0.6b-Q8_0.gguf | 805 | `75a964d8f5b1404b63086fdec95ae666099a97315d5f04fd8338261a9b07c04c` |
| qwen3-1.7b-F16.gguf | 4070 | `e1a47f1c4ab94ba47ceca2feda766f3bd3d70c846c98bb421bee36ec62af182d` |
| qwen3-1.7b-Q3_K_M.gguf | 1073 | `b1686bb0422c852170eaaa56cba826aba2ac21ea42a9609c6b34e2d7629eea2e` |
| qwen3-1.7b-Q4_K_M.gguf | 1282 | `7e71153c1134106ee8c957ee08b99c6d6baa9d1c21ff8b4d054717a921d5d873` |
| qwen3-1.7b-Q5_K_M.gguf | 1472 | `dcb4dbc18b466e894473cc378866101c6817cb0842e88fa8c8c1e0710d0c1683` |
| qwen3-1.7b-Q8_0.gguf | 2165 | `dff84b825885aa5cd00a0e49069d1a3e86157470774bc5eb52eed8b3494148c1` |
| qwen3-8b-F16.gguf | 16388 | `48b1f24d63179194435274bddb4d8a257e1b111809eb3480a2e1d583f2b521e2` |
| qwen3-8b-Q4_K_M.gguf | 5028 | `4d056e38325c0be25ee5ae014c1b6de755f08584c59d860173c828af6dbada51` |
| smollm2-1.7b-F16.gguf | 3425 | `b3067f36a7eca26f164d0c0e242e6b80f0a57b18d60afb38ddc9b43947d5d06a` |
| smollm2-1.7b-Q3_K_M.gguf | 860 | `5eb734960ad1eb039bd5bf13192d041a113ffa59d50a0dbdf3e19aa003223c2b` |
| smollm2-1.7b-Q4_K_M.gguf | 1056 | `d5c98ef94eeda37e8e16041a5f84f1f52b7ce756ab3126e4ba10225f16603a9b` |
| smollm2-1.7b-Q5_K_M.gguf | 1225 | `7f7aac63bac8453c0227a68fa229322f7903a3192c2aea72761d87f48281c476` |
| smollm2-1.7b-Q8_0.gguf | 1820 | `8405250ead5b81977a0da2820780f3cefcfb4f419eac067ab9ffa78fbefc728d` |
