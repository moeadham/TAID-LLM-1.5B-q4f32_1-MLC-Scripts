# TAID-LLM-1.5B-q4f32_1-MLC

Convert [SakanaAI/TAID-LLM-1.5B](https://huggingface.co/SakanaAI/TAID-LLM-1.5B) to MLC format with q4f32_1 quantization, so it can run in-browser via WebGPU using [web-llm](https://github.com/mlc-ai/web-llm).

## Why MLC format?

MLC (Machine Learning Compilation) is the format required by web-llm to run LLMs in the browser using WebGPU/WebAssembly. The conversion quantizes the model weights from BFloat16 to 4-bit integers with float32 scales (q4f32_1), reducing size from ~3GB to ~880MB while generating the config files web-llm needs to load and run the model.

## Why no WASM compilation step?

TAID-LLM-1.5B is a Qwen2 architecture model (identical to Qwen2-1.5B-Instruct except for vocab_size: 151,646 vs 151,936). A precompiled WebGPU WASM library already exists for Qwen2-1.5B and supports dynamic vocab sizes, so we only need to convert weights and generate config — no custom WASM compilation required.

The precompiled library: `Qwen2-1.5B-Instruct-q4f32_1_cs1k-webgpu.wasm`

## Remote machine requirements

- **CPU:** 4-8 vCPUs (e.g. AWS `c5.2xlarge`, GCP `e2-standard-8`)
- **RAM:** 16 GB
- **Disk:** 50 GB
- **GPU:** Not needed
- **OS:** Ubuntu 22.04 or 24.04

## Scripts

Run these in order on the remote machine. Each script is self-contained and activates its own conda environment.

| Script | What it does | Time |
|--------|-------------|------|
| `01-setup-env.sh` | Install Miniconda, Python 3.13, mlc_llm (CPU), git-lfs | ~3 min |
| `02-download-model.sh` | Clone TAID-LLM-1.5B from HuggingFace | ~2-5 min |
| `03-convert-weights.sh` | Quantize weights to q4f32_1 → `params_shard_*.bin` + `ndarray-cache.json` | ~5-15 min |
| `04-gen-config.sh` | Generate `mlc-chat-config.json` + copy tokenizer files | ~30 sec |
| `05-upload-hf.sh` | Upload converted model to your HuggingFace account | ~5 min |

```bash
bash 01-setup-env.sh
bash 02-download-model.sh
bash 03-convert-weights.sh
bash 04-gen-config.sh
HF_USERNAME=your-username bash 05-upload-hf.sh
```

## Output files

```
TAID-LLM-1.5B-q4f32_1-MLC/
├── mlc-chat-config.json      (~2 KB)     # model config, conv template, tokenizer refs
├── ndarray-cache.json         (~124 KB)   # weight shard manifest for web-llm
├── tokenizer.json             (~7 MB)
├── tokenizer_config.json      (~1.3 KB)
├── vocab.json                 (~2.8 MB)
├── merges.txt                 (~1.7 MB)
└── params_shard_*.bin         (~880 MB)   # quantized weight shards
```

## web-llm integration

```javascript
import * as webllm from "@mlc-ai/web-llm";

const appConfig = {
  model_list: [{
    model: "https://huggingface.co/YOUR_USERNAME/TAID-LLM-1.5B-q4f32_1-MLC",
    model_id: "TAID-LLM-1.5B-q4f32_1-MLC",
    model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_83/base/Qwen2-1.5B-Instruct-q4f32_1_cs1k-webgpu.wasm",
    vram_required_MB: 1889,
    low_resource_required: true,
  }],
};

const engine = await webllm.CreateMLCEngine("TAID-LLM-1.5B-q4f32_1-MLC", {
  appConfig,
});
```

## Reference

This conversion follows the same process used for [SakanaAI/TinySwallow-1.5B-Instruct-q4f32_1-MLC](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct-q4f32_1-MLC), another Qwen2-based model converted to MLC format.
