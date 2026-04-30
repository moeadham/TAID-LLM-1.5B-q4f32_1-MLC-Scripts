# TAID-LLM-1.5B-q4f32_1-MLC

Convert [SakanaAI/TAID-LLM-1.5B](https://huggingface.co/SakanaAI/TAID-LLM-1.5B) to MLC format with q4f32_1 quantization, so it can run in-browser via WebGPU using [web-llm](https://github.com/mlc-ai/web-llm).

## Why MLC format?

MLC (Machine Learning Compilation) is the format required by web-llm to run LLMs in the browser using WebGPU/WebAssembly. The conversion quantizes the model weights from BFloat16 to 4-bit integers with float32 scales (q4f32_1), reducing size from ~3GB to ~880MB while generating the config files web-llm needs to load and run the model.

## Why no WASM compilation step?

TAID-LLM-1.5B is a Qwen2 architecture model. A precompiled WebGPU WASM library already exists for Qwen2-1.5B-Instruct. TAID's original vocab_size (151,646) is smaller than Qwen2-1.5B-Instruct's (151,936), so we pad the embedding and lm_head weights with zeros during conversion to match the precompiled WASM.

The precompiled library: `Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm`

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
| `01-setup-env.sh` | Install Miniconda, Python 3.13, mlc_llm (CPU), torch, git-lfs | ~45 min |
| `02-download-model.sh` | Clone TAID-LLM-1.5B from HuggingFace | ~2 min |
| `03-pad-vocab.sh` | Pad vocab from 151,646 → 151,936 to match precompiled WASM | ~1 min |
| `04-convert-weights.sh` | Quantize weights to q4f32_1 → `params_shard_*.bin` + cache JSON | ~5 min |
| `05-gen-config.sh` | Generate `mlc-chat-config.json` + copy tokenizer files | ~3 mins |
| `06-upload-hf.sh` | Upload converted model to your HuggingFace account | ~5 min |

```bash
bash 01-setup-env.sh
bash 02-download-model.sh
bash 03-pad-vocab.sh
bash 04-convert-weights.sh
bash 05-gen-config.sh
HF_USERNAME=your-username HF_TOKEN=hf_xxxx bash 06-upload-hf.sh
```

## Output files

```
TAID-LLM-1.5B-q4f32_1-MLC/
├── mlc-chat-config.json      (~2 KB)     # model config, conv template, tokenizer refs
├── ndarray-cache.json         (~124 KB)   # weight shard manifest (web-llm ≤0.2.48)
├── tensor-cache.json          (~124 KB)   # weight shard manifest (web-llm ≥0.2.83)
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
    model_lib:
      webllm.modelLibURLPrefix +
      webllm.modelVersion +
      "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
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
