---
license: apache-2.0
base_model: SakanaAI/TAID-LLM-1.5B
tags:
  - mlc
  - webgpu
  - web-llm
  - qwen2
  - quantized
library_name: mlc-llm
pipeline_tag: text-generation
---

# TAID-LLM-1.5B-q4f32_1-MLC

This is the [SakanaAI/TAID-LLM-1.5B](https://huggingface.co/SakanaAI/TAID-LLM-1.5B) model converted to MLC format with q4f32_1 quantization for in-browser inference via WebGPU using [web-llm](https://github.com/mlc-ai/web-llm). Conversion scripts are [here](https://github.com/moeadham/TAID-LLM-1.5B-q4f32_1-MLC-Scripts)

## Model Details

- **Base model:** [SakanaAI/TAID-LLM-1.5B](https://huggingface.co/SakanaAI/TAID-LLM-1.5B)
- **Architecture:** Qwen2 (1.5B parameters)
- **Quantization:** q4f32_1 (4-bit weights, float32 scales)
- **Quantized size:** ~880 MB
- **Training method:** TAID (Temporally Adaptive Interpolated Distillation) from Qwen2.5-32B-Instruct
- **Context window:** 32,768 tokens
- **License:** Apache 2.0

## Usage with web-llm

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
  }],
};

const engine = await webllm.CreateMLCEngine("TAID-LLM-1.5B-q4f32_1-MLC", { appConfig });
```

## Disclaimer

This model is provided for research and development purposes only. It is an experimental prototype and is not intended for commercial use or deployment in mission-critical environments. See the [original model card](https://huggingface.co/SakanaAI/TAID-LLM-1.5B) for full details.
