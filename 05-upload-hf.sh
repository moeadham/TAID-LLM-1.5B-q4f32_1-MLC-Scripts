#!/bin/bash
set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mlc-convert

WORKDIR="$HOME/mlc-workspace"
cd "$WORKDIR"

# Set your HuggingFace username here
HF_USERNAME="${HF_USERNAME:-}"
if [ -z "$HF_USERNAME" ]; then
    echo "Usage: HF_USERNAME=your-username bash 05-upload-hf.sh"
    echo "  or:  export HF_USERNAME=your-username && bash 05-upload-hf.sh"
    exit 1
fi

REPO_NAME="TAID-LLM-1.5B-q4f32_1-MLC"

echo "=== Step 5: Upload to HuggingFace ==="
echo "Uploading to: ${HF_USERNAME}/${REPO_NAME}"

pip install -q huggingface-hub

# Login (will prompt for token if not already logged in)
huggingface-cli whoami || huggingface-cli login

# Create repo (ignore error if already exists)
huggingface-cli repo create "$REPO_NAME" --type model || true

# Upload all files
echo "Uploading files (this may take a while for ~880MB)..."
huggingface-cli upload "${HF_USERNAME}/${REPO_NAME}" "./${REPO_NAME}/" .

echo ""
echo "=== Step 5 complete ==="
echo "Model uploaded to: https://huggingface.co/${HF_USERNAME}/${REPO_NAME}"
echo ""
echo "To use in web-llm / localLlm.ts:"
echo ""
cat <<JSEOF
const appConfig = {
  model_list: [{
    model: "https://huggingface.co/${HF_USERNAME}/${REPO_NAME}",
    model_id: "${REPO_NAME}",
    model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_83/base/Qwen2-1.5B-Instruct-q4f32_1_cs1k-webgpu.wasm",
    vram_required_MB: 1889,
    low_resource_required: true,
  }],
};
JSEOF
