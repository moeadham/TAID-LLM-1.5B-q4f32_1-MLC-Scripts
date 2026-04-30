#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
export PATH="$BASEDIR/.conda-envs/mlc-convert/bin:$PATH"

# Set your HuggingFace username and token
HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_USERNAME" ] || [ -z "$HF_TOKEN" ]; then
    echo "Usage: HF_USERNAME=your-username HF_TOKEN=hf_xxxx bash 05-upload-hf.sh"
    exit 1
fi
export HF_TOKEN

REPO_NAME="TAID-LLM-1.5B-q4f32_1-MLC"

echo "=== Step 5: Upload to HuggingFace ==="
echo "Uploading to: ${HF_USERNAME}/${REPO_NAME}"

# Login with token
hf auth login --token "$HF_TOKEN"

# Create repo (ignore error if already exists)
hf repos create "${HF_USERNAME}/${REPO_NAME}" || true

# Upload all files
echo "Uploading files (this may take a while for ~880MB)..."
hf upload "${HF_USERNAME}/${REPO_NAME}" "./${REPO_NAME}/" .

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
