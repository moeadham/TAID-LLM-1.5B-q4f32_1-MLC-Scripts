#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
export PATH="$BASEDIR/.conda-envs/mlc-convert/bin:$PATH"

# Set your HuggingFace username and token
HF_USERNAME="${HF_USERNAME:-}"
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_USERNAME" ] || [ -z "$HF_TOKEN" ]; then
    echo "Usage: HF_USERNAME=your-username HF_TOKEN=hf_xxxx bash 06-upload-hf.sh"
    exit 1
fi
export HF_TOKEN

REPO_NAME="TAID-LLM-1.5B-q4f32_1-MLC"

echo "=== Step 6: Upload to HuggingFace ==="
echo "Uploading to: ${HF_USERNAME}/${REPO_NAME}"

# Copy model card and license into output dir
cp "$BASEDIR/MODEL_CARD.md" "./${REPO_NAME}/README.md"
if [ -f "./TAID-LLM-1.5B/LICENSE" ]; then
    cp "./TAID-LLM-1.5B/LICENSE" "./${REPO_NAME}/LICENSE"
elif [ -f "./TAID-LLM-1.5B/LICENCE" ]; then
    cp "./TAID-LLM-1.5B/LICENCE" "./${REPO_NAME}/LICENSE"
fi
echo "Copied MODEL_CARD.md -> README.md and LICENSE"

# Login with token
hf auth login --token "$HF_TOKEN"

# Create repo (ignore error if already exists)
hf repos create "${HF_USERNAME}/${REPO_NAME}" || true

# Upload all files, removing any stale files on the remote
echo "Uploading files (this may take a while for ~880MB)..."
hf upload "${HF_USERNAME}/${REPO_NAME}" "./${REPO_NAME}/" . --delete="*"

echo ""
echo "=== Step 6 complete ==="
echo "Model uploaded to: https://huggingface.co/${HF_USERNAME}/${REPO_NAME}"
echo ""
echo "To use in web-llm / localLlm.ts:"
echo ""
cat <<JSEOF
const appConfig = {
  model_list: [{
    model: "https://huggingface.co/${HF_USERNAME}/${REPO_NAME}",
    model_id: "${REPO_NAME}",
    model_lib:
      webllm.modelLibURLPrefix +
      webllm.modelVersion +
      "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    vram_required_MB: 1889,
    low_resource_required: true,
  }],
};
JSEOF
