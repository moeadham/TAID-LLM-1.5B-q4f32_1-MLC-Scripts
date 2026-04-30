#!/bin/bash
set -euo pipefail

# TAID-LLM-1.5B has vocab_size=151,646 but the precompiled
# Qwen2-1.5B-Instruct WASM expects vocab_size=151,936.
# This script pads the embedding/lm_head weights with zeros
# and adds the missing special tokens so the model matches
# the precompiled WASM at inference time.

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

MODEL_DIR="TAID-LLM-1.5B"

echo "=== Step 3: Pad vocab to match precompiled WASM ==="
echo "TAID vocab_size=151,646 -> Qwen2-1.5B-Instruct vocab_size=151,936"
echo ""

# Download the tokenizer files from the original Qwen2.5-1.5B-Instruct
# (the model the precompiled WASM was built for)
QWEN_BASE="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"
echo "Downloading Qwen2.5-1.5B-Instruct tokenizer files..."
wget -q -O "$MODEL_DIR/tokenizer_config.json" "$QWEN_BASE/tokenizer_config.json"
wget -q -O "$MODEL_DIR/tokenizer.json" "$QWEN_BASE/tokenizer.json"
echo "  tokenizer_config.json and tokenizer.json replaced with Qwen2.5-1.5B-Instruct versions"

python << 'PYEOF'
import json
import os
import torch
from safetensors.torch import load_file, save_file

MODEL_DIR = "TAID-LLM-1.5B"
OLD_VOCAB = 151646
NEW_VOCAB = 151936
PAD_ROWS = NEW_VOCAB - OLD_VOCAB  # 290

# --- 1. Update config.json ---
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
assert config["vocab_size"] == OLD_VOCAB, \
    f"Expected vocab_size={OLD_VOCAB}, got {config['vocab_size']}"
config["vocab_size"] = NEW_VOCAB
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[1/3] config.json: vocab_size {OLD_VOCAB} -> {NEW_VOCAB}")

# --- 2. Pad embedding + lm_head weights ---
EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

safetensor_files = sorted([
    f for f in os.listdir(MODEL_DIR)
    if f.endswith(".safetensors")
])
print(f"[2/3] Found safetensors: {safetensor_files}")

padded = set()
for sf_file in safetensor_files:
    sf_path = os.path.join(MODEL_DIR, sf_file)
    tensors = load_file(sf_path)

    modified = False
    for key in [EMBED_KEY, LM_HEAD_KEY]:
        if key in tensors:
            old = tensors[key]
            pad = torch.zeros(PAD_ROWS, old.shape[1], dtype=old.dtype)
            tensors[key] = torch.cat([old, pad], dim=0)
            print(f"      {key}: {list(old.shape)} -> {list(tensors[key].shape)}")
            padded.add(key)
            modified = True

    if modified:
        save_file(tensors, sf_path)
        print(f"      Saved {sf_file}")

assert EMBED_KEY in padded, f"{EMBED_KEY} not found!"
assert LM_HEAD_KEY in padded, f"{LM_HEAD_KEY} not found!"

# --- 3. Verify tokenizer matches ---
tc_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
with open(tc_path) as f:
    tc = json.load(f)
num_added = len(tc["added_tokens_decoder"])
max_id = max(int(k) for k in tc["added_tokens_decoder"])
print(f"[3/3] tokenizer_config.json: {num_added} added tokens, max id {max_id}")

print()
print("Vocab padding complete.")
PYEOF

echo ""
echo "=== Verification ==="
python -c "
import json
c = json.load(open('$MODEL_DIR/config.json'))
print(f'  vocab_size:       {c[\"vocab_size\"]}')
tc = json.load(open('$MODEL_DIR/tokenizer_config.json'))
print(f'  added_tokens:     {len(tc[\"added_tokens_decoder\"])}')
max_id = max(int(k) for k in tc['added_tokens_decoder'])
print(f'  max token id:     {max_id}')
print(f'  special_tokens:   {len(tc[\"additional_special_tokens\"])}')
"
echo ""
echo "=== Step 3 complete ==="
