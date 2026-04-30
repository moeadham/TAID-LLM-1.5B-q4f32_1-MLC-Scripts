#!/bin/bash
set -euo pipefail

# TAID-LLM-1.5B has vocab_size=151,646 but the precompiled
# Qwen2-1.5B-Instruct WASM expects vocab_size=151,936.
# This script pads the embedding weights with zeros and replaces
# the tokenizer with Qwen2.5-1.5B-Instruct's full tokenizer.

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

MODEL_DIR="TAID-LLM-1.5B"

echo "=== Step 3: Pad vocab to match precompiled WASM ==="
echo "TAID vocab_size=151,646 -> Qwen2-1.5B-Instruct vocab_size=151,936"
echo ""

# Download tokenizer files from the original Qwen2.5-1.5B-Instruct
# (the model the precompiled WASM was built for)
QWEN_BASE="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"
echo "Downloading Qwen2.5-1.5B-Instruct tokenizer files..."
wget -q -O "$MODEL_DIR/tokenizer_config.json" "$QWEN_BASE/tokenizer_config.json"
wget -q -O "$MODEL_DIR/tokenizer.json" "$QWEN_BASE/tokenizer.json"
echo "  Replaced tokenizer_config.json and tokenizer.json"

# Regenerate added_tokens.json from tokenizer_config.json
# (TAID's original only has 3 tokens, need all 22)
python -c "
import json
tc = json.load(open('$MODEL_DIR/tokenizer_config.json'))
added = {v['content']: int(k) for k, v in tc['added_tokens_decoder'].items()}
with open('$MODEL_DIR/added_tokens.json', 'w') as f:
    json.dump(added, f, indent=2, ensure_ascii=False)
print(f'  Regenerated added_tokens.json: {len(added)} tokens')
"

python << 'PYEOF'
import json
import os
import torch
from safetensors.torch import load_file, save_file

MODEL_DIR = "TAID-LLM-1.5B"
OLD_VOCAB = 151646
NEW_VOCAB = 151936

# --- 1. Update config.json ---
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)

if config["vocab_size"] == NEW_VOCAB:
    print("Already padded (vocab_size=151936). Skipping.")
    raise SystemExit(0)

assert config["vocab_size"] == OLD_VOCAB, \
    f"Unexpected vocab_size={config['vocab_size']}, expected {OLD_VOCAB}"
config["vocab_size"] = NEW_VOCAB
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"[1/2] config.json: vocab_size {OLD_VOCAB} -> {NEW_VOCAB}")

# --- 2. Pad model.embed_tokens.weight ---
# Both TAID and Qwen2.5-1.5B-Instruct use tie_word_embeddings=True,
# so lm_head shares embed_tokens — only one tensor to pad.
EMBED_KEY = "model.embed_tokens.weight"
PAD_ROWS = NEW_VOCAB - OLD_VOCAB  # 290

safetensor_files = sorted([
    f for f in os.listdir(MODEL_DIR)
    if f.endswith(".safetensors")
])
print(f"[2/2] Found safetensors: {safetensor_files}")

found = False
for sf_file in safetensor_files:
    sf_path = os.path.join(MODEL_DIR, sf_file)
    tensors = load_file(sf_path)

    if EMBED_KEY in tensors:
        old = tensors[EMBED_KEY]
        pad = torch.zeros(PAD_ROWS, old.shape[1], dtype=old.dtype)
        tensors[EMBED_KEY] = torch.cat([old, pad], dim=0)
        print(f"      {EMBED_KEY}: {list(old.shape)} -> {list(tensors[EMBED_KEY].shape)}")
        save_file(tensors, sf_path)
        found = True

assert found, f"{EMBED_KEY} not found in any safetensors file!"
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
"
echo ""
echo "=== Step 3 complete ==="
