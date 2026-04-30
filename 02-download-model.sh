#!/bin/bash
set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mlc-convert

WORKDIR="$HOME/mlc-workspace"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Step 2: Download TAID-LLM-1.5B ==="

if [ -d "TAID-LLM-1.5B" ]; then
    echo "Model directory already exists. Skipping download."
else
    echo "Cloning from HuggingFace (this may take a few minutes)..."
    git clone https://huggingface.co/SakanaAI/TAID-LLM-1.5B
fi

echo ""
echo "=== Verification ==="
ls -lh TAID-LLM-1.5B/
echo ""
python -c "
import json
c = json.load(open('TAID-LLM-1.5B/config.json'))
print(f'model_type:    {c[\"model_type\"]}')
print(f'vocab_size:    {c[\"vocab_size\"]}')
print(f'hidden_size:   {c[\"hidden_size\"]}')
print(f'num_layers:    {c[\"num_hidden_layers\"]}')
print(f'architecture:  {c[\"architectures\"][0]}')
"
echo ""
echo "=== Step 2 complete ==="
