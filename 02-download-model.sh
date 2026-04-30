#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

echo "=== Step 2: Download TAID-LLM-1.5B ==="

# Clean previous runs
if [ -d "TAID-LLM-1.5B-q4f32_1-MLC" ]; then
    echo "Cleaning previous output directory..."
    rm -rf TAID-LLM-1.5B-q4f32_1-MLC
fi
if [ -d "TAID-LLM-1.5B" ]; then
    echo "Cleaning previous source model (may have been modified by step 3)..."
    rm -rf TAID-LLM-1.5B
fi

echo "Cloning from HuggingFace (this may take a few minutes)..."
git clone https://huggingface.co/SakanaAI/TAID-LLM-1.5B

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
