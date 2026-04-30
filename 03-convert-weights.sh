#!/bin/bash
set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mlc-convert

WORKDIR="$HOME/mlc-workspace"
cd "$WORKDIR"

echo "=== Step 3: Convert weights (q4f32_1 quantization) ==="
echo "This may take 5-15 minutes depending on CPU..."

mlc_llm convert_weight \
    ./TAID-LLM-1.5B/ \
    --quantization q4f32_1 \
    -o ./TAID-LLM-1.5B-q4f32_1-MLC/

echo ""
echo "=== Verification ==="
echo "Output files:"
ls -lh TAID-LLM-1.5B-q4f32_1-MLC/
echo ""
echo "Total size:"
du -sh TAID-LLM-1.5B-q4f32_1-MLC/
echo ""
echo "ndarray-cache.json check:"
python -c "
import json
cache = json.load(open('TAID-LLM-1.5B-q4f32_1-MLC/ndarray-cache.json'))
meta = cache.get('metadata', {})
print(f'  ParamSize:    {meta.get(\"ParamSize\", \"N/A\")}')
print(f'  ParamBytes:   {meta.get(\"ParamBytes\", \"N/A\")}')
print(f'  BitsPerParam: {meta.get(\"BitsPerParam\", \"N/A\")}')
num_shards = len([r for r in cache.get('records', [])])
print(f'  Num shards:   {num_shards}')
"
echo ""
echo "=== Step 3 complete ==="
