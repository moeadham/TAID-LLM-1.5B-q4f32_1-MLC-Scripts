#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

echo "=== Step 3: Convert weights (q4f32_1 quantization) ==="
echo "This may take 5-15 minutes depending on CPU..."

python -m mlc_llm convert_weight \
    ./TAID-LLM-1.5B/ \
    --quantization q4f32_1 \
    -o ./TAID-LLM-1.5B-q4f32_1-MLC/

# Rename tensor-cache.json -> ndarray-cache.json for web-llm compatibility
# (mlc_llm renamed this in Sept 2025, but web-llm still expects the old name)
cd TAID-LLM-1.5B-q4f32_1-MLC/
if [ -f "tensor-cache.json" ] && [ ! -f "ndarray-cache.json" ]; then
    echo "Renaming tensor-cache.json -> ndarray-cache.json (web-llm compat)"
    mv tensor-cache.json ndarray-cache.json
fi
if [ -f "tensor-cache-b16.json" ] && [ ! -f "ndarray-cache-b16.json" ]; then
    mv tensor-cache-b16.json ndarray-cache-b16.json
fi
cd ..

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
