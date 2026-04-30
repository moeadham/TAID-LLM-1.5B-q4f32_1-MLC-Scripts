#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

echo "=== Step 4: Convert weights (q4f32_1 quantization) ==="
echo "This may take 5-15 minutes depending on CPU..."

python -m mlc_llm convert_weight \
    ./TAID-LLM-1.5B/ \
    --quantization q4f32_1 \
    -o ./TAID-LLM-1.5B-q4f32_1-MLC/

# Keep both names: mlc_llm outputs tensor-cache.json (web-llm 0.2.83+),
# but older web-llm versions expect ndarray-cache.json.
cd TAID-LLM-1.5B-q4f32_1-MLC/
if [ -f "tensor-cache.json" ] && [ ! -f "ndarray-cache.json" ]; then
    echo "Copying tensor-cache.json -> ndarray-cache.json (compat with older web-llm)"
    cp tensor-cache.json ndarray-cache.json
fi
if [ -f "ndarray-cache.json" ] && [ ! -f "tensor-cache.json" ]; then
    echo "Copying ndarray-cache.json -> tensor-cache.json (compat with web-llm 0.2.83+)"
    cp ndarray-cache.json tensor-cache.json
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
echo "tensor-cache.json check:"
python -c "
import json, os
name = 'tensor-cache.json' if os.path.exists('TAID-LLM-1.5B-q4f32_1-MLC/tensor-cache.json') else 'ndarray-cache.json'
cache = json.load(open(f'TAID-LLM-1.5B-q4f32_1-MLC/{name}'))
meta = cache.get('metadata', {})
print(f'  ParamSize:    {meta.get(\"ParamSize\", \"N/A\")}')
print(f'  ParamBytes:   {meta.get(\"ParamBytes\", \"N/A\")}')
print(f'  BitsPerParam: {meta.get(\"BitsPerParam\", \"N/A\")}')
num_shards = len([r for r in cache.get('records', [])])
print(f'  Num shards:   {num_shards}')
"
echo ""
echo "=== Step 4 complete ==="
