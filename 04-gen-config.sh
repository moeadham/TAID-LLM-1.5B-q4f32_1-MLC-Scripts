#!/bin/bash
set -euo pipefail

BASEDIR="$(pwd)"
eval "$($BASEDIR/.conda/bin/conda shell.bash hook)"
conda activate mlc-convert

echo "=== Step 4: Generate MLC config ==="

python -m mlc_llm gen_config \
    ./TAID-LLM-1.5B/ \
    --quantization q4f32_1 \
    --conv-template qwen2 \
    -o ./TAID-LLM-1.5B-q4f32_1-MLC/

echo ""
echo "=== Verification ==="
echo "All output files:"
ls -lh TAID-LLM-1.5B-q4f32_1-MLC/
echo ""
echo "Config validation:"
python -c "
import json, os

c = json.load(open('TAID-LLM-1.5B-q4f32_1-MLC/mlc-chat-config.json'))
print(f'  model_type:       {c[\"model_type\"]}')
print(f'  quantization:     {c[\"quantization\"]}')
print(f'  vocab_size:       {c[\"vocab_size\"]}')
print(f'  context_window:   {c[\"context_window_size\"]}')
print(f'  conv_template:    {c[\"conv_template\"][\"name\"]}')
print(f'  stop_token_ids:   {c[\"conv_template\"][\"stop_token_ids\"]}')
print(f'  system_message:   {c[\"conv_template\"][\"system_message\"]}')
print()

# Check tokenizer files
for f in c.get('tokenizer_files', []):
    path = f'TAID-LLM-1.5B-q4f32_1-MLC/{f}'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  tokenizer: {f} ({size:,} bytes) OK')
    else:
        print(f'  tokenizer: {f} MISSING!')

# Assertions
assert c['model_type'] == 'qwen2', f'Wrong model_type: {c[\"model_type\"]}'
assert c['quantization'] == 'q4f32_1', f'Wrong quantization: {c[\"quantization\"]}'
assert c['vocab_size'] == 151646, f'Wrong vocab_size: {c[\"vocab_size\"]}'
assert c['model_config']['num_hidden_layers'] == 28
assert 151643 in c['conv_template']['stop_token_ids']
assert 151645 in c['conv_template']['stop_token_ids']
print()
print('All assertions passed!')
"
echo ""
echo "=== Step 4 complete ==="
echo ""
echo "Full mlc-chat-config.json:"
cat TAID-LLM-1.5B-q4f32_1-MLC/mlc-chat-config.json
