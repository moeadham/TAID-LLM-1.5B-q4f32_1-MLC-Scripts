#!/bin/bash
set -euo pipefail

echo "=== Step 1: Install Miniconda + Python 3.13 + mlc_llm ==="

# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
else
    echo "Miniconda already installed."
fi

# Initialize conda for current shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Create conda environment
if conda env list | grep -q "mlc-convert"; then
    echo "Conda env 'mlc-convert' already exists."
else
    echo "Creating conda env with Python 3.13..."
    conda create --name mlc-convert python=3.13 -y
fi

conda activate mlc-convert

# Install mlc_llm (CPU-only)
echo "Installing mlc_llm nightly (CPU)..."
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu

# Install git-lfs
echo "Installing git-lfs..."
conda install -c conda-forge git-lfs -y
git lfs install

# Verify
echo ""
echo "=== Verification ==="
python --version
python -c "import mlc_llm; print(f'mlc_llm installed at: {mlc_llm.__path__}')"
mlc_llm --help | head -5
echo ""
echo "=== Step 1 complete ==="
