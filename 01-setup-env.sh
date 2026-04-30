#!/bin/bash
set -euo pipefail

# Install everything in the current working directory (network volume)
BASEDIR="$(pwd)"
CONDA_DIR="$BASEDIR/.conda"
PIP_CACHE="$BASEDIR/.pip-cache"

export TMPDIR="$BASEDIR/.tmp"
mkdir -p "$TMPDIR" "$PIP_CACHE"

echo "=== Step 1: Install Miniconda + Python 3.13 + mlc_llm ==="
echo "Installing to: $BASEDIR"

# Install Miniconda into pwd
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMPDIR/miniconda.sh"
    bash "$TMPDIR/miniconda.sh" -b -p "$CONDA_DIR"
    rm "$TMPDIR/miniconda.sh"
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# Initialize conda for current shell
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# Keep conda packages on the network volume too
conda config --append pkgs_dirs "$BASEDIR/.conda-pkgs"
conda config --append envs_dirs "$BASEDIR/.conda-envs"
mkdir -p "$BASEDIR/.conda-pkgs" "$BASEDIR/.conda-envs"

# Accept Anaconda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
if conda env list | grep -q "mlc-convert"; then
    echo "Conda env 'mlc-convert' already exists."
else
    echo "Creating conda env with Python 3.13..."
    conda create --name mlc-convert python=3.13 -y
fi

conda activate mlc-convert

# Install mlc_llm (CPU-only), cache to network volume
echo "Installing mlc_llm nightly (CPU)..."
pip install --cache-dir "$PIP_CACHE" --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu pytest

# Install torch (CPU-only) + safetensors for vocab padding step
echo "Installing torch (CPU) + safetensors for vocab padding..."
pip install --cache-dir "$PIP_CACHE" torch --index-url https://download.pytorch.org/whl/cpu
pip install --cache-dir "$PIP_CACHE" safetensors

# Install git-lfs
echo "Installing git-lfs..."
conda install -c conda-forge git-lfs -y
git lfs install

# Verify
echo ""
echo "=== Verification ==="
python --version
python -c "import mlc_llm; print(f'mlc_llm installed at: {mlc_llm.__path__}')"
python -m mlc_llm --help | head -5
echo ""
echo "=== Step 1 complete ==="
