#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://gitlab.mpcdf.mpg.de/ift/resolve"
REPO_DIR="resolve"

python -m pip install --upgrade pip pybind11 setuptools

if [ ! -d "$REPO_DIR" ]; then
  git clone --recursive "$REPO_URL" "$REPO_DIR"
fi

python -m pip install numpy matplotlib tqdm
python -m pip install "./$REPO_DIR"