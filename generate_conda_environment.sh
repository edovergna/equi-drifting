#!/usr/bin/env bash
set -euo pipefail

# Repository dir
REPO_DIR="${1:-$(pwd)}"

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
PIP_BIN="$REPO_DIR/.venv/bin/pip"
ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
  else
    echo "Error: Python/pip not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Run sync env from $REPO_DIR
echo "cd $REPO_DIR && uv sync"
cd "$REPO_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync

# Determine Python major.minor version from the env
echo ""$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --format freeze"
PIP_FREEZE=$(uv pip list --format freeze)

# Remove problematic packages
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch==')
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^pyg-lib==')
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch-scatter==')
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch-sparse==')
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch-cluster==')
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch-spline-conv==')

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

# Write conda YAML
{
  echo "name: equi"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - pip:"
  # ---- PyTorch (CUDA 12.6) ----
  echo '    - "torch==2.8.0"'
  echo '    - "--index-url=https://download.pytorch.org/whl/cu126"'
  echo '    - "--extra-index-url=https://pypi.org/simple"'
  # ---- PyG ecosystem (Torch 2.8, CUDA 12.6) ----
  echo '    - "pyg-lib"'
  echo '    - "torch-scatter"'
  echo '    - "torch-sparse"'
  echo '    - "torch-cluster"'
  echo '    - "torch-spline-conv"'
  echo '    - "--find-links=https://data.pyg.org/whl/torch-2.8.0+cu126.html"'

  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no additional pip packages detected"
  else
    while IFS= read -r pkg; do
      [ -z "$pkg" ] && continue
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

# Deactivate current uv env
echo "deactivate"
deactivate || true