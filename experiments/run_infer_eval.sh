#!/usr/bin/env bash
# Run atom-type inference benchmark for both methods sequentially.
# Each method produces its own W&B run under project "infer-method-evaluation".
#
# Env overrides:
#   QM9_ROOT    path to QM9 data directory   (default: data/QM9)
#   N_MOLS      molecules to evaluate         (default: 5000)
#   OFFLINE     set to 1 to use W&B offline   (default: online)
#
# Usage:
#   bash experiments/run_infer_eval.sh
#   N_MOLS=1000 OFFLINE=1 bash experiments/run_infer_eval.sh

set -euo pipefail

ROOT="${QM9_ROOT:-data/QM9}"
N_MOLS="${N_MOLS:-5000}"
OFFLINE_FLAG=""
if [[ "${OFFLINE:-0}" == "1" ]]; then
    OFFLINE_FLAG="--offline"
fi

METHODS=("degree" "heuristic")

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "=============================="
    echo "  infer_method = ${METHOD}"
    echo "=============================="
    python experiments/evaluate_infer_methods.py \
        --infer_method "${METHOD}"  \
        --n_molecules  "${N_MOLS}"  \
        --root         "${ROOT}"    \
        ${OFFLINE_FLAG}
done

echo ""
echo "All evaluations complete."
