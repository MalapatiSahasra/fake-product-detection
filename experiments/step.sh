#!/bin/bash
# one resumable chunk (<3 min) of the revision experiments
cd "$(dirname "$0")/.."
export OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout 170 python3 experiments/compute_scores.py --workers 4 "$@" 2>&1 | grep -E "DONE|LEFT|BUDGET|Error|error"
