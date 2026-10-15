#!/bin/bash
cd "$(dirname "$0")/.."
python3 experiments/compute_scores.py --perturb none --workers 3 > experiments/results/log_none.txt 2>&1
for p in blur jpeg lowlight lowres rotate perspective occlusion colorcast; do
  python3 experiments/compute_scores.py --perturb $p --originals_only --workers 3 > experiments/results/log_$p.txt 2>&1
done
echo ALLDONE > experiments/results/ALLDONE
