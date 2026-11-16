#!/bin/zsh
# Fixed-noise, cycle-only Steane/adaptive comparison; rerunning resumes batches.
set -euo pipefail

repo_root="${0:A:h:h}"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
experiment_python="${SPIDERWARP_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/qec/bin/python}"

# Keep macOS awake while this command runs. The display may sleep normally.
exec /usr/bin/caffeinate -i "$experiment_python" -u -m spiderwarp.se_comparison \
    --only adaptive_history steane_fixed_ps \
    --cycles 1 3 5 7 10 --basis both \
    --shots 10000 --batch-size 250 --jobs 4 \
    --seed 20261113 --repump \
    --results "$repo_root/benchmarks/sol_v2_overnight_10k" \
    "$@"
