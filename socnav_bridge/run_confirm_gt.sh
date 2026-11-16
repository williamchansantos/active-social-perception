#!/bin/bash
# Ground-truth batches of results/socnav/confirm/PLAN.md: 3 lanes (one per method), seeds 1-3 in sequence.
cd "c:/Users/James/Desktop/active-social-perception-main"
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
lane () {
  m=$1; base=$2
  for seed in 1 2 3; do
    DELETE_SIM=1 bash socnav_bridge/run_batch.sh conf_gt_${m}_s$seed $((base + seed)) "--camera $m --walls 8 --seed $seed" $EPS \
      >> results/socnav/confirm/progress.log 2>&1
    echo "done conf_gt_${m}_s$seed"
  done
}
lane active 2180 & lane infogain 2185 & lane fixed 2190 &
wait
echo "all gt done"
