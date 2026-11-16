#!/bin/bash
# Phase A: stale-track handling on the tuning set (seed 11, Zara01+Zara02, 100 episodes each).
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 100 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --set unseen_joint=False --planners supermpc supermpc-scan supermpc-nearest supermpc-infogain supermpc-active supermpc-360"
for cfg in prune_pos_std=2.5 prune_pos_std=1.5 prune_pos_std=1.0 prune_pos_std=0.75 prune_pos_std=0.5 cap_pos_std=1.5 cap_pos_std=1.0 cap_pos_std=0.75; do
  python evaluate_nav.py $COMMON --pmb-set $cfg --out results/stale/$cfg.csv > results/stale/$cfg.log 2>&1
  echo "done $cfg $(grep -c FAILED results/stale/$cfg.log) failures"
done
