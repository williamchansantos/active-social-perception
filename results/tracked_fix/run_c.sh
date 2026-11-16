#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
until grep -qs "Saved" results/tracked_fix/B_nofwd.log; do sleep 30; done
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --planners supermpc-active"
for w in 1.0 3.0; do
  python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True w_approach=$w --out results/tracked_fix/C_approach$w.csv > results/tracked_fix/C_approach$w.log 2>&1
  echo "done C $w $(grep -c FAILED results/tracked_fix/C_approach$w.log)"
done
