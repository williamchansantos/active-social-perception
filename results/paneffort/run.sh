#!/bin/bash
# Elementary sweep of the camera pan-effort weight R[2] (base 0.05), tuning seed 11, 100 episodes per Zara scene.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 100 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --planners supermpc-active supermpc-infogain"
for w in 0.005 0.5 2.0; do
  python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True "R=(0.1,0.3,$w)" --out results/paneffort/R$w.csv > results/paneffort/R$w.log 2>&1
  echo "done $w $(grep -c FAILED results/paneffort/R$w.log) failures"
done
