#!/bin/bash
# Camera pan-rate limit sweep on the held-out episodes (seed 23), proposed-method settings, 0.4 s everywhere.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 23 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --set unseen_joint=False unseen_obs_aware=True"
python evaluate_nav.py $COMMON --planners supermpc-nearest --out results/panrate/rate2.5.csv > results/panrate/rate2.5.log 2>&1; echo "done 2.5 $(grep -c FAILED results/panrate/rate2.5.log)"
for w in 5.0 10.0; do
  python evaluate_nav.py $COMMON --robot-set "omega_cam_bounds=(-$w,$w)" --planners supermpc-active supermpc-infogain supermpc-nearest --out results/panrate/rate$w.csv > results/panrate/rate$w.log 2>&1
  echo "done $w $(grep -c FAILED results/panrate/rate$w.log)"
done
