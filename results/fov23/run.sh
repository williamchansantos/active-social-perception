#!/bin/bash
# Field-of-view sweep on the held-out Zara episodes (seed 23, 300 per scene), current clean protocol.
# 100 deg already exists in results/confirm23; the 360 camera does not depend on --fov.
cd "c:/Users/James/Desktop/active-social-perception-main"
until grep -qs "Saved" results/speed/open_R10.0.log; do sleep 60; done
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 23 --workers 13 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for fov in 87 60 120; do
  python evaluate_nav.py $COMMON --fov $fov --set unseen_joint=False unseen_obs_aware=True --planners supermpc-active --out results/fov23/obs_fov$fov.csv > results/fov23/obs_fov$fov.log 2>&1
  python evaluate_nav.py $COMMON --fov $fov --set unseen_joint=False --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle --out results/fov23/open_fov$fov.csv > results/fov23/open_fov$fov.log 2>&1
  echo "done fov=$fov failures: $(cat results/fov23/obs_fov$fov.log results/fov23/open_fov$fov.log | grep -c FAILED)"
done
