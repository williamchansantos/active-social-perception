#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 31 --workers 13 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for fov in 60 45 70; do
  python evaluate_nav.py $COMMON --fov $fov --set unseen_joint=False unseen_obs_aware=True --planners supermpc-active --out results/narrow31/obs_fov$fov.csv > results/narrow31/obs_fov$fov.log 2>&1
  python evaluate_nav.py $COMMON --fov $fov --set unseen_joint=False --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle --out results/narrow31/open_fov$fov.csv > results/narrow31/open_fov$fov.log 2>&1
  echo "done fov=$fov failures: $(cat results/narrow31/obs_fov$fov.log results/narrow31/open_fov$fov.log | grep -c FAILED)"
done
python evaluate_nav.py $COMMON --fov 60 --set unseen_joint=False --planners supermpc-360 --out results/narrow31/open_360.csv > results/narrow31/open_360.log 2>&1
echo "done 360 failures: $(grep -c FAILED results/narrow31/open_360.log)"
