#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes eth hotel --episodes 300 --seed 23 --workers 13 --fov 60 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True --planners supermpc-active --out results/eth60/obs.csv > results/eth60/obs.log 2>&1
python evaluate_nav.py $COMMON --set unseen_joint=False --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle --out results/eth60/open.csv > results/eth60/open.log 2>&1
echo "done failures: $(cat results/eth60/obs.log results/eth60/open.log | grep -c FAILED)"
