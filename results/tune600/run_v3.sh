#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
until grep -qs "Saved" results/tune600/swept_m10_wpu4.log; do sleep 30; done
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --set unseen_joint=False"
python evaluate_nav.py $COMMON --planners supermpc-gazeoracle supermpc-nearest supermpc-oracle --out results/tune600/oracles.csv > results/tune600/oracles.log 2>&1; echo done oracles
