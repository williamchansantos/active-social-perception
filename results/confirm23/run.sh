#!/bin/bash
# Held-out confirmation: seed 23 (never used), Zara01 + Zara02, 300 episodes each, clean protocol.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 23 --workers 13 --fov 100 --range 4 --delta-unseen 0.02"
# proposed method + camera modes with the observation-aware unseen constraint
python evaluate_nav.py $COMMON --pmb-set prune_pos_std=1.0 --set unseen_joint=False unseen_obs_aware=True \
  --planners supermpc supermpc-infogain supermpc-active supermpc-360 --out results/confirm23/obs.csv > results/confirm23/obs.log 2>&1; echo done obs
# all baselines and oracles with the open-loop constraint (as evaluated so far)
python evaluate_nav.py $COMMON --pmb-set prune_pos_std=1.0 --set unseen_joint=False \
  --planners dmpc supermpc supermpc-scan supermpc-nearest supermpc-infogain supermpc-active supermpc-gazeoracle supermpc-360 supermpc-oracle \
  --out results/confirm23/open.csv > results/confirm23/open.log 2>&1; echo done open
# original stale-track threshold, for reference
python evaluate_nav.py $COMMON --pmb-set prune_pos_std=2.5 --set unseen_joint=False \
  --planners supermpc supermpc-infogain supermpc-active --out results/confirm23/prune25.csv > results/confirm23/prune25.log 2>&1; echo done prune25
