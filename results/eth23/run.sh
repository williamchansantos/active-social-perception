#!/bin/bash
# All methods on the ETH-dataset scenes (ETH, Hotel), clean protocol, fresh seed 23, 300 episodes per scene.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes eth hotel --episodes 300 --seed 23 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True \
  --planners supermpc supermpc-infogain supermpc-active supermpc-360 --out results/eth23/obs.csv > results/eth23/obs.log 2>&1
echo "done obs $(grep -c FAILED results/eth23/obs.log) failures"
python evaluate_nav.py $COMMON --set unseen_joint=False \
  --planners dmpc supermpc supermpc-scan supermpc-nearest supermpc-infogain supermpc-active supermpc-gazeoracle supermpc-360 supermpc-oracle \
  --out results/eth23/open.csv > results/eth23/open.log 2>&1
echo "done open $(grep -c FAILED results/eth23/open.log) failures"
