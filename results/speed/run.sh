#!/bin/bash
# Forward-speed effort weight R[0] (base 0.1): slower robot, all camera strategies, tuning seed 11, 300 episodes per Zara scene.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for r in 3.0 10.0; do
  python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True "R=($r,0.3,0.05)" --planners supermpc-active --out results/speed/obs_R$r.csv > results/speed/obs_R$r.log 2>&1
  python evaluate_nav.py $COMMON --set unseen_joint=False "R=($r,0.3,0.05)" --planners supermpc-infogain supermpc-nearest supermpc supermpc-360 --out results/speed/open_R$r.csv > results/speed/open_R$r.log 2>&1
  echo "done R=$r failures: $(cat results/speed/obs_R$r.log results/speed/open_R$r.log | grep -c FAILED)"
done
