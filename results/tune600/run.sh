#!/bin/bash
# Re-screen on a larger tuning set: seed 11, 300 episodes per Zara scene (first 100 reused), prune 1.0.
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
python evaluate_nav.py $COMMON --planners supermpc supermpc-infogain supermpc-active supermpc-360 --set unseen_joint=False --out results/tune600/base.csv > results/tune600/base.log 2>&1; echo done base
python evaluate_nav.py $COMMON --planners supermpc-hybrid --set unseen_joint=False risk_form=\"log\" w_entropy=1.0 unseen_floor=0.0 --out results/tune600/hyb_log_we1_uf0.csv > results/tune600/hyb_log_we1_uf0.log 2>&1; echo done h1
python evaluate_nav.py $COMMON --planners supermpc-hybrid --set unseen_joint=False risk_form=\"sqrt\" w_entropy=0.5 unseen_floor=1.0 --out results/tune600/hyb_sqrt_we05_uf1.csv > results/tune600/hyb_sqrt_we05_uf1.log 2>&1; echo done h2
python evaluate_nav.py $COMMON --planners supermpc-hybrid --set unseen_joint=False risk_form=\"log\" w_entropy=0.5 unseen_floor=1.0 --out results/tune600/hyb_log_we05_uf1.csv > results/tune600/hyb_log_we05_uf1.log 2>&1; echo done h3
python evaluate_nav.py $COMMON --planners supermpc-active --set unseen_joint=False perception_peds=6 --out results/tune600/active_pp6.csv > results/tune600/active_pp6.log 2>&1; echo done pp6
