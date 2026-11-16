#!/bin/bash
# Perception-aware motion: unseen-constraint kernel width and risk budget (planner-level, all camera modes).
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --pmb-set prune_pos_std=1.0 --planners supermpc supermpc-infogain supermpc-active"
python evaluate_nav.py $COMMON --delta-unseen 0.02 --set unseen_joint=False unseen_kernel_width=0.5 --out results/tune600/mot_kw05_du02.csv > results/tune600/mot_kw05_du02.log 2>&1; echo done 1
python evaluate_nav.py $COMMON --delta-unseen 0.01 --set unseen_joint=False --out results/tune600/mot_kw025_du01.csv > results/tune600/mot_kw025_du01.log 2>&1; echo done 2
python evaluate_nav.py $COMMON --delta-unseen 0.01 --set unseen_joint=False unseen_kernel_width=0.5 --out results/tune600/mot_kw05_du01.csv > results/tune600/mot_kw05_du01.log 2>&1; echo done 3
