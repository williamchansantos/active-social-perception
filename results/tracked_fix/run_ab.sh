#!/bin/bash
# Variants of the proposed method aimed at flank crossers (tuning seed 11, 300 episodes per Zara scene).
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --planners supermpc-active"
python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True unseen_relevance=\"swept\" unseen_swept_margin=1.0 --out results/tracked_fix/A_swept.csv > results/tracked_fix/A_swept.log 2>&1; echo "done A $(grep -c FAILED results/tracked_fix/A_swept.log)"
python evaluate_nav.py $COMMON --set unseen_joint=False unseen_obs_aware=True w_cam_forward=0.0 --out results/tracked_fix/B_nofwd.csv > results/tracked_fix/B_nofwd.log 2>&1; echo "done B $(grep -c FAILED results/tracked_fix/B_nofwd.log)"
