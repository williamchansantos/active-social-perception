#!/bin/bash
# Hybrid v2 (swept-path relevance of undetected masses) on the 600-episode tuning set.
cd "c:/Users/James/Desktop/active-social-perception-main"
until grep -qs "Saved" results/tune600/active_pp6.log; do sleep 30; done
COMMON="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0 --planners supermpc-active"
run () { name=$1; shift; python evaluate_nav.py $COMMON --set unseen_joint=False unseen_relevance=\"swept\" "$@" --out results/tune600/$name.csv > results/tune600/$name.log 2>&1; echo "done $name"; }
run swept_m05 unseen_swept_margin=0.5
run swept_m10 unseen_swept_margin=1.0
run swept_m05_wpu4 unseen_swept_margin=0.5 w_perception_unseen=4.0
run swept_m10_wpu4 unseen_swept_margin=1.0 w_perception_unseen=4.0
