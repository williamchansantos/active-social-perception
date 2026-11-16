#!/bin/bash
# Phase B stage 2: one-factor-at-a-time around the active objective (tuning set, seed 11, prune 1.0).
cd "c:/Users/James/Desktop/active-social-perception-main"
COMMON="--protocol clean --scenes zara01 zara02 --episodes 100 --seed 11 --workers 13 --fov 100 --range 4 --pmb-set prune_pos_std=1.0"
run () {  # name, delta_unseen, planner, overrides...
  name=$1; du=$2; pl=$3; shift 3
  python evaluate_nav.py $COMMON --delta-unseen $du --planners $pl --set unseen_joint=False "$@" --out results/hybrid/$name.csv > results/hybrid/$name.log 2>&1
  echo "done $name $(grep -c FAILED results/hybrid/$name.log) failures"
}
run s2_wpu4 0.02 supermpc-active w_perception_unseen=4.0
run s2_wpu8 0.02 supermpc-active w_perception_unseen=8.0
run s2_pus16 0.02 supermpc-active perception_unseen_slots=16
run s2_rl3 0.02 supermpc-active relevance_length=3.0
run s2_rl5 0.02 supermpc-active relevance_length=5.0
run s2_pp6 0.02 supermpc-active perception_peds=6
run s2_fwd02 0.02 supermpc-active w_cam_forward=0.2
run s2_fwd05 0.02 supermpc-active w_cam_forward=0.5
run s2_wp1 0.02 supermpc-active w_perception=1.0
run s2_wp4 0.02 supermpc-active w_perception=4.0
run s2_du01 0.01 supermpc-active
run s2_du005 0.005 supermpc-active
