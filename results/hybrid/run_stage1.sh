#!/bin/bash
# Phase B stage 1: hybrid objective grid on the tuning set (seed 11), with the stale-track setting from phase A.
# usage: bash results/hybrid/run_stage1.sh prune_pos_std=1.0
cd "c:/Users/James/Desktop/active-social-perception-main"
STALE=$1
COMMON="--protocol clean --scenes zara01 zara02 --episodes 100 --seed 11 --workers 13 --fov 100 --range 4 --delta-unseen 0.02 --planners supermpc-hybrid --pmb-set $STALE"
for form in sqrt log; do
 for we in 0.5 1.0 2.0 4.0; do
  for uf in 0.0 0.5 1.0; do
   name="s1_${form}_we${we}_uf${uf}"
   python evaluate_nav.py $COMMON --set unseen_joint=False risk_form=\"$form\" w_entropy=$we unseen_floor=$uf --out results/hybrid/$name.csv > results/hybrid/$name.log 2>&1
   echo "done $name $(grep -c FAILED results/hybrid/$name.log) failures"
  done
 done
done
