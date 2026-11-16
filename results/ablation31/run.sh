#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
C="--protocol clean --scenes zara01 zara02 --episodes 300 --seed 31 --workers 13 --fov 60 --range 4 --delta-unseen 0.02"
OBS="unseen_joint=False unseen_obs_aware=True"
run () { name=$1; shift; python evaluate_nav.py $C "$@" --out results/ablation31/$name.csv > results/ablation31/$name.log 2>&1; echo "done $name $(grep -c FAILED results/ablation31/$name.log)"; }
run open_constraint      --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners supermpc-active
run no_unseen_constraint --pmb-set prune_pos_std=1.0 --set $OBS unseen_constraint=False --planners supermpc-active
run no_unseen_term       --pmb-set prune_pos_std=1.0 --set $OBS --planners active-tracked-only
run no_tracked_term      --pmb-set prune_pos_std=1.0 --set $OBS --planners active-unseen-only
run fixed_camera         --pmb-set prune_pos_std=1.0 --set $OBS --planners supermpc
run kf_tracker           --pmb-set prune_pos_std=1.0 --set $OBS --planners active-kf
run prune25              --pmb-set prune_pos_std=2.5 --set $OBS --planners supermpc-active
run no_existence         --pmb-set prune_pos_std=1.0 --set $OBS existence_weighting=False --planners supermpc-active
run no_robot_sens        --pmb-set prune_pos_std=1.0 --set $OBS --planners active-no-sens
run no_state_cov         --pmb-set prune_pos_std=1.0 --set $OBS --planners active-no-statecov
