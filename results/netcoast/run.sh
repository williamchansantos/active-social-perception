#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
NC="state_cov_mode=\"predictor\" coast_mode=\"network\""
C="--protocol clean --episodes 300 --workers 13 --fov 60 --range 4 --delta-unseen 0.02"
# 1. core comparison
for set in "zara01 zara02:31:zara" "eth hotel:23:eth"; do
  scenes=${set%%:*}; rest=${set#*:}; seed=${rest%%:*}; tag=${rest#*:}
  python evaluate_nav.py $C --scenes $scenes --seed $seed --pmb-set prune_pos_std=1.0 --set unseen_joint=False unseen_obs_aware=True $NC --planners supermpc-active --out results/netcoast/${tag}_obs.csv > results/netcoast/${tag}_obs.log 2>&1
  python evaluate_nav.py $C --scenes $scenes --seed $seed --pmb-set prune_pos_std=1.0 --set unseen_joint=False $NC --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle supermpc-360 --out results/netcoast/${tag}_open.csv > results/netcoast/${tag}_open.log 2>&1
  echo "done core $tag failures: $(cat results/netcoast/${tag}_*.log | grep -c FAILED)"
done
# 2. ablations (Zara, seed 31)
Z="$C --scenes zara01 zara02 --seed 31"
OBS="unseen_joint=False unseen_obs_aware=True $NC"
run () { name=$1; shift; python evaluate_nav.py $Z "$@" --out results/netcoast/abl_$name.csv > results/netcoast/abl_$name.log 2>&1; echo "done abl $name $(grep -c FAILED results/netcoast/abl_$name.log)"; }
run open_constraint      --pmb-set prune_pos_std=1.0 --set unseen_joint=False $NC --planners supermpc-active
run no_unseen_constraint --pmb-set prune_pos_std=1.0 --set $OBS unseen_constraint=False --planners supermpc-active
run no_unseen_term       --pmb-set prune_pos_std=1.0 --set $OBS --planners active-tracked-only
run no_tracked_term      --pmb-set prune_pos_std=1.0 --set $OBS --planners active-unseen-only
run fixed_camera         --pmb-set prune_pos_std=1.0 --set $OBS --planners supermpc
run kf_tracker           --pmb-set prune_pos_std=1.0 --set $OBS --planners active-kf
run prune25              --pmb-set prune_pos_std=2.5 --set $OBS --planners supermpc-active
run no_existence         --pmb-set prune_pos_std=1.0 --set $OBS existence_weighting=False --planners supermpc-active
run no_robot_sens        --pmb-set prune_pos_std=1.0 --set $OBS --planners active-no-sens
run no_state_cov         --pmb-set prune_pos_std=1.0 --set $OBS --planners active-no-statecov
