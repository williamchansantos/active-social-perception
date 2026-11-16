#!/bin/bash
# results/final61/PLAN.md: part A (2D, seed 61), automatic selection of the 3 SocNavBench alternatives, part B.
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
D=results/final61
C="--protocol clean --episodes 450 --workers 13 --fov 60 --range 4 --delta-unseen 0.02 --scenes zara01 zara02 eth hotel --seed 61"
ev () { out=$1; shift; python evaluate_nav.py $C "$@" --out $D/$out.csv > $D/$out.log 2>&1; echo "done A $out failures: $(grep -c FAILED $D/$out.log)"; }
ev main --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners supermpc-active supermpc-infogain supermpc-ttc \
   supermpc-nearest supermpc-decoupled supermpc-scan supermpc supermpc-gazeoracle supermpc-360
ev abl_no_unseen_constraint --pmb-set prune_pos_std=1.0 --set unseen_joint=False unseen_constraint=False --planners supermpc-active
ev abl_no_unseen_term       --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners active-tracked-only
ev abl_no_tracked_term      --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners active-unseen-only
ev abl_kf_tracker           --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners active-kf
ev abl_prune25              --pmb-set prune_pos_std=2.5 --set unseen_joint=False --planners supermpc-active
ev abl_no_existence         --pmb-set prune_pos_std=1.0 --set unseen_joint=False existence_weighting=False --planners supermpc-active
ev abl_no_robot_sens        --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners active-no-sens
ev abl_no_state_cov         --pmb-set prune_pos_std=1.0 --set unseen_joint=False --planners active-no-statecov
python $D/analyze.py A > $D/summary_A.txt 2>&1
echo "selected for SocNavBench: $(cat $D/socnav_methods.txt)"
# part B: 10 lanes (one per seed), each runs the proposed method and the 3 selected alternatives
export TORCH_THREADS=2
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
METHODS="active $(cat $D/socnav_methods.txt)"
args () { case $1 in active) echo "--camera active --mpc-set unseen_obs_aware=False";; *) echo "--camera $1";; esac; }
lane () { seed=$1; base=$2; i=0
  for m in $METHODS; do i=$((i + 1))
    DELETE_SIM=1 bash socnav_bridge/run_batch.sh f61_vis_${m}_s$seed $((base + i)) "--vision --walls 8 --seed $seed $(args $m)" $EPS \
      >> $D/socnav_progress.log 2>&1
  done; echo "done B seed $seed"; }
for seed in 7 8 9 10 11 12 13 14 15 16; do lane $seed $((2400 + 10 * seed)) & done
wait
python $D/analyze.py B > $D/summary_B.txt 2>&1
echo "all done"
