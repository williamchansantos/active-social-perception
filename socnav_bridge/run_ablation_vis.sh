#!/bin/bash
# Batches of results/socnav/ablation_vis/PLAN.md: one lane per ablation (seeds 1-3 in sequence), 10 lanes in parallel.
cd "c:/Users/James/Desktop/active-social-perception-main"
export TORCH_THREADS=2
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
lane () {
  name=$1; base=$2; shift 2
  for seed in 1 2 3; do
    DELETE_SIM=1 bash socnav_bridge/run_batch.sh abl_vis_${name}_s$seed $((base + seed)) \
      "--vision --walls 8 --seed $seed $*" $EPS >> results/socnav/ablation_vis/progress.log 2>&1
    echo "done abl_vis_${name}_s$seed"
  done
}
lane open_constraint      2200 --camera active --mpc-set unseen_obs_aware=False &
lane no_unseen_constraint 2205 --camera active --mpc-set unseen_constraint=False &
lane no_unseen_term       2210 --camera active --mpc-set w_perception_unseen=0.0 &
lane no_tracked_term      2215 --camera active --mpc-set w_perception=0.0 &
lane fixed_camera         2220 --camera fixed --mpc-set unseen_obs_aware=True &
lane kf_tracker           2225 --camera active --tracker kf &
lane prune25              2230 --camera active --pmb-set prune_pos_std=2.5 &
lane no_existence         2235 --camera active --mpc-set existence_weighting=False &
lane no_robot_sens        2240 --camera active --mpc-set robot_sensitivity=False &
lane no_state_cov         2245 --camera active --no-state-cov &
wait
echo "all done"
