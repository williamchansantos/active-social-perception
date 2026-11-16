#!/bin/bash
# Finish results/socnav/ablation_vis after the server crashes: kf_tracker s2 (rest) + s3, no_state_cov s3 (rest).
cd "c:/Users/James/Desktop/active-social-perception-main"
export TORCH_THREADS=3
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
run () { DELETE_SIM=1 bash socnav_bridge/run_batch.sh "$@" $EPS >> results/socnav/ablation_vis/progress.log 2>&1; echo "done $1"; }
run abl_vis_kf_tracker_s2 2252 "--vision --walls 8 --seed 2 --camera active --tracker kf" &
run abl_vis_kf_tracker_s3 2253 "--vision --walls 8 --seed 3 --camera active --tracker kf" &
run abl_vis_no_state_cov_s3 2254 "--vision --walls 8 --seed 3 --camera active --no-state-cov" &
wait
echo "all done"
