#!/bin/bash
# Fill the gaps left by the Docker Desktop restart (completed episodes are skipped by run_batch.sh).
cd "c:/Users/James/Desktop/active-social-perception-main"
export TORCH_THREADS=3
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
run () { DELETE_SIM=1 bash socnav_bridge/run_batch.sh "$@" $EPS >> results/socnav/ablation_vis/progress.log 2>&1; echo "done $1"; }
run abl_vis_kf_tracker_s2 2256 "--vision --walls 8 --seed 2 --camera active --tracker kf" &
run abl_vis_no_state_cov_s3 2257 "--vision --walls 8 --seed 3 --camera active --no-state-cov" &
wait
# then the seed-3 Kalman batch, after its first pass (run_ablation_vis_rest.sh) has finished
while pgrep -f "run_batch.sh abl_vis_kf_tracker_s3" > /dev/null 2>&1 || ps -ef | grep -q "[r]un_batch.sh abl_vis_kf_tracker_s3"; do sleep 60; done
run abl_vis_kf_tracker_s3 2258 "--vision --walls 8 --seed 3 --camera active --tracker kf"
echo "all done"
