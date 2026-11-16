#!/bin/bash
# results/final53/PLAN.md: A (2D, seed 53), then B (SocNavBench seeds 4-6).
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
C="--protocol clean --episodes 300 --workers 13 --fov 60 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for set in "zara01 zara02:zara" "eth hotel:eth"; do
  scenes=${set%%:*}; tag=${set#*:}
  python evaluate_nav.py $C --scenes $scenes --seed 53 --set unseen_joint=False \
    --planners supermpc-active supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle supermpc-360 \
    --out results/final53/${tag}.csv > results/final53/${tag}.log 2>&1
  echo "done 2D $tag failures: $(grep -c FAILED results/final53/${tag}.log)"
done
# B: SocNavBench, 9 vision batches in parallel plus 3 ground-truth lanes
export TORCH_THREADS=2
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
args () { case $1 in active) echo "--camera active --mpc-set unseen_obs_aware=False";; *) echo "--camera $1";; esac; }
port=2300
for seed in 4 5 6; do for m in active infogain fixed; do
  port=$((port + 1))
  DELETE_SIM=1 bash socnav_bridge/run_batch.sh fin_vis_${m}_s$seed $port "--vision --walls 8 --seed $seed $(args $m)" $EPS \
    >> results/final53/socnav_progress.log 2>&1 &
done; done
gtlane () { m=$1; base=$2
  for seed in 4 5 6; do
    DELETE_SIM=1 bash socnav_bridge/run_batch.sh fin_gt_${m}_s$seed $((base + seed)) "--walls 8 --seed $seed $(args $m)" $EPS \
      >> results/final53/socnav_progress.log 2>&1
  done; }
gtlane active 2320 & gtlane infogain 2330 & gtlane fixed 2340 &
wait
echo "all done"
