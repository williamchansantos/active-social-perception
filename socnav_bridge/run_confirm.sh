#!/bin/bash
# Batches of results/socnav/confirm/PLAN.md, 10 in parallel (vision first).
cd "c:/Users/James/Desktop/active-social-perception-main"
export TORCH_THREADS=2
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
port=2160
jobs=""
for mode in vis gt; do for seed in 1 2 3; do for m in active infogain fixed; do
  port=$((port + 1)); v=""; [ $mode = vis ] && v="--vision"
  keep=0; [ $mode = vis ] && [ $m = active ] && [ $seed = 1 ] && keep=1   # keep sim states of one batch (videos)
  jobs="$jobs$mode $m $seed $port $keep $v\n"
done; done; done
printf "$jobs" | xargs -P 10 -L 1 bash -c 'DELETE_SIM=$((1 - $4)) bash socnav_bridge/run_batch.sh conf_$0_$1_s$2 $3 "--camera $1 --walls 8 --seed $2 $5" '"$EPS"' >> results/socnav/confirm/progress.log 2>&1; echo "done conf_$0_$1_s$2"'
echo "all done"
