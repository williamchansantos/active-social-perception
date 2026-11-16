#!/bin/bash
# All batches of results/socnav/val30w/PLAN.md, in parallel.
cd "c:/Users/James/Desktop/active-social-perception-main"
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
run () { bash socnav_bridge/run_batch.sh "$@" >> results/socnav/val30w/progress.log 2>&1; echo "done $1"; }
export TORCH_THREADS=2
run val30w_gt_active   2141 "--camera active --walls 8"           $EPS &
run val30w_gt_infogain 2142 "--camera infogain --walls 8"         $EPS &
run val30w_gt_nearest  2143 "--camera nearest --walls 8"          $EPS &
run val30w_gt_fixed    2144 "--camera fixed --walls 8"            $EPS &
run val30w_gt_360      2145 "--camera fixed --fov 360 --walls 8"  $EPS &
run val30w_vis_active   2146 "--camera active --vision --walls 8 --save-frames"   $EPS &
run val30w_vis_infogain 2147 "--camera infogain --vision --walls 8" $EPS &
run val30w_vis_nearest  2148 "--camera nearest --vision --walls 8"  $EPS &
run val30w_vis_fixed    2149 "--camera fixed --vision --walls 8"    $EPS &
wait
echo "all done"
