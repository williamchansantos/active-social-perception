#!/bin/bash
# All batches of results/socnav/val30/PLAN.md, in parallel.
cd "c:/Users/James/Desktop/active-social-perception-main"
EPS=$(python -c "from socnav_bridge.episodes import EPISODES; print(' '.join(EPISODES))")
run () { bash socnav_bridge/run_batch.sh "$@" >> results/socnav/val30/progress.log 2>&1; echo "done $1"; }
export TORCH_THREADS=2
run val30_gt_active   2121 "--camera active"           $EPS &
run val30_gt_infogain 2122 "--camera infogain"         $EPS &
run val30_gt_nearest  2123 "--camera nearest"          $EPS &
run val30_gt_fixed    2124 "--camera fixed"            $EPS &
run val30_gt_360      2125 "--camera fixed --fov 360"  $EPS &
run val30_vis_active   2126 "--camera active --vision"   $EPS &
run val30_vis_infogain 2127 "--camera infogain --vision" $EPS &
run val30_vis_nearest  2128 "--camera nearest --vision"  $EPS &
run val30_vis_fixed    2129 "--camera fixed --vision"    $EPS &
wait
echo "all done"
