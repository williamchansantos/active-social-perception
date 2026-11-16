#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
D=results/lever12
C="--protocol clean --episodes 300 --workers 13 --fov 60 --range 4 --scenes zara01 zara02 eth hotel --seed 11 --pmb-set prune_pos_std=1.0 --planners supermpc-active supermpc"
python evaluate_nav.py $C --delta-unseen 0.02 --set unseen_joint=False --out $D/base.csv > $D/base.log 2>&1; echo "done base $(grep -c FAILED $D/base.log)"
python evaluate_nav.py $C --delta-unseen 0.01 --set unseen_joint=False --out $D/du01.csv > $D/du01.log 2>&1; echo "done du01 $(grep -c FAILED $D/du01.log)"
python evaluate_nav.py $C --delta-unseen 0.02 --set unseen_joint=False state_cov_mode=\"predictor\" --out $D/gcov.csv > $D/gcov.log 2>&1; echo "done gcov $(grep -c FAILED $D/gcov.log)"
python evaluate_nav.py $C --delta-unseen 0.01 --set unseen_joint=False state_cov_mode=\"predictor\" --out $D/both.csv > $D/both.log 2>&1; echo "done both $(grep -c FAILED $D/both.log)"
python $D/analyze.py > $D/summary.txt 2>&1
echo "all done"
