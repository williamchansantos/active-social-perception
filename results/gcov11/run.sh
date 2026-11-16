#!/bin/bash
# Tuning set (seed 11): predictor-sensitivity state covariance vs CV, 60 deg, rotation-augmented predictors.
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
C="--protocol clean --episodes 300 --seed 11 --workers 13 --fov 60 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for set in "zara01 zara02:zara" "eth hotel:eth"; do
  scenes=${set%%:*}; tag=${set#*:}
  for mode in predictor cv; do
    python evaluate_nav.py $C --scenes $scenes --set unseen_joint=False unseen_obs_aware=True state_cov_mode=\"$mode\" --planners supermpc-active --out results/gcov11/${tag}_${mode}_obs.csv > results/gcov11/${tag}_${mode}_obs.log 2>&1
    if [ $mode = predictor ]; then pl="supermpc-infogain supermpc-nearest supermpc"; else pl="supermpc-infogain"; fi
    python evaluate_nav.py $C --scenes $scenes --set unseen_joint=False state_cov_mode=\"$mode\" --planners $pl --out results/gcov11/${tag}_${mode}_open.csv > results/gcov11/${tag}_${mode}_open.log 2>&1
    echo "done $tag $mode failures: $(cat results/gcov11/${tag}_${mode}_*.log | grep -c FAILED)"
  done
done
