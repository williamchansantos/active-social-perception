#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
export LOO_PREFIX=loorot
NC="state_cov_mode=\"cv\" coast_mode=\"network\""
C="--protocol clean --episodes 300 --workers 13 --fov 60 --range 4 --delta-unseen 0.02"
# 1. core comparison
for set in "zara01 zara02:31:zara" "eth hotel:23:eth"; do
  scenes=${set%%:*}; rest=${set#*:}; seed=${rest%%:*}; tag=${rest#*:}
  python evaluate_nav.py $C --scenes $scenes --seed $seed --pmb-set prune_pos_std=1.0 --set unseen_joint=False unseen_obs_aware=True $NC --planners supermpc-active --out results/netcoast_cv/${tag}_obs.csv > results/netcoast_cv/${tag}_obs.log 2>&1
  python evaluate_nav.py $C --scenes $scenes --seed $seed --pmb-set prune_pos_std=1.0 --set unseen_joint=False $NC --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle supermpc-360 --out results/netcoast_cv/${tag}_open.csv > results/netcoast_cv/${tag}_open.log 2>&1
  echo "done core $tag failures: $(cat results/netcoast_cv/${tag}_*.log | grep -c FAILED)"
done
