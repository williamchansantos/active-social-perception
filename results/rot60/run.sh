#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
until [ $(ls checkpoints/loorot_*.pt 2>/dev/null | wc -l) -eq 5 ] && ! tasklist //FI "IMAGENAME eq python.exe" //V 2>/dev/null | grep -q "main.py"; do sleep 60; done
sleep 30
python check_predictor_rotation.py loo loorot 2>&1 | grep -v Warning > results/rot60/predictor_quality.txt
export LOO_PREFIX=loorot
C="--protocol clean --episodes 300 --workers 13 --fov 60 --range 4 --delta-unseen 0.02 --pmb-set prune_pos_std=1.0"
for set in "zara01 zara02:31:zara" "eth hotel:23:eth"; do
  scenes=${set%%:*}; rest=${set#*:}; seed=${rest%%:*}; tag=${rest#*:}
  python evaluate_nav.py $C --scenes $scenes --seed $seed --set unseen_joint=False unseen_obs_aware=True --planners supermpc-active --out results/rot60/${tag}_obs.csv > results/rot60/${tag}_obs.log 2>&1
  python evaluate_nav.py $C --scenes $scenes --seed $seed --set unseen_joint=False --planners supermpc-infogain supermpc-nearest supermpc supermpc-gazeoracle supermpc-360 --out results/rot60/${tag}_open.csv > results/rot60/${tag}_open.log 2>&1
  echo "done $tag failures: $(cat results/rot60/${tag}_obs.log results/rot60/${tag}_open.log | grep -c FAILED)"
done
