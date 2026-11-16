#!/bin/bash
cd "c:/Users/James/Desktop/active-social-perception-main"
until grep -qs "Saved" results/eth23/open.log; do sleep 60; done
python diagnose_origins.py --scenes eth hotel --episodes 150 --seed 23 2>&1 | grep -v Warning > results/eth23/collision_origins.txt
echo done origins
