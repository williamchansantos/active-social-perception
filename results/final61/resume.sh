#!/bin/bash
# Resume results/final61/run.sh after the controlling shell died (VS Code closed, 2026-10-13): wait until the
# orphaned workers of the running 2D evaluation have exited, then rerun run.sh; evaluate_nav.py and
# run_batch.sh skip everything already completed, so finished work is not repeated.
cd "c:/Users/James/Desktop/active-social-perception-main"
echo "$(date) resume: waiting for running python workers to exit" >> results/final61/run.log
while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -qi python; do sleep 60; done
sleep 30
echo "$(date) resume: starting run.sh" >> results/final61/run.log
bash results/final61/run.sh >> results/final61/run.log 2>&1
