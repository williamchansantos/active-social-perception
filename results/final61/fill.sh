#!/bin/bash
# After the Docker Desktop outage (2026-10-13 morning): wait for the current run (resume.sh, bash PID 23516) to
# finish, then rerun run.sh once more to fill the SocNavBench episodes that failed during the outage
# (completed episodes and 2D runs are skipped). No change to the design.
cd "c:/Users/James/Desktop/active-social-perception-main"
while tasklist //FI "PID eq 23516" 2>/dev/null | grep -qi bash; do sleep 120; done
echo "$(date) fill: rerunning run.sh to fill episodes that failed during the Docker outage" >> results/final61/run.log
bash results/final61/run.sh >> results/final61/run.log 2>&1
