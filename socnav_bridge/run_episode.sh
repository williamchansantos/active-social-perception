#!/bin/bash
# Run one SocNavBench test episode against a SupeR-MPC server already listening on the host.
#   bash socnav_bridge/run_episode.sh [test_script=tests/test_socnav.py]
#   VISION=1 bash socnav_bridge/run_episode.sh     # the joystick renders RGB-D for the server's --vision mode
TEST=${1:-tests/test_socnav.py}
VISION=${VISION:-0}
MSYS_NO_PATHCONV=1 docker run --rm -e SUPERMPC_HOST=host.docker.internal -e SUPERMPC_VISION=$VISION \
  -e EGL_PLATFORM=surfaceless -v "C:/Users/James/Desktop/SocNavBench:/socnavbench" socnavbench:patched bash -c "
cd /socnavbench && cp params/user_params.ini /tmp/user_params.bak && trap 'cp /tmp/user_params.bak params/user_params.ini' EXIT
sed -i 's/^use_system_dynamics=False/use_system_dynamics=True/' params/user_params.ini
(PYTHONPATH=. timeout 1500 python3 -u $TEST > /tmp/sim.log 2>&1 &); sleep 10
PYTHONPATH=. timeout 1480 python3 -u joystick/joystick_client.py --algo supermpc > /tmp/joy.log 2>&1
tail -4 /tmp/joy.log; grep -i 'terminat\|powering\|success\|collision' /tmp/sim.log | tail -5"
