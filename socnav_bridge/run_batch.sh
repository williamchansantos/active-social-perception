#!/bin/bash
# Run curated SocNavBench episodes against one persistent SupeR-MPC server, one fresh container (simulator +
# joystick) per episode: SocNavBench's multi-episode handover can deadlock, and the joystick's renderer cannot
# switch maps within a process. Episodes that already have results are skipped; a failed episode is retried once.
#   bash socnav_bridge/run_batch.sh TAG PORT "SERVER ARGS" EPISODE...
#   e.g. bash socnav_bridge/run_batch.sh gt_active 2111 "--camera active" t_univ1 t_zara2
# Server args containing --vision make the joystick render RGB-D. Results: results/socnav/TAG/
#   <episode>_<camera>[_vision].json (server log), <episode>.episode_log.txt (SocNavBench), server.log,
#   <episode>.sim.log / .joy.log (tails). Each episode gets its own copy of SocNavBench/params (episode file,
#   use_system_dynamics=True, no movie), mounted over the original, so batches can run in parallel.
set -u
TAG=$1; PORT=$2; SARGS=$3; shift 3
REPO="c:/Users/James/Desktop/active-social-perception-main"
SNB="C:/Users/James/Desktop/SocNavBench"
OUT="$REPO/results/socnav/$TAG"
mkdir -p "$OUT"
cd "$REPO"
VISION=0; [[ "$SARGS" == *--vision* ]] && VISION=1
LIMIT=$(( VISION ? 2400 : 900 ))  # s per episode
start_server () {
  LOO_PREFIX=loorot python -u -m socnav_bridge.server --port $PORT --log-dir "$OUT" --persistent $SARGS \
    >> "$OUT/server.log" 2>&1 &
  SERVER=$!
  sleep 12
}
start_server
done_ep () { ls "$OUT/$1"_*.json > /dev/null 2>&1 && [ -f "$OUT/$1.episode_log.txt" ]; }
for e in "$@"; do
  for attempt in 1 2; do
    done_ep "$e" && break
    for w in $(seq 1 60); do docker info > /dev/null 2>&1 && break; echo "$TAG: Docker not available, waiting" >> "$OUT/server.log"; sleep 30; done
    if ! kill -0 $SERVER 2>/dev/null; then  # the server can die without a traceback (native crash): restart it
      echo "$TAG: server died before $e (attempt $attempt), restarting" | tee -a "$OUT/server.log"
      start_server
    fi
    rm -f "$OUT/$e"_*.json
    PARAMS="$SNB/ours/params_${TAG}_$e"
    rm -rf "$PARAMS"; cp -r "$SNB/params" "$PARAMS"
    sed -i -e 's/^use_system_dynamics=False/use_system_dynamics=True/' -e 's/^render_movie=True/render_movie=False/' \
        "$PARAMS/user_params.ini"
    python -c "import sys; from socnav_bridge.episodes import write_ini; write_ini(sys.argv[1:], '$PARAMS/episodes.ini')" "$e"
    MSYS_NO_PATHCONV=1 docker run --rm -e SUPERMPC_HOST=host.docker.internal -e SUPERMPC_PORT=$PORT \
      -e SUPERMPC_VISION=$VISION -e EGL_PLATFORM=surfaceless -e SOCNAV_EPISODES=params/episodes.ini \
      -e SUPERMPC_ALGO=supermpc_$TAG -v "$SNB:/socnavbench" -v "$PARAMS:/socnavbench/params" socnavbench:patched bash -c "
cd /socnavbench
(PYTHONPATH=. timeout $LIMIT python3 -u tests/test_episodes.py > /tmp/sim.log 2>&1 &); sleep 10
PYTHONPATH=. timeout $LIMIT python3 -u joystick/joystick_client.py --algo supermpc > /tmp/joy.log 2>&1
sleep 3; grep -v 'Frames:' /tmp/sim.log | tail -30 > /socnavbench/params/sim_tail.log; tail -30 /tmp/joy.log > /socnavbench/params/joy_tail.log" \
      > /dev/null 2>&1
    cp "$PARAMS/sim_tail.log" "$OUT/$e.sim.log" 2>/dev/null; cp "$PARAMS/joy_tail.log" "$OUT/$e.joy.log" 2>/dev/null
    cp "$SNB/tests/socnav/test_supermpc_$TAG/$e/episode_log.txt" "$OUT/$e.episode_log.txt" 2>/dev/null
    rm -rf "$PARAMS"
    SIMD="$SNB/tests/socnav/test_supermpc_$TAG/$e/sim_state_data"
    if [ -d "$SIMD" ] && [ -f "$OUT/$e.episode_log.txt" ]; then
      python -c "import sys; from socnav_bridge.check_spawns import summarize_dir; summarize_dir(sys.argv[1], sys.argv[2])"         "$SIMD" "$OUT/$e.collisions.json"
      [ "${DELETE_SIM:-0}" = "1" ] && rm -rf "$SIMD"
    fi
    done_ep "$e" && echo "$TAG $e ok (attempt $attempt)" || echo "$TAG $e FAILED (attempt $attempt)"
  done
done
kill $SERVER 2>/dev/null
echo "batch $TAG done: $(ls "$OUT"/*.episode_log.txt 2>/dev/null | wc -l) episodes with logs"
