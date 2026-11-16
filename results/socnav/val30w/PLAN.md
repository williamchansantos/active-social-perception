# SocNavBench curated episodes with walls in the MPC (written before any results)

As results/socnav/val30/PLAN.md, with one change: static obstacles are MPC constraints (server --walls 8):
per horizon step, the 8 obstacle-boundary points nearest the warm-start position within 2 m give half-plane
constraints n^T (p - o) >= 0.30 m + kappa * robot position sigma (n: towards the warm-start position, or the
free-space distance gradient), softened with slack weight 1e4; the waypoint is the farthest point of the geodesic
path within 2.5 m that is in line of sight. Developed on 5 val30 episodes that ended in obstacle collisions
(t_zara1_wrap, t_univ_trapTL, t_univ_obstacle, t_eth_dense_flow, t_zara2_wrap; ground-truth observation,
proposed method): all 5 reached the goal afterwards. Nothing else changed or tuned.

Same 30 episodes, methods, perception modes, outcome (success = Success and 0 pedestrian collisions) and
comparisons (exact McNemar, per perception mode: proposed vs information gain two-sided, vs nearest, vs fixed;
gt vs vision per method). val30 (no walls) is kept as the baseline; also reported: change in obstacle collisions.
