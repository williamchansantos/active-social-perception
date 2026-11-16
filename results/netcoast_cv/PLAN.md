# Network coasting with constant-velocity uncertainty propagation (written before any results)

Change (user's intended version): a pedestrian out of view is predicted by the network's rollout from its last
real history, shifted to now; its uncertainty is the network's predicted spread plus the last-detection covariance
propagated with the constant-velocity Jacobian J_(m+k) P_det J_(m+k)^T, as in planning
(coast_mode="network", state_cov_mode="cv"). Everything else as in results/netcoast and results/rot60: loorot
predictors, 60 deg, 4 m, delta_unseen 0.02, prune_pos_std 1.0, N = 8, clean protocol, nothing tuned.

Core comparison only (no ablations): Zara seed 31 (600) and ETH + Hotel seed 23 (600); proposed (obs-aware),
infogain, nearest, fixed, gaze oracle, 360.
Primary, per dataset (success, exact McNemar, Holm over the two): proposed vs infogain (two-sided); proposed > nearest.
Secondary: proposed > fixed; never-seen / tracked collisions; change vs results/rot60 (CV coasting) and
results/netcoast (G propagation).
