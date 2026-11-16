# Rerun with network coasting (written before any network-coasting results existed)

Change (user decision): a pedestrian out of view is predicted by the network's rollout from its last real history,
shifted to now, with the network's predicted spread plus the last-detection covariance propagated through the
network's sensitivity (MPCConfig coast_mode="network", state_cov_mode="predictor"). Predictors: rotation-augmented
(LOO_PREFIX=loorot). Everything else as in results/rot60: 60 deg FOV, 4 m range, delta_unseen 0.02,
prune_pos_std 1.0, N = 8, clean protocol, nothing tuned.

1. Core comparison: Zara seed 31 (600) and ETH + Hotel seed 23 (600). Planners: proposed (supermpc-active,
   unseen_obs_aware=True); supermpc-infogain, supermpc-nearest, supermpc, supermpc-gazeoracle, supermpc-360
   (open-loop unseen constraint, as before).
   Primary, per dataset (success, exact McNemar, Holm over the two): proposed vs information gain (two-sided:
   the earlier conclusion is that they do not differ); proposed > look-at-nearest.
   Secondary: proposed > fixed; never-seen / tracked collisions; change from results/rot60 (CV coasting).
2. Ablations: the 10 of results/ablation31, on Zara seed 31 with these settings, each paired with the full method.
