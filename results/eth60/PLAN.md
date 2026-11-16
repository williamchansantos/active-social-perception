# 60 deg replication on ETH and Hotel (written before any 60 deg ETH/Hotel results existed)

The 60 deg advantage was found on Zara (seed 23) and confirmed on fresh Zara episodes (seed 31). This tests whether it
transfers to the two ETH-dataset scenes.

- Episodes: seed 23, ETH + Hotel, 300 each (600 paired) -- the same episodes as the 100 deg run in results/eth23,
  on which nothing was tuned. Clean protocol, leave-one-scene-out predictors, range 4 m, delta_unseen 0.02,
  prune_pos_std 1.0, N = 8.
- Planners: proposed (supermpc-active, unseen_obs_aware=True); supermpc-infogain, supermpc-nearest, supermpc (fixed),
  supermpc-gazeoracle (open-loop unseen constraint, as in all earlier baselines). The 360 camera from results/eth23
  (independent of the FOV setting).
- Primary hypotheses (success, pooled over both scenes, exact McNemar, Holm over the two): proposed > information gain;
  proposed > look-at-nearest.
- Secondary: per scene; episodes with never-seen collisions; proposed vs fixed and vs gaze oracle; 60 vs 100 deg.
- No parameter is changed after seeing these results.
