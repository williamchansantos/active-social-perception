# 60 deg comparisons with rotation-augmented predictors (written before any loorot results existed)

Why: the leave-one-scene-out Social LSTMs take global-frame displacements and are not rotation-invariant. ETH's
pedestrians walk along the other map axis from every training scene, so the ETH predictor's error doubled
(ADE 0.94 m; 0.47 m when ETH is rotated 90 deg). New predictors (checkpoints/loorot_<scene>.pt) are trained the same
way plus random-rotation augmentation (main.py --rotate-aug). Nothing else changes and nothing is tuned.

- Zara: seed 31, 600 episodes (the pre-registered 60 deg episodes). ETH + Hotel: seed 23, 600 episodes.
- 60 deg FOV, range 4 m, delta_unseen 0.02, prune_pos_std 1.0, N = 8, clean protocol, LOO_PREFIX=loorot.
- Planners: proposed (supermpc-active, unseen_obs_aware=True); supermpc-infogain, supermpc-nearest, supermpc,
  supermpc-gazeoracle, supermpc-360 (open-loop unseen constraint, as before). All are rerun: every planner uses the predictor.
- Primary hypotheses, per dataset (success, exact McNemar, Holm over the two per dataset):
  proposed > information gain; proposed > look-at-nearest.
- Secondary: never-seen collisions; per scene; change from the old predictors (results/narrow31, results/eth60).
