# Ablations of the proposed method (written before any ablation results existed)

Setting: 60 deg FOV, range 4 m, seed 31 (the 600 confirmation episodes), clean protocol, delta_unseen 0.02,
prune_pos_std 1.0, N = 8. The full method's results are results/narrow31/obs_fov60.csv (success 0.663).
Each ablation removes one component; everything else is identical. Paired comparison with the full method
on success (exact McNemar), plus collisions with never-seen / tracked pedestrians. No parameter is tuned.

| name | change |
|---|---|
| open_constraint | unseen_obs_aware=False (open-loop unseen constraint; the previous method) |
| no_unseen_constraint | unseen_constraint=False |
| no_unseen_term | w_perception_unseen=0 |
| no_tracked_term | w_perception=0 |
| fixed_camera | camera_mode fixed (supermpc), observation-aware constraint kept |
| kf_tracker | plain Kalman tracker (no PMB: no unseen model, constraint or term) |
| prune25 | prune_pos_std=2.5 (stale-track fix removed) |
| no_existence | existence_weighting=False |
| no_robot_sens | robot_sensitivity=False |
| no_state_cov | CV state covariance dropped from predictions |
