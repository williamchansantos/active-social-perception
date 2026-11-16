# Ablations of the proposed method with RGB-D vision on SocNavBench (written before any ablation results)

Setting: exactly the confirmation run's vision setting (results/socnav/confirm/PLAN.md): walls in the MPC, 60 deg /
4 m, PMB prune 1.0, loorot predictors, leakage-free prior, SocNavBench radii, the 30 curated episodes x seeds 1, 2, 3.
Reference: the full method's confirmation runs conf_vis_active_s{1,2,3} (same episodes and seeds).
Each ablation removes one component; nothing else changes; nothing is tuned. Same ablations as the 2D study
(results/ablation31/PLAN.md):

| name | server flags (added to --camera active --vision --walls 8) |
|---|---|
| open_constraint | --mpc-set unseen_obs_aware=False (open-loop unseen constraint) |
| no_unseen_constraint | --mpc-set unseen_constraint=False |
| no_unseen_term | --mpc-set w_perception_unseen=0.0 |
| no_tracked_term | --mpc-set w_perception=0.0 |
| fixed_camera | --camera fixed --mpc-set unseen_obs_aware=True (observation-aware constraint kept) |
| kf_tracker | --tracker kf (plain Kalman tracker: no existence, no undetected-pedestrian model/constraint/term) |
| prune25 | --pmb-set prune_pos_std=2.5 (stale-track pruning removed) |
| no_existence | --mpc-set existence_weighting=False |
| no_robot_sens | --mpc-set robot_sensitivity=False |
| no_state_cov | --no-state-cov |

Outcome: success excluding spawn (as the confirmation run's primary); also success with all collisions, collisions.
Test per ablation: full method vs ablation, two-sided exact McNemar over the 90 paired episode-seeds, Holm over
the 10 ablations; per-episode check (mean over seeds) reported alongside. Expectation from 2D: the PMB tracker,
active gaze and stale-track pruning matter most; 90 pairs detect only differences of roughly 15 points or more.
